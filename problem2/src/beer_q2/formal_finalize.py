"""Finalize formal Question 2 candidates without relaxing locked gates.

The formal synthesis combines 7x3-fold feature permutation, TreeSHAP,
stable-community joint permutation, and style-direction checks. Network
de-redundancy is applied only after candidates pass the declared gates.
Attachment 6/8 are used afterward as same-source audits, never for ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from .source_audit import build_source_consistency


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_stable_network_nodes(
    project_root: Path, paths: Sequence[str], variant: str
) -> pd.DataFrame:
    """Load one complete stable-network assignment for every frozen fold."""
    frames = []
    for value in paths:
        frame = pd.read_csv(_resolve(project_root, value))
        if "variant" in frame:
            frame = frame[frame["variant"] == variant].copy()
        frames.append(frame)
    nodes = pd.concat(frames, ignore_index=True)
    if nodes.duplicated(["repeat", "fold", "feature"]).any():
        raise ValueError("Duplicate repeat/fold/feature network assignments")
    fold_sizes = nodes.groupby(["repeat", "fold"]).size()
    if len(fold_sizes) != 21 or not fold_sizes.eq(231).all():
        raise ValueError("Expected 21 complete stable-network folds with 231 nodes")
    return nodes.sort_values(["repeat", "fold", "feature"]).reset_index(drop=True)


def assemble_formal_candidate_evidence(
    permutation: pd.DataFrame,
    shap_stability: pd.DataFrame,
    group_importance: pd.DataFrame,
    style_heterogeneity: pd.DataFrame,
    nodes: pd.DataFrame,
    shap_top10_minimum: float = 0.60,
    group_positive_fold_minimum: float = 2 / 3,
    style_reversal_maximum: float = 0.10,
) -> pd.DataFrame:
    """Combine the four locked evidence layers for every target-feature pair."""
    primary = permutation[permutation["mode"] == "within_style"].copy()
    group_map = nodes[["repeat", "fold", "feature", "community"]].merge(
        group_importance[
            [
                "target",
                "repeat",
                "fold",
                "community",
                "community_size",
                "mean_delta_rmse",
                "positive_frequency",
            ]
        ],
        on=["repeat", "fold", "community"],
        how="inner",
        validate="many_to_many",
    )
    group_support = (
        group_map.groupby(["target", "feature"], as_index=False)
        .agg(
            mean_group_delta_rmse=("mean_delta_rmse", "mean"),
            group_positive_fold_frequency=(
                "mean_delta_rmse", lambda values: float((values > 0).mean())
            ),
            group_mean_positive_permutation_frequency=("positive_frequency", "mean"),
            mean_community_size=("community_size", "mean"),
        )
    )
    style_support = (
        style_heterogeneity.groupby(["target", "feature"], as_index=False)
        .agg(
            style_tests=("direction_reversal", "size"),
            style_reversals=("direction_reversal", "sum"),
        )
    )
    style_support["style_reversal_rate"] = (
        style_support["style_reversals"] / style_support["style_tests"]
    )
    evidence = (
        primary.merge(
            shap_stability,
            on=["target", "grade", "source_model", "feature"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            group_support,
            on=["target", "feature"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            style_support,
            on=["target", "feature"],
            how="left",
            validate="one_to_one",
        )
    )
    stable = evidence["passes_pilot_stability"]
    if evidence.loc[stable, "style_tests"].isna().any():
        raise ValueError("A stable feature is missing the formal style audit")
    evidence["shap_support"] = evidence["top10_shap_frequency"].ge(
        float(shap_top10_minimum)
    )
    evidence["group_support"] = evidence["mean_group_delta_rmse"].gt(0) & evidence[
        "group_positive_fold_frequency"
    ].ge(float(group_positive_fold_minimum))
    evidence["style_direction_ok"] = evidence["style_reversal_rate"].le(
        float(style_reversal_maximum)
    )
    evidence["formal_evidence_complete"] = (
        evidence["passes_pilot_stability"]
        & evidence["shap_support"]
        & evidence["group_support"]
        & evidence["style_direction_ok"]
    )
    evidence["interpretation_ceiling"] = evidence["grade"].map(
        {"A": "high", "B": "medium", "C": "exploratory"}
    )
    return evidence.sort_values(
        ["target", "formal_evidence_complete", "mean_delta_rmse"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def method_agreement_by_fold(
    feature_permutation_fold: pd.DataFrame,
    shap_fold: pd.DataFrame,
    group_importance: pd.DataFrame,
    nodes: pd.DataFrame,
    top_ks: Iterable[int] = (3, 5, 10),
) -> pd.DataFrame:
    """Compare feature- and fold-community agreement at reporting depths."""
    primary = feature_permutation_fold[
        feature_permutation_fold["mode"] == "within_style"
    ]
    shap_with_modules = shap_fold.merge(
        nodes[["repeat", "fold", "feature", "community"]],
        on=["repeat", "fold", "feature"],
        how="left",
        validate="many_to_one",
    )
    if shap_with_modules["community"].isna().any():
        raise ValueError("Missing community assignments in formal SHAP folds")
    module_shap = (
        shap_with_modules.groupby(
            ["target", "grade", "repeat", "fold", "community"], as_index=False
        )["mean_abs_shap"].sum()
    )
    rows: List[Dict[str, Any]] = []
    keys = shap_fold[["target", "grade", "repeat", "fold"]].drop_duplicates()
    for target, grade, repeat, fold in keys.itertuples(index=False, name=None):
        selector = (
            (shap_fold["target"] == target)
            & (shap_fold["repeat"] == repeat)
            & (shap_fold["fold"] == fold)
        )
        sf = shap_fold[selector]
        pf = primary[
            (primary["target"] == target)
            & (primary["repeat"] == repeat)
            & (primary["fold"] == fold)
        ]
        sm = module_shap[
            (module_shap["target"] == target)
            & (module_shap["repeat"] == repeat)
            & (module_shap["fold"] == fold)
        ]
        pm = group_importance[
            (group_importance["target"] == target)
            & (group_importance["repeat"] == repeat)
            & (group_importance["fold"] == fold)
        ]
        for top_k in top_ks:
            feature_value = _jaccard(
                set(sf.nlargest(top_k, "mean_abs_shap")["feature"]),
                set(pf.nlargest(top_k, "mean_delta_rmse")["feature"]),
            )
            module_value = _jaccard(
                set(sm.nlargest(top_k, "mean_abs_shap")["community"]),
                set(pm.nlargest(top_k, "mean_delta_rmse")["community"]),
            )
            rows.append(
                {
                    "target": target,
                    "grade": grade,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "top_k": int(top_k),
                    "feature_level_jaccard": feature_value,
                    "module_level_jaccard": module_value,
                    "module_minus_feature": module_value - feature_value,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["top_k", "target", "repeat", "fold"]
    ).reset_index(drop=True)


def dereduplicate_candidates(
    evidence: pd.DataFrame,
    nodes: pd.DataFrame,
    co_community_minimum: float = 2 / 3,
    maximum_candidates: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse stable features repeatedly occupying the same community."""
    selected = evidence[evidence["formal_evidence_complete"]].copy()
    overlap_rows: List[Dict[str, Any]] = []
    module_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    targets = evidence[["target", "grade", "source_model"]].drop_duplicates()
    for target_row in targets.itertuples(index=False):
        target = str(target_row.target)
        grade = str(target_row.grade)
        group = selected[selected["target"] == target].copy()
        graph = nx.Graph()
        graph.add_nodes_from(group["feature"].astype(str))
        features = sorted(graph.nodes)
        for index, left in enumerate(features):
            for right in features[index + 1 :]:
                pair = nodes[nodes["feature"].isin([left, right])].pivot(
                    index=["repeat", "fold"], columns="feature", values="community"
                )
                valid = pair.dropna(subset=[left, right])
                frequency = (
                    float((valid[left] == valid[right]).mean())
                    if len(valid)
                    else np.nan
                )
                redundant = bool(
                    pd.notna(frequency) and frequency >= float(co_community_minimum)
                )
                if redundant:
                    graph.add_edge(left, right)
                overlap_rows.append(
                    {
                        "target": target,
                        "feature_left": left,
                        "feature_right": right,
                        "folds_compared": int(len(valid)),
                        "co_community_frequency": frequency,
                        "network_redundant": redundant,
                    }
                )
        components = sorted(
            [sorted(component) for component in nx.connected_components(graph)],
            key=lambda component: component[0],
        )
        ranked_modules = []
        for component in components:
            ranked = group[group["feature"].isin(component)].sort_values(
                ["mean_delta_rmse", "feature"], ascending=[False, True]
            )
            representative = ranked.iloc[0]
            ranked_modules.append(
                {
                    "target": target,
                    "grade": grade,
                    "source_model": representative["source_model"],
                    "representative_feature": representative["feature"],
                    "related_stable_alternatives": ";".join(
                        ranked["feature"].iloc[1:].astype(str)
                    ),
                    "module_candidate_count": int(len(component)),
                    "representative_mean_delta_rmse": float(
                        representative["mean_delta_rmse"]
                    ),
                }
            )
        ranked_modules.sort(
            key=lambda row: (
                -row["representative_mean_delta_rmse"],
                row["representative_feature"],
            )
        )
        for rank, row in enumerate(ranked_modules, start=1):
            row["module_rank"] = rank
            row["reported_as_top3"] = rank <= int(maximum_candidates)
            module_rows.append(row)
        reported = min(len(ranked_modules), int(maximum_candidates))
        if grade == "C":
            decision = "EXPLORATORY_PARTIAL" if reported else "INSUFFICIENT_EVIDENCE"
        elif reported >= int(maximum_candidates):
            decision = "FORMAL_TOP3_COMPLETE"
        elif reported:
            decision = "FORMAL_PARTIAL_EVIDENCE"
        else:
            decision = "INSUFFICIENT_EVIDENCE"
        decision_rows.append(
            {
                "target": target,
                "grade": grade,
                "source_model": target_row.source_model,
                "stable_feature_count": int(len(group)),
                "nonredundant_module_count": int(len(ranked_modules)),
                "reported_candidate_count": int(reported),
                "decision": decision,
                "forced_to_three": False,
            }
        )
    return (
        pd.DataFrame(module_rows),
        pd.DataFrame(overlap_rows),
        pd.DataFrame(decision_rows).sort_values("target").reset_index(drop=True),
    )


def _reliability(grade: str, module_jaccard: float) -> str:
    if grade == "C":
        return "exploratory"
    if grade == "A" and module_jaccard >= 0.50:
        return "high"
    if grade == "B" and module_jaccard >= 0.50:
        return "medium"
    return "medium" if grade == "A" else "exploratory"


def run_formal_finalize(project_root: Path, config_path: Path) -> Mapping[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = _resolve(root, config["base_run"])
    advanced = _resolve(root, config["advanced_run"])
    output = _resolve(root, config["output_root"])
    if (root / "problem2").resolve() not in output.resolve().parents:
        raise ValueError("Question 2 output must stay under problem2")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    base_audit = json.loads((base / "data_audit.json").read_text(encoding="utf-8"))
    advanced_summary = json.loads((advanced / "summary.json").read_text(encoding="utf-8"))
    if base_audit["status"] != "PASS" or advanced_summary["status"] != "PASS":
        raise ValueError("Formal source runs must pass before finalization")
    if base_audit["targets"] != advanced_summary["targets"]:
        raise ValueError("Base and advanced target order differs")

    nodes = load_stable_network_nodes(
        root, config["network_nodes_paths"], config["network_variant"]
    )
    permutation = pd.read_csv(base / "importance_stability.csv")
    shap_stability = pd.read_csv(advanced / "shap_stability.csv")
    group_importance = pd.read_csv(advanced / "group_importance_summary.csv")
    style = pd.read_csv(advanced / "style_direction_heterogeneity.csv")
    evidence = assemble_formal_candidate_evidence(
        permutation,
        shap_stability,
        group_importance,
        style,
        nodes,
        shap_top10_minimum=config["candidate_gates"]["shap_top10_frequency"],
        group_positive_fold_minimum=config["candidate_gates"]["group_positive_fold_frequency"],
        style_reversal_maximum=config["candidate_gates"]["style_reversal_rate_maximum"],
    )
    agreement = method_agreement_by_fold(
        pd.read_parquet(base / "importance_fold_level.parquet"),
        pd.read_parquet(advanced / "shap_fold_level.parquet"),
        group_importance,
        nodes,
        config["agreement_top_k"],
    )
    modules, overlaps, decisions = dereduplicate_candidates(
        evidence,
        nodes,
        config["co_community_minimum"],
        config["maximum_candidates_per_target"],
    )
    target_agreement = (
        agreement[agreement["top_k"] == 3]
        .groupby("target", as_index=False)
        .agg(
            median_feature_top3_jaccard=("feature_level_jaccard", "median"),
            median_module_top3_jaccard=("module_level_jaccard", "median"),
            median_module_top3_improvement=("module_minus_feature", "median"),
        )
    )
    modules = modules.merge(target_agreement, on="target", how="left")
    modules["interpretation_reliability"] = [
        _reliability(str(grade), float(jaccard))
        for grade, jaccard in zip(
            modules["grade"], modules["median_module_top3_jaccard"]
        )
    ]

    audit_input = evidence.copy()
    audit_input["pilot_evidence_complete"] = audit_input["formal_evidence_complete"]
    attachments = root / "2026年MathorCup大数据挑战赛-赛道A初赛/附件"
    source = build_source_consistency(
        audit_input,
        pd.read_csv(root / "problem2/artifacts/preflight_v1/field_dictionary.csv"),
        pd.read_csv(attachments / "1.csv"),
        pd.read_csv(attachments / "4.csv"),
        pd.read_csv(attachments / "6.csv"),
        pd.read_excel(attachments / "8.xlsx", sheet_name="Sensory names"),
        pd.read_excel(attachments / "8.xlsx", sheet_name="Sensory thresholds"),
    )
    top3 = modules[modules["reported_as_top3"]].copy()
    top3 = top3.merge(
        evidence,
        left_on=["target", "representative_feature"],
        right_on=["target", "feature"],
        how="left",
        suffixes=("", "_evidence"),
        validate="one_to_one",
    ).merge(
        source,
        left_on=["target", "representative_feature"],
        right_on=["target", "feature"],
        how="left",
        suffixes=("", "_source"),
        validate="one_to_one",
    )
    top3 = top3.sort_values(["target", "module_rank"]).reset_index(drop=True)
    top3["formal_candidate_status"] = top3["interpretation_reliability"].map(
        {
            "high": "HIGH_CONFIDENCE_CANDIDATE",
            "medium": "MEDIUM_CONFIDENCE_CANDIDATE",
            "exploratory": "EXPLORATORY_CANDIDATE",
        }
    )
    decisions = decisions.merge(target_agreement, on="target", how="left")
    failure_columns = [
        "target", "grade", "feature", "mean_delta_rmse",
        "passes_pilot_stability", "shap_support", "group_support",
        "style_direction_ok", "formal_evidence_complete",
    ]
    failures = (
        evidence.sort_values(["target", "mean_delta_rmse"], ascending=[True, False])
        .groupby("target", group_keys=False).head(10)[failure_columns]
    )

    evidence.to_csv(output / "formal_candidate_evidence.csv", index=False)
    agreement.to_csv(output / "formal_method_agreement.csv", index=False)
    overlaps.to_csv(output / "formal_candidate_module_overlap.csv", index=False)
    modules.to_csv(output / "formal_candidate_modules.csv", index=False)
    decisions.to_csv(output / "target_decisions.csv", index=False)
    failures.to_csv(output / "candidate_failure_diagnostics.csv", index=False)
    source.to_csv(output / "source_consistency.csv", index=False)
    top3.to_csv(output / "top3_candidates.csv", index=False)

    complete = evidence[evidence["formal_evidence_complete"]]
    top3_agreement = agreement[agreement["top_k"] == 3]
    mapped = int(source["attachment6_rho"].notna().sum())
    mapped_consistent = int(
        source.loc[source["attachment6_rho"].notna(), "attachment6_direction_consistent"].sum()
    )
    summary: Dict[str, Any] = {
        "status": "PASS",
        "scope": "formal_all30_candidate_finalization",
        "targets": int(decisions["target"].nunique()),
        "stable_feature_candidates_before_advanced_checks": int(evidence["passes_pilot_stability"].sum()),
        "formal_evidence_complete_features": int(len(complete)),
        "targets_with_complete_feature": int(complete["target"].nunique()),
        "nonredundant_modules": int(len(modules)),
        "reported_candidates": int(len(top3)),
        "targets_with_three_reported_candidates": int((decisions["reported_candidate_count"] == 3).sum()),
        "targets_with_partial_candidate_sets": int(decisions["reported_candidate_count"].between(1, 2).sum()),
        "targets_with_insufficient_evidence": int((decisions["reported_candidate_count"] == 0).sum()),
        "forced_top3_count": 0,
        "formal_module_top3_jaccard_median": float(top3_agreement["module_level_jaccard"].median()),
        "formal_feature_top3_jaccard_median": float(top3_agreement["feature_level_jaccard"].median()),
        "formal_module_top3_paired_improvement_median": float(top3_agreement["module_minus_feature"].median()),
        "attachment6_mapped": mapped,
        "attachment6_unmapped": int(len(source) - mapped),
        "attachment6_direction_consistent_among_mapped": mapped_consistent,
        "attachment8_metadata_mapped": int(source["chemical_name"].notna().sum()),
        "attachment8_importance_used_for_ranking": False,
        "same_source_not_external_validation": True,
        "question1_limit_preserved": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    insufficient = decisions.loc[
        decisions["reported_candidate_count"] == 0, "target"
    ].tolist()
    lines = [
        "# Problem 2 formal candidate decision", "",
        f"- Pipeline status: **{summary['status']}**",
        f"- Complete stable features: {len(complete)}/{int(evidence['passes_pilot_stability'].sum())}.",
        f"- Network de-redundancy: {len(complete)} features -> {len(modules)} modules -> {len(top3)} reported representatives.",
        f"- Targets with three / one-to-two / zero candidates: {summary['targets_with_three_reported_candidates']} / {summary['targets_with_partial_candidate_sets']} / {summary['targets_with_insufficient_evidence']}.",
        f"- Feature-to-module Top-3 median Jaccard: {summary['formal_feature_top3_jaccard_median']:.3f} -> {summary['formal_module_top3_jaccard_median']:.3f}.",
        f"- Attachment 6 mapped {mapped}/{len(source)} complete candidates; {mapped_consistent}/{mapped} mapped directions agree.",
        "- Attachment 8 importance was excluded from ranking because its importance sheets are not target-traceable.",
        "- Attachment 6/8 are same-source audits, not independent validation.",
        "- No target was forced to have three candidates.", "",
        "## Targets with insufficient evidence", "",
    ]
    lines.extend([f"- `{target}`" for target in insufficient])
    lines.extend([
        "",
        "All chemical statements remain associative. Question 1 grades cap interpretation reliability, and C-grade targets remain exploratory even when a candidate is stable.",
    ])
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

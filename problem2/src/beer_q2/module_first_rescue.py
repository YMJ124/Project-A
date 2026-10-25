"""Cross-repeat module-first rescue for formally insufficient Q2 targets.

The mechanism was independently motivated before the formal run: correlated
features can exchange single-feature rank while their stable community remains
important.  Each frozen repeat is held out in turn; selection uses the other
six repeats and the held repeat must independently pass the locked module
criteria.  Results supplement, and never replace, stronger feature-level
candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import yaml

from .formal_finalize import dereduplicate_candidates, load_stable_network_nodes
from .source_audit import build_source_consistency


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _aggregate_module_anchor(frame: pd.DataFrame, training: bool) -> pd.DataFrame:
    result = (
        frame.groupby(["target", "feature"], as_index=False)
        .agg(
            group_top_frequency=("group_rank", lambda values: (values <= 5).mean()),
            shap_top_frequency=("shap_rank", lambda values: (values <= 5).mean()),
            positive_group_fold_frequency=(
                "mean_delta_rmse", lambda values: (values > 0).mean()
            ),
            positive_group_permutation_frequency=("positive_frequency", "mean"),
            mean_group_delta_rmse=("mean_delta_rmse", "mean"),
            direction_consistency=(
                "heldout_feature_target_spearman",
                lambda values: max((values > 0).mean(), (values < 0).mean()),
            ),
        )
    )
    if training:
        result["passes"] = (
            result["group_top_frequency"].ge(0.60)
            & result["shap_top_frequency"].ge(0.60)
            & result["positive_group_fold_frequency"].ge(0.80)
            & result["positive_group_permutation_frequency"].ge(0.80)
            & result["mean_group_delta_rmse"].gt(0)
            & result["direction_consistency"].ge(0.70)
        )
    else:
        result["passes"] = (
            result["group_top_frequency"].ge(2 / 3)
            & result["shap_top_frequency"].ge(2 / 3)
            & result["positive_group_fold_frequency"].ge(2 / 3)
            & result["positive_group_permutation_frequency"].ge(0.80)
            & result["mean_group_delta_rmse"].gt(0)
            & result["direction_consistency"].ge(2 / 3)
        )
    return result


def build_leave_one_repeat_out_evidence(
    feature_fold: pd.DataFrame,
    shap_fold: pd.DataFrame,
    group_importance: pd.DataFrame,
    style: pd.DataFrame,
    nodes: pd.DataFrame,
    targets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build module-anchor evidence with every repeat held out once."""
    primary = feature_fold[
        (feature_fold["mode"] == "within_style")
        & feature_fold["target"].isin(targets)
    ]
    group = group_importance[group_importance["target"].isin(targets)].copy()
    module_shap = (
        shap_fold[shap_fold["target"].isin(targets)]
        .merge(
            nodes[["repeat", "fold", "feature", "community"]],
            on=["repeat", "fold", "feature"],
            validate="many_to_one",
        )
        .groupby(["target", "repeat", "fold", "community"], as_index=False)[
            "mean_abs_shap"
        ]
        .sum()
    )
    group["group_rank"] = group.groupby(["target", "repeat", "fold"])[
        "mean_delta_rmse"
    ].rank(method="min", ascending=False)
    module_shap["shap_rank"] = module_shap.groupby(
        ["target", "repeat", "fold"]
    )["mean_abs_shap"].rank(method="min", ascending=False)
    anchors = (
        nodes[["repeat", "fold", "feature", "community"]]
        .merge(
            group[
                [
                    "target", "repeat", "fold", "community",
                    "mean_delta_rmse", "positive_frequency", "group_rank",
                ]
            ],
            on=["repeat", "fold", "community"],
        )
        .merge(
            module_shap[
                ["target", "repeat", "fold", "community", "shap_rank"]
            ],
            on=["target", "repeat", "fold", "community"],
        )
        .merge(
            primary[
                [
                    "target", "repeat", "fold", "feature",
                    "heldout_feature_target_spearman",
                ]
            ],
            on=["target", "repeat", "fold", "feature"],
        )
    )
    style_rate = (
        style[style["target"].isin(targets)]
        .groupby(["target", "feature"], as_index=False)["direction_reversal"]
        .mean()
        .rename(columns={"direction_reversal": "style_reversal_rate"})
    )
    rows = []
    for held_repeat in range(7):
        training = _aggregate_module_anchor(
            anchors[anchors["repeat"] != held_repeat], True
        ).rename(columns={"passes": "selected_on_six_repeats"})
        held = _aggregate_module_anchor(
            anchors[anchors["repeat"] == held_repeat], False
        )[["target", "feature", "passes"]].rename(
            columns={"passes": "passes_held_repeat"}
        )
        combined = training.merge(held, on=["target", "feature"])
        combined = combined.merge(
            style_rate, on=["target", "feature"], how="left"
        )
        combined["passes_both"] = (
            combined["selected_on_six_repeats"]
            & combined["passes_held_repeat"]
            & combined["style_reversal_rate"].le(0.10)
        )
        combined["held_repeat"] = held_repeat
        rows.append(combined)
    loo = pd.concat(rows, ignore_index=True)
    stability = (
        loo.groupby(["target", "feature"], as_index=False)
        .agg(
            train_selection_repeats=("selected_on_six_repeats", "sum"),
            heldout_pass_repeats=("passes_held_repeat", "sum"),
            both_pass_repeats=("passes_both", "sum"),
            mean_group_top5_frequency=("group_top_frequency", "mean"),
            mean_shap_top5_frequency=("shap_top_frequency", "mean"),
            style_reversal_rate=("style_reversal_rate", "first"),
        )
    )
    stability["module_first_qualifies"] = (
        stability["train_selection_repeats"].ge(5)
        & stability["heldout_pass_repeats"].ge(5)
        & stability["both_pass_repeats"].ge(5)
    )
    return loo, stability


def run_module_first_rescue(
    project_root: Path, config_path: Path
) -> Mapping[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = _resolve(root, config["base_run"])
    advanced = _resolve(root, config["advanced_run"])
    formal = _resolve(root, config["formal_candidate_run"])
    output = _resolve(root, config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    targets = list(map(str, config["targets"]))
    decisions = pd.read_csv(formal / "target_decisions.csv")
    actual = sorted(
        decisions.loc[
            decisions["reported_candidate_count"] == 0, "target"
        ].astype(str)
    )
    if sorted(targets) != actual:
        raise ValueError("Configured rescue targets differ from formal insufficient set")
    nodes = load_stable_network_nodes(
        root, config["network_nodes_paths"], config["network_variant"]
    )
    loo, stability = build_leave_one_repeat_out_evidence(
        pd.read_parquet(base / "importance_fold_level.parquet"),
        pd.read_parquet(advanced / "shap_fold_level.parquet"),
        pd.read_csv(advanced / "group_importance_summary.csv"),
        pd.read_csv(advanced / "style_direction_heterogeneity.csv"),
        nodes,
        targets,
    )
    feature_summary = pd.read_csv(base / "importance_stability.csv")
    feature_summary = feature_summary[
        (feature_summary["mode"] == "within_style")
        & feature_summary["target"].isin(targets)
    ]
    grades = feature_summary[["target", "grade", "source_model"]].drop_duplicates()
    shap_stability = pd.read_csv(advanced / "shap_stability.csv")
    rescue_evidence = (
        feature_summary.merge(stability, on=["target", "feature"])
        .merge(grades, on=["target", "grade", "source_model"], validate="many_to_one")
        .merge(
            shap_stability[
                [
                    "target", "grade", "source_model", "feature",
                    "median_shap_direction_rho",
                ]
            ],
            on=["target", "grade", "source_model", "feature"],
            validate="one_to_one",
        )
    )
    rescue_evidence["formal_evidence_complete"] = rescue_evidence[
        "module_first_qualifies"
    ]
    modules, overlaps, rescue_decisions = dereduplicate_candidates(
        rescue_evidence,
        nodes,
        co_community_minimum=float(config["co_community_minimum"]),
        maximum_candidates=int(config["maximum_candidates_per_target"]),
    )
    modules = modules[modules["reported_as_top3"]].copy()
    modules = modules.merge(
        stability,
        left_on=["target", "representative_feature"],
        right_on=["target", "feature"],
        validate="one_to_one",
    )
    modules["candidate_status"] = modules["grade"].map(
        {
            "A": "MEDIUM_CONFIDENCE_MODULE_CANDIDATE",
            "B": "MEDIUM_CONFIDENCE_MODULE_CANDIDATE",
            "C": "EXPLORATORY_MODULE_CANDIDATE",
        }
    )
    audit_input = rescue_evidence.copy()
    audit_input["pilot_evidence_complete"] = audit_input[
        "module_first_qualifies"
    ]
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
    representative_source = source.rename(columns={"feature": "representative_feature"})
    modules = modules.merge(
        representative_source[
            [
                "target", "representative_feature", "readable_name",
                "chemical_name", "chemical_name_short", "unit",
                "origin_paper", "attachment6_rho",
                "attachment6_direction_consistent", "evidence_scope",
            ]
        ],
        on=["target", "representative_feature"],
        how="left",
        validate="one_to_one",
    )
    loo.to_csv(output / "leave_one_repeat_out_evidence.csv", index=False)
    stability.to_csv(output / "module_anchor_stability.csv", index=False)
    rescue_evidence.to_csv(output / "rescue_candidate_evidence.csv", index=False)
    overlaps.to_csv(output / "rescue_module_overlap.csv", index=False)
    modules.to_csv(output / "rescue_module_candidates.csv", index=False)
    rescue_decisions.to_csv(output / "rescue_target_decisions.csv", index=False)
    source.to_csv(output / "source_consistency.csv", index=False)
    rescued_targets = sorted(modules["target"].unique())
    summary: Dict[str, Any] = {
        "status": "ADOPT_AS_SECONDARY_MODULE_EVIDENCE" if rescued_targets else "REJECT",
        "scope": "predeclared_formal_insufficient_targets_only",
        "targets_tested": len(targets),
        "qualifying_anchor_features": int(stability["module_first_qualifies"].sum()),
        "rescued_nonredundant_modules": int(len(modules)),
        "rescued_targets": rescued_targets,
        "targets_still_insufficient": sorted(set(targets) - set(rescued_targets)),
        "selection_repeats_minimum": 5,
        "heldout_pass_repeats_minimum": 5,
        "both_pass_repeats_minimum": 5,
        "single_feature_gates_relaxed": False,
        "primary_formal_candidates_replaced": False,
        "independent_external_validation": False,
        "attachment6_mapped": int(source["attachment6_rho"].notna().sum()),
        "attachment6_direction_consistent_among_mapped": int(
            source.loc[
                source["attachment6_rho"].notna(),
                "attachment6_direction_consistent",
            ].sum()
        ),
        "attachment8_metadata_mapped": int(source["chemical_name"].notna().sum()),
        "attachment8_importance_used_for_ranking": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Module-first rescue decision", "",
        f"- Decision: **{summary['status']}**",
        f"- Insufficient targets tested: {len(targets)}.",
        f"- Qualifying anchors: {summary['qualifying_anchor_features']}.",
        f"- Rescued targets/modules: {len(rescued_targets)}/{len(modules)}.",
        f"- Rescued targets: {', '.join(rescued_targets) if rescued_targets else 'none'}.",
        f"- Still insufficient: {', '.join(summary['targets_still_insufficient'])}.",
        "- Every accepted anchor was selected using six repeats and passed the held repeat in at least 5/7 rotations.",
        "- This is secondary module-level evidence; it does not overwrite stronger feature-level candidates or raise Question 1 grades.",
        f"- Attachment 6 mapped {summary['attachment6_mapped']}/{len(source)} qualifying anchors; attachment 8 metadata mapped {summary['attachment8_metadata_mapped']}/{len(source)}.",
    ]
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

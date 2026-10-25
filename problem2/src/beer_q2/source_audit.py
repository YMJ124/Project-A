"""Same-source attachment audit and pilot network de-redundancy.

Attachment 6/8 evidence is deliberately added only after the pilot candidates
have been selected from frozen-fold OOF explanations.  It is never used to
rank candidates and is not described as external validation.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx
import numpy as np
import pandas as pd


def _sign(value: float, tolerance: float = 1e-12) -> str:
    if pd.isna(value) or abs(float(value)) <= tolerance:
        return "zero_or_missing"
    return "positive" if value > 0 else "negative"


def _mode(values: Iterable[str]) -> str:
    clean = [str(value) for value in values if pd.notna(value) and str(value)]
    if not clean:
        return ""
    counts = Counter(clean)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def build_source_consistency(
    candidates: pd.DataFrame,
    field_dictionary: pd.DataFrame,
    attachment1: pd.DataFrame,
    attachment4: pd.DataFrame,
    attachment6: pd.DataFrame,
    sensory_names: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Audit already-selected pilot candidates against attachments 6 and 8."""
    selected = candidates[candidates["pilot_evidence_complete"]].copy()
    if selected.empty:
        return pd.DataFrame()

    target_names = sensory_names[
        ["R_name", "CleanName_PanelFeedback"]
    ].drop_duplicates("R_name")
    selected = selected.merge(
        target_names,
        left_on="target",
        right_on="R_name",
        how="left",
        validate="many_to_one",
    )
    feature_names = field_dictionary[
        ["field_1_csv", "node_2_csv", "mapping_status"]
    ].drop_duplicates("field_1_csv")
    selected = selected.merge(
        feature_names,
        left_on="feature",
        right_on="field_1_csv",
        how="left",
        validate="many_to_one",
    )
    metadata_columns = [
        "field_name_R",
        "chemical_name",
        "chemical_name_short",
        "calibrated",
        "origin_paper",
        "unit",
        "aging",
        "off_flavors",
        "threshold_mean",
        "threshold_median",
    ]
    selected = selected.merge(
        thresholds[metadata_columns].drop_duplicates("field_name_R"),
        left_on="feature",
        right_on="field_name_R",
        how="left",
        validate="many_to_one",
    )

    reference = attachment6.copy()
    reference = reference.set_index(reference.columns[0])
    rows: List[Dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        if pd.isna(row.CleanName_PanelFeedback) or pd.isna(row.node_2_csv):
            reference_rho = np.nan
        elif (
            row.node_2_csv not in reference.index
            or row.CleanName_PanelFeedback not in reference.columns
        ):
            reference_rho = np.nan
        else:
            reference_rho = float(
                reference.loc[row.node_2_csv, row.CleanName_PanelFeedback]
            )
        if row.feature in attachment1 and row.target in attachment4:
            recalculated = float(
                attachment1[row.feature].corr(
                    attachment4[row.target], method="spearman"
                )
            )
        else:
            recalculated = np.nan
        local_direction = _sign(row.median_heldout_feature_target_spearman)
        reference_direction = _sign(reference_rho)
        direction_consistent = (
            local_direction == reference_direction
            and local_direction != "zero_or_missing"
        )
        threshold_status = (
            "not_available"
            if pd.isna(row.threshold_median)
            else "not_compared_scale_and_transform_require_manual_audit"
        )
        rows.append(
            {
                "target": row.target,
                "grade": row.grade,
                "source_model": row.source_model,
                "feature": row.feature,
                "readable_name": row.node_2_csv,
                "target_reference_name": row.CleanName_PanelFeedback,
                "field_mapping_status": row.mapping_status,
                "local_oof_direction_rho": row.median_heldout_feature_target_spearman,
                "local_shap_direction_rho": row.median_shap_direction_rho,
                "attachment6_rho": reference_rho,
                "attachment6_recalculated_rho": recalculated,
                "attachment6_recalculation_abs_error": (
                    abs(reference_rho - recalculated)
                    if pd.notna(reference_rho) and pd.notna(recalculated)
                    else np.nan
                ),
                "attachment6_direction_consistent": direction_consistent,
                "chemical_name": row.chemical_name,
                "chemical_name_short": row.chemical_name_short,
                "calibrated": row.calibrated,
                "origin_paper": row.origin_paper,
                "unit": row.unit,
                "aging": row.aging,
                "off_flavors": row.off_flavors,
                "threshold_mean": row.threshold_mean,
                "threshold_median": row.threshold_median,
                "threshold_comparison_status": threshold_status,
                "attachment8_importance_status": (
                    "not_target_traceable_not_used_for_ranking"
                ),
                "evidence_scope": "same_source_consistency_not_external_validation",
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "feature"])


def build_network_deredundancy(
    candidates: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    co_community_minimum: float = 2 / 3,
) -> Dict[str, pd.DataFrame]:
    """Place pilot candidates in independently confirmed stable communities."""
    selected = candidates[candidates["pilot_evidence_complete"]].copy()
    nodes = nodes[nodes["variant"] == "stable_filtered"].copy()
    edges = edges[edges["variant"] == "stable_filtered"].copy()
    folds = sorted(nodes["fold"].unique())

    context_rows: List[Dict[str, Any]] = []
    for candidate in selected.itertuples(index=False):
        for fold in folds:
            fold_nodes = nodes[nodes["fold"] == fold]
            node_row = fold_nodes[fold_nodes["feature"] == candidate.feature]
            if node_row.empty:
                continue
            community = int(node_row.iloc[0]["community"])
            members = fold_nodes.loc[
                fold_nodes["community"] == community, "feature"
            ].tolist()
            incident = edges[
                (edges["fold"] == fold)
                & (
                    (edges["feature_left"] == candidate.feature)
                    | (edges["feature_right"] == candidate.feature)
                )
            ].copy()
            if incident.empty:
                top_neighbor = ""
                rho = np.nan
                frequency = np.nan
                sign = ""
            else:
                incident["abs_rho"] = incident["rho"].abs()
                best = incident.sort_values(
                    ["abs_rho", "bootstrap_frequency"], ascending=False
                ).iloc[0]
                top_neighbor = (
                    best["feature_right"]
                    if best["feature_left"] == candidate.feature
                    else best["feature_left"]
                )
                rho = float(best["rho"])
                frequency = float(best["bootstrap_frequency"])
                sign = str(best["sign"])
            context_rows.append(
                {
                    "target": candidate.target,
                    "grade": candidate.grade,
                    "feature": candidate.feature,
                    "repeat": int(node_row.iloc[0]["repeat"]),
                    "fold": int(fold),
                    "community": community,
                    "community_size": len(members),
                    "top_stable_neighbor": top_neighbor,
                    "top_neighbor_rho": rho,
                    "top_neighbor_sign": sign,
                    "top_neighbor_bootstrap_frequency": frequency,
                }
            )
    context = pd.DataFrame(context_rows)

    overlap_rows: List[Dict[str, Any]] = []
    for target, group in selected.groupby("target"):
        features = sorted(group["feature"].unique())
        for left_index, left in enumerate(features):
            for right in features[left_index + 1 :]:
                pair = context[
                    (context["target"] == target)
                    & (context["feature"].isin([left, right]))
                ].pivot(index="fold", columns="feature", values="community")
                valid = pair.dropna(subset=[left, right])
                fraction = (
                    float((valid[left] == valid[right]).mean())
                    if len(valid)
                    else np.nan
                )
                overlap_rows.append(
                    {
                        "target": target,
                        "feature_left": left,
                        "feature_right": right,
                        "folds_compared": int(len(valid)),
                        "co_community_frequency": fraction,
                        "network_redundant": bool(
                            pd.notna(fraction) and fraction >= co_community_minimum
                        ),
                    }
                )
    overlaps = pd.DataFrame(overlap_rows)

    summary_rows: List[Dict[str, Any]] = []
    module_rows: List[Dict[str, Any]] = []
    for target, group in selected.groupby("target"):
        graph = nx.Graph()
        graph.add_nodes_from(group["feature"])
        target_overlaps = overlaps[overlaps["target"] == target]
        for row in target_overlaps.itertuples(index=False):
            if row.network_redundant:
                graph.add_edge(row.feature_left, row.feature_right)
        components = sorted(
            [sorted(component) for component in nx.connected_components(graph)],
            key=lambda component: component[0],
        )
        module_lookup = {
            feature: module_id
            for module_id, component in enumerate(components, start=1)
            for feature in component
        }
        for feature in sorted(group["feature"]):
            rows = context[
                (context["target"] == target) & (context["feature"] == feature)
            ]
            module_features = components[module_lookup[feature] - 1]
            neighbor_rho = rows["top_neighbor_rho"].dropna()
            summary_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "stable_module": module_lookup[feature],
                    "median_community_size": float(rows["community_size"].median()),
                    "top_stable_neighbor_mode": _mode(rows["top_stable_neighbor"]),
                    "median_abs_top_neighbor_rho": (
                        float(neighbor_rho.abs().median())
                        if len(neighbor_rho)
                        else np.nan
                    ),
                    "same_candidate_module_members": ";".join(
                        item for item in module_features if item != feature
                    ),
                }
            )
        for module_id, module_features in enumerate(components, start=1):
            ranked = group[group["feature"].isin(module_features)].sort_values(
                ["mean_delta_rmse", "feature"], ascending=[False, True]
            )
            representative = ranked.iloc[0]
            alternatives = [
                feature for feature in ranked["feature"] if feature != representative.feature
            ]
            module_rows.append(
                {
                    "target": target,
                    "grade": representative.grade,
                    "pilot_module": module_id,
                    "representative_feature": representative.feature,
                    "related_pilot_candidates": ";".join(alternatives),
                    "representative_mean_delta_rmse": representative.mean_delta_rmse,
                    "interpretation_ceiling": representative.interpretation_ceiling,
                    "formal_top3_status": "NOT_EVALUATED_PILOT_ONLY",
                }
            )
    return {
        "context": context.sort_values(["target", "feature", "fold"]),
        "overlap": overlaps.sort_values(
            ["target", "feature_left", "feature_right"]
        ),
        "summary": pd.DataFrame(summary_rows).sort_values(["target", "feature"]),
        "modules": pd.DataFrame(module_rows).sort_values(
            ["target", "pilot_module"]
        ),
    }


def run_source_and_network_audit(
    project_root: Path,
    candidate_path: Path,
    network_confirmation: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    root = project_root.resolve()
    output = output_dir.resolve()
    if (root / "problem2").resolve() not in output.parents:
        raise ValueError("Question 2 output must stay under problem2")
    output.mkdir(parents=True, exist_ok=False)
    attachments = root / "2026年MathorCup大数据挑战赛-赛道A初赛/附件"

    candidates = pd.read_csv(candidate_path)
    field_dictionary = pd.read_csv(
        root / "problem2/artifacts/preflight_v1/field_dictionary.csv"
    )
    attachment1 = pd.read_csv(attachments / "1.csv")
    attachment4 = pd.read_csv(attachments / "4.csv")
    attachment6 = pd.read_csv(attachments / "6.csv")
    sensory_names = pd.read_excel(attachments / "8.xlsx", sheet_name="Sensory names")
    thresholds = pd.read_excel(
        attachments / "8.xlsx", sheet_name="Sensory thresholds"
    )
    source = build_source_consistency(
        candidates,
        field_dictionary,
        attachment1,
        attachment4,
        attachment6,
        sensory_names,
        thresholds,
    )
    nodes = pd.read_csv(network_confirmation / "network_nodes_confirmation.csv")
    edges = pd.read_parquet(
        network_confirmation / "network_edges_confirmation.parquet"
    )
    network = build_network_deredundancy(candidates, nodes, edges)

    source.to_csv(output / "source_consistency.csv", index=False)
    network["context"].to_csv(
        output / "candidate_network_context_by_fold.csv", index=False
    )
    network["overlap"].to_csv(
        output / "candidate_module_overlap.csv", index=False
    )
    network["summary"].to_csv(
        output / "candidate_network_deredundancy.csv", index=False
    )
    network["modules"].to_csv(output / "pilot_candidate_modules.csv", index=False)

    summary = {
        "scope": "pilot_only_not_formal_top3",
        "same_source_not_external_validation": True,
        "pilot_candidates_audited": int(len(source)),
        "attachment6_mapped": int(source["attachment6_rho"].notna().sum()),
        "attachment6_direction_consistent": int(
            source["attachment6_direction_consistent"].sum()
        ),
        "attachment6_max_recalculation_abs_error": float(
            source["attachment6_recalculation_abs_error"].max()
        ),
        "attachment8_metadata_mapped": int(source["chemical_name"].notna().sum()),
        "attachment8_importance_used_for_ranking": False,
        "attachment8_importance_exclusion_reason": (
            "GBM MDI and SHAP sheets do not identify a sensory target"
        ),
        "pilot_modules_by_target": {
            str(key): int(value)
            for key, value in network["modules"].groupby("target").size().items()
        },
        "redundant_candidate_pairs": int(
            network["overlap"]["network_redundant"].sum()
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Pilot source consistency and network de-redundancy",
        "",
        "- Scope: pilot-only; no formal Top-3 is produced.",
        f"- Attachment 6 mapping: {summary['attachment6_mapped']}/{summary['pilot_candidates_audited']} candidates.",
        f"- Same-direction attachment 6 associations: {summary['attachment6_direction_consistent']}/{summary['pilot_candidates_audited']}.",
        f"- Attachment 6 maximum recomputation error: {summary['attachment6_max_recalculation_abs_error']:.3e}.",
        f"- Attachment 8 metadata mapping: {summary['attachment8_metadata_mapped']}/{summary['pilot_candidates_audited']}.",
        "- Attachment 8 GBM importance was not used because its sheets do not identify a sensory target.",
        f"- Stable-network redundant pilot pairs: {summary['redundant_candidate_pairs']}.",
        "",
        "Attachment 6/8 are same-source audit material, not independent external validation. Threshold comparisons remain disabled until each feature's stored transformation and unit are manually verified.",
    ]
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

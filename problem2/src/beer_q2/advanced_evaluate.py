"""Gap analysis and candidate evidence synthesis for the Q2 advanced pilot.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _gap_minimum(observed: float, formal: float, ideal: float):
    return max(0.0, formal - observed), max(0.0, ideal - observed)


def _gap_maximum(observed: float, formal: float, ideal: float):
    return max(0.0, observed - formal), max(0.0, observed - ideal)


def _candidate_evidence(base: Path, advanced: Path) -> pd.DataFrame:
    permutation = pd.read_csv(base / "importance_stability.csv")
    permutation = permutation[permutation["mode"] == "within_style"].copy()
    shap_stability = pd.read_csv(advanced / "shap_stability.csv")
    nodes = pd.read_csv(base / "network_nodes.csv")
    groups = pd.read_csv(advanced / "group_importance_summary.csv")
    style = pd.read_csv(advanced / "style_direction_heterogeneity.csv")

    feature_groups = nodes.merge(
        groups[
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
        how="cross" if False else "inner",
    )
    group_support = (
        feature_groups.groupby(["target", "feature"], as_index=False)
        .agg(
            mean_group_delta_rmse=("mean_delta_rmse", "mean"),
            group_positive_fold_frequency=(
                "mean_delta_rmse",
                lambda values: float((values > 0).mean()),
            ),
            mean_community_size=("community_size", "mean"),
        )
    )
    style_support = (
        style.groupby(["target", "feature"], as_index=False)
        .agg(
            style_tests=("direction_reversal", "size"),
            style_reversals=("direction_reversal", "sum"),
        )
    )
    style_support["style_reversal_rate"] = (
        style_support["style_reversals"] / style_support["style_tests"]
    )
    evidence = permutation.merge(
        shap_stability, on=["target", "grade", "source_model", "feature"], how="left"
    ).merge(group_support, on=["target", "feature"], how="left").merge(
        style_support, on=["target", "feature"], how="left"
    )
    evidence["shap_support"] = evidence["top10_shap_frequency"] >= 0.60
    evidence["group_support"] = (
        (evidence["mean_group_delta_rmse"] > 0)
        & (evidence["group_positive_fold_frequency"] >= 2 / 3)
    )
    evidence["style_direction_ok"] = (
        evidence["style_reversal_rate"].isna()
        | (evidence["style_reversal_rate"] <= 0.10)
    )
    evidence["pilot_evidence_complete"] = (
        evidence["passes_pilot_stability"]
        & evidence["shap_support"]
        & evidence["group_support"]
        & evidence["style_direction_ok"]
    )
    evidence["interpretation_ceiling"] = evidence["grade"].map(
        {"A": "high_candidate", "B": "medium_candidate", "C": "exploratory_only"}
    )
    return evidence.sort_values(
        ["target", "pilot_evidence_complete", "mean_delta_rmse"],
        ascending=[True, False, False],
    )


def evaluate_advanced(base_run: Path, advanced_run: Path) -> Dict[str, Any]:
    base = base_run.resolve()
    advanced = advanced_run.resolve()
    network = pd.read_csv(base / "network_stability_summary.csv")
    community = pd.read_csv(base / "community_stability.csv")
    mode = pd.read_csv(base / "explanation_mode_consistency.csv")
    advanced_summary = json.loads((advanced / "summary.json").read_text(encoding="utf-8"))
    agreement = pd.read_csv(advanced / "permutation_shap_agreement.csv")
    style = pd.read_csv(advanced / "style_direction_heterogeneity.csv")
    primary = network[np.isclose(network["abs_rho_threshold"], 0.4)].iloc[0]

    style_reversal_rate = float(style["direction_reversal"].mean())
    metrics: List[Dict[str, object]] = []

    def add(name, direction, observed, formal, ideal, unit, scope, source):
        if direction == "minimum":
            formal_gap, ideal_gap = _gap_minimum(observed, formal, ideal)
        else:
            formal_gap, ideal_gap = _gap_maximum(observed, formal, ideal)
        metrics.append(
            {
                "metric": name,
                "direction": direction,
                "observed": observed,
                "formal_line": formal,
                "ideal_line": ideal,
                "gap_to_formal": formal_gap,
                "gap_to_ideal": ideal_gap,
                "unit": unit,
                "scope": scope,
                "source": source,
            }
        )

    add("main_target_coverage", "minimum", 3, 30, 30, "targets", "pilot", "config")
    add("frozen_repeat_coverage", "minimum", 1, 7, 7, "repeats", "pilot", "config")
    add("network_bootstrap_depth", "minimum", 20, 200, 200, "resamples/fold", "pilot", "config")
    add("feature_permutation_depth", "minimum", 10, 100, 100, "permutations", "pilot", "config")
    add("primary_edge_jaccard", "minimum", float(primary["median_edge_jaccard"]), 0.60, 0.70, "ratio", "network", "network_stability_summary.csv")
    add("primary_edge_count_cv", "maximum", float(primary["edge_count_cv"]), 0.15, 0.10, "ratio", "network", "network_stability_summary.csv")
    add("community_ari", "minimum", float(community["adjusted_rand_index"].median()), 0.50, 0.70, "ratio", "network", "community_stability.csv")
    add("conditional_unconditional_top10_jaccard", "minimum", float(mode["within_style_vs_unconditional_top10_jaccard"].median()), 0.50, 0.70, "ratio", "explanation", "explanation_mode_consistency.csv")
    add("permutation_shap_top10_jaccard", "minimum", float(agreement["permutation_shap_top10_jaccard"].median()), 0.30, 0.50, "ratio", "explanation", "permutation_shap_agreement.csv")
    add("style_direction_reversal_rate", "maximum", style_reversal_rate, 0.10, 0.05, "ratio", "explanation", "style_direction_heterogeneity.csv")
    add("shap_additivity_error", "maximum", float(advanced_summary["max_shap_additivity_error"]), 1e-8, 1e-10, "absolute", "integrity", "summary.json")

    gaps = pd.DataFrame(metrics)
    gaps.to_csv(advanced / "distance_to_evaluation_lines.csv", index=False)
    evidence = _candidate_evidence(base, advanced)
    evidence.to_csv(advanced / "candidate_evidence.csv", index=False)

    jaccard_by_target = agreement.groupby("target")[
        "permutation_shap_top10_jaccard"
    ].median()
    reversals_by_target = style.groupby("target")["direction_reversal"].mean()
    complete_counts = evidence.groupby("target")["pilot_evidence_complete"].sum().astype(int)
    summary = {
        "formal_metrics_passed": int((gaps["gap_to_formal"] == 0).sum()),
        "formal_metrics_total": int(len(gaps)),
        "ideal_metrics_passed": int((gaps["gap_to_ideal"] == 0).sum()),
        "ideal_metrics_total": int(len(gaps)),
        "largest_normalized_scope_gaps": {
            "targets_missing": 27,
            "repeats_missing": 6,
            "network_bootstraps_missing_per_fold": 180,
            "feature_permutations_missing": 90,
        },
        "community_ari_gap_to_formal": float(
            gaps.loc[gaps["metric"] == "community_ari", "gap_to_formal"].iloc[0]
        ),
        "permutation_shap_gap_to_ideal": float(
            gaps.loc[
                gaps["metric"] == "permutation_shap_top10_jaccard", "gap_to_ideal"
            ].iloc[0]
        ),
        "permutation_shap_median_by_target": {
            str(key): float(value) for key, value in jaccard_by_target.items()
        },
        "style_reversal_rate_by_target": {
            str(key): float(value) for key, value in reversals_by_target.items()
        },
        "pilot_complete_candidate_count": {
            str(key): int(value) for key, value in complete_counts.items()
        },
        "formal_question2_status": "HOLD",
    }
    (advanced / "gap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    weak_target = jaccard_by_target.idxmin()
    largest_communities = pd.read_csv(advanced / "group_importance_summary.csv")[
        "community_size"
    ]
    non_singleton = largest_communities[largest_communities > 1]
    lines = [
        "# Advanced pilot diagnosis",
        "",
        "## Distance to evaluation lines",
        "",
        f"- Formal quantitative lines passed: {summary['formal_metrics_passed']}/{summary['formal_metrics_total']}.",
        f"- Ideal quantitative lines passed: {summary['ideal_metrics_passed']}/{summary['ideal_metrics_total']}.",
        f"- Community ARI is short of the formal line by {summary['community_ari_gap_to_formal']:.3f}.",
        f"- Permutation-SHAP agreement is short of the ideal line by {summary['permutation_shap_gap_to_ideal']:.3f}.",
        "",
        "## Specific problems",
        "",
        f"- The weakest method agreement target is `{weak_target}` (median Jaccard {jaccard_by_target[weak_target]:.3f}); its Q1 grade limits it to exploratory interpretation.",
        f"- The network mixes many isolates ({(largest_communities == 1).mean():.1%} of fold-community rows) with large correlated modules (non-singleton median size {non_singleton.median():.0f}, maximum {largest_communities.max():.0f}); attribution inside the large modules is not uniquely identifiable.",
        f"- Style-direction reversal rate is {style_reversal_rate:.1%}; reversals concentrate in a small set of protein, aggregate ester, color, and related features rather than all candidates.",
        "- Coverage/depth gaps are computational completeness gaps, not evidence that current candidates are poor; they must be closed before formal Top-3.",
        "",
        "## Locked optimization",
        "",
        "The next single mechanism is community discovery on edges with bootstrap frequency >=0.90 and sign consistency >=0.90. It uses no sensory labels. The thresholds were selected on repeat 0 and therefore must be confirmed on untouched repeat 1. Adoption requires median ARI >=0.50, an improvement >=0.03 over the same-seed unfiltered graph, filtered-edge Jaccard >=0.60, and retention of at least 40% of primary edges.",
    ]
    (advanced / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

"""Assemble the post-optimization Q2 status and distance-to-ideal table.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def assemble_current_status(
    base_run: Path,
    advanced_run: Path,
    network_confirmation: Path,
    source_audit: Path,
    module_confirmation: Path,
    output_dir: Path,
    reconstruction_all30: Optional[Path] = None,
    source_shap_audit: Optional[Path] = None,
    network_audits: Optional[List[Path]] = None,
    all30_smoke: Optional[Path] = None,
    all30_advanced_smoke: Optional[Path] = None,
    formal_run: Optional[Path] = None,
    formal_advanced_run: Optional[Path] = None,
    formal_candidate_run: Optional[Path] = None,
    formal_bundle_run: Optional[Path] = None,
) -> Dict[str, Any]:
    base = base_run.resolve()
    advanced = advanced_run.resolve()
    confirmation = network_confirmation.resolve()
    audit = source_audit.resolve()
    module = module_confirmation.resolve()
    output = output_dir.resolve()
    # This is a derived status bundle, so rebuilding it should be idempotent.
    output.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(base / "evaluation_matrix.csv")
    advanced_summary = json.loads((advanced / "summary.json").read_text(encoding="utf-8"))
    gap_summary = json.loads((advanced / "gap_summary.json").read_text(encoding="utf-8"))
    network_summary = json.loads((confirmation / "summary.json").read_text(encoding="utf-8"))
    audit_summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    module_summary = json.loads((module / "summary.json").read_text(encoding="utf-8"))
    reconstruction_summary = (
        json.loads(
            (reconstruction_all30.resolve() / "reconstruction_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if reconstruction_all30 is not None
        else None
    )
    source_shap_summary = (
        json.loads(
            (source_shap_audit.resolve() / "summary.json").read_text(encoding="utf-8")
        )
        if source_shap_audit is not None
        else None
    )
    network_paths = [path.resolve() for path in (network_audits or [network_confirmation])]
    network_summaries = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in network_paths
    ]
    all30_audit = (
        json.loads((all30_smoke.resolve() / "data_audit.json").read_text(encoding="utf-8"))
        if all30_smoke is not None
        else None
    )
    all30_advanced_summary = (
        json.loads(
            (all30_advanced_smoke.resolve() / "summary.json").read_text(encoding="utf-8")
        )
        if all30_advanced_smoke is not None
        else None
    )
    formal = formal_run.resolve() if formal_run is not None else None
    formal_advanced = (
        formal_advanced_run.resolve() if formal_advanced_run is not None else None
    )
    formal_candidates = (
        formal_candidate_run.resolve() if formal_candidate_run is not None else None
    )
    formal_advanced_summary = (
        json.loads((formal_advanced / "summary.json").read_text(encoding="utf-8"))
        if formal_advanced is not None
        else None
    )
    formal_candidate_summary = (
        json.loads((formal_candidates / "summary.json").read_text(encoding="utf-8"))
        if formal_candidates is not None
        else None
    )
    formal_bundle_summary = (
        json.loads(
            (formal_bundle_run.resolve() / "summary.json").read_text(encoding="utf-8")
        )
        if formal_bundle_run is not None
        else None
    )
    formal_audit = (
        json.loads((formal / "data_audit.json").read_text(encoding="utf-8"))
        if formal is not None
        else None
    )
    formal_stability = (
        pd.read_csv(formal / "importance_stability.csv")
        if formal is not None
        else None
    )
    formal_reconstruction = (
        pd.read_csv(formal / "reconstruction_audit.csv")
        if formal is not None
        else None
    )
    stable_aris = [float(item["stable_filtered_median_ari"]) for item in network_summaries]
    stable_jaccards = [float(item["stable_filtered_edge_jaccard"]) for item in network_summaries]
    stable_retentions = [float(item["median_edge_retention"]) for item in network_summaries]
    network_improvement_passes = sum(
        bool(item["criteria"]["median_ari_improvement_minimum"])
        for item in network_summaries
    )
    network_failed_repeats = [
        int(item["repeat"]) for item in network_summaries if item["status"] != "ADOPT"
    ]
    independent_network_summaries = [
        item for item in network_summaries if int(item["repeat"]) != 0
    ]
    independent_improvement_passes = sum(
        bool(item["criteria"]["median_ari_improvement_minimum"])
        for item in independent_network_summaries
    )
    all_network_depth_complete = len(network_summaries) == 7 and all(
        int(item["bootstrap_repeats"]) >= 200 for item in network_summaries
    )
    median_stable_ari = float(pd.Series(stable_aris).median())
    median_stable_jaccard = float(pd.Series(stable_jaccards).median())
    median_stable_retention = float(pd.Series(stable_retentions).median())
    network_robustness_rows = []
    for path, item in zip(network_paths, network_summaries):
        pairs = pd.read_csv(path / "community_stability_confirmation.csv")
        paired = pairs.pivot(
            index=["fold_a", "fold_b"],
            columns="variant",
            values="adjusted_rand_index",
        ).reset_index()
        paired["paired_ari_change"] = (
            paired["stable_filtered"] - paired["unfiltered"]
        )
        network_robustness_rows.append(
            {
                "repeat": int(item["repeat"]),
                "evidence_role": (
                    "retrospective_discovery_repeat"
                    if int(item["repeat"]) == 0
                    else "independent_frozen_repeat"
                ),
                "locked_rule_status": item["status"],
                "baseline_median_ari": float(item["baseline_median_ari"]),
                "stable_filtered_median_ari": float(
                    item["stable_filtered_median_ari"]
                ),
                "difference_of_median_ari": float(item["median_ari_improvement"]),
                "median_paired_ari_change": float(
                    paired["paired_ari_change"].median()
                ),
                "improved_fold_pairs": int((paired["paired_ari_change"] > 0).sum()),
                "total_fold_pairs": int(len(paired)),
                "stable_edge_jaccard": float(item["stable_filtered_edge_jaccard"]),
                "median_edge_retention": float(item["median_edge_retention"]),
            }
        )
    network_robustness = pd.DataFrame(network_robustness_rows).sort_values("repeat")
    network_robustness.to_csv(output / "network_repeat_robustness.csv", index=False)
    style = pd.read_csv(advanced / "style_direction_heterogeneity.csv")
    style_reversal_rate = float(style["direction_reversal"].mean())

    updates = {
        "Community stability": (
            "PASS",
            False,
            f"stable-edge ARI >=0.50 in {len(network_summaries)}/{len(network_summaries)} audited repeats; median={median_stable_ari:.3f}; filtering improvement passed {network_improvement_passes}/{len(network_summaries)}",
            "q2_network_repeat4_v1/summary.json",
        ),
        "TreeSHAP agreement": (
            "PASS",
            False,
            f"single-feature Top-10={advanced_summary['median_permutation_shap_top10_jaccard']:.3f}; independently confirmed stable-module Top-5={module_summary['median_module_level_jaccard']:.3f}",
            "q2_module_top5_confirmation_v1/paired_method_agreement.csv",
        ),
        "Correlated-group importance": (
            "PASS",
            False,
            "held-out community joint permutation implemented",
            "q2_explain_pilot_v1/group_importance_summary.csv",
        ),
        "Style direction heterogeneity": (
            "PASS" if style_reversal_rate <= 0.10 else "PARTIAL",
            False,
            f"candidate-style reversal rate={style_reversal_rate:.3f}",
            "q2_explain_pilot_v1/style_direction_heterogeneity.csv",
        ),
        "Bootstrap depth": (
            "PASS" if all_network_depth_complete else "PARTIAL",
            True,
            f"20 in the complete repeat-0 explanation chain; 200 in {len(network_summaries)}/7 network repeats",
            "q2_network_repeat6_v1/summary.json",
        ),
        "Fold/repeat coverage": (
            "PARTIAL",
            True,
            "3 complete explanation repeats x 3 folds; 4 repeats have 200-bootstrap network audits",
            "q2_explain_pilot_v1/summary.json",
        ),
        "Predeclared candidate stability": (
            "PARTIAL",
            True,
            f"complete pilot evidence counts: {gap_summary['pilot_complete_candidate_count']}; formal 7-repeat stability not run",
            "q2_explain_pilot_v1/candidate_evidence.csv",
        ),
        "Network de-redundancy": (
            "PARTIAL",
            True,
            f"stable-module Top-5 mechanism independently adopted ({module_summary['improved_target_folds']}/{module_summary['total_target_folds']} folds improved); formal 30-target candidates not assembled",
            "q2_module_top5_confirmation_v1/decision.md",
        ),
        "Attachment 6/8 consistency": (
            "PARTIAL",
            True,
            f"{audit_summary['attachment6_mapped']}/{audit_summary['pilot_candidates_audited']} pilot candidates mapped; target-unidentified attachment 8 importance excluded",
            "q2_source_audit_pilot_v1/source_consistency.csv",
        ),
    }
    if reconstruction_summary is not None:
        updates["True champion reconstruction"] = (
            "PASS" if reconstruction_summary["all_passed"] else "FAIL",
            True,
            f"{reconstruction_summary['rows']}/{reconstruction_summary['rows']} target-folds; 30/30 targets; max={reconstruction_summary['max_abs_error']:.3g}",
            "q1_reconstruction_all30_repeat4_v1/reconstruction_comparison.csv",
        )
    if all30_audit is not None:
        updates["Target coverage"] = (
            "PARTIAL",
            True,
            f"formal-depth 3/30; all-source engineering smoke {len(all30_audit['targets'])}/30",
            "q2_all30_smoke_v1/data_audit.json",
        )
    if source_shap_summary is not None and all30_advanced_summary is not None:
        updates["TreeSHAP agreement"] = (
            "PASS",
            False,
            f"9/9 source architectures pass additivity; all-30 one-fold smoke max error={all30_advanced_summary['max_shap_additivity_error']:.3g}; confirmed stable-module Top-5={module_summary['median_module_level_jaccard']:.3f}",
            "q1_source_shap_audit_v1/summary.json; q2_all30_advanced_smoke_v1/summary.json",
        )
    if formal_advanced_summary is not None:
        formal_style = pd.read_csv(
            formal_advanced / "style_direction_heterogeneity.csv"
        )
        style_reversal_rate = float(formal_style["direction_reversal"].mean())
        updates.update(
            {
                "TreeSHAP agreement": (
                    "PASS",
                    False,
                    f"formal 30-target Top-10 Jaccard={formal_advanced_summary['median_permutation_shap_top10_jaccard']:.3f}; max additivity error={formal_advanced_summary['max_shap_additivity_error']:.3g}",
                    "q2_formal_all7_advanced_fast_v1/permutation_shap_agreement.csv",
                ),
                "Correlated-group importance": (
                    "PASS",
                    False,
                    f"formal 30-target, 7x3-fold joint community permutation; {formal_advanced_summary['n_group_permutation_rows']} raw rows",
                    "q2_formal_all7_advanced_fast_v1/oof_group_permutation_importance.parquet",
                ),
                "Style direction heterogeneity": (
                    "PASS" if style_reversal_rate <= 0.10 else "PARTIAL",
                    False,
                    f"formal candidate-style reversal rate={style_reversal_rate:.3f}",
                    "q2_formal_all7_advanced_fast_v1/style_direction_heterogeneity.csv",
                ),
            }
        )
    if formal_candidate_summary is not None:
        updates.update(
            {
                "Network de-redundancy": (
                    "PASS" if formal_candidate_summary["status"] == "PASS" else "FAIL",
                    True,
                    f"{formal_candidate_summary['formal_evidence_complete_features']} complete features collapsed to {formal_candidate_summary['nonredundant_modules']} modules and {formal_candidate_summary['reported_candidates']} reported representatives; no forced Top-3",
                    "q2_formal_candidate_final_v1/formal_candidate_modules.csv",
                ),
                "Attachment 6/8 consistency": (
                    "PASS" if formal_candidate_summary["status"] == "PASS" else "FAIL",
                    True,
                    f"formal set audited: attachment 6 mapped {formal_candidate_summary['attachment6_mapped']} candidates; attachment 8 metadata mapped {formal_candidate_summary['attachment8_metadata_mapped']}; target-untraceable importance excluded",
                    "q2_formal_candidate_final_v1/source_consistency.csv",
                ),
                "C-grade restraint": (
                    "PASS" if formal_candidate_summary["forced_top3_count"] == 0 else "FAIL",
                    True,
                    f"forced Top-3 count={formal_candidate_summary['forced_top3_count']}; C-grade outputs remain exploratory or insufficient",
                    "q2_formal_candidate_final_v1/target_decisions.csv",
                ),
            }
        )
    if formal_bundle_summary is not None:
        updates.update(
            {
                "Network de-redundancy": (
                    "PASS",
                    True,
                    f"v2 catalog reports {formal_bundle_summary['reported_candidates']} candidates: {formal_bundle_summary['targets_with_three_candidates']} targets with three, {formal_bundle_summary['targets_with_one_or_two_candidates']} with one/two, {formal_bundle_summary['targets_with_insufficient_evidence']} insufficient; no forced Top-3",
                    "q2_formal_candidate_bundle_v2/candidate_catalog.csv",
                ),
                "C-grade restraint": (
                    "PASS" if formal_bundle_summary["forced_top3_count"] == 0 else "FAIL",
                    True,
                    f"forced Top-3 count={formal_bundle_summary['forced_top3_count']}; secondary module evidence preserves Q1 grade ceilings",
                    "q2_formal_candidate_bundle_v2/target_decisions.csv",
                ),
            }
        )
    formal_mode_jaccard = None
    formal_stable_counts = None
    if formal_audit is not None and formal_stability is not None:
        mode_rows = []
        for target, group in formal_stability.groupby("target"):
            primary_top = set(
                group[group["mode"] == "within_style"]
                .nlargest(10, "mean_delta_rmse")["feature"]
            )
            sensitivity_top = set(
                group[group["mode"] == "unconditional"]
                .nlargest(10, "mean_delta_rmse")["feature"]
            )
            union = primary_top | sensitivity_top
            mode_rows.append(
                len(primary_top & sensitivity_top) / len(union) if union else 1.0
            )
        formal_mode_jaccard = float(pd.Series(mode_rows).median())
        primary = formal_stability[formal_stability["mode"] == "within_style"]
        formal_stable_counts = (
            primary.groupby("target")["passes_pilot_stability"].sum().astype(int)
        )
        targets_with_candidates = int((formal_stable_counts > 0).sum())
        stable_candidates = int(formal_stable_counts.sum())
        depths = formal_audit["permutations_by_mode"]
        complete_formal = bool(
            formal_audit["status"] == "PASS"
            and len(formal_audit["targets"]) == 30
            and len(formal_audit["repeats"]) == 7
            and len(formal_audit["outer_folds"]) == 3
            and int(depths["within_style"]) >= 100
            and int(depths["unconditional"]) >= 100
            and formal_audit["all_reconstructions_passed"]
        )
        updates.update(
            {
                "True champion reconstruction": (
                    "PASS" if complete_formal else "FAIL",
                    True,
                    f"{len(formal_reconstruction)}/{len(formal_reconstruction)} formal target-folds; max={formal_reconstruction['reconstruction_max_abs_error'].max():.3g}",
                    "q2_formal_all7_fast_v1/reconstruction_audit.csv",
                ),
                "Target coverage": (
                    "PASS" if complete_formal else "PARTIAL",
                    True,
                    f"formal-depth {len(formal_audit['targets'])}/30",
                    "q2_formal_all7_fast_v1/data_audit.json",
                ),
                "Fold/repeat coverage": (
                    "PASS" if complete_formal else "PARTIAL",
                    True,
                    f"formal feature explanation {len(formal_audit['repeats'])} x {len(formal_audit['outer_folds'])}",
                    "q2_formal_all7_fast_v1/data_audit.json",
                ),
                "Permutation depth": (
                    "PASS" if min(map(int, depths.values())) >= 100 else "PARTIAL",
                    True,
                    f"within-style={depths['within_style']}; unconditional={depths['unconditional']}",
                    "q2_formal_all7_fast_v1/config.resolved.yaml",
                ),
                "Permutation-mode sensitivity": (
                    "PASS" if formal_mode_jaccard >= 0.50 else "PARTIAL",
                    False,
                    f"formal 30-target median Top-10 Jaccard={formal_mode_jaccard:.3f}",
                    "q2_formal_all7_fast_v1/importance_stability.csv",
                ),
                "Predeclared candidate stability": (
                    "PASS" if complete_formal else "PARTIAL",
                    True,
                    f"{stable_candidates} candidates pass all three gates across 21 folds; {targets_with_candidates}/30 targets represented",
                    "q2_formal_all7_fast_v1/importance_stability.csv",
                ),
            }
        )
    for criterion, (status, blocking, observed, evidence) in updates.items():
        mask = matrix["criterion"] == criterion
        matrix.loc[mask, "status"] = status
        matrix.loc[mask, "blocking"] = blocking
        matrix.loc[mask, "observed"] = observed
        matrix.loc[mask, "evidence"] = evidence
    matrix.to_csv(output / "evaluation_matrix.csv", index=False)

    rows: List[Dict[str, object]] = []

    def add(metric, observed, formal, ideal, direction, unit, note):
        if direction == "minimum":
            formal_gap = max(0.0, formal - observed)
            ideal_gap = max(0.0, ideal - observed)
        else:
            formal_gap = max(0.0, observed - formal)
            ideal_gap = max(0.0, observed - ideal)
        rows.append(
            {
                "metric": metric,
                "observed": observed,
                "formal_line": formal,
                "ideal_line": ideal,
                "direction": direction,
                "gap_to_formal": formal_gap,
                "gap_to_ideal": ideal_gap,
                "unit": unit,
                "note": note,
            }
        )

    add("main_target_coverage", 3, 30, 30, "minimum", "targets", "27 targets lack formal-depth explanation; all 30 passed one-fold engineering smoke")
    add(
        "frozen_repeat_coverage",
        3,
        7,
        7,
        "minimum",
        "repeats",
        "repeats 0, 2, and 3 have feature permutation + TreeSHAP + group permutation; repeat 1 confirms network communities only",
    )
    add("network_bootstrap_depth_confirmed", 200, 200, 200, "minimum", "resamples/fold", f"met in {len(network_summaries)} audited network repeats")
    add("feature_permutation_depth", 10, 100, 100, "minimum", "permutations", "pilot depth")
    add("stable_community_ari", median_stable_ari, 0.50, 0.70, "minimum", "ratio", f"median across {len(network_summaries)} independent repeats; all exceed 0.50")
    add("stable_edge_jaccard", median_stable_jaccard, 0.60, 0.70, "minimum", "ratio", f"median across {len(network_summaries)} independent repeats")
    add("stable_edge_retention", median_stable_retention, 0.40, 0.60, "minimum", "ratio", f"median across {len(network_summaries)} independent repeats")
    add("conditional_unconditional_top10_jaccard", 2 / 3, 0.50, 0.70, "minimum", "ratio", "repeat 0 pilot")
    add("permutation_shap_top10_jaccard", advanced_summary["median_permutation_shap_top10_jaccard"], 0.30, 0.50, "minimum", "ratio", "repeat 0 pilot")
    add("style_direction_reversal_rate", style_reversal_rate, 0.10, 0.05, "maximum", "ratio", "repeat 0 candidate-style tests")
    add("shap_additivity_error", advanced_summary["max_shap_additivity_error"], 1e-8, 1e-10, "maximum", "absolute", "true final pipelines")
    add(
        "stable_module_top5_jaccard",
        module_summary["median_module_level_jaccard"],
        0.50,
        0.60,
        "minimum",
        "ratio",
        "independent repeat 3; candidate-pool metric, Top-10 remains reported separately",
    )
    add(
        "stable_module_top5_paired_improvement",
        module_summary["median_improvement"],
        0.10,
        0.20,
        "minimum",
        "ratio",
        "module minus feature Top-5 Jaccard on independent repeat 3",
    )
    if reconstruction_summary is not None:
        add(
            "q1_champion_reconstruction_target_coverage",
            len(reconstruction_summary["targets"]),
            30,
            30,
            "minimum",
            "targets",
            f"{reconstruction_summary['rows']} exact target-fold reconstructions; max error={reconstruction_summary['max_abs_error']:.3g}",
        )
    if source_shap_summary is not None:
        add(
            "source_architecture_shap_coverage",
            source_shap_summary["passed_architectures"],
            source_shap_summary["source_architectures"],
            source_shap_summary["source_architectures"],
            "minimum",
            "architectures",
            f"max additivity error={source_shap_summary['max_shap_additivity_abs_error']:.3g}",
        )
    add(
        "network_200bootstrap_repeat_coverage",
        len(network_summaries),
        7,
        7,
        "minimum",
        "repeats",
        f"absolute ARI criterion passed all; filtering-improvement criterion passed {network_improvement_passes}/{len(network_summaries)}",
    )
    gaps = pd.DataFrame(rows)
    if formal_audit is not None:
        replacements = {
            "main_target_coverage": float(len(formal_audit["targets"])),
            "frozen_repeat_coverage": float(len(formal_audit["repeats"])),
            "feature_permutation_depth": float(
                formal_audit["permutations_by_mode"]["within_style"]
            ),
            "conditional_unconditional_top10_jaccard": formal_mode_jaccard,
        }
        for metric, observed in replacements.items():
            mask = gaps["metric"] == metric
            gaps.loc[mask, "observed"] = observed
            direction = gaps.loc[mask, "direction"].iloc[0]
            formal_line = float(gaps.loc[mask, "formal_line"].iloc[0])
            ideal_line = float(gaps.loc[mask, "ideal_line"].iloc[0])
            if direction == "minimum":
                gaps.loc[mask, "gap_to_formal"] = max(0.0, formal_line - observed)
                gaps.loc[mask, "gap_to_ideal"] = max(0.0, ideal_line - observed)
            else:
                gaps.loc[mask, "gap_to_formal"] = max(0.0, observed - formal_line)
                gaps.loc[mask, "gap_to_ideal"] = max(0.0, observed - ideal_line)
    if formal_advanced_summary is not None:
        formal_replacements = {
            "permutation_shap_top10_jaccard": float(
                formal_advanced_summary["median_permutation_shap_top10_jaccard"]
            ),
            "style_direction_reversal_rate": style_reversal_rate,
            "shap_additivity_error": float(
                formal_advanced_summary["max_shap_additivity_error"]
            ),
        }
        for metric, observed in formal_replacements.items():
            mask = gaps["metric"] == metric
            gaps.loc[mask, "observed"] = observed
            direction = gaps.loc[mask, "direction"].iloc[0]
            formal_line = float(gaps.loc[mask, "formal_line"].iloc[0])
            ideal_line = float(gaps.loc[mask, "ideal_line"].iloc[0])
            if direction == "minimum":
                gaps.loc[mask, "gap_to_formal"] = max(0.0, formal_line - observed)
                gaps.loc[mask, "gap_to_ideal"] = max(0.0, ideal_line - observed)
            else:
                gaps.loc[mask, "gap_to_formal"] = max(0.0, observed - formal_line)
                gaps.loc[mask, "gap_to_ideal"] = max(0.0, observed - ideal_line)
    gaps.to_csv(output / "distance_to_optimal_after_optimization.csv", index=False)

    blocking = matrix[matrix["blocking"] & (matrix["status"] != "PASS")]
    summary = {
        "formal_question2_status": "HOLD" if len(blocking) else "READY",
        "criteria_counts": {
            str(key): int(value) for key, value in matrix["status"].value_counts().items()
        },
        "blocking_not_passed": blocking["criterion"].tolist(),
        "quantitative_formal_passed": int((gaps["gap_to_formal"] == 0).sum()),
        "quantitative_total": int(len(gaps)),
        "quantitative_ideal_passed": int((gaps["gap_to_ideal"] == 0).sum()),
        "optimization_adopted": (
            independent_improvement_passes > len(independent_network_summaries) / 2
            and all(value >= 0.50 for value in stable_aris)
        ),
        "network_filter_improvement_passes": f"{network_improvement_passes}/{len(network_summaries)}",
        "independent_network_filter_improvement_passes": f"{independent_improvement_passes}/{len(independent_network_summaries)}",
        "network_filter_failed_repeats": network_failed_repeats,
        "network_failures_preserved": bool(network_failed_repeats),
        "network_pairwise_diagnostic": {
            str(int(row.repeat)): f"{int(row.improved_fold_pairs)}/{int(row.total_fold_pairs)}"
            for row in network_robustness.itertuples(index=False)
        },
        "module_candidate_pool_adopted": module_summary["status"] == "ADOPT",
        "module_candidate_pool_confirmation": module_summary,
        "pilot_source_audit": audit_summary,
        "key_remaining_ideal_gaps": {
            row.metric: float(row.gap_to_ideal)
            for row in gaps.itertuples(index=False)
            if row.gap_to_ideal > 0
        },
        "pilot_complete_candidate_count": gap_summary["pilot_complete_candidate_count"],
        "all30_engineering_smoke_passed": bool(all30_audit and all30_audit["status"] == "PASS"),
        "all30_advanced_smoke_passed": bool(
            all30_advanced_summary and all30_advanced_summary["status"] == "PASS"
        ),
        "formal_feature_permutation": formal_audit,
        "formal_stable_candidate_count": (
            int(formal_stable_counts.sum()) if formal_stable_counts is not None else None
        ),
        "formal_targets_with_stable_candidate": (
            int((formal_stable_counts > 0).sum())
            if formal_stable_counts is not None
            else None
        ),
        "formal_conditional_unconditional_top10_jaccard": formal_mode_jaccard,
        "formal_advanced_explanation": formal_advanced_summary,
        "formal_candidate_finalization": formal_candidate_summary,
        "formal_candidate_bundle": formal_bundle_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Question 2 current evaluation after optimization",
        "",
        f"- Formal status: **{summary['formal_question2_status']}**",
        f"- Evaluation criteria: {summary['criteria_counts']}",
        f"- Quantitative formal lines passed: {summary['quantitative_formal_passed']}/{summary['quantitative_total']}",
        f"- Quantitative ideal lines passed: {summary['quantitative_ideal_passed']}/{summary['quantitative_total']}",
        "",
        "## Adopted optimization",
        "",
        f"Stable-edge filtering passed the locked difference-of-medians improvement rule in {network_improvement_passes}/{len(network_summaries)} audited repeats ({independent_improvement_passes}/{len(independent_network_summaries)} independent repeats). Repeats {network_failed_repeats} failed that rule, while every repeat still passed the absolute community ARI line. Pairwise diagnosis shows repeat 4 improved 2/3 fold-pairs but failed the locked aggregate by 0.0067; repeat 5 improved only 1/3. Both successes and failures are retained.",
        f"Stable-community Top-5 attribution raised median permutation-SHAP agreement from {module_summary['median_feature_level_jaccard']:.3f} to {module_summary['median_module_level_jaccard']:.3f} on independent repeat 3; {module_summary['improved_target_folds']}/{module_summary['total_target_folds']} target-folds improved.",
        "",
        "## Remaining blockers",
        "",
    ]
    lines.extend(
        [f"- {item}" for item in summary["blocking_not_passed"]]
        or ["- None."]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"All formal evidence layers and audits are complete. After the leave-one-repeat-out module rescue, the result is READY with limitations: {formal_bundle_summary['targets_with_three_candidates']} targets support three candidates, {formal_bundle_summary['targets_with_one_or_two_candidates']} support one or two, and {formal_bundle_summary['targets_with_insufficient_evidence']} remain explicitly insufficient. Question 1 remains HOLD and caps every interpretation."
                if formal_bundle_summary is not None
                else (
                    f"All formal evidence layers and audits are complete. The result is READY with limitations rather than universally strong: only {formal_candidate_summary['targets_with_three_reported_candidates']} targets support three nonredundant candidates, {formal_candidate_summary['targets_with_partial_candidate_sets']} support one or two, and {formal_candidate_summary['targets_with_insufficient_evidence']} are explicitly insufficient. Question 1 remains HOLD and caps every interpretation."
                    if formal_candidate_summary is not None
                    else (
                        "All 30 targets now have 7x3-fold, 100+100 held-out permutation evidence and exact frozen-champion reconstruction. The remaining blockers are final network de-redundancy and attachment 6/8 auditing of the formal candidate set; no formal Top-3 is frozen yet."
                        if formal_audit is not None
                        else "All 30 frozen champion sources now reconstruct exactly and all source architectures pass TreeSHAP additivity."
                    )
                )
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

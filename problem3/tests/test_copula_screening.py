import pandas as pd

from beer_q3.copula_screening import (
    _factor_method_name,
    _moment_mixture_method_name,
    _rank_stratified_method_name,
    _skeptic_method_name,
    apply_relaxed_pareto_confirmation,
    build_selection_audit,
)


def test_selection_audit_requires_every_protection_condition():
    scorecard = pd.DataFrame(
        [
            {
                "method": "base",
                "all_hard_gates_all_cells": False,
                "wasserstein_core_pass_rate": 0.4,
                "real_reference_distance_pass_rate": 0.6,
                "style_hit_rate_generated_mean": 0.1,
            },
            {
                "method": "candidate",
                "all_hard_gates_all_cells": True,
                "wasserstein_core_pass_rate": 0.5,
                "real_reference_distance_pass_rate": 0.7,
                "style_hit_rate_generated_mean": 0.4,
            },
        ]
    )
    distances = pd.DataFrame(
        {
            "method": ["base", "base", "candidate", "candidate"],
            "metric": [
                "spearman_structure_rmse",
                "q2_edge_strength_mae",
                "spearman_structure_rmse",
                "q2_edge_strength_mae",
            ],
            "status": ["PASS", "PASS", "PASS", "FAIL"],
        }
    )
    audit = build_selection_audit(
        scorecard,
        distances,
        baseline_method="base",
        candidate_methods=["candidate"],
        minimum_style_hit_gain=0.15,
        maximum_structure_pass_rate_drop=0.05,
        minimum_q2_edge_strength_pass_rate=0.8,
        minimum_q2_edge_strength_gain=0.05,
    )
    assert not bool(audit.loc[0, "q2_edge_strength_protected"])
    assert not bool(audit.loc[0, "q2_edge_strength_recovered"])
    assert not bool(audit.loc[0, "all_predeclared_conditions"])


def test_factor_method_name_records_poet_and_distance_controls():
    assert _factor_method_name(5.0, 10, 0.05, 1.0) == (
        "factor_rank_gaussian_copula_prior_5_factors_10_poet_c1_distance_q0.05"
    )


def test_skeptic_method_name_records_fixed_distance_control():
    assert _skeptic_method_name(5.0, 0.05, 1.0) == (
        "skeptic_spearman_rank_gaussian_copula_prior_5_distance_q0.05"
    )
    assert _skeptic_method_name(5.0, 0.05, 0.5) == (
        "skeptic_spearman_rank_gaussian_copula_prior_5_blend_0.5_distance_q0.05"
    )
    assert _skeptic_method_name(5.0, 0.05, 0.5, 10).endswith(
        "distance_q0.05_sensory_anchor10"
    )


def test_moment_mixture_method_name_records_all_locked_controls():
    assert _moment_mixture_method_name(5.0, 0.5, 2, 5, 0.5, 0.05) == (
        "moment_preserving_mixture2_pc5_cap0.5_skeptic_blend0.5"
        "_prior5_distance_q0.05"
    )


def test_rank_stratified_method_name_records_parent_controls():
    assert _rank_stratified_method_name(5.0, 0.5, 0.05) == (
        "rank_stratified_skeptic_blend0.5_prior5_distance_q0.05"
    )


def test_relaxed_pareto_confirmation_requires_gains_and_repeat_stability():
    methods = ["base", "candidate"]
    repeats = list(range(7))
    scorecard = pd.DataFrame(
        [
            {
                "method": "base",
                "real_reference_distance_pass_rate": 0.50,
                "wasserstein_core_pass_rate": 0.50,
                "style_hit_rate_generated_mean": 0.40,
            },
            {
                "method": "candidate",
                "real_reference_distance_pass_rate": 0.65,
                "wasserstein_core_pass_rate": 0.75,
                "style_hit_rate_generated_mean": 0.52,
            },
        ]
    )
    distance_rows = []
    gate_rows = []
    metric_rows = []
    distance_metrics = [
        "wasserstein_median",
        "wasserstein_q75",
        "wasserstein_max",
        "spearman_structure_rmse",
        "q2_edge_strength_mae",
        "q2_edge_loss_rate",
        "q2_edge_sign_agreement",
    ]
    for repeat in repeats:
        for method in methods:
            gate_rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "fold": 0,
                    "target_style": "Tripel",
                    "gate": "all",
                    "pass": True,
                }
            )
            for metric in distance_metrics:
                candidate = method == "candidate"
                status = "PASS" if candidate or metric in {
                    "wasserstein_max",
                    "spearman_structure_rmse",
                    "q2_edge_strength_mae",
                } else "FAIL"
                distance_rows.append(
                    {
                        "method": method,
                        "repeat": repeat,
                        "fold": 0,
                        "target_style": "Tripel",
                        "metric": metric,
                        "value": (
                            1.005
                            if candidate and metric == "spearman_structure_rmse"
                            else 0.995
                            if candidate and metric == "q2_edge_sign_agreement"
                            else 1.01
                            if candidate and metric in {
                                "q2_edge_strength_mae",
                                "q2_edge_loss_rate",
                            }
                            else 1.0
                        ),
                        "minimum_line": 1.1,
                        "status": status,
                    }
                )
            style_value = 0.55 if method == "candidate" else 0.40
            for metric, value in {
                "style_hit_rate_generated": style_value,
                "style_hit_rate_real": 0.60,
                "style_probability_gap": 0.10,
                "generated_unique_rate": 1.0,
            }.items():
                metric_rows.append(
                    {
                        "method": method,
                        "repeat": repeat,
                        "fold": 0,
                        "target_style": "Tripel",
                        "metric": metric,
                        "value": value,
                    }
                )
    audit = pd.DataFrame(
        [{"method": "candidate", "all_predeclared_conditions": True}]
    )
    result = apply_relaxed_pareto_confirmation(
        audit,
        scorecard,
        pd.DataFrame(distance_rows),
        pd.DataFrame(gate_rows),
        pd.DataFrame(metric_rows),
        baseline_method="base",
        candidate_methods=["candidate"],
        minimum_overall_distance_gain=0.08,
        minimum_wasserstein_core_gain=0.15,
        maximum_spearman_mean_relative_degradation=0.01,
        minimum_joint_repeat_wins=5,
        maximum_q2_edge_loss_pass_rate_drop=2 / 63,
        maximum_q2_edge_strength_mean_relative_degradation=0.02,
        maximum_q2_edge_loss_mean_relative_degradation=0.02,
        maximum_q2_edge_sign_mean_relative_degradation=0.02,
    )
    assert int(result.loc[0, "joint_repeat_wins"]) == 7
    assert bool(result.loc[0, "spearman_mean_within_relative_limit"])
    assert bool(result.loc[0, "q2_edge_strength_mean_within_relative_limit"])
    assert bool(result.loc[0, "q2_edge_loss_mean_within_relative_limit"])
    assert bool(result.loc[0, "q2_edge_sign_mean_within_relative_limit"])
    assert bool(result.loc[0, "all_predeclared_conditions"])

    pass_count_drop = pd.DataFrame(distance_rows)
    baseline_loss = (
        (pass_count_drop["method"] == "base")
        & (pass_count_drop["metric"] == "q2_edge_loss_rate")
    )
    pass_count_drop.loc[baseline_loss, "status"] = "PASS"
    candidate_loss = (
        (pass_count_drop["method"] == "candidate")
        & (pass_count_drop["metric"] == "q2_edge_loss_rate")
        & (pass_count_drop["repeat"] == repeats[0])
    )
    pass_count_drop.loc[candidate_loss, "status"] = "FAIL"
    strict_result = apply_relaxed_pareto_confirmation(
        audit,
        scorecard,
        pass_count_drop,
        pd.DataFrame(gate_rows),
        pd.DataFrame(metric_rows),
        baseline_method="base",
        candidate_methods=["candidate"],
        minimum_overall_distance_gain=0.08,
        minimum_wasserstein_core_gain=0.15,
        maximum_spearman_mean_relative_degradation=0.01,
        minimum_joint_repeat_wins=5,
        maximum_q2_edge_loss_pass_rate_drop=0.0,
        maximum_q2_edge_strength_mean_relative_degradation=0.02,
        maximum_q2_edge_loss_mean_relative_degradation=0.02,
        maximum_q2_edge_sign_mean_relative_degradation=0.02,
    )
    assert not bool(strict_result.loc[0, "q2_edge_loss_pass_rate_protected"])
    assert not bool(strict_result.loc[0, "all_predeclared_conditions"])

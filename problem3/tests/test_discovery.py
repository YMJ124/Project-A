import pandas as pd

from beer_q3.discovery import build_method_scorecard, summarize_discovery_metrics


def test_discovery_summary_bootstraps_outer_fold_values_only():
    metrics = pd.DataFrame(
        {
            "method": ["m"] * 3,
            "target_style": ["Tripel"] * 3,
            "metric": ["wasserstein_median"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    summary = summarize_discovery_metrics(metrics, bootstrap_resamples=100, seed=7)
    assert summary.loc[0, "n_outer_fold_values"] == 3
    assert summary.loc[0, "mean"] == 2.0
    assert summary.loc[0, "bootstrap_mean_ci_low"] <= 2.0
    assert summary.loc[0, "bootstrap_mean_ci_high"] >= 2.0


def test_method_scorecard_keeps_hard_gates_separate_from_effect_metrics():
    gates = pd.DataFrame(
        {
            "method": ["m", "m"],
            "repeat": [0, 0],
            "fold": [0, 0],
            "target_style": ["Tripel", "Tripel"],
            "pass": [True, False],
        }
    )
    distances = pd.DataFrame(
        {
            "method": ["m"],
            "metric": ["wasserstein_median"],
            "minimum_line": [1.0],
            "status": ["PASS"],
        }
    )
    metrics = pd.DataFrame(
        {
            "method": ["m"] * 4,
            "repeat": [0] * 4,
            "fold": [0] * 4,
            "target_style": ["Tripel"] * 4,
            "metric": [
                "style_hit_rate_generated",
                "style_hit_rate_real",
                "style_probability_gap",
                "generated_unique_rate",
            ],
            "value": [0.8, 0.7, -0.1, 1.0],
        }
    )
    scorecard = build_method_scorecard(gates, distances, metrics)
    assert not bool(scorecard.loc[0, "all_hard_gates_all_cells"])
    assert scorecard.loc[0, "real_reference_distance_pass_rate"] == 1.0
    assert not bool(scorecard.loc[0, "champion_eligible"])

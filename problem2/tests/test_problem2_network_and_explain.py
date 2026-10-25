import numpy as np
import pandas as pd

from beer_q2.advanced import aggregate_indicator_shap, grouped_permutation_source_indices
from beer_q2.explain import (
    _permutation_groups,
    batched_regression_metrics,
    permute_values_within_groups,
    safe_spearman,
    summarize_importance,
)
from beer_q2.formal_finalize import dereduplicate_candidates
from beer_q2.reconstruct import FrozenTargetPredictor, predict_frozen_targets
from beer_q2.network import benjamini_hochberg, build_fold_network, spearman_with_pvalues
from beer_q2.module_optimize import module_method_agreement
from beer_q2.source_audit import build_network_deredundancy


def test_benjamini_hochberg_known_values_and_nan():
    result = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(result[:3], [0.03, 0.04, 0.04])
    assert np.isnan(result[3])


def test_spearman_re_ranks_on_pairwise_complete_rows():
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 100.0, 4.0],
            "y": [1.0, 4.0, np.nan, 2.0],
            "z": [4.0, 3.0, 2.0, 1.0],
        }
    )
    rho, _, n_pair = spearman_with_pvalues(frame)
    assert np.allclose(rho, frame.corr(method="spearman"), equal_nan=True)
    assert n_pair[0, 1] == 3


def test_safe_spearman_returns_nan_for_constant_input_without_warning():
    assert np.isnan(safe_spearman([1, 1, 1], [1, 2, 3]))
    assert np.isclose(safe_spearman([1, 2, 3], [3, 2, 1]), -1.0)


def test_batched_metrics_match_scalar_metrics():
    truth = np.array([0.2, -0.4, 1.1, 0.7])
    predictions = np.array([[0.1, -0.2, 0.9, 0.8], [0.7, 0.3, -0.1, -0.4]])
    batched = batched_regression_metrics(truth, predictions)
    scalar = [
        {
            "rmse": np.sqrt(np.mean((row - truth) ** 2)),
            "r2": 1.0 - np.sum((row - truth) ** 2) / np.sum((truth - truth.mean()) ** 2),
            "spearman": safe_spearman(truth, row),
        }
        for row in predictions
    ]
    for metric in ["rmse", "r2", "spearman"]:
        assert np.allclose(batched[metric], [item[metric] for item in scalar])


def test_shared_predictor_evaluates_common_model_once():
    class CountingModel:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            return np.tile(self.values, (len(X), 1))

    shared = CountingModel([1.0, 2.0])
    predictors = [
        FrozenTargetPredictor("a", 0, 2, "fold_tuned_extra_trees", shared),
        FrozenTargetPredictor("b", 1, 2, "fixed_scale_1.25", shared, prediction_scale=1.25),
    ]
    result = predict_frozen_targets(predictors, pd.DataFrame({"x": [0, 1, 2]}))
    assert shared.calls == 1
    assert result.tolist() == [[1.0, 2.5], [1.0, 2.5], [1.0, 2.5]]


def test_within_style_permutation_pools_predeclared_small_cells():
    styles = np.array(["A", "A", "A", "B", "C"], dtype=object)
    groups = _permutation_groups(styles, "within_style", minimum_group_n=3)
    assert groups.tolist() == ["A", "A", "A", "__pooled_rare__", "__pooled_rare__"]
    values = np.arange(5)
    permuted = permute_values_within_groups(values, groups, np.random.default_rng(3))
    assert sorted(permuted[:3]) == [0, 1, 2]
    assert sorted(permuted[3:]) == [3, 4]


def test_small_fold_network_has_signed_edges_and_connected_communities():
    rng = np.random.default_rng(7)
    base = rng.normal(size=30)
    frame = pd.DataFrame(
        {
            "x": base,
            "x_copy": base + rng.normal(scale=0.01, size=30),
            "x_negative": -base + rng.normal(scale=0.01, size=30),
            "noise": rng.normal(size=30),
        }
    )
    styles = np.repeat(["A", "B", "C"], 10)
    result = build_fold_network(
        frame,
        styles,
        repeat=0,
        fold=0,
        thresholds=[0.3, 0.5],
        primary_threshold=0.5,
        fdr_alpha=0.05,
        bootstrap_repeats=3,
        bootstrap_seed=11,
        community_method="greedy",
    )
    assert {"positive", "negative"}.issubset(set(result.edges["sign"]))
    assert result.nodes["community_connected"].all()
    assert result.sensitivity["n_edges"].max() >= 2


def test_importance_summary_marks_pilot_only_candidates():
    rows = []
    for fold in range(3):
        for feature, delta, rho in [("strong", 0.2, 0.5), ("weak", -0.01, -0.1)]:
            for permutation in range(2):
                rows.append(
                    {
                        "target": "A_hops_all",
                        "grade": "A",
                        "source_model": "affine_calibration",
                        "mode": "within_style",
                        "feature": feature,
                        "repeat": 0,
                        "fold": fold,
                        "permutation": permutation,
                        "delta_rmse": delta,
                        "relative_delta_rmse": delta,
                        "heldout_feature_target_spearman": rho,
                    }
                )
    summary, view = summarize_importance(
        pd.DataFrame(rows),
        primary_mode="within_style",
        top_k=[3, 10],
        top10_frequency_minimum=0.6,
        positive_importance_frequency_minimum=0.8,
        direction_consistency_minimum=0.7,
    )
    strong = summary[summary["feature"] == "strong"].iloc[0]
    assert bool(strong["passes_pilot_stability"])
    assert view["formal_top3_status"].eq("NOT_EVALUATED_PILOT_ONLY").all()


def test_missing_indicator_shap_is_returned_to_source_feature():
    transformed = np.array([[1.0, 2.0, 3.0, 10.0, 20.0]])
    result = aggregate_indicator_shap(
        transformed, n_original_features=3, indicator_features=[0, 2]
    )
    assert result.tolist() == [[11.0, 2.0, 23.0]]
    assert np.isclose(result.sum(), transformed.sum())


def test_missing_indicator_shap_supports_multioutput_cache_shape():
    transformed = np.array(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ]
    )
    result = aggregate_indicator_shap(
        transformed, n_original_features=3, indicator_features=[1]
    )
    assert result.shape == (1, 3, 2)
    assert result.tolist() == [[[1.0, 10.0], [6.0, 60.0], [3.0, 30.0]]]
    assert np.allclose(result.sum(axis=1), transformed.sum(axis=1))


def test_grouped_permutation_preserves_joint_rows_within_style_groups():
    groups = np.array(["A", "A", "A", "B", "B"], dtype=object)
    source = grouped_permutation_source_indices(groups, np.random.default_rng(9))
    assert sorted(source[:3].tolist()) == [0, 1, 2]
    assert sorted(source[3:].tolist()) == [3, 4]


def test_network_deredundancy_groups_candidates_that_co_cluster_in_two_of_three_folds():
    candidates = pd.DataFrame(
        {
            "target": ["A", "A", "A"],
            "grade": ["A", "A", "A"],
            "feature": ["x", "y", "z"],
            "pilot_evidence_complete": [True, True, True],
            "mean_delta_rmse": [0.3, 0.2, 0.1],
            "interpretation_ceiling": ["high_candidate"] * 3,
        }
    )
    nodes = pd.DataFrame(
        [
            {"repeat": 1, "fold": fold, "feature": feature, "variant": "stable_filtered", "community": community}
            for fold, assignments in enumerate(
                [
                    {"x": 0, "y": 0, "z": 1},
                    {"x": 2, "y": 2, "z": 3},
                    {"x": 4, "y": 5, "z": 6},
                ]
            )
            for feature, community in assignments.items()
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "fold": fold,
                "variant": "stable_filtered",
                "feature_left": "x",
                "feature_right": "y",
                "rho": 0.8,
                "sign": "positive",
                "bootstrap_frequency": 1.0,
            }
            for fold in range(3)
        ]
    )
    result = build_network_deredundancy(candidates, nodes, edges)
    pair = result["overlap"].query("feature_left == 'x' and feature_right == 'y'").iloc[0]
    assert bool(pair["network_redundant"])
    assert len(result["modules"]) == 2
    xy = result["modules"].query("representative_feature == 'x'").iloc[0]
    assert xy["related_pilot_candidates"] == "y"


def test_module_agreement_aggregates_correlated_feature_shap_before_ranking():
    shap = pd.DataFrame(
        {
            "target": ["A"] * 4,
            "repeat": [2] * 4,
            "fold": [0] * 4,
            "feature": ["x1", "x2", "y", "z"],
            "abs_shap_value": [0.4, 0.3, 0.5, 0.1],
        }
    )
    nodes = pd.DataFrame(
        {
            "repeat": [2] * 4,
            "fold": [0] * 4,
            "feature": ["x1", "x2", "y", "z"],
            "community": [0, 0, 1, 2],
            "variant": ["stable_filtered"] * 4,
        }
    )
    group = pd.DataFrame(
        {
            "target": ["A"] * 3,
            "repeat": [2] * 3,
            "fold": [0] * 3,
            "community": [0, 1, 2],
            "delta_rmse": [0.8, 0.4, 0.1],
        }
    )
    feature = pd.DataFrame(
        {
            "target": ["A"],
            "repeat": [2],
            "fold": [0],
            "permutation_shap_top2_jaccard": [1 / 3],
        }
    )
    result = module_method_agreement(shap, group, nodes, feature, top_k=2)
    assert np.isclose(result.iloc[0]["module_level_jaccard"], 1.0)
    assert result.iloc[0]["module_minus_feature"] > 0


def test_formal_deredundancy_does_not_force_three_candidates():
    evidence = pd.DataFrame(
        {
            "target": ["A", "A", "B"],
            "grade": ["A", "A", "B"],
            "source_model": ["tree", "tree", "tree"],
            "feature": ["x", "y", "z"],
            "mean_delta_rmse": [0.2, 0.1, 0.3],
            "formal_evidence_complete": [True, True, False],
        }
    )
    rows = []
    for repeat in range(7):
        for fold in range(3):
            rows.extend(
                [
                    {"repeat": repeat, "fold": fold, "feature": "x", "community": 1},
                    {"repeat": repeat, "fold": fold, "feature": "y", "community": 1},
                    {"repeat": repeat, "fold": fold, "feature": "z", "community": 2},
                ]
            )
    modules, overlap, decisions = dereduplicate_candidates(
        evidence, pd.DataFrame(rows)
    )
    assert len(modules) == 1
    assert modules.iloc[0]["representative_feature"] == "x"
    assert modules.iloc[0]["related_stable_alternatives"] == "y"
    assert bool(overlap.iloc[0]["network_redundant"])
    by_target = decisions.set_index("target")
    assert by_target.loc["A", "reported_candidate_count"] == 1
    assert by_target.loc["A", "decision"] == "FORMAL_PARTIAL_EVIDENCE"
    assert by_target.loc["B", "reported_candidate_count"] == 0
    assert by_target.loc["B", "decision"] == "INSUFFICIENT_EVIDENCE"
    assert not decisions["forced_to_three"].any()

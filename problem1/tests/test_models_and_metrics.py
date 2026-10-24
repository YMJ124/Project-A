"""Baseline isolation and metric tests.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

import numpy as np
import pandas as pd
import pytest

from beer_q1.evaluation import bootstrap_metric_intervals, regression_metrics
from beer_q1.models import (
    AromaFlavourPairedExtraTreesRegressor,
    CrossFittedAffineCalibratedRegressor,
    CrossFittedMechanismResidualExtraTreesRegressor,
    CrossFittedTargetStackingExtraTreesRegressor,
    FeatureSubsetTargetwiseExtraTreesRegressor,
    FoldStandardizedGroupSummaryTransformer,
    FoldSkewLogTransformer,
    RankBlendedTargetExtraTreesRegressor,
    SensoryFamilyExtraTreesRegressor,
    ThresholdMechanismTransformer,
    build_model,
    ensure_2d_prediction,
    predict_style_mean,
)
from beer_q1.runner import _validate_fixed_mode


def test_style_mean_never_uses_validation_targets() -> None:
    y_train = pd.DataFrame({"a": [1.0, 3.0, 10.0], "b": [2.0, 4.0, 20.0]})
    styles_train = pd.Series(["x", "x", "y"])
    styles_valid = pd.Series(["x", "missing", "y"])
    prediction = predict_style_mean(y_train, styles_train, styles_valid)
    np.testing.assert_allclose(prediction[0], [2.0, 3.0])
    np.testing.assert_allclose(prediction[1], [14.0 / 3.0, 26.0 / 3.0])
    np.testing.assert_allclose(prediction[2], [10.0, 20.0])


def test_regression_metrics_are_exact_for_perfect_prediction() -> None:
    values = np.array([-1.0, 0.0, 1.0])
    metrics = regression_metrics(values, values.copy())
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["spearman"] == 1.0


def test_pls1_prediction_has_samples_by_targets_shape() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, 8))
    y = rng.normal(size=(30, 3))
    model = build_model("pls1", {"n_components": 2}, seed=9, n_jobs=1)
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:4]), n_targets=3)
    assert prediction.shape == (4, 3)


def test_targetwise_extra_trees_prediction_shape() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(24, 6))
    y = rng.normal(size=(24, 3))
    model = build_model(
        "extra_trees_targetwise",
        {"n_estimators": 8, "max_features": 0.5, "min_samples_leaf": 2},
        seed=19,
        n_jobs=1,
    )
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:5]), n_targets=3)
    assert prediction.shape == (5, 3)


def test_targetwise_selectk_extra_trees_is_fold_local_and_predicts() -> None:
    rng = np.random.default_rng(29)
    X = rng.normal(size=(32, 10))
    y = np.column_stack(
        [X[:, 0] + rng.normal(scale=0.1, size=32), X[:, 7] ** 2]
    )
    model = build_model(
        "extra_trees_targetwise_selectk",
        {"k": 4, "n_estimators": 8, "max_features": 0.5, "min_samples_leaf": 2},
        seed=29,
        n_jobs=1,
    )
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:5]), n_targets=2)
    assert prediction.shape == (5, 2)
    assert all(
        estimator.named_steps["selector"].get_support().sum() == 4
        for estimator in model.estimators_
    )


def test_fixed_feature_subset_extra_trees_uses_only_declared_columns() -> None:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(36, 8))
    y = np.column_stack([X[:, 1] ** 2, X[:, 6] + rng.normal(scale=0.1, size=36)])
    model = build_model(
        "extra_trees_targetwise_subset",
        {
            "feature_indices": [1, 3, 6],
            "n_estimators": 8,
            "max_features": 1.0,
            "min_samples_leaf": 2,
        },
        seed=43,
        n_jobs=1,
    )
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:4]), n_targets=2)
    subset_model = model.named_steps["model"]
    assert isinstance(subset_model, FeatureSubsetTargetwiseExtraTreesRegressor)
    assert prediction.shape == (4, 2)
    np.testing.assert_array_equal(subset_model.feature_indices_, [1, 3, 6])


def test_fold_standardized_group_summaries_append_declared_features() -> None:
    rng = np.random.default_rng(47)
    X = rng.normal(size=(40, 7))
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 4] - X[:, 5]])
    model = build_model(
        "extra_trees_targetwise_group_summaries",
        {
            "groups": [[0, 1], [4, 5, 6]],
            "n_estimators": 8,
            "max_features": 0.5,
            "min_samples_leaf": 2,
        },
        seed=47,
        n_jobs=1,
    )
    model.fit(X, y)
    transformed = model.named_steps["group_summaries"].transform(
        model.named_steps["imputer"].transform(X[:3])
    )
    prediction = ensure_2d_prediction(model.predict(X[:3]), n_targets=2)
    assert transformed.shape[1] == X.shape[1] + 2
    assert prediction.shape == (3, 2)


def test_threshold_mechanism_features_are_finite_and_fold_local() -> None:
    rng = np.random.default_rng(53)
    X = np.abs(rng.normal(size=(42, 9)))
    transformer = ThresholdMechanismTransformer(
        feature_indices=[1, 4, 7],
        thresholds=[0.03, 0.02, 0.04],
        matrix_indices=[0, 2],
        append_original=False,
    )
    transformed = transformer.fit_transform(X[:30])
    held_out = transformer.transform(X[30:])
    assert transformed.shape == (30, 22)
    assert held_out.shape == (12, 22)
    assert np.isfinite(transformed).all()
    assert np.isfinite(held_out).all()
    np.testing.assert_allclose(
        transformer.matrix_means_, X[:30, [0, 2]].mean(axis=0)
    )


def test_threshold_mechanism_models_predict_multiple_targets() -> None:
    rng = np.random.default_rng(59)
    X = np.abs(rng.normal(size=(54, 9)))
    y = np.column_stack(
        [
            np.log1p(X[:, 1] / 0.03) + rng.normal(scale=0.1, size=54),
            X[:, 4] * X[:, 7] + rng.normal(scale=0.1, size=54),
        ]
    )
    common = {
        "feature_indices": [1, 4, 7],
        "thresholds": [0.03, 0.02, 0.04],
        "matrix_indices": [0, 2],
    }
    appended = build_model(
        "extra_trees_targetwise_threshold_mechanism",
        {
            **common,
            "n_estimators": 8,
            "max_features": 0.5,
            "min_samples_leaf": 2,
        },
        seed=59,
        n_jobs=1,
    )
    appended.fit(X, y)
    assert ensure_2d_prediction(appended.predict(X[:5]), 2).shape == (5, 2)

    shared = build_model(
        "extra_trees_threshold_mechanism",
        {
            **common,
            "n_estimators": 8,
            "max_features": 0.5,
            "min_samples_leaf": 2,
        },
        seed=59,
        n_jobs=1,
    )
    shared.fit(X, y)
    assert ensure_2d_prediction(shared.predict(X[:5]), 2).shape == (5, 2)

    residual = build_model(
        "threshold_mechanism_residual_extra_trees",
        {
            **common,
            "ridge_alpha": 5.0,
            "residual_splits": 3,
            "n_estimators": 8,
            "max_features": 0.5,
            "min_samples_leaf": 2,
        },
        seed=59,
        n_jobs=1,
    )
    residual.fit(X, y)
    prediction = ensure_2d_prediction(residual.predict(X[:5]), 2)
    residual_model = residual.named_steps["model"]
    assert isinstance(
        residual_model, CrossFittedMechanismResidualExtraTreesRegressor
    )
    assert residual_model.mechanism_oof_predictions_.shape == y.shape
    assert prediction.shape == (5, 2)


def test_targetwise_mutual_information_selectk_detects_nonlinear_signal() -> None:
    rng = np.random.default_rng(37)
    X = rng.normal(size=(48, 10))
    y = np.column_stack(
        [X[:, 0] ** 2 + rng.normal(scale=0.05, size=48), np.sin(X[:, 7])]
    )
    model = build_model(
        "extra_trees_targetwise_selectk_mi",
        {"k": 4, "n_estimators": 8, "max_features": 0.5, "min_samples_leaf": 2},
        seed=37,
        n_jobs=1,
    )
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:5]), n_targets=2)
    assert prediction.shape == (5, 2)
    assert all(
        estimator.named_steps["selector"].get_support().sum() == 4
        for estimator in model.estimators_
    )


def test_cross_fitted_affine_calibration_is_bounded_and_finite() -> None:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(48, 7))
    y = np.column_stack(
        [2.0 * X[:, 0] + rng.normal(scale=0.2, size=48), X[:, 1] ** 2]
    )
    model = build_model(
        "extra_trees_affine_calibrated",
        {
            "n_estimators": 12,
            "max_features": 0.5,
            "min_samples_leaf": 3,
            "calibration_splits": 3,
            "identity_weight": 0.5,
            "slope_min": 0.8,
            "slope_max": 1.5,
            "clip_to_training_range": True,
        },
        seed=31,
        n_jobs=1,
    )
    assert isinstance(model, CrossFittedAffineCalibratedRegressor)
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:6]), n_targets=2)
    assert prediction.shape == (6, 2)
    assert np.isfinite(prediction).all()
    assert np.all(model.calibration_slopes_ >= 0.8)
    assert np.all(model.calibration_slopes_ <= 1.5)
    assert np.all(prediction <= y.max(axis=0) + 1e-12)
    assert np.all(prediction >= y.min(axis=0) - 1e-12)


def test_aroma_flavour_paired_extra_trees_preserves_target_layout() -> None:
    rng = np.random.default_rng(35)
    X = rng.normal(size=(42, 6))
    y = pd.DataFrame(
        {
            "A_alpha": X[:, 0] + rng.normal(scale=0.1, size=42),
            "A_beta": X[:, 1] + rng.normal(scale=0.1, size=42),
            "F_alpha": X[:, 0] - X[:, 2] + rng.normal(scale=0.1, size=42),
            "F_beta": X[:, 1] + X[:, 3] + rng.normal(scale=0.1, size=42),
        }
    )
    model = build_model(
        "extra_trees_paired",
        {"n_estimators": 12, "max_features": 0.5, "min_samples_leaf": 2},
        seed=35,
        n_jobs=1,
    )
    model.fit(X, y)
    paired = model.named_steps["model"]
    assert isinstance(paired, AromaFlavourPairedExtraTreesRegressor)
    assert len(paired.models_) == 2
    prediction = ensure_2d_prediction(model.predict(X[:5]), n_targets=4)
    assert prediction.shape == (5, 4)
    assert np.isfinite(prediction).all()


def test_rank_blended_extra_trees_only_replaces_declared_target() -> None:
    rng = np.random.default_rng(41)
    X = rng.normal(size=(60, 8))
    skewed = np.exp(1.5 * X[:, 0])
    y = np.column_stack([X[:, 1], X[:, 2], skewed])
    model = build_model(
        "extra_trees_rank_blended",
        {
            "target_index": 2,
            "rank_weight": 0.5,
            "n_estimators": 16,
            "max_features": 0.5,
            "min_samples_leaf": 2,
            "random_state_override": 41,
        },
        seed=99,
        n_jobs=1,
    )
    model.fit(X, y)
    specialist = model.named_steps["model"]
    assert isinstance(specialist, RankBlendedTargetExtraTreesRegressor)
    transformed = model.named_steps["imputer"].transform(X[:7])
    base = ensure_2d_prediction(specialist.base_model_.predict(transformed), 3)
    prediction = ensure_2d_prediction(model.predict(X[:7]), 3)
    np.testing.assert_allclose(prediction[:, :2], base[:, :2])
    assert np.isfinite(prediction).all()
    assert not np.allclose(prediction[:, 2], base[:, 2])


def test_family_grouped_extra_trees_preserves_30_target_layout() -> None:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(48, 7))
    y = rng.normal(size=(48, 30))
    model = build_model(
        "extra_trees_family_grouped",
        {
            "n_estimators": 12,
            "max_features": 0.5,
            "min_samples_leaf": 2,
            "random_state_override": 43,
        },
        seed=99,
        n_jobs=1,
    )
    model.fit(X, y)
    grouped = model.named_steps["model"]
    assert isinstance(grouped, SensoryFamilyExtraTreesRegressor)
    assert [columns.tolist() for columns in grouped.groups_] == [
        [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
        [5, 6, 7, 8, 9, 20, 21, 22, 23, 24],
        [10, 11, 12, 13, 14, 25, 26, 27, 28, 29],
    ]
    prediction = ensure_2d_prediction(model.predict(X[:6]), n_targets=30)
    assert prediction.shape == (6, 30)
    assert np.isfinite(prediction).all()


def test_cross_fitted_target_stacking_uses_complete_oof_meta_features() -> None:
    rng = np.random.default_rng(47)
    X = rng.normal(size=(54, 8))
    latent = X[:, 0] - 0.5 * X[:, 1]
    y = rng.normal(scale=0.4, size=(54, 30))
    y[:, :5] += latent[:, None]
    y[:, 15:20] += latent[:, None]
    model = build_model(
        "extra_trees_target_stacked",
        {
            "n_estimators": 12,
            "max_features": 0.5,
            "min_samples_leaf": 2,
            "stacking_splits": 3,
            "ridge_alpha": 10.0,
            "identity_weight": 0.5,
            "random_state_override": 47,
        },
        seed=99,
        n_jobs=1,
    )
    model.fit(X, y)
    stacked = model.named_steps["model"]
    assert isinstance(stacked, CrossFittedTargetStackingExtraTreesRegressor)
    assert stacked.cross_fitted_predictions_.shape == y.shape
    assert np.isfinite(stacked.cross_fitted_predictions_).all()
    prediction = ensure_2d_prediction(model.predict(X[:6]), n_targets=30)
    assert prediction.shape == (6, 30)
    assert np.isfinite(prediction).all()


def test_unclipped_positive_affine_calibration_preserves_rank_order() -> None:
    rng = np.random.default_rng(37)
    X = rng.normal(size=(54, 8))
    y = np.column_stack(
        [1.8 * X[:, 0] + rng.normal(scale=0.3, size=54), X[:, 1] - X[:, 2]]
    )
    model = build_model(
        "extra_trees_affine_calibrated",
        {
            "n_estimators": 16,
            "max_features": 0.5,
            "min_samples_leaf": 3,
            "calibration_splits": 3,
            "identity_weight": 0.5,
            "slope_min": 0.8,
            "slope_max": 1.5,
            "clip_to_training_range": False,
        },
        seed=37,
        n_jobs=1,
    )
    model.fit(X, y)
    raw = ensure_2d_prediction(model.base_estimator_.predict(X), n_targets=2)
    calibrated = ensure_2d_prediction(model.predict(X), n_targets=2)
    for target_index in range(2):
        np.testing.assert_array_equal(
            np.argsort(raw[:, target_index]),
            np.argsort(calibrated[:, target_index]),
        )


def test_fold_skew_log_transformer_is_safe_for_held_out_negative() -> None:
    ordinary = np.linspace(0.0, 1.0, 30)
    extreme = ordinary.copy()
    extreme[-1] = 10_000.0
    X = np.column_stack([ordinary, extreme])
    transformer = FoldSkewLogTransformer(skew_threshold=2.0).fit(X)
    assert transformer.transform_mask_.tolist() == [False, True]
    held_out = np.array([[0.5, -2.0], [0.7, 20_000.0]])
    transformed = transformer.transform(held_out)
    assert np.isfinite(transformed).all()
    assert transformed[0, 1] < 0
    assert transformed[1, 1] < held_out[1, 1]


def test_skew_robust_ridge_prediction_is_finite() -> None:
    rng = np.random.default_rng(23)
    X = rng.lognormal(size=(36, 7))
    X[-1, 2] = 100_000.0
    y = rng.normal(size=(36, 3))
    model = build_model("ridge_skew_robust", {"alpha": 100.0}, seed=23, n_jobs=1)
    model.fit(X, y)
    prediction = ensure_2d_prediction(model.predict(X[:4]), n_targets=3)
    assert prediction.shape == (4, 3)
    assert np.isfinite(prediction).all()


def test_bootstrap_interval_is_degenerate_for_one_repeat() -> None:
    metrics = pd.DataFrame(
        {
            "target": ["t"],
            "model": ["m"],
            "rmse": [0.8],
            "mae": [0.6],
            "r2": [0.2],
            "spearman": [0.4],
        }
    )
    intervals = bootstrap_metric_intervals(metrics, n_bootstrap=10, seed=1)
    assert np.allclose(intervals["estimate"], intervals["ci95_low"])
    assert np.allclose(intervals["estimate"], intervals["ci95_high"])


def test_fixed_mode_rejects_multiple_non_baseline_families() -> None:
    with pytest.raises(ValueError, match="exactly one non-baseline model"):
        _validate_fixed_mode(
            ["ridge", "extra_trees"],
            {"ridge": [{"alpha": 1.0}], "extra_trees": [{"n_estimators": 10}]},
            enabled=True,
        )


def test_fixed_mode_rejects_parameter_search() -> None:
    with pytest.raises(ValueError, match="exactly one parameter candidate"):
        _validate_fixed_mode(
            ["ridge"],
            {"ridge": [{"alpha": 1.0}, {"alpha": 10.0}]},
            enabled=True,
        )

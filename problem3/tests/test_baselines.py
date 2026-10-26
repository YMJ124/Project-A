from pathlib import Path

import numpy as np
import pandas as pd
import warnings

from beer_q1.data import load_problem1_data
from beer_q3.baselines import (
    _factor_copula_covariance,
    _moment_preserving_mixture_parameters,
    _rank_stratified_gaussian,
    _module_correlation_covariance,
    _sample_covariance,
    _spearman_skeptic_covariance,
    calibrated_knn_jitter,
    factor_rank_gaussian_copula,
    knn_jitter,
    knn_correlated_innovation,
    moment_preserving_mixture_rank_gaussian_copula,
    module_block_rank_gaussian_copula,
    module_correlation_rank_gaussian_copula,
    pca_shrink_gaussian,
    rank_gaussian_copula,
    rank_stratified_skeptic_copula,
)
from beer_q3.derived import load_formula_config, recompute_derived_features
from beer_q3.evaluation import DERIVED_FEATURES, FoldGeometry, exact_copy_mask


ROOT = Path(__file__).resolve().parents[2]


def test_noncopy_baselines_recompute_derived_fields_and_preserve_schema():
    config = {
        "data": {
            "chemicals_csv": str(
                ROOT / "2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
            ),
            "sensory_csv": str(
                ROOT / "2026年MathorCup大数据挑战赛-赛道A初赛/附件/4.csv"
            ),
            "id_column": "beer",
            "style_column": "tasting_category_fine",
            "forbidden_input_columns": ["beer", "beer_id", "tasting_category_fine"],
            "target_scope": "main",
        },
        "runtime": {},
    }
    data = load_problem1_data(config)
    geometry = FoldGeometry.fit(data.X)
    defects = pd.read_csv(
        ROOT / "problem3/artifacts/preflight_v1/defect_constraint_registry.csv"
    )
    modules = pd.read_csv(
        ROOT / "problem3/interfaces/q2_handoff_v1/consensus_communities.csv"
    )
    formulas = load_formula_config(
        ROOT / "problem3/configs/derived_feature_formulas_v1.yaml"
    )
    common = {
        "training": data.X,
        "training_styles": data.styles,
        "target_style": "Tripel",
        "n_samples": 2,
        "full_geometry": geometry,
        "defect_registry": defects,
        "formula_config": formulas,
    }
    batches = [
        knn_jitter(**common, training_ids=data.ids, seed=31),
        calibrated_knn_jitter(**common, training_ids=data.ids, seed=34),
        knn_correlated_innovation(**common, training_ids=data.ids, seed=35),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        batches.extend(
            [
                pca_shrink_gaussian(**common, seed=32),
                rank_gaussian_copula(**common, seed=33),
                rank_gaussian_copula(
                    **common,
                    seed=41,
                    dependence_estimator="spearman_skeptic",
                    minimum_training_distance_quantile=0.05,
                ),
                moment_preserving_mixture_rank_gaussian_copula(
                    **common,
                    seed=42,
                    prior_strength=5.0,
                    skeptic_blend_weight=0.5,
                    n_components=2,
                    n_factors=5,
                    between_variance_cap=0.5,
                    min_component_size=2,
                    minimum_training_distance_quantile=0.05,
                ),
                rank_stratified_skeptic_copula(
                    **common,
                    seed=43,
                    prior_strength=5.0,
                    skeptic_blend_weight=0.5,
                    minimum_training_distance_quantile=0.05,
                ),
                module_block_rank_gaussian_copula(
                    **common, module_registry=modules, seed=36
                ),
                module_correlation_rank_gaussian_copula(
                    **common, module_registry=modules, seed=37
                ),
                factor_rank_gaussian_copula(**common, seed=38, n_factors=5),
                factor_rank_gaussian_copula(
                    **common,
                    seed=39,
                    n_factors=10,
                    minimum_training_distance_quantile=0.05,
                ),
                factor_rank_gaussian_copula(
                    **common,
                    seed=40,
                    n_factors=10,
                    minimum_training_distance_quantile=0.05,
                    poet_threshold_constant=1.0,
                ),
            ]
        )
    for batch in batches:
        assert list(batch.chemicals.columns) == list(data.X.columns)
        assert np.isfinite(batch.chemicals.to_numpy()).all()
        assert not exact_copy_mask(batch.chemicals, data.X).any()
        expected = recompute_derived_features(batch.chemicals, formulas)
        assert np.allclose(
            batch.chemicals.loc[:, list(DERIVED_FEATURES)],
            expected,
            rtol=0.0,
            atol=1e-10,
        )


def test_module_correlation_covariance_preserves_base_variances_and_is_psd():
    rng = np.random.default_rng(20261026)
    global_values = rng.normal(size=(40, 8))
    target_values = global_values[:9] + np.linspace(-0.4, 0.4, 8)
    modules = np.asarray(["a"] * 4 + ["b"] * 4)
    _, base_covariance = _sample_covariance(global_values, target_values, 5.0)
    _, candidate_covariance = _module_correlation_covariance(
        global_values,
        target_values,
        modules,
        prior_strength=5.0,
        module_weight=0.75,
    )
    assert np.allclose(
        np.diag(candidate_covariance),
        np.diag(base_covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.linalg.eigvalsh(candidate_covariance).min() >= -1e-8


def test_covariance_prior_can_change_covariance_without_changing_mean():
    rng = np.random.default_rng(20261027)
    global_values = rng.normal(size=(40, 8))
    target_values = global_values[:9] + 0.5
    mean_joint, covariance_joint = _sample_covariance(
        global_values, target_values, 5.0
    )
    mean_decoupled, covariance_decoupled = _sample_covariance(
        global_values,
        target_values,
        5.0,
        covariance_prior_strength=20.0,
    )
    assert np.allclose(mean_joint, mean_decoupled, rtol=0.0, atol=1e-12)
    assert not np.allclose(covariance_joint, covariance_decoupled)
    assert np.linalg.eigvalsh(covariance_decoupled).min() > 0.0


def test_factor_copula_covariance_preserves_base_mean_and_variances():
    rng = np.random.default_rng(20261028)
    global_values = rng.normal(size=(40, 12))
    target_values = global_values[:9] + np.linspace(-0.5, 0.5, 12)
    base_mean, base_covariance = _sample_covariance(
        global_values, target_values, 5.0
    )
    factor_mean, factor_covariance = _factor_copula_covariance(
        global_values,
        target_values,
        prior_strength=5.0,
        n_factors=5,
    )
    assert np.allclose(factor_mean, base_mean, rtol=0.0, atol=1e-12)
    assert np.allclose(
        np.diag(factor_covariance),
        np.diag(base_covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.linalg.eigvalsh(factor_covariance).min() > 0.0
    poet_mean, poet_covariance = _factor_copula_covariance(
        global_values,
        target_values,
        prior_strength=5.0,
        n_factors=5,
        poet_threshold_constant=1.0,
    )
    assert np.allclose(poet_mean, base_mean, rtol=0.0, atol=1e-12)
    assert np.allclose(
        np.diag(poet_covariance),
        np.diag(base_covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.linalg.eigvalsh(poet_covariance).min() > -1e-8


def test_spearman_skeptic_covariance_preserves_base_mean_variances_and_psd():
    rng = np.random.default_rng(20261029)
    global_values = rng.normal(size=(45, 9))
    global_values[:, 1] = np.exp(global_values[:, 0]) + rng.normal(0.0, 0.05, 45)
    target_values = global_values[:10].copy()
    target_values[:, -1] = 2.0  # exercise the constant-style fallback
    base_mean, base_covariance = _sample_covariance(
        global_values, target_values, 5.0
    )
    candidate_mean, candidate_covariance = _spearman_skeptic_covariance(
        global_values,
        target_values,
        prior_strength=5.0,
    )
    assert np.allclose(candidate_mean, base_mean, rtol=0.0, atol=1e-12)
    assert np.allclose(
        np.diag(candidate_covariance),
        np.diag(base_covariance),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.linalg.eigvalsh(candidate_covariance).min() > -1e-8


def test_moment_preserving_mixture_retains_parent_mean_and_covariance():
    rng = np.random.default_rng(20261030)
    global_values = rng.normal(size=(80, 10))
    target_values = np.vstack(
        [
            rng.normal(-1.5, 0.25, size=(8, 10)),
            rng.normal(1.5, 0.25, size=(8, 10)),
        ]
    )
    parent_mean, parent_covariance = _spearman_skeptic_covariance(
        global_values,
        target_values,
        prior_strength=5.0,
        blend_weight=0.5,
    )
    mean, within, shifts, weights, diagnostics = (
        _moment_preserving_mixture_parameters(
            global_values,
            target_values,
            prior_strength=5.0,
            blend_weight=0.5,
            n_components=2,
            n_factors=5,
            between_variance_cap=0.5,
            min_component_size=2,
            cluster_seed=20261030,
        )
    )
    reconstructed_mean = mean + np.sum(weights[:, None] * shifts, axis=0)
    between = sum(
        weights[index] * np.outer(shifts[index], shifts[index])
        for index in range(len(weights))
    )
    assert bool(diagnostics["mixture_active"])
    assert np.allclose(mean, parent_mean, rtol=0.0, atol=1e-12)
    assert np.allclose(reconstructed_mean, parent_mean, rtol=0.0, atol=1e-10)
    assert np.allclose(within + between, parent_covariance, rtol=0.0, atol=1e-9)
    assert np.linalg.eigvalsh(within).min() >= -1e-8
    assert float(diagnostics["latent_covariance_reconstruction_error"]) < 1e-9


def test_moment_preserving_mixture_falls_back_for_tiny_component_rule():
    rng = np.random.default_rng(20261031)
    global_values = rng.normal(size=(30, 6))
    target_values = rng.normal(size=(6, 6))
    parent_mean, parent_covariance = _spearman_skeptic_covariance(
        global_values,
        target_values,
        prior_strength=5.0,
        blend_weight=0.5,
    )
    mean, within, shifts, weights, diagnostics = (
        _moment_preserving_mixture_parameters(
            global_values,
            target_values,
            prior_strength=5.0,
            blend_weight=0.5,
            n_components=2,
            n_factors=5,
            between_variance_cap=0.5,
            min_component_size=4,
            cluster_seed=20261031,
        )
    )
    assert not bool(diagnostics["mixture_active"])
    assert np.allclose(mean, parent_mean)
    assert np.allclose(within, parent_covariance)
    assert shifts.shape == (1, 6)
    assert np.array_equal(weights, np.ones(1))


def test_rank_stratified_gaussian_matches_parent_marginal_moments():
    rng = np.random.default_rng(20261101)
    raw = rng.normal(size=(30, 7))
    covariance = np.cov(raw, rowvar=False) + np.eye(7) * 0.2
    mean = np.linspace(-0.4, 0.4, 7)
    sampled = _rank_stratified_gaussian(
        np.random.default_rng(20261102), mean, covariance, 9
    )
    assert np.allclose(sampled.mean(axis=0), mean, rtol=0.0, atol=1e-12)
    assert np.allclose(
        np.mean((sampled - mean) ** 2, axis=0),
        np.diag(covariance),
        rtol=0.0,
        atol=1e-12,
    )
    for column in range(sampled.shape[1]):
        assert len(np.unique(sampled[:, column])) == len(sampled)

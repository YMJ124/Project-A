"""Small-sample conditional generation baselines for Question 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .derived import apply_derived_features, recompute_derived_features
from .evaluation import DERIVED_FEATURES, FoldGeometry, exact_copy_mask


@dataclass(frozen=True)
class GeneratedBatch:
    chemicals: pd.DataFrame
    derived_expected: pd.DataFrame
    source_training_ids: tuple[object, ...]
    method: str
    diagnostics: Optional[Mapping[str, Any]] = None


def _model_limit(value: float, inverse_transform: str) -> float:
    if inverse_transform == "identity":
        return float(value)
    if inverse_transform == "pow10":
        return float(np.log10(value))
    if inverse_transform == "exp":
        return float(np.log(value))
    raise ValueError(f"Unsupported inverse transform: {inverse_transform}")


def project_chemical_defects(
    candidates: pd.DataFrame,
    defect_registry: pd.DataFrame,
    target_style_training: pd.DataFrame,
    *,
    relative_generation_quantile: float = 0.95,
) -> pd.DataFrame:
    """Apply the common conservative fold-local chemical projection."""
    result = candidates.copy()
    chemical = defect_registry.loc[defect_registry["chemical_field_present"].astype(bool)]
    for row in chemical.itertuples(index=False):
        feature = str(row.chemical_field)
        if row.constraint_mode == "absolute_excess_nonincrease":
            cap = _model_limit(float(row.threshold_used), str(row.inverse_transform))
        elif row.constraint_mode == "relative_concentration_nonincrease":
            cap = float(target_style_training[feature].quantile(relative_generation_quantile))
        else:
            raise ValueError(f"Defect constraint is not projectable: {row.defect}")
        result[feature] = np.minimum(result[feature].to_numpy(dtype=float), cap)
    return result


def _finalize_candidates(
    decision_candidates: pd.DataFrame,
    training: pd.DataFrame,
    target_style_training: pd.DataFrame,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    minimum_training_distance: Optional[float] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    decision_columns = [c for c in full_geometry.feature_names if c not in DERIVED_FEATURES]
    ordered = decision_candidates.loc[:, decision_columns].copy()
    ordered = ordered.clip(
        lower=full_geometry.q01.loc[decision_columns],
        upper=full_geometry.q99.loc[decision_columns],
        axis=1,
    )
    ordered = project_chemical_defects(ordered, defect_registry, target_style_training)
    for feature in DERIVED_FEATURES:
        ordered[feature] = np.nan
    completed = apply_derived_features(ordered, formula_config).loc[
        :, list(full_geometry.feature_names)
    ]
    values = completed.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    within = (
        completed.ge(full_geometry.minimum - 1e-12)
        & completed.le(full_geometry.maximum + 1e-12)
    ).all(axis=1).to_numpy()
    copied = exact_copy_mask(completed, training)
    distance_ok = np.ones(len(completed), dtype=bool)
    if minimum_training_distance is not None:
        candidate_z = full_geometry.transform(completed)
        training_z = full_geometry.transform(training)
        distance_ok = (
            pairwise_distances(candidate_z, training_z).min(axis=1)
            >= float(minimum_training_distance) - 1e-12
        )
    return completed, finite & within & ~copied & distance_ok


def _collect_valid(
    sampler: Callable[[int], tuple[pd.DataFrame, Sequence[object]]],
    n_samples: int,
    method: str,
    target_style: str,
    training: pd.DataFrame,
    target_style_training: pd.DataFrame,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    minimum_training_distance: Optional[float] = None,
) -> GeneratedBatch:
    accepted = []
    accepted_sources: list[object] = []
    for _ in range(20):
        needed = n_samples - sum(len(frame) for frame in accepted)
        if needed <= 0:
            break
        raw, sources = sampler(max(needed * 5, 20))
        finalized, valid = _finalize_candidates(
            raw,
            training,
            target_style_training,
            full_geometry,
            defect_registry,
            formula_config,
            minimum_training_distance,
        )
        accepted.append(finalized.loc[valid])
        accepted_sources.extend(np.asarray(sources, dtype=object)[valid].tolist())
    if not accepted or sum(len(frame) for frame in accepted) < n_samples:
        raise RuntimeError(f"Could not obtain {n_samples} feasible {method} rows")
    chemicals = pd.concat(accepted, ignore_index=True).iloc[:n_samples].copy()
    chemicals.index = pd.Index(
        [f"{target_style}__{method}__{i:04d}" for i in range(n_samples)],
        name="sample_id",
    )
    expected = recompute_derived_features(chemicals, formula_config)
    return GeneratedBatch(
        chemicals=chemicals,
        derived_expected=expected,
        source_training_ids=tuple(accepted_sources[:n_samples]),
        method=method,
    )


def empirical_resample(
    training: pd.DataFrame,
    training_styles: pd.Series,
    training_ids: Sequence[object],
    target_style: str,
    n_samples: int,
    *,
    seed: int,
) -> GeneratedBatch:
    """Resample target-style rows as a deliberately non-eligible reference.

    This method preserves the observed joint distribution and the four derived
    fields, but every row is an exact training copy.  The hard-gate evaluator
    must therefore reject it as a usable generator.
    """
    if len(training) != len(training_styles) or len(training) != len(training_ids):
        raise ValueError("Training features, styles, and ids are not aligned")
    positions = np.flatnonzero(training_styles.astype(str).to_numpy() == str(target_style))
    if not len(positions):
        raise ValueError(f"No training rows for target style {target_style}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(positions, size=int(n_samples), replace=True)
    sampled = training.iloc[selected].copy().reset_index(drop=True)
    sampled.index = pd.Index(
        [f"{target_style}__empirical__{i:04d}" for i in range(len(sampled))],
        name="sample_id",
    )
    expected = sampled.loc[:, list(DERIVED_FEATURES)].copy()
    ids = tuple(np.asarray(training_ids, dtype=object)[selected].tolist())
    return GeneratedBatch(
        chemicals=sampled,
        derived_expected=expected,
        source_training_ids=ids,
        method="empirical_resample_reference_only",
    )


def knn_jitter(
    training: pd.DataFrame,
    training_styles: pd.Series,
    training_ids: Sequence[object],
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    neighbors: int = 3,
    jitter_min: float = 0.05,
    jitter_max: float = 0.35,
) -> GeneratedBatch:
    """Jitter target-style rows along fold-training local-neighbor directions."""
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target = decision_geometry.impute(training.loc[mask, decision_columns])
    target_ids = np.asarray(training_ids, dtype=object)[mask]
    if len(target) < 2:
        raise ValueError(f"Insufficient target-style rows for KNN jitter: {target_style}")
    target_z = decision_geometry.transform(target)
    distances = pairwise_distances(target_z)
    np.fill_diagonal(distances, np.inf)
    k = min(int(neighbors), len(target) - 1)
    neighbor_order = np.argsort(distances, axis=1)[:, :k]
    rng = np.random.default_rng(seed)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        base = rng.integers(0, len(target), size=count)
        neighbor_choice = rng.integers(0, k, size=count)
        neighbor = neighbor_order[base, neighbor_choice]
        magnitude = rng.uniform(jitter_min, jitter_max, size=count)
        sign = rng.choice(np.array([-1.0, 1.0]), size=count)
        alpha = magnitude * sign
        sampled_z = target_z[base] + alpha[:, None] * (
            target_z[neighbor] - target_z[base]
        )
        sampled = pd.DataFrame(
            decision_geometry.scaler.inverse_transform(sampled_z),
            columns=decision_columns,
        )
        return sampled, target_ids[base]

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "knn_jitter",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
    )


def calibrated_knn_jitter(
    training: pd.DataFrame,
    training_styles: pd.Series,
    training_ids: Sequence[object],
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    neighbors: int = 3,
    calibration_quantile: float = 0.25,
    step_min_multiplier: float = 1.0,
    step_max_multiplier: float = 1.75,
) -> GeneratedBatch:
    """KNN extrapolation calibrated only from fold-training real distances.

    The original fixed small jitter can sit much closer to a training row than
    a held-out real beer.  This variant estimates the target-style leave-one-
    out nearest-neighbour distribution inside the outer training fold, moves
    away from a local neighbour, and rejects candidates below the locked
    training-only distance quantile.
    """
    if not 0.0 <= calibration_quantile <= 1.0:
        raise ValueError("calibration_quantile must be in [0, 1]")
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    positions = np.flatnonzero(mask)
    target = decision_geometry.impute(training.loc[mask, decision_columns])
    target_ids = np.asarray(training_ids, dtype=object)[mask]
    if len(target) < 2:
        raise ValueError(f"Insufficient target-style rows for calibrated KNN: {target_style}")

    target_z = decision_geometry.transform(target)
    local_distances = pairwise_distances(target_z)
    np.fill_diagonal(local_distances, np.inf)
    k = min(int(neighbors), len(target) - 1)
    neighbor_order = np.argsort(local_distances, axis=1)[:, :k]
    decision_loo = local_distances.min(axis=1)

    full_z = full_geometry.transform(training)
    full_target_z = full_z[positions]
    full_distances = pairwise_distances(full_target_z, full_z)
    full_distances[np.arange(len(positions)), positions] = np.inf
    full_loo = full_distances.min(axis=1)
    minimum_training_distance = float(np.quantile(full_loo, calibration_quantile))
    base_step = float(np.quantile(decision_loo, calibration_quantile))
    rng = np.random.default_rng(seed)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        base = rng.integers(0, len(target), size=count)
        neighbor_choice = rng.integers(0, k, size=count)
        neighbor = neighbor_order[base, neighbor_choice]
        direction = target_z[base] - target_z[neighbor]
        norm_value = np.linalg.norm(direction, axis=1)
        safe_norm = np.where(norm_value > 1e-12, norm_value, 1.0)
        unit = direction / safe_norm[:, None]
        step = rng.uniform(
            step_min_multiplier * base_step,
            step_max_multiplier * base_step,
            size=count,
        )
        sampled_z = target_z[base] + step[:, None] * unit
        sampled = pd.DataFrame(
            decision_geometry.scaler.inverse_transform(sampled_z),
            columns=decision_columns,
        )
        return sampled, target_ids[base]

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "calibrated_knn_jitter",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
        minimum_training_distance=minimum_training_distance,
    )


def knn_correlated_innovation(
    training: pd.DataFrame,
    training_styles: pd.Series,
    training_ids: Sequence[object],
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    n_components: int = 15,
    calibration_quantile: float = 0.75,
    step_min_multiplier: float = 0.9,
    step_max_multiplier: float = 1.25,
) -> GeneratedBatch:
    """Add fold-calibrated correlated innovation to target-style real bases.

    Unlike one-dimensional neighbour extrapolation, the direction is sampled
    from the global fold-training PCA covariance.  Its length is calibrated to
    the target-style leave-one-out distance distribution using training rows
    only.  This tests one mechanism: escaping memorization in many correlated
    dimensions while retaining a target-style starting point.
    """
    if not 0.0 <= calibration_quantile <= 1.0:
        raise ValueError("calibration_quantile must be in [0, 1]")
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    scaled = decision_geometry.transform(training.loc[:, decision_columns])
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    positions = np.flatnonzero(mask)
    target_scaled = scaled[positions]
    target_ids = np.asarray(training_ids, dtype=object)[positions]
    if len(target_scaled) < 2:
        raise ValueError(f"Insufficient target-style rows for correlated innovation: {target_style}")

    loo = pairwise_distances(target_scaled, scaled)
    loo[np.arange(len(positions)), positions] = np.inf
    base_step = float(np.quantile(loo.min(axis=1), calibration_quantile))
    component_count = min(int(n_components), len(training) - 1, len(decision_columns))
    pca = PCA(n_components=component_count, svd_solver="full")
    pca.fit(scaled)
    direction_factor = pca.components_ * np.sqrt(
        np.maximum(pca.explained_variance_, 1e-10)
    )[:, None]
    rng = np.random.default_rng(seed)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        base = rng.integers(0, len(target_scaled), size=count)
        latent_noise = rng.standard_normal((count, component_count))
        direction = latent_noise @ direction_factor
        direction_norm = np.linalg.norm(direction, axis=1)
        safe_norm = np.where(direction_norm > 1e-12, direction_norm, 1.0)
        unit = direction / safe_norm[:, None]
        step = rng.uniform(
            step_min_multiplier * base_step,
            step_max_multiplier * base_step,
            size=count,
        )
        sampled_scaled = target_scaled[base] + step[:, None] * unit
        sampled = pd.DataFrame(
            decision_geometry.scaler.inverse_transform(sampled_scaled),
            columns=decision_columns,
        )
        return sampled, target_ids[base]

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "knn_correlated_innovation",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
    )


def _sample_covariance(
    global_values: np.ndarray,
    target_values: np.ndarray,
    prior_strength: float,
    covariance_prior_strength: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    mean_weight = len(target_values) / (len(target_values) + float(prior_strength))
    covariance_prior = (
        float(prior_strength)
        if covariance_prior_strength is None
        else float(covariance_prior_strength)
    )
    covariance_weight = len(target_values) / (len(target_values) + covariance_prior)
    global_mean = global_values.mean(axis=0)
    target_mean = target_values.mean(axis=0)
    mean = mean_weight * target_mean + (1.0 - mean_weight) * global_mean
    global_covariance = LedoitWolf().fit(global_values).covariance_
    target_covariance = LedoitWolf().fit(target_values).covariance_
    covariance = (
        covariance_weight * target_covariance
        + (1.0 - covariance_weight) * global_covariance
    )
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
    covariance = (covariance + covariance.T) / 2.0
    return mean, covariance


def _covariance_to_correlation(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a numerically safe correlation matrix and marginal scales."""
    symmetric = (np.asarray(covariance, dtype=float) + covariance.T) / 2.0
    scales = np.sqrt(np.maximum(np.diag(symmetric), 1e-12))
    correlation = symmetric / np.outer(scales, scales)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation, scales


def _nearest_correlation(
    matrix: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 50,
) -> np.ndarray:
    """Project a symmetric matrix to a PSD, unit-diagonal correlation matrix.

    This is Higham's alternating-projection construction with Dykstra's
    correction. A final spectral cleanup removes only numerical residue and
    renormalizes the diagonal exactly to one.
    """
    current = (np.asarray(matrix, dtype=float) + matrix.T) / 2.0
    np.fill_diagonal(current, 1.0)
    correction = np.zeros_like(current)
    for _ in range(int(max_iterations)):
        previous = current
        residual = previous - correction
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        correction = projected - residual
        current = (projected + projected.T) / 2.0
        np.fill_diagonal(current, 1.0)
        relative_change = np.linalg.norm(current - previous, ord="fro") / max(
            np.linalg.norm(current, ord="fro"), 1e-12
        )
        if relative_change <= float(tolerance):
            break

    eigenvalues, eigenvectors = np.linalg.eigh((current + current.T) / 2.0)
    positive = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    scales = np.sqrt(np.maximum(np.diag(positive), 1e-12))
    result = positive / np.outer(scales, scales)
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def _module_correlation_covariance(
    global_values: np.ndarray,
    target_values: np.ndarray,
    modules: np.ndarray,
    *,
    prior_strength: float,
    module_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only within-module correlations while preserving base variances."""
    mean, base_covariance = _sample_covariance(
        global_values, target_values, prior_strength
    )
    base_correlation, base_scales = _covariance_to_correlation(base_covariance)
    candidate_correlation = base_correlation.copy()
    for module in sorted(set(modules)):
        indices = np.flatnonzero(modules == module)
        _, block_covariance = _sample_covariance(
            global_values[:, indices], target_values[:, indices], prior_strength
        )
        block_correlation, _ = _covariance_to_correlation(block_covariance)
        current_block = base_correlation[np.ix_(indices, indices)]
        candidate_correlation[np.ix_(indices, indices)] = (
            (1.0 - module_weight) * current_block
            + module_weight * block_correlation
        )
    repaired_correlation = _nearest_correlation(candidate_correlation)
    covariance = repaired_correlation * np.outer(base_scales, base_scales)
    covariance = (covariance + covariance.T) / 2.0
    return mean, covariance


def _factor_copula_covariance(
    global_values: np.ndarray,
    target_values: np.ndarray,
    *,
    prior_strength: float,
    n_factors: int,
    poet_threshold_constant: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate dependence by shared factors while keeping base variances."""
    mean, base_covariance = _sample_covariance(
        global_values, target_values, prior_strength
    )
    factor_count = min(int(n_factors), global_values.shape[1], len(global_values) - 1)
    if factor_count < 1:
        raise ValueError("n_factors must be positive")

    global_center = global_values.mean(axis=0)
    global_covariance = LedoitWolf().fit(global_values).covariance_
    eigenvalues, eigenvectors = np.linalg.eigh(global_covariance)
    loadings = eigenvectors[:, np.argsort(eigenvalues)[::-1][:factor_count]]
    global_scores = (global_values - global_center) @ loadings
    target_scores = (target_values - global_center) @ loadings
    _, factor_covariance = _sample_covariance(
        global_scores, target_scores, prior_strength
    )

    global_residual = (
        global_values - global_center - global_scores @ loadings.T
    )
    target_residual = (
        target_values - global_center - target_scores @ loadings.T
    )
    residual_weight = len(target_values) / (
        len(target_values) + float(prior_strength)
    )
    if poet_threshold_constant is None:
        global_residual_variance = np.var(global_residual, axis=0, ddof=1)
        target_residual_variance = np.var(target_residual, axis=0, ddof=1)
        residual_variance = (
            residual_weight * target_residual_variance
            + (1.0 - residual_weight) * global_residual_variance
        )
        residual_covariance = np.diag(np.maximum(residual_variance, 1e-10))
    else:
        threshold_constant = float(poet_threshold_constant)
        if threshold_constant <= 0.0:
            raise ValueError("poet_threshold_constant must be positive")

        def covariance_and_product_variance(
            values: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            centered = values - values.mean(axis=0)
            count = len(centered)
            cross_sum = centered.T @ centered
            covariance = cross_sum / max(count - 1, 1)
            raw_product_mean = cross_sum / max(count, 1)
            squared = centered**2
            second_product_moment = squared.T @ squared / max(count, 1)
            product_variance = (
                second_product_moment
                - 2.0 * covariance * raw_product_mean
                + covariance**2
            )
            return covariance, np.maximum(product_variance, 0.0)

        global_residual_covariance, global_product_variance = (
            covariance_and_product_variance(global_residual)
        )
        target_residual_covariance, target_product_variance = (
            covariance_and_product_variance(target_residual)
        )
        residual_covariance = (
            residual_weight * target_residual_covariance
            + (1.0 - residual_weight) * global_residual_covariance
        )
        estimated_variance = (
            residual_weight**2
            * target_product_variance
            / max(len(target_residual), 1)
            + (1.0 - residual_weight) ** 2
            * global_product_variance
            / max(len(global_residual), 1)
        )
        dimension_penalty = np.log(global_values.shape[1])
        threshold = threshold_constant * np.sqrt(
            np.maximum(estimated_variance, 0.0) * dimension_penalty
        )
        off_diagonal = ~np.eye(global_values.shape[1], dtype=bool)
        residual_covariance[
            off_diagonal & (np.abs(residual_covariance) < threshold)
        ] = 0.0
        residual_covariance = (
            residual_covariance + residual_covariance.T
        ) / 2.0
        diagonal = np.maximum(np.diag(residual_covariance), 1e-10)
        np.fill_diagonal(residual_covariance, diagonal)

    reconstructed = (
        loadings @ factor_covariance @ loadings.T + residual_covariance
    )
    factor_correlation, _ = _covariance_to_correlation(reconstructed)
    if poet_threshold_constant is not None:
        factor_correlation = _nearest_correlation(factor_correlation)
    _, base_scales = _covariance_to_correlation(base_covariance)
    covariance = factor_correlation * np.outer(base_scales, base_scales)
    covariance = (covariance + covariance.T) / 2.0
    return mean, covariance


def _spearman_skeptic_correlation(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate latent-Gaussian correlation from Spearman ranks.

    The boolean mask marks pairs whose rank correlation is defined. Constant
    columns can occur inside a small target style; callers can then fall back
    to the corresponding global training-fold correlation.
    """
    matrix = np.asarray(values, dtype=float)
    ranks = np.column_stack(
        [rankdata(matrix[:, column], method="average") for column in range(matrix.shape[1])]
    )
    centered = ranks - ranks.mean(axis=0)
    norms = np.sqrt(np.sum(centered**2, axis=0))
    denominator = np.outer(norms, norms)
    defined = denominator > 1e-12
    spearman = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros_like(denominator),
        where=defined,
    )
    spearman = np.clip((spearman + spearman.T) / 2.0, -1.0, 1.0)
    latent = 2.0 * np.sin(np.pi * spearman / 6.0)
    latent = np.clip((latent + latent.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(latent, 1.0)
    np.fill_diagonal(defined, True)
    return latent, defined


def _spearman_skeptic_covariance(
    global_values: np.ndarray,
    target_values: np.ndarray,
    *,
    prior_strength: float,
    blend_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior-blend global/style SKEPTIC dependence and keep base scales."""
    if not 0.0 <= float(blend_weight) <= 1.0:
        raise ValueError("blend_weight must be in [0, 1]")
    mean, base_covariance = _sample_covariance(
        global_values, target_values, prior_strength
    )
    global_correlation, global_defined = _spearman_skeptic_correlation(
        global_values
    )
    target_correlation, target_defined = _spearman_skeptic_correlation(
        target_values
    )
    global_correlation = np.where(global_defined, global_correlation, 0.0)
    np.fill_diagonal(global_correlation, 1.0)
    target_correlation = np.where(
        target_defined, target_correlation, global_correlation
    )
    weight = len(target_values) / (len(target_values) + float(prior_strength))
    skeptic_correlation = (
        weight * target_correlation + (1.0 - weight) * global_correlation
    )
    base_correlation, base_scales = _covariance_to_correlation(base_covariance)
    correlation = (
        float(blend_weight) * skeptic_correlation
        + (1.0 - float(blend_weight)) * base_correlation
    )
    correlation = _nearest_correlation(correlation)
    covariance = correlation * np.outer(base_scales, base_scales)
    covariance = (covariance + covariance.T) / 2.0
    return mean, covariance


def _moment_preserving_mixture_parameters(
    global_values: np.ndarray,
    target_values: np.ndarray,
    *,
    prior_strength: float,
    blend_weight: float,
    n_components: int,
    n_factors: int,
    between_variance_cap: float,
    min_component_size: int,
    cluster_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Add low-dimensional component means while preserving parent moments.

    K-means is fitted only in leading, parent-whitened directions. Component
    offsets are centered and capped, and their between-component covariance is
    removed from a shared within-component covariance. The resulting mixture
    therefore retains the half-shrunk SKEPTIC parent's latent mean and total
    covariance while allowing higher-order multimodal structure.
    """
    if int(n_components) != 2:
        raise ValueError("The pre-registered mixture requires exactly 2 components")
    if int(n_factors) < 1:
        raise ValueError("n_factors must be positive")
    if not 0.0 < float(between_variance_cap) < 1.0:
        raise ValueError("between_variance_cap must be in (0, 1)")
    if int(min_component_size) < 2:
        raise ValueError("min_component_size must be at least 2")

    mean, parent_covariance = _spearman_skeptic_covariance(
        global_values,
        target_values,
        prior_strength=prior_strength,
        blend_weight=blend_weight,
    )
    dimension = parent_covariance.shape[0]
    fallback_diagnostics = {
        "mixture_active": False,
        "target_count": int(len(target_values)),
        "component_count_0": int(len(target_values)),
        "component_count_1": 0,
        "raw_relative_between_max": 0.0,
        "shift_scale": 0.0,
        "latent_mean_reconstruction_error": 0.0,
        "latent_covariance_reconstruction_error": 0.0,
    }
    if len(target_values) < int(n_components) * int(min_component_size):
        return (
            mean,
            parent_covariance,
            np.zeros((1, dimension), dtype=float),
            np.ones(1, dtype=float),
            fallback_diagnostics,
        )

    eigenvalues, eigenvectors = np.linalg.eigh(parent_covariance)
    factor_count = min(int(n_factors), dimension)
    order = np.argsort(eigenvalues)[::-1][:factor_count]
    selected_values = np.maximum(eigenvalues[order], 1e-10)
    selected_vectors = eigenvectors[:, order]
    whitened = (
        (np.asarray(target_values, dtype=float) - mean)
        @ selected_vectors
        / np.sqrt(selected_values)
    )
    clustering = KMeans(
        n_clusters=int(n_components),
        n_init=20,
        random_state=int(cluster_seed),
    ).fit(whitened)
    counts = np.bincount(clustering.labels_, minlength=int(n_components))
    if int(counts.min()) < int(min_component_size):
        fallback_diagnostics.update(
            {
                "component_count_0": int(counts[0]),
                "component_count_1": int(counts[1]),
            }
        )
        return (
            mean,
            parent_covariance,
            np.zeros((1, dimension), dtype=float),
            np.ones(1, dtype=float),
            fallback_diagnostics,
        )

    weights = counts.astype(float) / float(counts.sum())
    centers = clustering.cluster_centers_
    centered_offsets = centers - np.sum(weights[:, None] * centers, axis=0)
    between_whitened = sum(
        weights[index] * np.outer(centered_offsets[index], centered_offsets[index])
        for index in range(int(n_components))
    )
    raw_relative_between_max = float(
        np.linalg.eigvalsh((between_whitened + between_whitened.T) / 2.0).max()
    )
    shift_scale = min(
        1.0,
        np.sqrt(float(between_variance_cap) / max(raw_relative_between_max, 1e-12)),
    )
    component_shifts = (
        shift_scale * centered_offsets * np.sqrt(selected_values)
    ) @ selected_vectors.T
    between_covariance = sum(
        weights[index] * np.outer(component_shifts[index], component_shifts[index])
        for index in range(int(n_components))
    )
    within_covariance = (
        parent_covariance - between_covariance
        + (parent_covariance - between_covariance).T
    ) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(within_covariance).min())
    if minimum_eigenvalue < -1e-8:
        raise ArithmeticError(
            "Moment-preserving mixture produced a non-PSD within covariance"
        )
    if minimum_eigenvalue < 1e-10:
        values, vectors = np.linalg.eigh(within_covariance)
        within_covariance = (
            vectors * np.maximum(values, 1e-10)
        ) @ vectors.T
        within_covariance = (within_covariance + within_covariance.T) / 2.0

    reconstructed_mean = mean + np.sum(weights[:, None] * component_shifts, axis=0)
    reconstructed_covariance = within_covariance + sum(
        weights[index] * np.outer(component_shifts[index], component_shifts[index])
        for index in range(int(n_components))
    )
    diagnostics = {
        "mixture_active": True,
        "target_count": int(len(target_values)),
        "component_count_0": int(counts[0]),
        "component_count_1": int(counts[1]),
        "raw_relative_between_max": raw_relative_between_max,
        "shift_scale": float(shift_scale),
        "latent_mean_reconstruction_error": float(
            np.max(np.abs(reconstructed_mean - mean))
        ),
        "latent_covariance_reconstruction_error": float(
            np.max(np.abs(reconstructed_covariance - parent_covariance))
        ),
    }
    return mean, within_covariance, component_shifts, weights, diagnostics


def moment_preserving_mixture_rank_gaussian_copula(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    prior_strength: float = 5.0,
    skeptic_blend_weight: float = 0.5,
    n_components: int = 2,
    n_factors: int = 5,
    between_variance_cap: float = 0.5,
    min_component_size: int = 2,
    minimum_training_distance_quantile: Optional[float] = 0.05,
) -> GeneratedBatch:
    """Half-shrunk SKEPTIC copula with a moment-preserving latent mixture."""
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    imputed = decision_geometry.impute(training.loc[:, decision_columns]).to_numpy()
    gaussian = _rank_gaussian(imputed)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_gaussian = gaussian[mask]
    if len(target_gaussian) < 2:
        raise ValueError(f"Insufficient target-style rows for mixture: {target_style}")
    mean, within_covariance, component_shifts, weights, diagnostics = (
        _moment_preserving_mixture_parameters(
            gaussian,
            target_gaussian,
            prior_strength=prior_strength,
            blend_weight=skeptic_blend_weight,
            n_components=n_components,
            n_factors=n_factors,
            between_variance_cap=between_variance_cap,
            min_component_size=min_component_size,
            cluster_seed=int(seed) + 1543,
        )
    )
    minimum_training_distance = None
    if minimum_training_distance_quantile is not None:
        quantile = float(minimum_training_distance_quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("minimum_training_distance_quantile must be in [0, 1]")
        training_z = full_geometry.transform(training)
        training_distances = pairwise_distances(training_z)
        np.fill_diagonal(training_distances, np.inf)
        minimum_training_distance = float(
            np.quantile(training_distances.min(axis=1), quantile)
        )

    noise_rng = np.random.default_rng(seed)
    component_rng = np.random.default_rng(int(seed) + 7919)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        component = component_rng.choice(len(weights), size=count, p=weights)
        innovation = _draw_psd_gaussian(
            noise_rng,
            np.zeros_like(mean),
            within_covariance,
            count,
        )
        sampled_gaussian = mean + component_shifts[component] + innovation
        sampled = pd.DataFrame(
            _inverse_empirical_gaussian(sampled_gaussian, imputed),
            columns=decision_columns,
        )
        return sampled, [None] * count

    target_full = training.loc[mask].reset_index(drop=True)
    batch = _collect_valid(
        sampler,
        n_samples,
        "moment_preserving_mixture_rank_gaussian_copula",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
        minimum_training_distance=minimum_training_distance,
    )
    return GeneratedBatch(
        chemicals=batch.chemicals,
        derived_expected=batch.derived_expected,
        source_training_ids=batch.source_training_ids,
        method=batch.method,
        diagnostics=diagnostics,
    )


def _draw_psd_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    count: int,
) -> np.ndarray:
    """Draw from a covariance through an explicit PSD square root.

    ``Generator.multivariate_normal`` can warn on numerically tiny negative
    eigenvalues when the covariance spectrum is very wide.  Reconstructing the
    square root explicitly makes the stabilization deterministic and auditable.
    """
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 1e-10))
    noise = rng.standard_normal((int(count), len(mean)))
    return mean + noise @ factor.T


def _rank_stratified_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    count: int,
) -> np.ndarray:
    """Create one normal-score-stratified batch with parent rank ordering."""
    count = int(count)
    if count < 2:
        raise ValueError("Rank-stratified sampling requires at least 2 rows")
    correlation, scales = _covariance_to_correlation(covariance)
    auxiliary = _draw_psd_gaussian(
        rng,
        np.zeros_like(mean, dtype=float),
        correlation,
        count,
    )
    probabilities = (np.arange(count, dtype=float) + 0.5) / float(count)
    scores = norm.ppf(probabilities)
    scores = (scores - scores.mean()) / np.sqrt(np.mean(scores**2))
    result = np.empty_like(auxiliary)
    for column in range(auxiliary.shape[1]):
        order = np.argsort(auxiliary[:, column], kind="mergesort")
        result[order, column] = scores
    return np.asarray(mean, dtype=float) + result * scales


def rank_stratified_skeptic_copula(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    prior_strength: float = 5.0,
    skeptic_blend_weight: float = 0.5,
    minimum_training_distance_quantile: Optional[float] = 0.05,
    maximum_batch_attempts: int = 100,
) -> GeneratedBatch:
    """Sample the protected SKEPTIC parent with full-batch rank stratification."""
    if int(n_samples) < 2:
        raise ValueError("Rank-stratified generator requires at least 2 samples")
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    imputed = decision_geometry.impute(training.loc[:, decision_columns]).to_numpy()
    gaussian = _rank_gaussian(imputed)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_gaussian = gaussian[mask]
    if len(target_gaussian) < 2:
        raise ValueError(
            f"Insufficient target-style rows for stratified copula: {target_style}"
        )
    mean, covariance = _spearman_skeptic_covariance(
        gaussian,
        target_gaussian,
        prior_strength=prior_strength,
        blend_weight=skeptic_blend_weight,
    )
    minimum_training_distance = None
    if minimum_training_distance_quantile is not None:
        quantile = float(minimum_training_distance_quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("minimum_training_distance_quantile must be in [0, 1]")
        training_z = full_geometry.transform(training)
        training_distances = pairwise_distances(training_z)
        np.fill_diagonal(training_distances, np.inf)
        minimum_training_distance = float(
            np.quantile(training_distances.min(axis=1), quantile)
        )

    target_full = training.loc[mask].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    accepted = None
    latent = None
    attempts_used = 0
    for attempt in range(1, int(maximum_batch_attempts) + 1):
        latent_candidate = _rank_stratified_gaussian(
            rng, mean, covariance, int(n_samples)
        )
        raw = pd.DataFrame(
            _inverse_empirical_gaussian(latent_candidate, imputed),
            columns=decision_columns,
        )
        finalized, valid = _finalize_candidates(
            raw,
            training,
            target_full,
            full_geometry,
            defect_registry,
            formula_config,
            minimum_training_distance,
        )
        if bool(valid.all()):
            accepted = finalized
            latent = latent_candidate
            attempts_used = attempt
            break
    if accepted is None or latent is None:
        raise RuntimeError(
            f"Could not obtain a fully feasible stratified batch in "
            f"{maximum_batch_attempts} attempts"
        )

    method = "rank_stratified_skeptic_copula"
    accepted = accepted.reset_index(drop=True)
    accepted.index = pd.Index(
        [f"{target_style}__{method}__{i:04d}" for i in range(int(n_samples))],
        name="sample_id",
    )
    expected = recompute_derived_features(accepted, formula_config)
    marginal_mean_error = float(np.max(np.abs(latent.mean(axis=0) - mean)))
    marginal_variance_error = float(
        np.max(
            np.abs(
                np.mean((latent - mean) ** 2, axis=0)
                - np.diag(covariance)
            )
        )
    )
    return GeneratedBatch(
        chemicals=accepted,
        derived_expected=expected,
        source_training_ids=tuple([None] * int(n_samples)),
        method=method,
        diagnostics={
            "batch_attempts": int(attempts_used),
            "latent_marginal_mean_error": marginal_mean_error,
            "latent_marginal_variance_error": marginal_variance_error,
        },
    )


def pca_shrink_gaussian(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    n_components: int = 15,
    prior_strength: float = 20.0,
) -> GeneratedBatch:
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    scaled = decision_geometry.transform(training.loc[:, decision_columns])
    component_count = min(int(n_components), len(training) - 1, len(decision_columns))
    pca = PCA(n_components=component_count, svd_solver="full")
    latent = pca.fit_transform(scaled)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_latent = latent[mask]
    if len(target_latent) < 2:
        raise ValueError(f"Insufficient target-style rows for PCA Gaussian: {target_style}")
    mean, covariance = _sample_covariance(latent, target_latent, prior_strength)
    rng = np.random.default_rng(seed)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        sampled_latent = _draw_psd_gaussian(rng, mean, covariance, count)
        sampled_scaled = pca.inverse_transform(sampled_latent)
        sampled = pd.DataFrame(
            decision_geometry.scaler.inverse_transform(sampled_scaled),
            columns=decision_columns,
        )
        return sampled, [None] * count

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "pca_shrink_gaussian",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
    )


def _rank_gaussian(matrix: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix, dtype=float)
    n_rows = len(matrix)
    for column in range(matrix.shape[1]):
        ranks = rankdata(matrix[:, column], method="average")
        result[:, column] = norm.ppf((ranks - 0.5) / n_rows)
    return result


def _inverse_empirical_gaussian(
    gaussian: np.ndarray,
    training_values: np.ndarray,
) -> np.ndarray:
    probabilities = np.clip(norm.cdf(gaussian), 1e-6, 1.0 - 1e-6)
    result = np.empty_like(gaussian, dtype=float)
    for column in range(gaussian.shape[1]):
        result[:, column] = np.quantile(
            training_values[:, column], probabilities[:, column]
        )
    return result


def rank_gaussian_copula(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    prior_strength: float = 20.0,
    covariance_prior_strength: Optional[float] = None,
    dependence_estimator: str = "ledoit_wolf",
    minimum_training_distance_quantile: Optional[float] = None,
    skeptic_blend_weight: float = 1.0,
) -> GeneratedBatch:
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    imputed = decision_geometry.impute(training.loc[:, decision_columns]).to_numpy()
    gaussian = _rank_gaussian(imputed)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_gaussian = gaussian[mask]
    if len(target_gaussian) < 2:
        raise ValueError(f"Insufficient target-style rows for copula: {target_style}")
    if dependence_estimator == "ledoit_wolf":
        mean, covariance = _sample_covariance(
            gaussian,
            target_gaussian,
            prior_strength,
            covariance_prior_strength=covariance_prior_strength,
        )
    elif dependence_estimator == "spearman_skeptic":
        if covariance_prior_strength is not None:
            raise ValueError(
                "spearman_skeptic does not support a separate covariance prior"
            )
        mean, covariance = _spearman_skeptic_covariance(
            gaussian,
            target_gaussian,
            prior_strength=prior_strength,
            blend_weight=skeptic_blend_weight,
        )
    else:
        raise ValueError(f"Unsupported dependence_estimator: {dependence_estimator}")
    rng = np.random.default_rng(seed)
    minimum_training_distance = None
    if minimum_training_distance_quantile is not None:
        quantile = float(minimum_training_distance_quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("minimum_training_distance_quantile must be in [0, 1]")
        training_z = full_geometry.transform(training)
        training_distances = pairwise_distances(training_z)
        np.fill_diagonal(training_distances, np.inf)
        minimum_training_distance = float(
            np.quantile(training_distances.min(axis=1), quantile)
        )

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        sampled_gaussian = _draw_psd_gaussian(rng, mean, covariance, count)
        sampled = pd.DataFrame(
            _inverse_empirical_gaussian(sampled_gaussian, imputed),
            columns=decision_columns,
        )
        return sampled, [None] * count

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        (
            "skeptic_spearman_rank_gaussian_copula"
            if dependence_estimator == "spearman_skeptic"
            else "rank_gaussian_copula"
        ),
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
        minimum_training_distance=minimum_training_distance,
    )


def module_block_rank_gaussian_copula(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    module_registry: pd.DataFrame,
    *,
    seed: int,
    prior_strength: float = 5.0,
    module_weight: float = 1.0,
) -> GeneratedBatch:
    """Rank-Gaussian copula with Q2 module-specific covariance shrinkage.

    The ordinary full covariance remains the starting point.  Each within-Q2-
    module block is re-estimated separately with the same target/global shrink
    weight, while cross-module entries remain unchanged.  The final matrix is
    projected to PSD only for numerical validity.
    """
    if not 0.0 <= module_weight <= 1.0:
        raise ValueError("module_weight must be in [0, 1]")
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    registry = module_registry.loc[:, ["feature", "consensus_module"]].copy()
    if registry["feature"].duplicated().any():
        raise ValueError("Q2 module registry contains duplicate features")
    module_by_feature = registry.set_index("feature")["consensus_module"]
    missing = sorted(set(decision_columns).difference(module_by_feature.index))
    if missing:
        raise ValueError(f"Q2 module registry is missing decision features: {missing}")

    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    imputed = decision_geometry.impute(training.loc[:, decision_columns]).to_numpy()
    gaussian = _rank_gaussian(imputed)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_gaussian = gaussian[mask]
    if len(target_gaussian) < 2:
        raise ValueError(f"Insufficient target-style rows for module copula: {target_style}")

    mean, covariance = _sample_covariance(gaussian, target_gaussian, prior_strength)
    modules = module_by_feature.loc[decision_columns].astype(str).to_numpy()
    for module in sorted(set(modules)):
        indices = np.flatnonzero(modules == module)
        _, block_covariance = _sample_covariance(
            gaussian[:, indices], target_gaussian[:, indices], prior_strength
        )
        current_block = covariance[np.ix_(indices, indices)]
        covariance[np.ix_(indices, indices)] = (
            (1.0 - module_weight) * current_block
            + module_weight * block_covariance
        )
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (
        eigenvectors * np.maximum(eigenvalues, 1e-10)
    ) @ eigenvectors.T
    covariance = (covariance + covariance.T) / 2.0
    rng = np.random.default_rng(seed)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        sampled_gaussian = _draw_psd_gaussian(rng, mean, covariance, count)
        sampled = pd.DataFrame(
            _inverse_empirical_gaussian(sampled_gaussian, imputed),
            columns=decision_columns,
        )
        return sampled, [None] * count

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "module_block_rank_gaussian_copula",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
    )


def module_correlation_rank_gaussian_copula(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    module_registry: pd.DataFrame,
    *,
    seed: int,
    prior_strength: float = 5.0,
    module_weight: float = 1.0,
) -> GeneratedBatch:
    """Copula that changes Q2 within-module correlations, not variances.

    The prior=5 full-copula mean, marginal latent variances and empirical
    inverse margins stay fixed. Only module-local correlations are blended
    toward separately estimated Q2 blocks, followed by a valid-correlation
    projection. This isolates structure repair from the marginal mechanism.
    """
    if not 0.0 <= module_weight <= 1.0:
        raise ValueError("module_weight must be in [0, 1]")
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    registry = module_registry.loc[:, ["feature", "consensus_module"]].copy()
    if registry["feature"].duplicated().any():
        raise ValueError("Q2 module registry contains duplicate features")
    module_by_feature = registry.set_index("feature")["consensus_module"]
    missing = sorted(set(decision_columns).difference(module_by_feature.index))
    if missing:
        raise ValueError(f"Q2 module registry is missing decision features: {missing}")

    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    imputed = decision_geometry.impute(training.loc[:, decision_columns]).to_numpy()
    gaussian = _rank_gaussian(imputed)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_gaussian = gaussian[mask]
    if len(target_gaussian) < 2:
        raise ValueError(
            f"Insufficient target-style rows for module correlation copula: {target_style}"
        )

    modules = module_by_feature.loc[decision_columns].astype(str).to_numpy()
    mean, covariance = _module_correlation_covariance(
        gaussian,
        target_gaussian,
        modules,
        prior_strength=prior_strength,
        module_weight=module_weight,
    )
    rng = np.random.default_rng(seed)

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        sampled_gaussian = _draw_psd_gaussian(rng, mean, covariance, count)
        sampled = pd.DataFrame(
            _inverse_empirical_gaussian(sampled_gaussian, imputed),
            columns=decision_columns,
        )
        return sampled, [None] * count

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "module_correlation_rank_gaussian_copula",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
    )


def factor_rank_gaussian_copula(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    n_samples: int,
    full_geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    *,
    seed: int,
    prior_strength: float = 5.0,
    n_factors: int = 10,
    minimum_training_distance_quantile: Optional[float] = None,
    poet_threshold_constant: Optional[float] = None,
) -> GeneratedBatch:
    """Rank-Gaussian copula with shared low-rank dependence plus diagonal noise."""
    decision_columns = [c for c in training.columns if c not in DERIVED_FEATURES]
    decision_geometry = FoldGeometry.fit(training.loc[:, decision_columns])
    imputed = decision_geometry.impute(training.loc[:, decision_columns]).to_numpy()
    gaussian = _rank_gaussian(imputed)
    mask = training_styles.astype(str).to_numpy() == str(target_style)
    target_gaussian = gaussian[mask]
    if len(target_gaussian) < 2:
        raise ValueError(f"Insufficient target-style rows for factor copula: {target_style}")
    mean, covariance = _factor_copula_covariance(
        gaussian,
        target_gaussian,
        prior_strength=prior_strength,
        n_factors=n_factors,
        poet_threshold_constant=poet_threshold_constant,
    )
    rng = np.random.default_rng(seed)
    minimum_training_distance = None
    if minimum_training_distance_quantile is not None:
        quantile = float(minimum_training_distance_quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("minimum_training_distance_quantile must be in [0, 1]")
        training_z = full_geometry.transform(training)
        training_distances = pairwise_distances(training_z)
        np.fill_diagonal(training_distances, np.inf)
        minimum_training_distance = float(
            np.quantile(training_distances.min(axis=1), quantile)
        )

    def sampler(count: int) -> tuple[pd.DataFrame, Sequence[object]]:
        sampled_gaussian = _draw_psd_gaussian(rng, mean, covariance, count)
        sampled = pd.DataFrame(
            _inverse_empirical_gaussian(sampled_gaussian, imputed),
            columns=decision_columns,
        )
        return sampled, [None] * count

    target_full = training.loc[mask].reset_index(drop=True)
    return _collect_valid(
        sampler,
        n_samples,
        "factor_rank_gaussian_copula",
        target_style,
        training,
        target_full,
        full_geometry,
        defect_registry,
        formula_config,
        minimum_training_distance=minimum_training_distance,
    )

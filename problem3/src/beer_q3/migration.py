"""Constrained Pils/Lager to Tripel migration primitives.

The module is deliberately independent from orchestration.  Every fitted
quantity is supplied from the caller's training partition, and the returned
rows retain the original 231-column chemical schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .derived import apply_derived_features, recompute_derived_features
from .evaluation import (
    DERIVED_FEATURES,
    FoldGeometry,
    defect_violation_mask,
    exact_copy_mask,
)


REFERENCE_METHODS = {"identity_reference", "nearest_tripel_reference"}


@dataclass(frozen=True)
class MigrationResult:
    """All candidates plus their audit metrics and deterministic selection."""

    candidates: pd.DataFrame
    metrics: pd.DataFrame
    pareto: pd.DataFrame
    selected: pd.DataFrame


def _symmetric_power(matrix: np.ndarray, power: float, floor: float = 1e-8) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor) ** float(power)
    return (vectors * values) @ vectors.T


def _shrunk_covariance(values: np.ndarray, shrinkage: float) -> np.ndarray:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Covariance fitting requires at least two rows")
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance).astype(float)
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal


def _sinkhorn_plan(cost: np.ndarray, epsilon: float, iterations: int) -> np.ndarray:
    """Return a small, deterministic entropic optimal-transport plan."""
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Sinkhorn cost matrix must be finite and two-dimensional")
    positive = cost[cost > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    regularization = max(float(epsilon) * scale, 1e-8)
    kernel = np.exp(-cost / regularization)
    kernel = np.maximum(kernel, 1e-300)
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(int(iterations)):
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
    plan = u[:, None] * kernel * v[None, :]
    total = float(plan.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Sinkhorn transport plan is degenerate")
    return plan / total


def _model_threshold(threshold: float, inverse_transform: str) -> float:
    if inverse_transform == "identity":
        return float(threshold)
    if inverse_transform == "pow10":
        return float(np.log10(threshold))
    if inverse_transform == "exp":
        return float(np.log(threshold))
    raise ValueError(f"Unsupported inverse transform: {inverse_transform}")


def project_transfer_constraints(
    candidate: pd.DataFrame,
    source: pd.DataFrame,
    geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    actionability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project a transfer candidate onto the locked fold-local feasible set.

    Non-decision fields are held at their source values.  Absolute defect
    constraints enforce non-increasing excess above the threshold; t2n is
    capped at its paired source value.  The four derived fields are then
    recomputed from their registered components.
    """
    if len(candidate) != len(source):
        raise ValueError("Candidate and source row counts differ")
    ordered_source = geometry.validate_schema(source).reset_index(drop=True)
    result = geometry.validate_schema(candidate).reset_index(drop=True).copy()
    roles = actionability.set_index("feature")
    if set(geometry.feature_names) != set(roles.index):
        raise ValueError("Actionability registry does not match the chemical schema")
    decision = roles.loc[list(geometry.feature_names), "generator_decision_variable"].astype(bool)
    immutable = decision.index[~decision].difference(DERIVED_FEATURES)
    result.loc[:, immutable] = ordered_source.loc[:, immutable].to_numpy()

    decision_columns = decision.index[decision].tolist()
    result.loc[:, decision_columns] = result.loc[:, decision_columns].clip(
        lower=geometry.q01.loc[decision_columns],
        upper=geometry.q99.loc[decision_columns],
        axis=1,
    )
    chemical = defect_registry.loc[defect_registry["chemical_field_present"].astype(bool)]
    projection_rows: list[dict[str, Any]] = []
    for row in chemical.itertuples(index=False):
        feature = str(row.chemical_field)
        before = result[feature].to_numpy(dtype=float).copy()
        source_values = ordered_source[feature].to_numpy(dtype=float)
        if row.constraint_mode == "absolute_excess_nonincrease":
            threshold = _model_threshold(float(row.threshold_used), str(row.inverse_transform))
            cap = np.maximum(source_values, threshold)
            policy = "source_excess_nonincrease"
        elif row.constraint_mode == "relative_concentration_nonincrease":
            cap = source_values
            policy = "source_relative_nonincrease"
        else:
            raise ValueError(f"Unsupported migration defect rule: {row.defect}")
        result[feature] = np.minimum(before, cap)
        projection_rows.append(
            {
                "defect": str(row.defect),
                "chemical_field": feature,
                "policy": policy,
                "projected_rows": int(np.sum(before > cap + 1e-12)),
            }
        )

    for feature in DERIVED_FEATURES:
        result[feature] = np.nan
    result = apply_derived_features(result, formula_config).loc[:, list(geometry.feature_names)]
    return result, pd.DataFrame(projection_rows)


def _latent_endpoints(
    training: pd.DataFrame,
    training_styles: pd.Series,
    source: pd.DataFrame,
    decision_columns: Sequence[str],
    *,
    source_style: str,
    target_style: str,
    n_components: int,
    coral_shrinkage: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> dict[str, pd.DataFrame]:
    geometry = FoldGeometry.fit(training.loc[:, list(decision_columns)])
    train_z = geometry.transform(training.loc[:, list(decision_columns)])
    source_z = geometry.transform(source.loc[:, list(decision_columns)])
    component_count = min(int(n_components), len(training) - 1, len(decision_columns))
    if component_count < 2:
        raise ValueError("Migration latent space requires at least two components")
    pca = PCA(n_components=component_count, svd_solver="full")
    train_latent = pca.fit_transform(train_z)
    query_latent = pca.transform(source_z)
    styles = training_styles.astype(str).to_numpy()
    source_mask = styles == str(source_style)
    target_mask = styles == str(target_style)
    source_train = train_latent[source_mask]
    target_train = train_latent[target_mask]
    if len(source_train) < 2 or len(target_train) < 2:
        raise ValueError("Migration styles require at least two training rows")

    source_mean = source_train.mean(axis=0)
    target_mean = target_train.mean(axis=0)
    centroid = query_latent + (target_mean - source_mean)

    source_cov = _shrunk_covariance(source_train, coral_shrinkage)
    target_cov = _shrunk_covariance(target_train, coral_shrinkage)
    coral_map = _symmetric_power(source_cov, -0.5) @ _symmetric_power(target_cov, 0.5)
    coral = (query_latent - source_mean) @ coral_map + target_mean

    cost = pairwise_distances(source_train, target_train, metric="sqeuclidean")
    plan = _sinkhorn_plan(cost, sinkhorn_epsilon, sinkhorn_iterations)
    row_mass = np.maximum(plan.sum(axis=1), 1e-300)
    mapped_source_train = (plan @ target_train) / row_mass[:, None]
    displacement = mapped_source_train - source_train
    query_cost = pairwise_distances(query_latent, source_train, metric="sqeuclidean")
    positive = query_cost[query_cost > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-query_cost / max(bandwidth, 1e-8))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    sinkhorn = query_latent + weights @ displacement

    endpoints = {}
    for name, latent in {
        "pca_centroid_shift": centroid,
        "shrink_coral": coral,
        "sinkhorn_barycentric": sinkhorn,
    }.items():
        inverse = geometry.scaler.inverse_transform(pca.inverse_transform(latent))
        endpoints[name] = pd.DataFrame(inverse, columns=list(decision_columns))
    return endpoints


def _module_endpoint(
    source: pd.DataFrame,
    target_training: pd.DataFrame,
    actionability: pd.DataFrame,
    geometry: FoldGeometry,
    module_budget: int,
) -> tuple[pd.DataFrame, list[list[str]]]:
    roles = actionability.set_index("feature")
    target_center = geometry.impute(target_training).median()
    result = source.copy().reset_index(drop=True)
    chosen_per_row: list[list[str]] = []
    for row_index in range(len(result)):
        standardized_gap = (
            (target_center - result.iloc[row_index]) / geometry.robust_scale
        ).abs()
        candidates = roles.loc[
            roles["generator_decision_variable"].astype(bool)
            & ~roles.index.isin(DERIVED_FEATURES)
        ].copy()
        candidates["gap"] = standardized_gap.loc[candidates.index]
        ranking = (
            candidates.groupby("consensus_module", dropna=False)["gap"]
            .mean()
            .sort_values(ascending=False)
        )
        selected_modules = ranking.head(int(module_budget)).index.astype(str).tolist()
        selected_features = candidates.index[
            candidates["consensus_module"].astype(str).isin(selected_modules)
        ].tolist()
        result.loc[row_index, selected_features] = target_center.loc[selected_features].to_numpy()
        chosen_per_row.append(selected_modules)
    return result, chosen_per_row


def _pareto_flags(frame: pd.DataFrame) -> np.ndarray:
    """Nondominance for gain/gain/cost/cost objectives."""
    values = frame[
        ["style_probability_gain", "target_distance_improvement", "standardized_l1", "changed_modules"]
    ].to_numpy(dtype=float)
    objectives = np.column_stack([values[:, 0], values[:, 1], -values[:, 2], -values[:, 3]])
    flags = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        dominated = np.all(objectives >= objectives[i] - 1e-12, axis=1) & np.any(
            objectives > objectives[i] + 1e-12, axis=1
        )
        dominated[i] = False
        if dominated.any():
            flags[i] = False
    return flags


def source_aware_range_mask(
    candidate: pd.DataFrame,
    source: pd.DataFrame,
    geometry: FoldGeometry,
) -> np.ndarray:
    """Allow pre-existing source extremes but prohibit further excursions.

    A held-out migration source can legitimately lie outside an outer-training
    empirical min/max on immutable context fields.  The feasible envelope is
    therefore the union of that training range and the paired source value.
    """
    candidate = geometry.validate_schema(candidate).reset_index(drop=True)
    source = geometry.validate_schema(source).reset_index(drop=True)
    if len(candidate) != len(source):
        raise ValueError("Candidate and paired source row counts differ")
    lower = np.minimum(
        geometry.minimum.to_numpy(dtype=float)[None, :],
        source.to_numpy(dtype=float),
    )
    upper = np.maximum(
        geometry.maximum.to_numpy(dtype=float)[None, :],
        source.to_numpy(dtype=float),
    )
    values = candidate.to_numpy(dtype=float)
    return ((values >= lower - 1e-12) & (values <= upper + 1e-12)).all(axis=1)


def build_migration_candidates(
    training: pd.DataFrame,
    training_styles: pd.Series,
    training_ids: Sequence[object],
    source: pd.DataFrame,
    source_ids: Sequence[object],
    actionability: pd.DataFrame,
    defect_registry: pd.DataFrame,
    formula_config: Mapping[str, Any],
    style_classifier: Any,
    *,
    source_style: str = "Pils/Lager",
    target_style: str = "Tripel",
    interpolation_budgets: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    module_budgets: Sequence[int] = (3, 6, 12),
    n_components: int = 10,
    coral_shrinkage: float = 0.5,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_iterations: int = 300,
    selection_gain_fraction: float = 0.5,
    eligible_method_prefixes: Sequence[str] | None = None,
    range_policy: str = "training_or_paired_source_envelope_no_further_excursion",
) -> MigrationResult:
    """Build the locked migration grid and select one Pareto point per source."""
    if len(source) != len(source_ids):
        raise ValueError("Source rows and source identifiers are not aligned")
    geometry = FoldGeometry.fit(training)
    source = geometry.impute(source).reset_index(drop=True)
    target_mask = training_styles.astype(str).to_numpy() == str(target_style)
    source_train_mask = training_styles.astype(str).to_numpy() == str(source_style)
    target_training = geometry.impute(training.loc[target_mask]).reset_index(drop=True)
    source_training = geometry.impute(training.loc[source_train_mask]).reset_index(drop=True)
    if target_training.empty or source_training.empty:
        raise ValueError("Source or target migration style is absent from training")
    roles = actionability.set_index("feature")
    decision_columns = roles.index[
        roles["generator_decision_variable"].astype(bool)
        & ~roles.index.isin(DERIVED_FEATURES)
    ].tolist()
    endpoints = _latent_endpoints(
        training,
        training_styles,
        source,
        decision_columns,
        source_style=source_style,
        target_style=target_style,
        n_components=n_components,
        coral_shrinkage=coral_shrinkage,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iterations=sinkhorn_iterations,
    )

    class_names = list(style_classifier.classes_)
    if target_style not in class_names:
        raise ValueError(f"Style classifier lacks target class {target_style}")
    target_index = class_names.index(target_style)
    source_probability = style_classifier.predict_proba(source)[:, target_index]
    source_z = geometry.transform(source)
    target_center = target_training.median().to_frame().T
    target_center_z = geometry.transform(target_center)[0]
    source_target_distance = np.sqrt(np.mean((source_z - target_center_z) ** 2, axis=1))
    training_z = geometry.transform(training)
    pairwise = pairwise_distances(training_z)
    np.fill_diagonal(pairwise, np.inf)
    near_line = float(np.quantile(pairwise.min(axis=1), 0.05))

    candidate_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    projection_rows: list[pd.DataFrame] = []

    def append(method: str, budget: float, raw: pd.DataFrame, selected_modules: list[list[str]] | None = None) -> None:
        projected, projection = project_transfer_constraints(
            raw, source, geometry, defect_registry, formula_config, actionability
        )
        start = len(metadata_rows)
        candidate_frames.append(projected)
        projection.insert(0, "method", method)
        projection.insert(1, "budget", float(budget))
        projection_rows.append(projection)
        method_allowed = method not in REFERENCE_METHODS and (
            eligible_method_prefixes is None
            or any(method.startswith(str(prefix)) for prefix in eligible_method_prefixes)
        )
        for position, source_id in enumerate(source_ids):
            metadata_rows.append(
                {
                    "candidate_index": start + position,
                    "source_beer": source_id,
                    "method": method,
                    "budget": float(budget),
                    "preselected_modules": "" if selected_modules is None else ";".join(selected_modules[position]),
                    "champion_eligible_method": method_allowed,
                }
            )

    append("identity_reference", 0.0, source)
    target_z = geometry.transform(target_training)
    nearest = pairwise_distances(source_z, target_z).argmin(axis=1)
    append("nearest_tripel_reference", 1.0, target_training.iloc[nearest].reset_index(drop=True))
    for method, endpoint in endpoints.items():
        for budget in interpolation_budgets:
            raw = source.copy()
            raw.loc[:, decision_columns] = (
                source.loc[:, decision_columns].to_numpy()
                + float(budget)
                * (endpoint.loc[:, decision_columns].to_numpy() - source.loc[:, decision_columns].to_numpy())
            )
            append(method, float(budget), raw)
    for module_budget in module_budgets:
        endpoint, modules = _module_endpoint(
            source, target_training, actionability, geometry, int(module_budget)
        )
        for budget in interpolation_budgets:
            raw = source.copy()
            raw.loc[:, decision_columns] = (
                source.loc[:, decision_columns].to_numpy()
                + float(budget)
                * (endpoint.loc[:, decision_columns].to_numpy() - source.loc[:, decision_columns].to_numpy())
            )
            append(f"module_counterfactual_{int(module_budget)}", float(budget), raw, modules)

    candidates = pd.concat(candidate_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows).sort_values("candidate_index").reset_index(drop=True)
    if len(candidates) != len(metadata):
        raise RuntimeError("Migration candidate metadata lost row alignment")
    candidates.insert(0, "candidate_id", [f"migration_{i:05d}" for i in range(len(candidates))])
    candidates = pd.concat([metadata, candidates], axis=1)
    chemistry = candidates.loc[:, list(geometry.feature_names)]
    paired_source = source.iloc[np.arange(len(candidates)) % len(source)].reset_index(drop=True)
    paired_source.index = chemistry.index
    defect_failed, _ = defect_violation_mask(
        chemistry,
        defect_registry,
        source=paired_source,
        target_style_training=target_training,
    )
    finite = np.isfinite(chemistry.to_numpy(dtype=float)).all(axis=1)
    within_training = chemistry.ge(geometry.minimum - 1e-12).all(axis=1).to_numpy() & chemistry.le(
        geometry.maximum + 1e-12
    ).all(axis=1).to_numpy()
    within_source_envelope = source_aware_range_mask(chemistry, paired_source, geometry)
    expected = recompute_derived_features(chemistry, formula_config)
    derived_ok = np.isclose(
        chemistry.loc[:, list(DERIVED_FEATURES)].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        atol=1e-10,
        rtol=0.0,
    ).all(axis=1)
    copied = exact_copy_mask(chemistry, geometry.impute(training))
    nearest_training = pairwise_distances(geometry.transform(chemistry), training_z).min(axis=1)
    candidate_z = geometry.transform(chemistry)
    target_distance = np.sqrt(np.mean((candidate_z - target_center_z) ** 2, axis=1))
    probability = style_classifier.predict_proba(chemistry)[:, target_index]

    repeated_source_probability = np.tile(source_probability, len(candidates) // len(source))
    repeated_source_distance = np.tile(source_target_distance, len(candidates) // len(source))
    paired_source_values = paired_source.to_numpy(dtype=float)
    delta_z = (chemistry.to_numpy(dtype=float) - paired_source_values) / geometry.robust_scale.to_numpy()
    changed = np.abs(delta_z) > 1e-6
    module_by_feature = roles.loc[list(geometry.feature_names), "consensus_module"].astype(str).to_numpy()
    candidate_flags = roles.loc[list(geometry.feature_names), "q2_candidate_count"].fillna(0).to_numpy() > 0
    changed_modules = [len(set(module_by_feature[row])) for row in changed]
    metrics = metadata.copy()
    metrics["candidate_id"] = candidates["candidate_id"]
    metrics["target_style_probability_before"] = repeated_source_probability
    metrics["target_style_probability_after"] = probability
    metrics["style_probability_gain"] = probability - repeated_source_probability
    metrics["target_distance_before"] = repeated_source_distance
    metrics["target_distance_after"] = target_distance
    metrics["target_distance_improvement"] = repeated_source_distance - target_distance
    metrics["standardized_l1"] = np.mean(np.abs(delta_z), axis=1)
    metrics["standardized_l2"] = np.sqrt(np.mean(delta_z**2, axis=1))
    metrics["changed_fields"] = changed.sum(axis=1).astype(int)
    metrics["changed_modules"] = np.asarray(changed_modules, dtype=int)
    metrics["changed_q2_candidate_fields"] = (changed & candidate_flags[None, :]).sum(axis=1).astype(int)
    metrics["numeric_valid"] = finite
    metrics["within_training_minmax"] = within_training
    metrics["within_training_or_source_envelope"] = within_source_envelope
    metrics["derived_fields_verified"] = derived_ok
    metrics["chemical_defects_pass"] = ~defect_failed
    metrics["exact_training_copy"] = copied
    metrics["nearest_training_distance"] = nearest_training
    metrics["near_neighbor_danger"] = nearest_training < near_line - 1e-12
    if range_policy == "strict_outer_training_minmax_v1":
        range_pass = within_training
    elif range_policy == "training_or_paired_source_envelope_no_further_excursion":
        range_pass = within_source_envelope
    else:
        raise ValueError(f"Unsupported migration range policy: {range_policy}")
    metrics["range_policy"] = str(range_policy)
    metrics["range_policy_pass"] = range_pass
    metrics["hard_constraints_pass"] = finite & range_pass & derived_ok & ~defect_failed
    metrics["selection_eligible"] = (
        metrics["champion_eligible_method"]
        & metrics["hard_constraints_pass"]
        & ~metrics["exact_training_copy"]
        & ~metrics["near_neighbor_danger"]
    )

    pareto_frames = []
    selected_rows = []
    for source_id, group in metrics.groupby("source_beer", sort=False):
        group = group.copy()
        eligible = group.loc[group["selection_eligible"]].copy()
        group["pareto_optimal"] = False
        group["selected"] = False
        group["selection_status"] = "NOT_SELECTED"
        if len(eligible):
            flags = _pareto_flags(eligible)
            group.loc[eligible.index, "pareto_optimal"] = flags
            frontier = eligible.loc[flags].copy()
            positive = frontier.loc[
                (frontier["style_probability_gain"] > 1e-12)
                & (frontier["target_distance_improvement"] >= -1e-12)
            ]
            if len(positive):
                best_gain = float(positive["style_probability_gain"].max())
                threshold = float(selection_gain_fraction) * best_gain
                pool = positive.loc[positive["style_probability_gain"] >= threshold - 1e-12]
                chosen = pool.sort_values(
                    ["standardized_l1", "changed_modules", "target_distance_improvement", "style_probability_gain"],
                    ascending=[True, True, False, False],
                ).iloc[0]
                status = "SELECTED_MIN_COST_WITH_POSITIVE_STYLE_AND_CHEMISTRY_GAIN"
            else:
                chosen = frontier.sort_values(
                    ["style_probability_gain", "target_distance_improvement", "standardized_l1"],
                    ascending=[False, False, True],
                ).iloc[0]
                status = "SELECTED_BEST_AVAILABLE_WITHOUT_DUAL_GAIN"
            group.loc[chosen.name, "selected"] = True
            group.loc[chosen.name, "selection_status"] = status
            selected_rows.append(chosen.name)
        else:
            group["selection_status"] = "NO_ELIGIBLE_CANDIDATE"
        pareto_frames.append(group)
    pareto = pd.concat(pareto_frames).sort_index()
    selected_metrics = pareto.loc[pareto["selected"]].copy()
    selected = candidates.loc[selected_metrics.index].copy()
    selected = selected.merge(
        selected_metrics[[
            "candidate_id",
            "selection_status",
            "style_probability_gain",
            "target_distance_improvement",
            "standardized_l1",
            "standardized_l2",
            "changed_fields",
            "changed_modules",
            "nearest_training_distance",
            "near_neighbor_danger",
        ]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    return MigrationResult(candidates=candidates, metrics=metrics, pareto=pareto, selected=selected)

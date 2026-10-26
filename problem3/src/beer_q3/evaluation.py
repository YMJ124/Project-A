"""Leakage-safe fold utilities and evaluation primitives for Question 3.

The functions in this module are deliberately fold-local.  They accept the
outer-training and outer-validation frames explicitly and never read the full
competition data on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


DERIVED_FEATURES = ("esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum")


def _finite_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=float, na_value=np.nan)


def _safe_scale(frame: pd.DataFrame) -> pd.Series:
    """Return a fold-fitted robust denominator with deterministic fallbacks."""
    q25 = frame.quantile(0.25)
    q75 = frame.quantile(0.75)
    scale = q75 - q25
    standard = frame.std(ddof=0)
    scale = scale.where(scale > 1e-12, standard)
    return scale.where(scale > 1e-12, 1.0).astype(float)


@dataclass(frozen=True)
class FoldGeometry:
    """Training-fold-only geometry used by all distance and range checks."""

    feature_names: tuple[str, ...]
    imputer: SimpleImputer
    scaler: RobustScaler
    minimum: pd.Series
    maximum: pd.Series
    q01: pd.Series
    q99: pd.Series
    robust_scale: pd.Series

    @classmethod
    def fit(cls, training: pd.DataFrame) -> "FoldGeometry":
        if training.empty or training.columns.duplicated().any():
            raise ValueError("Training features must be non-empty with unique columns")
        values = _finite_matrix(training)
        if np.isinf(values).any():
            raise ValueError("Training features contain infinite values")
        if training.isna().all().any():
            missing = training.columns[training.isna().all()].tolist()
            raise ValueError(f"All-missing training features: {missing}")
        imputer = SimpleImputer(strategy="median")
        imputed = pd.DataFrame(
            imputer.fit_transform(training),
            columns=training.columns,
            index=training.index,
        )
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        scaler.fit(imputed)
        return cls(
            feature_names=tuple(training.columns),
            imputer=imputer,
            scaler=scaler,
            minimum=imputed.min(),
            maximum=imputed.max(),
            q01=imputed.quantile(0.01),
            q99=imputed.quantile(0.99),
            robust_scale=_safe_scale(imputed),
        )

    def validate_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Chemical profiles must be a pandas DataFrame")
        missing = sorted(set(self.feature_names).difference(frame.columns))
        extra = sorted(set(frame.columns).difference(self.feature_names))
        if missing or extra:
            raise ValueError(f"Chemical schema mismatch; missing={missing}, extra={extra}")
        return frame.loc[:, list(self.feature_names)]

    def impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        ordered = self.validate_schema(frame)
        return pd.DataFrame(
            self.imputer.transform(ordered),
            columns=list(self.feature_names),
            index=ordered.index,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self.impute(frame))


def map_relative_0_6(
    predictions: pd.DataFrame,
    training_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map standardized predictions to the locked relative 0--6 report scale."""
    missing = sorted(set(predictions.columns).difference(training_targets.columns))
    if missing:
        raise ValueError(f"Training targets are missing columns: {missing}")
    training = training_targets.loc[:, predictions.columns]
    center = training.median()
    scale = (training.quantile(0.75) - training.quantile(0.25)) / 1.349
    fallback = training.std(ddof=0)
    scale = scale.where(scale > 1e-12, fallback).where(lambda x: x > 1e-12, 1.0)
    continuous = (3.0 + predictions.subtract(center).divide(scale)).clip(0.0, 6.0)
    integer = np.rint(continuous).astype(int)
    return continuous, integer


def normalized_wasserstein_by_feature(
    left: pd.DataFrame,
    right: pd.DataFrame,
    scale: pd.Series,
) -> pd.Series:
    if list(left.columns) != list(right.columns):
        raise ValueError("Wasserstein inputs must have identical ordered columns")
    rows = {}
    for feature in left.columns:
        a = left[feature].dropna().to_numpy(dtype=float)
        b = right[feature].dropna().to_numpy(dtype=float)
        rows[feature] = (
            float(wasserstein_distance(a, b) / scale[feature])
            if len(a) and len(b)
            else np.nan
        )
    return pd.Series(rows, dtype=float)


def spearman_structure_error(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Root-mean-square difference between finite off-diagonal correlations."""
    if list(left.columns) != list(right.columns):
        raise ValueError("Structure inputs must have identical ordered columns")
    if len(left) < 3 or len(right) < 3:
        return np.nan
    a = left.corr(method="spearman", min_periods=3).to_numpy(dtype=float)
    b = right.corr(method="spearman", min_periods=3).to_numpy(dtype=float)
    off_diagonal = ~np.eye(len(a), dtype=bool)
    finite = np.isfinite(a) & np.isfinite(b) & off_diagonal
    return float(np.sqrt(np.mean((a[finite] - b[finite]) ** 2))) if finite.any() else np.nan


def consensus_edge_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    consensus_edges: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate only the locked Q2 consensus edge set."""
    edges = consensus_edges.loc[consensus_edges["consensus_edge"].astype(bool)]
    if edges.empty:
        raise ValueError("The Q2 consensus edge table contains no selected edges")
    corr_left = left.corr(method="spearman", min_periods=3)
    corr_right = right.corr(method="spearman", min_periods=3)
    left_values = []
    right_values = []
    for row in edges.itertuples(index=False):
        a = corr_left.at[row.feature_left, row.feature_right]
        b = corr_right.at[row.feature_left, row.feature_right]
        if np.isfinite(a) and np.isfinite(b):
            left_values.append(float(a))
            right_values.append(float(b))
    if not left_values:
        return {
            "q2_edges_evaluable": 0.0,
            "q2_edge_sign_agreement": np.nan,
            "q2_edge_strength_mae": np.nan,
            "q2_edge_loss_rate": np.nan,
        }
    a = np.asarray(left_values)
    b = np.asarray(right_values)
    return {
        "q2_edges_evaluable": float(len(a)),
        "q2_edge_sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
        "q2_edge_strength_mae": float(np.mean(np.abs(a - b))),
        "q2_edge_loss_rate": float(np.mean((np.abs(a) >= 0.3) & (np.abs(b) < 0.3))),
    }


def _aggregate_distribution_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    scale: pd.Series,
) -> dict[str, float]:
    distances = normalized_wasserstein_by_feature(left, right, scale)
    return {
        "wasserstein_median": float(distances.median()),
        "wasserstein_q75": float(distances.quantile(0.75)),
        "wasserstein_max": float(distances.max()),
        "spearman_structure_rmse": spearman_structure_error(left, right),
    }


def build_real_real_reference(
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    desired_size: int,
    geometry: FoldGeometry,
    consensus_edges: Optional[pd.DataFrame] = None,
    *,
    n_resamples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a disjoint real--real reference using only an outer training fold."""
    style_mask = training_styles.astype(str).to_numpy() == str(target_style)
    pool = geometry.impute(training.loc[style_mask])
    subset_size = min(int(desired_size), len(pool) // 2)
    if subset_size < 2:
        raise ValueError(f"Insufficient fold-training support for {target_style}: {len(pool)}")
    rng = np.random.default_rng(seed)
    rows = []
    for resample in range(int(n_resamples)):
        order = rng.permutation(len(pool))
        left = pool.iloc[order[:subset_size]]
        right = pool.iloc[order[subset_size : 2 * subset_size]]
        metrics = _aggregate_distribution_metrics(left, right, geometry.robust_scale)
        if consensus_edges is not None:
            metrics.update(consensus_edge_metrics(left, right, consensus_edges))
        rows.extend(
            {
                "target_style": target_style,
                "resample": resample,
                "subset_size": subset_size,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        )
    return pd.DataFrame(rows)


def summarize_reference(reference: pd.DataFrame) -> pd.DataFrame:
    return (
        reference.groupby(["target_style", "metric"], as_index=False)
        .agg(
            median=("value", "median"),
            q05=("value", lambda x: x.quantile(0.05)),
            q95=("value", lambda x: x.quantile(0.95)),
        )
    )


def _rows_equal_any(candidate: np.ndarray, training: np.ndarray, atol: float) -> bool:
    equal = np.isclose(training, candidate[None, :], rtol=0.0, atol=atol, equal_nan=True)
    return bool(np.any(np.all(equal, axis=1)))


def exact_copy_mask(
    generated: pd.DataFrame,
    training: pd.DataFrame,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    if list(generated.columns) != list(training.columns):
        raise ValueError("Exact-copy inputs must have identical ordered columns")
    train_values = _finite_matrix(training)
    return np.asarray(
        [_rows_equal_any(row, train_values, atol) for row in _finite_matrix(generated)],
        dtype=bool,
    )


def nearest_training_distances(
    query: pd.DataFrame,
    training: pd.DataFrame,
    geometry: FoldGeometry,
) -> np.ndarray:
    distances = pairwise_distances(geometry.transform(query), geometry.transform(training))
    return distances.min(axis=1)


def fit_style_classifier(
    training: pd.DataFrame,
    training_styles: pd.Series,
    *,
    seed: int,
):
    """Fit the locked fold-local style recognizer; no validation rows are used."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(quantile_range=(25.0, 75.0)),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    model.fit(training, training_styles.astype(str))
    return model


def _inverse_defect(values: pd.Series, inverse_transform: str) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    if inverse_transform == "identity":
        return array
    if inverse_transform == "pow10":
        return np.power(10.0, array)
    if inverse_transform == "exp":
        return np.exp(array)
    raise ValueError(f"Unsupported defect inverse transform: {inverse_transform}")


def defect_violation_mask(
    generated: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    source: Optional[pd.DataFrame] = None,
    target_style_training: Optional[pd.DataFrame] = None,
    relative_generation_quantile: float = 0.95,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return row-level chemical defect failures and an auditable detail table.

    For trans-2-nonenal, transfer rows use the locked source-relative rule.  A
    de-novo generation row has no source, so the fold-training target-style
    quantile is used as a distributional cap and is explicitly labelled as
    such; it is not presented as a sensory threshold.
    """
    failures = np.zeros(len(generated), dtype=bool)
    details = []
    chemical = registry.loc[registry["chemical_field_present"].astype(bool)]
    for row in chemical.itertuples(index=False):
        feature = str(row.chemical_field)
        values = _inverse_defect(generated[feature], str(row.inverse_transform))
        if row.constraint_mode == "absolute_excess_nonincrease":
            threshold = float(row.threshold_used)
            if source is not None:
                if len(source) != len(generated):
                    raise ValueError("Transfer source and generated row counts differ")
                source_values = _inverse_defect(
                    source[feature], str(row.inverse_transform)
                )
                limit_values = np.maximum(source_values, threshold)
                violation = values > limit_values + 1e-12
                limit = np.nan
                policy = "source_absolute_excess_nonincrease"
            else:
                limit = threshold
                violation = values > limit + 1e-12
                policy = "locked_absolute_threshold"
        elif row.constraint_mode == "relative_concentration_nonincrease":
            if source is not None:
                if len(source) != len(generated):
                    raise ValueError("Transfer source and generated row counts differ")
                limit_values = _inverse_defect(source[feature], str(row.inverse_transform))
                violation = values > limit_values + 1e-12
                limit = np.nan
                policy = "source_relative_nonincrease"
            else:
                if target_style_training is None:
                    raise ValueError("Target-style training data are required for de-novo t2n audit")
                reference = _inverse_defect(
                    target_style_training[feature], str(row.inverse_transform)
                )
                limit = float(np.quantile(reference[np.isfinite(reference)], relative_generation_quantile))
                violation = values > limit + 1e-12
                policy = "fold_target_style_distribution_cap_not_absolute_threshold"
        else:
            raise ValueError(f"Chemical defect constraint is not ready: {row.defect}")
        failures |= violation
        details.append(
            {
                "defect": row.defect,
                "chemical_field": feature,
                "policy": policy,
                "limit": limit,
                "violation_count": int(violation.sum()),
                "violation_rate": float(violation.mean()) if len(violation) else np.nan,
            }
        )
    return failures, pd.DataFrame(details)


def evaluate_hard_gates(
    generated: pd.DataFrame,
    training: pd.DataFrame,
    validation_real: pd.DataFrame,
    geometry: FoldGeometry,
    defect_registry: pd.DataFrame,
    *,
    target_style_training: pd.DataFrame,
    derived_expected: Optional[pd.DataFrame],
    source: Optional[pd.DataFrame] = None,
    exact_copy_atol: float = 1e-12,
    near_neighbor_quantile: float = 0.05,
    near_neighbor_rate_limit: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate non-negotiable row-level gates for one outer fold/style."""
    ordered = geometry.validate_schema(generated)
    numeric_valid = np.isfinite(_finite_matrix(ordered)).all(axis=1)
    imputed_train = geometry.impute(training)
    finite_generated = ordered.fillna(pd.Series(geometry.imputer.statistics_, index=ordered.columns))
    in_range = (
        finite_generated.ge(geometry.minimum - 1e-12)
        & finite_generated.le(geometry.maximum + 1e-12)
    ).all(axis=1).to_numpy()
    copied = exact_copy_mask(ordered, training, atol=exact_copy_atol)

    if derived_expected is None:
        derived_ok = np.zeros(len(ordered), dtype=bool)
        derived_detail = "missing_recomputation_evidence"
    else:
        expected = derived_expected.loc[ordered.index, list(DERIVED_FEATURES)]
        observed = ordered.loc[:, list(DERIVED_FEATURES)]
        derived_ok = np.isclose(
            observed.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        ).all(axis=1)
        derived_detail = "generator_supplied_recomputed_values_verified"

    defect_failed, defect_detail = defect_violation_mask(
        ordered,
        defect_registry,
        source=source,
        target_style_training=target_style_training,
    )
    generated_nn = nearest_training_distances(finite_generated, imputed_train, geometry)
    validation_nn = nearest_training_distances(validation_real, imputed_train, geometry)
    training_z = geometry.transform(imputed_train)
    training_pairwise = pairwise_distances(training_z)
    np.fill_diagonal(training_pairwise, np.inf)
    training_loo_nn = training_pairwise.min(axis=1)
    danger_line = float(np.quantile(training_loo_nn, near_neighbor_quantile))
    dangerous = generated_nn < danger_line - 1e-12
    validation_dangerous = validation_nn < danger_line - 1e-12
    dangerous_rate = float(dangerous.mean()) if len(dangerous) else np.nan
    validation_dangerous_rate = (
        float(validation_dangerous.mean()) if len(validation_dangerous) else np.nan
    )
    allowed_dangerous_rate = max(
        float(near_neighbor_rate_limit), validation_dangerous_rate
    )

    trace = pd.DataFrame(
        {
            "numeric_valid": numeric_valid,
            "within_training_minmax": in_range,
            "derived_fields_verified": derived_ok,
            "chemical_defects_pass": ~defect_failed,
            "exact_training_copy": copied,
            "nearest_training_distance": generated_nn,
            "near_neighbor_danger": dangerous,
        },
        index=ordered.index,
    )
    gate_rows = [
        ("numeric_validity", bool(numeric_valid.all()), int((~numeric_valid).sum()), "NaN/Inf must be zero"),
        ("physical_training_range", bool(in_range.all()), int((~in_range).sum()), "fold-training min/max"),
        ("derived_field_recomputation", bool(derived_ok.all()), int((~derived_ok).sum()), derived_detail),
        ("chemical_defect_constraints", bool((~defect_failed).all()), int(defect_failed.sum()), "DMS/Diacetyl/4VG absolute; t2n relative/distributional"),
        ("exact_copy_rate", bool((~copied).all()), int(copied.sum()), "required rate=0"),
        (
            "near_neighbor_rate",
            bool(dangerous_rate <= allowed_dangerous_rate + 1e-12),
            int(dangerous.sum()),
            f"training_loo_distance<{danger_line:.12g}; "
            f"validation_rate={validation_dangerous_rate:.6g}; "
            f"allowed_rate={allowed_dangerous_rate:.6g}",
        ),
    ]
    gates = pd.DataFrame(gate_rows, columns=["gate", "pass", "failure_count", "detail"])
    gates["n_samples"] = len(ordered)
    gates["failure_rate"] = gates["failure_count"] / max(len(ordered), 1)
    gates["reference_failure_rate"] = np.nan
    gates["allowed_failure_rate"] = np.nan
    near_row = gates["gate"] == "near_neighbor_rate"
    gates.loc[near_row, "reference_failure_rate"] = validation_dangerous_rate
    gates.loc[near_row, "allowed_failure_rate"] = allowed_dangerous_rate
    gates["all_hard_gates_pass"] = bool(gates["pass"].all())
    defect_detail["gate"] = "chemical_defect_constraints"
    return gates, trace.join(pd.Series(~defect_failed, index=trace.index, name="defect_row_pass")), defect_detail


def evaluate_generated_vs_real(
    generated: pd.DataFrame,
    validation_real: pd.DataFrame,
    training: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    geometry: FoldGeometry,
    style_classifier,
    consensus_edges: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    generated_i = geometry.impute(generated)
    validation_i = geometry.impute(validation_real)
    metrics = _aggregate_distribution_metrics(
        generated_i, validation_i, geometry.robust_scale
    )
    if consensus_edges is not None:
        metrics.update(consensus_edge_metrics(generated_i, validation_i, consensus_edges))

    classes = list(style_classifier.classes_)
    if target_style not in classes:
        raise ValueError(f"Style classifier lacks target class {target_style}")
    class_index = classes.index(target_style)
    generated_probability = style_classifier.predict_proba(generated)[:, class_index]
    real_probability = style_classifier.predict_proba(validation_real)[:, class_index]
    metrics.update(
        {
            "style_probability_generated_mean": float(generated_probability.mean()),
            "style_probability_real_mean": float(real_probability.mean()),
            "style_probability_gap": float(real_probability.mean() - generated_probability.mean()),
            "style_hit_rate_generated": float(
                np.mean(style_classifier.predict(generated) == target_style)
            ),
            "style_hit_rate_real": float(
                np.mean(style_classifier.predict(validation_real) == target_style)
            ),
        }
    )
    transformed = geometry.transform(generated)
    if len(generated) >= 2:
        pairwise = pairwise_distances(transformed)
        pairwise_values = pairwise[np.triu_indices_from(pairwise, k=1)]
        metrics["generated_pairwise_distance_median"] = float(np.median(pairwise_values))
    else:
        metrics["generated_pairwise_distance_median"] = np.nan
    metrics["generated_unique_rate"] = float(
        len(generated.drop_duplicates()) / max(len(generated), 1)
    )
    return pd.DataFrame(
        [{"target_style": target_style, "metric": key, "value": value} for key, value in metrics.items()]
    )


def attach_reference_targets(
    metrics: pd.DataFrame,
    reference_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the real--real 95% line and compute transparent PASS/FAIL gaps."""
    result = metrics.merge(
        reference_summary,
        on=["target_style", "metric"],
        how="left",
        validate="many_to_one",
    )
    result["minimum_line"] = result["q95"]
    result["ideal_line"] = result["median"]
    distance_metric = result["metric"].isin(
        [
            "wasserstein_median",
            "wasserstein_q75",
            "wasserstein_max",
            "spearman_structure_rmse",
            "q2_edge_strength_mae",
            "q2_edge_loss_rate",
        ]
    )
    result["relative_gap"] = np.where(
        distance_metric & result["minimum_line"].notna(),
        np.maximum(0.0, (result["value"] - result["minimum_line"]) / (result["minimum_line"].abs() + 1e-12)),
        np.nan,
    )
    result["status"] = np.where(
        result["relative_gap"].isna(),
        "AUDIT_ONLY",
        np.where(result["relative_gap"] == 0.0, "PASS", np.where(result["relative_gap"] < 0.05, "NEAR", "FAIL")),
    )
    return result

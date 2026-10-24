"""Leakage-safe model construction and nested tuning.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import json
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from .folds import style_balanced_fold_ids


BASELINE_MODELS = {"global_mean", "style_mean"}


class TargetwisePLSRegressor(BaseEstimator, RegressorMixin):
    """Fit one PLS1 model per target and always return (samples, targets)."""

    def __init__(self, n_components: int = 2, max_iter: int = 1000):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TargetwisePLSRegressor":
        target = np.asarray(y)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        self.models_ = []
        for target_index in range(target.shape[1]):
            model = PLSRegression(
                n_components=self.n_components, scale=False, max_iter=self.max_iter
            )
            model.fit(X, target[:, target_index])
            self.models_.append(model)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise ValueError("TargetwisePLSRegressor is not fitted")
        return np.column_stack(
            [np.asarray(model.predict(X)).reshape(-1) for model in self.models_]
        )


class FeatureSubsetTargetwiseExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """Fit one Extra Trees model per target on a fixed, declared column subset.

    The subset is supplied by configuration and must be defined independently
    of the current response data (for example, from chemistry literature).
    Fold-local imputation remains outside this estimator in the enclosing
    pipeline.  Original features precede any appended missing indicators, so
    stable original-column indices remain valid after imputation.
    """

    def __init__(
        self,
        feature_indices: Sequence[int],
        n_estimators: int = 100,
        max_features: Any = 1.0,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.feature_indices = feature_indices
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "FeatureSubsetTargetwiseExtraTreesRegressor":
        features = np.asarray(X)
        target = np.asarray(y)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        indices = np.asarray(self.feature_indices, dtype=int)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("feature_indices must be a non-empty one-dimensional list")
        if indices.min() < 0 or indices.max() >= features.shape[1]:
            raise ValueError("feature_indices contain an out-of-range column")
        if np.unique(indices).size != indices.size:
            raise ValueError("feature_indices must not contain duplicates")
        self.feature_indices_ = indices
        self.n_features_in_ = features.shape[1]
        selected = features[:, indices]
        self.estimators_ = []
        for target_index in range(target.shape[1]):
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            model.fit(selected, target[:, target_index])
            self.estimators_.append(model)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise ValueError("FeatureSubsetTargetwiseExtraTreesRegressor is not fitted")
        features = np.asarray(X)
        selected = features[:, self.feature_indices_]
        return np.column_stack([model.predict(selected) for model in self.estimators_])


class FoldStandardizedGroupSummaryTransformer(BaseEstimator, TransformerMixin):
    """Append fold-standardized means for fixed, literature-defined groups."""

    def __init__(self, groups: Sequence[Sequence[int]]):
        self.groups = groups

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "FoldStandardizedGroupSummaryTransformer":
        features = np.asarray(X, dtype=float)
        checked_groups = []
        group_means = []
        group_scales = []
        for group in self.groups:
            indices = np.asarray(group, dtype=int)
            if indices.ndim != 1 or indices.size == 0:
                raise ValueError("Every summary group must be non-empty and one-dimensional")
            if indices.min() < 0 or indices.max() >= features.shape[1]:
                raise ValueError("A summary group contains an out-of-range column")
            if np.unique(indices).size != indices.size:
                raise ValueError("A summary group must not contain duplicate columns")
            values = features[:, indices]
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale[scale == 0.0] = 1.0
            checked_groups.append(indices)
            group_means.append(mean)
            group_scales.append(scale)
        self.groups_ = checked_groups
        self.group_means_ = group_means
        self.group_scales_ = group_scales
        self.n_features_in_ = features.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "groups_"):
            raise ValueError("FoldStandardizedGroupSummaryTransformer is not fitted")
        features = np.asarray(X, dtype=float)
        summaries = []
        for indices, mean, scale in zip(
            self.groups_, self.group_means_, self.group_scales_
        ):
            standardized = (features[:, indices] - mean) / scale
            summaries.append(standardized.mean(axis=1))
        return np.column_stack([features, *summaries])


class ThresholdMechanismTransformer(BaseEstimator, TransformerMixin):
    """Build fixed, literature-shaped hop-aroma features without target leakage.

    Concentrations are converted to dimensionless threshold ratios using fixed
    sensory-threshold priors. Smooth logarithmic and Hill-type activations,
    pair/triple synergy terms, balance summaries, and optional fold-standardized
    matrix interactions are then derived. No response value is used here.

    The resulting variables are threshold-inspired proxies rather than claimed
    beer-specific OAVs because published thresholds vary with the beer matrix.
    """

    def __init__(
        self,
        feature_indices: Sequence[int],
        thresholds: Sequence[float],
        matrix_indices: Sequence[int] = (),
        append_original: bool = False,
    ):
        self.feature_indices = feature_indices
        self.thresholds = thresholds
        self.matrix_indices = matrix_indices
        self.append_original = append_original

    @staticmethod
    def _checked_indices(
        indices: Sequence[int], n_features: int, label: str, allow_empty: bool
    ) -> np.ndarray:
        result = np.asarray(indices, dtype=int)
        if result.ndim != 1 or (result.size == 0 and not allow_empty):
            qualifier = "possibly empty" if allow_empty else "non-empty"
            raise ValueError(f"{label} must be a {qualifier} one-dimensional list")
        if result.size and (result.min() < 0 or result.max() >= n_features):
            raise ValueError(f"{label} contain an out-of-range column")
        if np.unique(result).size != result.size:
            raise ValueError(f"{label} must not contain duplicates")
        return result

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "ThresholdMechanismTransformer":
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("ThresholdMechanismTransformer expects a 2D matrix")
        self.n_features_in_ = features.shape[1]
        self.feature_indices_ = self._checked_indices(
            self.feature_indices, self.n_features_in_, "feature_indices", False
        )
        self.matrix_indices_ = self._checked_indices(
            self.matrix_indices, self.n_features_in_, "matrix_indices", True
        )
        thresholds = np.asarray(self.thresholds, dtype=float)
        if thresholds.ndim != 1 or thresholds.size != self.feature_indices_.size:
            raise ValueError("thresholds must align one-to-one with feature_indices")
        if not np.isfinite(thresholds).all() or (thresholds <= 0.0).any():
            raise ValueError("thresholds must be finite and strictly positive")
        self.thresholds_ = thresholds

        if self.matrix_indices_.size:
            matrix = features[:, self.matrix_indices_]
            self.matrix_means_ = matrix.mean(axis=0)
            self.matrix_scales_ = matrix.std(axis=0)
            self.matrix_scales_[self.matrix_scales_ == 0.0] = 1.0
        else:
            self.matrix_means_ = np.empty(0, dtype=float)
            self.matrix_scales_ = np.empty(0, dtype=float)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "thresholds_"):
            raise ValueError("ThresholdMechanismTransformer is not fitted")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected a 2D matrix with {self.n_features_in_} columns"
            )

        concentrations = np.clip(features[:, self.feature_indices_], 0.0, None)
        ratio = concentrations / self.thresholds_
        log_ratio = np.log1p(ratio)
        activation = ratio / (1.0 + ratio)
        mean_log_load = log_ratio.mean(axis=1, keepdims=True)
        activation_mean = activation.mean(axis=1, keepdims=True)
        activation_min = activation.min(axis=1, keepdims=True)
        activation_max = activation.max(axis=1, keepdims=True)
        activation_sd = activation.std(axis=1, keepdims=True)
        geometric_synergy = np.exp(
            np.mean(np.log(np.clip(activation, 1e-12, None)), axis=1, keepdims=True)
        )

        interactions = []
        for left in range(activation.shape[1]):
            for right in range(left + 1, activation.shape[1]):
                interactions.append(
                    (activation[:, left] * activation[:, right]).reshape(-1, 1)
                )
        triple_synergy = np.prod(activation, axis=1, keepdims=True)
        mechanism = np.column_stack(
            [
                log_ratio,
                activation,
                mean_log_load,
                activation_mean,
                activation_min,
                activation_max,
                activation_sd,
                geometric_synergy,
                *interactions,
                triple_synergy,
            ]
        )

        if self.matrix_indices_.size:
            matrix = (
                features[:, self.matrix_indices_] - self.matrix_means_
            ) / self.matrix_scales_
            matrix_terms = [matrix]
            for summary in (mean_log_load, geometric_synergy):
                matrix_terms.append(matrix * summary)
            mechanism = np.column_stack([mechanism, *matrix_terms])

        if not np.isfinite(mechanism).all():
            raise ValueError("Threshold mechanism features contain non-finite values")
        if self.append_original:
            return np.column_stack([features, mechanism])
        return mechanism


class CrossFittedMechanismResidualExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """Add a full-chemistry Extra Trees residual to a low-dimensional mechanism.

    The mechanistic ridge prediction is cross-fitted on the current training
    fold before residuals are formed. This prevents the residual learner from
    seeing in-sample mechanism errors and keeps the entire two-stage procedure
    inside the enclosing outer fold.
    """

    def __init__(
        self,
        feature_indices: Sequence[int],
        thresholds: Sequence[float],
        matrix_indices: Sequence[int] = (),
        ridge_alpha: float = 10.0,
        residual_splits: int = 3,
        n_estimators: int = 100,
        max_features: Any = 1.0,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.feature_indices = feature_indices
        self.thresholds = thresholds
        self.matrix_indices = matrix_indices
        self.ridge_alpha = ridge_alpha
        self.residual_splits = residual_splits
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _new_mechanism_model(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "mechanism",
                    ThresholdMechanismTransformer(
                        feature_indices=self.feature_indices,
                        thresholds=self.thresholds,
                        matrix_indices=self.matrix_indices,
                        append_original=False,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(self.ridge_alpha))),
            ]
        )

    def _new_residual_model(self) -> MultiOutputRegressor:
        tree = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return MultiOutputRegressor(tree, n_jobs=self.n_jobs)

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "CrossFittedMechanismResidualExtraTreesRegressor":
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if not 2 <= int(self.residual_splits) <= len(target):
            raise ValueError("residual_splits must be between 2 and n_samples")

        splitter = KFold(
            n_splits=int(self.residual_splits),
            shuffle=True,
            random_state=self.random_state,
        )
        cross_fitted = np.empty_like(target, dtype=float)
        for train_index, valid_index in splitter.split(features):
            mechanism = self._new_mechanism_model()
            mechanism.fit(features[train_index], target[train_index])
            cross_fitted[valid_index] = ensure_2d_prediction(
                mechanism.predict(features[valid_index]), n_targets=target.shape[1]
            )

        self.mechanism_oof_predictions_ = cross_fitted
        self.residual_model_ = self._new_residual_model()
        self.residual_model_.fit(features, target - cross_fitted)
        self.mechanism_model_ = self._new_mechanism_model()
        self.mechanism_model_.fit(features, target)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mechanism_model_"):
            raise ValueError(
                "CrossFittedMechanismResidualExtraTreesRegressor is not fitted"
            )
        mechanism = ensure_2d_prediction(
            self.mechanism_model_.predict(X), n_targets=self.n_outputs_
        )
        residual = ensure_2d_prediction(
            self.residual_model_.predict(X), n_targets=self.n_outputs_
        )
        return mechanism + residual


class AromaFlavourPairedExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """Fit one two-output Extra Trees model per matched Aroma/Flavour pair.

    The official main-target layout contains 15 ``A_`` columns followed by the
    corresponding 15 ``F_`` columns in the same descriptor order.  Pairing
    keeps chemically related sensory outputs together without forcing all 30
    heterogeneous targets to share every tree split.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Any = 1.0,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: Any) -> "AromaFlavourPairedExtraTreesRegressor":
        target = np.asarray(y, dtype=float)
        if target.ndim != 2 or target.shape[1] % 2 != 0:
            raise ValueError("Paired Extra Trees requires an even 2D target matrix")
        half = target.shape[1] // 2
        if hasattr(y, "columns"):
            columns = list(y.columns)
            expected_flavour = [f"F_{name[2:]}" for name in columns[:half]]
            if not all(name.startswith("A_") for name in columns[:half]):
                raise ValueError("First half of paired targets must be Aroma columns")
            if columns[half:] != expected_flavour:
                raise ValueError("Aroma and Flavour target descriptors do not align")
        self.models_ = []
        for pair_index in range(half):
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=(
                    None
                    if self.random_state is None
                    else int(self.random_state) + pair_index
                ),
                n_jobs=self.n_jobs,
            )
            model.fit(X, target[:, [pair_index, pair_index + half]])
            self.models_.append(model)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise ValueError("AromaFlavourPairedExtraTreesRegressor is not fitted")
        half = self.n_outputs_ // 2
        prediction = np.empty((len(X), self.n_outputs_), dtype=float)
        for pair_index, model in enumerate(self.models_):
            pair_prediction = ensure_2d_prediction(model.predict(X), n_targets=2)
            prediction[:, pair_index] = pair_prediction[:, 0]
            prediction[:, pair_index + half] = pair_prediction[:, 1]
        return prediction


class SensoryFamilyExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """Fit one multi-output Extra Trees model per semantic sensory family.

    The official 30-target layout is six consecutive five-column blocks:
    aroma malt/hops/esters followed by flavour malt/hops/esters.  This model
    shares splits between the ten aroma/flavour outputs in one family, while
    preventing unrelated families from contributing to the same split loss.
    The grouping is fixed from column semantics and never learned from held-out
    target correlations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Any = 1.0,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: Any) -> "SensoryFamilyExtraTreesRegressor":
        target = np.asarray(y, dtype=float)
        if target.ndim != 2 or target.shape[1] != 30:
            raise ValueError("Family-grouped Extra Trees requires exactly 30 targets")
        self.groups_ = [
            np.r_[0:5, 15:20],
            np.r_[5:10, 20:25],
            np.r_[10:15, 25:30],
        ]
        self.models_ = []
        for group_index, columns in enumerate(self.groups_):
            seed = (
                None
                if self.random_state is None
                else int(self.random_state) + group_index
            )
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=seed,
                n_jobs=self.n_jobs,
            )
            model.fit(X, target[:, columns])
            self.models_.append(model)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise ValueError("SensoryFamilyExtraTreesRegressor is not fitted")
        prediction = np.empty((len(X), self.n_outputs_), dtype=float)
        for columns, model in zip(self.groups_, self.models_):
            group_prediction = ensure_2d_prediction(
                model.predict(X), n_targets=len(columns)
            )
            prediction[:, columns] = group_prediction
        return prediction


class CrossFittedTargetStackingExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """Use cross-fitted sensory predictions as leakage-safe meta-features.

    A multi-output Extra Trees model first produces out-of-fold predictions on
    the rows passed to ``fit``.  Within each fixed malt/hops/esters family, a
    ridge meta-model learns a regularized mapping from those ten predictions to
    the ten observed targets.  At inference, the meta prediction is blended
    with the original Extra Trees prediction to limit variance and protect the
    base model when target dependencies do not transfer.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Any = 1.0,
        min_samples_leaf: int = 1,
        stacking_splits: int = 3,
        ridge_alpha: float = 10.0,
        identity_weight: float = 0.5,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.stacking_splits = stacking_splits
        self.ridge_alpha = ridge_alpha
        self.identity_weight = identity_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _new_tree(self, seed: Optional[int]) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
            n_jobs=self.n_jobs,
        )

    def fit(
        self, X: np.ndarray, y: Any
    ) -> "CrossFittedTargetStackingExtraTreesRegressor":
        target = np.asarray(y, dtype=float)
        if target.ndim != 2 or target.shape[1] != 30:
            raise ValueError("Target-stacked Extra Trees requires exactly 30 targets")
        if not 0.0 <= float(self.identity_weight) <= 1.0:
            raise ValueError("identity_weight must be between 0 and 1")
        if int(self.stacking_splits) < 2:
            raise ValueError("stacking_splits must be at least 2")

        splitter = KFold(
            n_splits=int(self.stacking_splits),
            shuffle=True,
            random_state=self.random_state,
        )
        cross_fitted = np.empty_like(target, dtype=float)
        for fold, (train_index, valid_index) in enumerate(splitter.split(X)):
            seed = None if self.random_state is None else int(self.random_state) + fold
            model = self._new_tree(seed)
            model.fit(np.asarray(X)[train_index], target[train_index])
            cross_fitted[valid_index] = ensure_2d_prediction(
                model.predict(np.asarray(X)[valid_index]), n_targets=target.shape[1]
            )

        self.groups_ = [
            np.r_[0:5, 15:20],
            np.r_[5:10, 20:25],
            np.r_[10:15, 25:30],
        ]
        self.meta_models_ = []
        for columns in self.groups_:
            meta = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=float(self.ridge_alpha))),
                ]
            )
            meta.fit(cross_fitted[:, columns], target[:, columns])
            self.meta_models_.append(meta)

        self.cross_fitted_predictions_ = cross_fitted
        self.base_model_ = self._new_tree(self.random_state)
        self.base_model_.fit(X, target)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "base_model_"):
            raise ValueError("CrossFittedTargetStackingExtraTreesRegressor is not fitted")
        base = ensure_2d_prediction(
            self.base_model_.predict(X), n_targets=self.n_outputs_
        )
        stacked = np.empty_like(base)
        for columns, meta in zip(self.groups_, self.meta_models_):
            stacked[:, columns] = meta.predict(base[:, columns])
        weight = float(self.identity_weight)
        return weight * base + (1.0 - weight) * stacked


class RankBlendedTargetExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """Protect squared-error predictions while adding one rank-trained expert.

    The base model is the same multi-output Extra Trees regressor used by the
    current champion.  For one predeclared target, a second Extra Trees model
    is fitted to empirical fractional ranks learned from the current training
    fold only.  Its predictions are mapped back through that fold's empirical
    target quantiles and blended with the base prediction.

    This is intentionally a narrow diagnostic model for a highly skewed target:
    it gives ordinary observations more equal influence without discarding the
    original-scale model that protects RMSE and R2 on the upper tail.
    """

    def __init__(
        self,
        target_index: int,
        rank_weight: float = 0.5,
        n_estimators: int = 100,
        max_features: Any = 1.0,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.target_index = target_index
        self.rank_weight = rank_weight
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _new_tree(self, seed: Optional[int]) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, y: Any) -> "RankBlendedTargetExtraTreesRegressor":
        target = np.asarray(y, dtype=float)
        if target.ndim != 2:
            raise ValueError("Rank-blended Extra Trees requires a 2D target matrix")
        if not 0 <= int(self.target_index) < target.shape[1]:
            raise ValueError("target_index is outside the target matrix")
        if not 0.0 <= float(self.rank_weight) <= 1.0:
            raise ValueError("rank_weight must be between 0 and 1")

        self.n_outputs_ = target.shape[1]
        self.base_model_ = self._new_tree(self.random_state)
        self.base_model_.fit(X, target)

        rank_seed = (
            None if self.random_state is None else int(self.random_state) + 1_000_003
        )
        self.rank_model_ = self._new_tree(rank_seed)
        rank_target = rankdata(
            target[:, int(self.target_index)], method="average"
        ) / (len(target) + 1.0)
        self.rank_model_.fit(X, rank_target)

        self.sorted_target_ = np.sort(target[:, int(self.target_index)])
        self.quantile_grid_ = (np.arange(len(target), dtype=float) + 0.5) / len(
            target
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "base_model_"):
            raise ValueError("RankBlendedTargetExtraTreesRegressor is not fitted")
        prediction = ensure_2d_prediction(
            self.base_model_.predict(X), n_targets=self.n_outputs_
        )
        rank_prediction = np.asarray(self.rank_model_.predict(X), dtype=float)
        rank_prediction = np.clip(rank_prediction, 0.0, 1.0)
        mapped_prediction = np.interp(
            rank_prediction, self.quantile_grid_, self.sorted_target_
        )
        index = int(self.target_index)
        weight = float(self.rank_weight)
        prediction[:, index] = (
            (1.0 - weight) * prediction[:, index]
            + weight * mapped_prediction
        )
        return prediction


class FoldSkewLogTransformer(BaseEstimator, TransformerMixin):
    """Log strongly right-skewed nonnegative columns, fitted within a fold.

    Training columns are selected only when their minimum is nonnegative and
    sample skew exceeds ``skew_threshold``.  A signed log1p is used at transform
    time so an unexpected negative held-out value cannot create NaN or leakage-
    motivated clipping.
    """

    def __init__(self, skew_threshold: float = 2.0):
        self.skew_threshold = skew_threshold

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "FoldSkewLogTransformer":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("FoldSkewLogTransformer expects a 2D matrix")
        self.n_features_in_ = values.shape[1]
        column_skew = np.asarray(
            skew(values, axis=0, bias=False, nan_policy="omit"), dtype=float
        )
        column_min = np.nanmin(values, axis=0)
        self.transform_mask_ = (
            np.isfinite(column_skew)
            & (column_skew >= self.skew_threshold)
            & (column_min >= 0.0)
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "transform_mask_"):
            raise ValueError("FoldSkewLogTransformer is not fitted")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected a 2D matrix with {self.n_features_in_} columns"
            )
        transformed = values.copy()
        selected = transformed[:, self.transform_mask_]
        transformed[:, self.transform_mask_] = np.sign(selected) * np.log1p(
            np.abs(selected)
        )
        return transformed


class CrossFittedAffineCalibratedRegressor(BaseEstimator, RegressorMixin):
    """Conservatively recalibrate continuous predictions without fold leakage.

    Calibration targets are generated by cross-fitting the base estimator only
    on the rows passed to ``fit``.  Per-target affine slopes are shrunk halfway
    toward the identity and bounded, which limits variance inflation in this
    n=250 setting.  The final base estimator is then refit on all available
    training rows; an outer validation row never contributes to its calibration.
    """

    def __init__(
        self,
        base_estimator: RegressorMixin,
        calibration_splits: int = 3,
        identity_weight: float = 0.5,
        slope_min: float = 0.8,
        slope_max: float = 1.5,
        clip_to_training_range: bool = True,
        random_state: int = 0,
    ):
        self.base_estimator = base_estimator
        self.calibration_splits = calibration_splits
        self.identity_weight = identity_weight
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.clip_to_training_range = clip_to_training_range
        self.random_state = random_state

    @staticmethod
    def _take_rows(X: Any, indices: np.ndarray) -> Any:
        if hasattr(X, "iloc"):
            return X.iloc[indices]
        return np.asarray(X)[indices]

    def fit(self, X: Any, y: Any) -> "CrossFittedAffineCalibratedRegressor":
        target = np.asarray(y, dtype=float)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if not 0.0 <= self.identity_weight <= 1.0:
            raise ValueError("identity_weight must be between 0 and 1")
        if self.slope_min <= 0 or self.slope_max < self.slope_min:
            raise ValueError("Require 0 < slope_min <= slope_max")
        if not 2 <= self.calibration_splits <= len(target):
            raise ValueError("calibration_splits must be between 2 and n_samples")

        splitter = KFold(
            n_splits=self.calibration_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof = np.empty_like(target, dtype=float)
        for train_index, valid_index in splitter.split(np.arange(len(target))):
            estimator = clone(self.base_estimator)
            estimator.fit(self._take_rows(X, train_index), target[train_index])
            oof[valid_index] = ensure_2d_prediction(
                estimator.predict(self._take_rows(X, valid_index)),
                n_targets=target.shape[1],
            )

        raw_slopes = np.ones(target.shape[1], dtype=float)
        slopes = np.ones(target.shape[1], dtype=float)
        intercepts = np.zeros(target.shape[1], dtype=float)
        for target_index in range(target.shape[1]):
            prediction = oof[:, target_index]
            truth = target[:, target_index]
            centered_prediction = prediction - prediction.mean()
            centered_truth = truth - truth.mean()
            denominator = float(np.dot(centered_prediction, centered_prediction))
            if denominator > 1e-12:
                raw_slope = float(
                    np.dot(centered_prediction, centered_truth) / denominator
                )
            else:
                raw_slope = 1.0
            shrunk_slope = (
                self.identity_weight
                + (1.0 - self.identity_weight) * raw_slope
            )
            slope = float(np.clip(shrunk_slope, self.slope_min, self.slope_max))
            raw_slopes[target_index] = raw_slope
            slopes[target_index] = slope
            intercepts[target_index] = float(truth.mean() - slope * prediction.mean())

        self.calibration_raw_slopes_ = raw_slopes
        self.calibration_slopes_ = slopes
        self.calibration_intercepts_ = intercepts
        self.target_min_ = np.min(target, axis=0)
        self.target_max_ = np.max(target, axis=0)
        self.n_outputs_ = target.shape[1]
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, target)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "base_estimator_"):
            raise ValueError("CrossFittedAffineCalibratedRegressor is not fitted")
        prediction = ensure_2d_prediction(
            self.base_estimator_.predict(X), n_targets=self.n_outputs_
        )
        calibrated = (
            prediction * self.calibration_slopes_ + self.calibration_intercepts_
        )
        if self.clip_to_training_range:
            calibrated = np.clip(calibrated, self.target_min_, self.target_max_)
        return calibrated


def _linear_pipeline(estimator: RegressorMixin) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _tree_pipeline(estimator: RegressorMixin) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", estimator),
        ]
    )


def _skew_robust_pipeline(estimator: RegressorMixin) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("skew_log", FoldSkewLogTransformer(skew_threshold=2.0)),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            ("model", estimator),
        ]
    )


def build_model(
    name: str, params: Dict[str, Any], seed: int, n_jobs: int
) -> RegressorMixin:
    """Build a fresh model; learned transforms are always inside its pipeline."""
    if name == "ridge":
        return _linear_pipeline(Ridge(**params))
    if name == "ridge_skew_robust":
        return _skew_robust_pipeline(Ridge(**params))
    if name == "elastic_net":
        model = MultiTaskElasticNet(
            random_state=seed, max_iter=10000, selection="cyclic", **params
        )
        return _linear_pipeline(model)
    if name == "pls2":
        return _linear_pipeline(PLSRegression(scale=False, max_iter=1000, **params))
    if name == "pls2_skew_robust":
        return _skew_robust_pipeline(
            PLSRegression(scale=False, max_iter=1000, **params)
        )
    if name == "pls1":
        return _linear_pipeline(TargetwisePLSRegressor(max_iter=1000, **params))
    if name == "pls1_skew_robust":
        return _skew_robust_pipeline(
            TargetwisePLSRegressor(max_iter=1000, **params)
        )
    if name == "random_forest":
        model = RandomForestRegressor(
            random_state=seed, n_jobs=n_jobs, bootstrap=True, **params
        )
        return _tree_pipeline(model)
    if name == "extra_trees":
        tree_params = dict(params)
        tree_seed = int(tree_params.pop("random_state_override", seed))
        model = ExtraTreesRegressor(
            random_state=tree_seed, n_jobs=n_jobs, **tree_params
        )
        return _tree_pipeline(model)
    if name == "extra_trees_affine_calibrated":
        calibration_params = dict(params)
        calibration_splits = int(calibration_params.pop("calibration_splits", 3))
        identity_weight = float(calibration_params.pop("identity_weight", 0.5))
        slope_min = float(calibration_params.pop("slope_min", 0.8))
        slope_max = float(calibration_params.pop("slope_max", 1.5))
        clip_to_training_range = bool(
            calibration_params.pop("clip_to_training_range", True)
        )
        tree_seed = int(calibration_params.pop("random_state_override", seed))
        base = _tree_pipeline(
            ExtraTreesRegressor(
                random_state=tree_seed, n_jobs=n_jobs, **calibration_params
            )
        )
        return CrossFittedAffineCalibratedRegressor(
            base_estimator=base,
            calibration_splits=calibration_splits,
            identity_weight=identity_weight,
            slope_min=slope_min,
            slope_max=slope_max,
            clip_to_training_range=clip_to_training_range,
            random_state=seed,
        )
    if name == "extra_trees_targetwise":
        base = ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **params)
        return _tree_pipeline(MultiOutputRegressor(base, n_jobs=n_jobs))
    if name == "extra_trees_targetwise_subset":
        subset_params = dict(params)
        feature_indices = list(subset_params.pop("feature_indices"))
        base = FeatureSubsetTargetwiseExtraTreesRegressor(
            feature_indices=feature_indices,
            random_state=seed,
            n_jobs=n_jobs,
            **subset_params,
        )
        return _tree_pipeline(base)
    if name == "extra_trees_targetwise_group_summaries":
        summary_params = dict(params)
        groups = [list(group) for group in summary_params.pop("groups")]
        base = ExtraTreesRegressor(
            random_state=seed, n_jobs=n_jobs, **summary_params
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("group_summaries", FoldStandardizedGroupSummaryTransformer(groups)),
                ("model", MultiOutputRegressor(base, n_jobs=n_jobs)),
            ]
        )
    if name == "threshold_mechanism_ridge":
        mechanism_params = dict(params)
        feature_indices = list(mechanism_params.pop("feature_indices"))
        thresholds = list(mechanism_params.pop("thresholds"))
        matrix_indices = list(mechanism_params.pop("matrix_indices", []))
        alpha = float(mechanism_params.pop("alpha", 10.0))
        if mechanism_params:
            raise ValueError(
                f"Unknown threshold_mechanism_ridge parameters: {mechanism_params}"
            )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "mechanism",
                    ThresholdMechanismTransformer(
                        feature_indices=feature_indices,
                        thresholds=thresholds,
                        matrix_indices=matrix_indices,
                        append_original=False,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
    if name == "extra_trees_targetwise_threshold_mechanism":
        mechanism_params = dict(params)
        feature_indices = list(mechanism_params.pop("feature_indices"))
        thresholds = list(mechanism_params.pop("thresholds"))
        matrix_indices = list(mechanism_params.pop("matrix_indices", []))
        base = ExtraTreesRegressor(
            random_state=seed, n_jobs=n_jobs, **mechanism_params
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "mechanism",
                    ThresholdMechanismTransformer(
                        feature_indices=feature_indices,
                        thresholds=thresholds,
                        matrix_indices=matrix_indices,
                        append_original=True,
                    ),
                ),
                ("model", MultiOutputRegressor(base, n_jobs=n_jobs)),
            ]
        )
    if name == "extra_trees_threshold_mechanism":
        mechanism_params = dict(params)
        feature_indices = list(mechanism_params.pop("feature_indices"))
        thresholds = list(mechanism_params.pop("thresholds"))
        matrix_indices = list(mechanism_params.pop("matrix_indices", []))
        base = ExtraTreesRegressor(
            random_state=seed, n_jobs=n_jobs, **mechanism_params
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "mechanism",
                    ThresholdMechanismTransformer(
                        feature_indices=feature_indices,
                        thresholds=thresholds,
                        matrix_indices=matrix_indices,
                        append_original=True,
                    ),
                ),
                ("model", base),
            ]
        )
    if name == "threshold_mechanism_residual_extra_trees":
        residual_params = dict(params)
        feature_indices = list(residual_params.pop("feature_indices"))
        thresholds = list(residual_params.pop("thresholds"))
        matrix_indices = list(residual_params.pop("matrix_indices", []))
        ridge_alpha = float(residual_params.pop("ridge_alpha", 10.0))
        residual_splits = int(residual_params.pop("residual_splits", 3))
        base = CrossFittedMechanismResidualExtraTreesRegressor(
            feature_indices=feature_indices,
            thresholds=thresholds,
            matrix_indices=matrix_indices,
            ridge_alpha=ridge_alpha,
            residual_splits=residual_splits,
            random_state=seed,
            n_jobs=n_jobs,
            **residual_params,
        )
        return _tree_pipeline(base)
    if name == "extra_trees_paired":
        paired_params = dict(params)
        paired_seed = int(paired_params.pop("random_state_override", seed))
        return _tree_pipeline(
            AromaFlavourPairedExtraTreesRegressor(
                random_state=paired_seed,
                n_jobs=n_jobs,
                **paired_params,
            )
        )
    if name == "extra_trees_family_grouped":
        grouped_params = dict(params)
        grouped_seed = int(grouped_params.pop("random_state_override", seed))
        return _tree_pipeline(
            SensoryFamilyExtraTreesRegressor(
                random_state=grouped_seed,
                n_jobs=n_jobs,
                **grouped_params,
            )
        )
    if name == "extra_trees_target_stacked":
        stacking_params = dict(params)
        stacking_seed = int(stacking_params.pop("random_state_override", seed))
        return _tree_pipeline(
            CrossFittedTargetStackingExtraTreesRegressor(
                random_state=stacking_seed,
                n_jobs=n_jobs,
                **stacking_params,
            )
        )
    if name == "extra_trees_rank_blended":
        rank_params = dict(params)
        tree_seed = int(rank_params.pop("random_state_override", seed))
        return _tree_pipeline(
            RankBlendedTargetExtraTreesRegressor(
                random_state=tree_seed,
                n_jobs=n_jobs,
                **rank_params,
            )
        )
    if name == "extra_trees_targetwise_selectk":
        selector_params = dict(params)
        k = int(selector_params.pop("k"))
        base = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("selector", SelectKBest(score_func=f_regression, k=k)),
                (
                    "model",
                    ExtraTreesRegressor(
                        random_state=seed, n_jobs=n_jobs, **selector_params
                    ),
                ),
            ]
        )
        # Each target gets its own fold-local feature ranking.  The selector is
        # therefore never fitted on another target or on held-out samples.
        return MultiOutputRegressor(base, n_jobs=n_jobs)
    if name == "extra_trees_targetwise_selectk_mi":
        selector_params = dict(params)
        k = int(selector_params.pop("k"))
        # Mutual information can detect non-linear one-feature associations.
        # It is fitted separately for every target inside the training fold;
        # fixing its seed makes the nearest-neighbour estimate reproducible.
        mi_score = partial(mutual_info_regression, random_state=seed)
        base = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("selector", SelectKBest(score_func=mi_score, k=k)),
                (
                    "model",
                    ExtraTreesRegressor(
                        random_state=seed, n_jobs=n_jobs, **selector_params
                    ),
                ),
            ]
        )
        return MultiOutputRegressor(base, n_jobs=n_jobs)
    if name == "gradient_boosting":
        base = GradientBoostingRegressor(random_state=seed, loss="squared_error", **params)
        return _tree_pipeline(MultiOutputRegressor(base, n_jobs=n_jobs))
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost model requested; install it with "
                "`python -m pip install xgboost==1.7.3`"
            ) from exc
        xgb_params = dict(params)
        xgb_params.setdefault("tree_method", "hist")
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
            **xgb_params,
        )
        return _tree_pipeline(MultiOutputRegressor(base, n_jobs=n_jobs))
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost model requested; install it with "
                "`python -m pip install catboost==1.2.5`"
            ) from exc
        base = CatBoostRegressor(
            random_seed=seed, verbose=False, allow_writing_files=False, **params
        )
        return _tree_pipeline(MultiOutputRegressor(base, n_jobs=n_jobs))
    raise KeyError(f"Unknown model: {name}")


def ensure_2d_prediction(prediction: np.ndarray, n_targets: int) -> np.ndarray:
    result = np.asarray(prediction)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.shape[1] != n_targets:
        raise ValueError(
            f"Prediction has {result.shape[1]} targets, expected {n_targets}"
        )
    return result


def aggregate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_target = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return float(np.mean(per_target))


def tune_model_one_se(
    name: str,
    candidates: Sequence[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.DataFrame,
    styles: pd.Series,
    inner_splits: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Nested inner-CV tuning with a one-standard-error simplicity rule.

    Candidate order defines simplicity: place the simplest/most regularized
    configuration first in YAML. The aggregate score is the unweighted mean of
    target-wise RMSE; targets are standardized and therefore comparable.
    """
    if not candidates:
        raise ValueError(f"No parameter candidates configured for {name}")
    inner_fold = style_balanced_fold_ids(styles, n_splits=inner_splits, seed=seed)
    rows: List[Dict[str, Any]] = []
    for candidate_index, params in enumerate(candidates):
        for fold in range(inner_splits):
            train_mask = inner_fold != fold
            valid_mask = inner_fold == fold
            model = build_model(name, dict(params), seed + candidate_index * 97 + fold, n_jobs)
            model.fit(X.loc[train_mask], y.loc[train_mask])
            prediction = ensure_2d_prediction(
                model.predict(X.loc[valid_mask]), n_targets=y.shape[1]
            )
            target_rmse = np.sqrt(
                np.mean((y.loc[valid_mask].to_numpy() - prediction) ** 2, axis=0)
            )
            score = float(np.mean(target_rmse))
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "params_json": json.dumps(params, sort_keys=True),
                    "inner_fold": fold,
                    "score_scope": "aggregate",
                    "target": "__all__",
                    "rmse": score,
                }
            )
            for target, target_score in zip(y.columns, target_rmse):
                rows.append(
                    {
                        "candidate_index": candidate_index,
                        "params_json": json.dumps(params, sort_keys=True),
                        "inner_fold": fold,
                        "score_scope": "target",
                        "target": target,
                        "rmse": float(target_score),
                    }
                )
    trials = pd.DataFrame(rows)
    aggregate_trials = trials[trials["score_scope"] == "aggregate"]
    summary = (
        aggregate_trials.groupby(["candidate_index", "params_json"], as_index=False)["rmse"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    best_row = summary.loc[summary["mean"].idxmin()]
    threshold = float(best_row["mean"] + best_row["se"])
    eligible = summary[summary["mean"] <= threshold].sort_values("candidate_index")
    chosen = eligible.iloc[0]
    chosen_index = int(chosen["candidate_index"])
    trials["selected"] = trials["candidate_index"] == chosen_index
    trials["selection_threshold"] = threshold
    trials["selected_mean_rmse"] = float(chosen["mean"])
    trials["selected_se"] = float(chosen["se"])
    return dict(candidates[chosen_index]), trials


def predict_global_mean(y_train: pd.DataFrame, n_valid: int) -> np.ndarray:
    return np.tile(y_train.mean(axis=0).to_numpy(), (n_valid, 1))


def predict_style_mean(
    y_train: pd.DataFrame,
    styles_train: pd.Series,
    styles_valid: pd.Series,
) -> np.ndarray:
    """Style means use training rows only; unseen styles fall back globally."""
    global_mean = y_train.mean(axis=0).to_numpy()
    lookup = y_train.assign(__style=styles_train.to_numpy()).groupby("__style").mean()
    predictions = []
    for style in styles_valid:
        if style in lookup.index:
            predictions.append(lookup.loc[style].to_numpy())
        else:
            predictions.append(global_mean)
    return np.vstack(predictions)

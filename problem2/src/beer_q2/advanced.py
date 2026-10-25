"""TreeSHAP, community permutation, and style heterogeneity for Q2.

The implementation explains the true representative Q1 pipelines. Missing
indicator SHAP values are folded back into their source feature. Positive
affine post-processing scales SHAP values; the 50/50 blend combines component
SHAP values because both components predict the same standardized target.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
import yaml

from beer_q1.data import load_problem1_data

from .explain import _permutation_groups, regression_metrics, safe_spearman
from .reconstruct import (
    _align_frozen_rows,
    _problem1_config,
    fit_representative_target_predictor,
)


def aggregate_indicator_shap(
    transformed_shap: np.ndarray,
    n_original_features: int,
    indicator_features: Sequence[int],
) -> np.ndarray:
    """Fold imputer missing-indicator contributions into source features."""
    values = np.asarray(transformed_shap, dtype=float)
    expected_width = n_original_features + len(indicator_features)
    if values.ndim not in {2, 3} or values.shape[1] != expected_width:
        raise ValueError(
            f"Expected SHAP width {expected_width}, got {values.shape}"
        )
    result = values[:, :n_original_features, ...].copy()
    for offset, feature_index in enumerate(indicator_features):
        result[:, int(feature_index), ...] += values[
            :, n_original_features + offset, ...
        ]
    return result


def grouped_permutation_source_indices(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return row sources that preserve joint feature vectors within groups."""
    source = np.arange(len(groups), dtype=int)
    for group in pd.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) >= 2:
            source[positions] = rng.permutation(positions)
    return source


def _extract_tree_shap(
    pipeline,
    X: pd.DataFrame,
    output_index: int,
    cache: Optional[
        MutableMapping[Tuple[object, ...], Tuple[np.ndarray, np.ndarray]]
    ] = None,
) -> Tuple[np.ndarray, float]:
    # SelectK specialists wrap a complete one-target Pipeline in an outer
    # MultiOutputRegressor. Other tree sources wrap the estimator inside a
    # common imputer Pipeline.
    if pipeline.__class__.__name__ == "MultiOutputRegressor":
        pipeline = pipeline.estimators_[output_index]
        output_index = 0
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "estimators_") and estimator.__class__.__name__ == "MultiOutputRegressor":
        tree = estimator.estimators_[output_index]
        local_output_index = 0
        cache_key = ("single_output_tree", id(pipeline), int(output_index))
    else:
        tree = estimator
        local_output_index = output_index
        cache_key = ("all_output_tree", id(pipeline))
    value_cache = cache if cache is not None else {}
    if cache_key not in value_cache:
        imputer = pipeline.named_steps["imputer"]
        transformed = imputer.transform(X)
        selector = pipeline.named_steps.get("selector")
        tree_input = (
            selector.transform(transformed) if selector is not None else transformed
        )
        explainer = shap.TreeExplainer(tree)
        values = np.asarray(
            explainer.shap_values(tree_input, check_additivity=True), dtype=float
        )
        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if values.ndim == 2:
            values = values[:, :, None]
        elif values.ndim != 3:
            raise ValueError(f"Unexpected TreeSHAP shape: {values.shape}")
        if selector is not None:
            restored = np.zeros(
                (len(X), transformed.shape[1], values.shape[2]), dtype=float
            )
            restored[:, selector.get_support(indices=True), :] = values
            values = restored
        indicator = getattr(imputer, "indicator_", None)
        indicator_features = [] if indicator is None else list(indicator.features_)
        aggregated = aggregate_indicator_shap(
            values,
            n_original_features=X.shape[1],
            indicator_features=indicator_features,
        )
        if len(expected) == 1 and aggregated.shape[2] > 1:
            expected = np.repeat(expected, aggregated.shape[2])
        if len(expected) != aggregated.shape[2]:
            raise ValueError(
                "TreeSHAP expected-value width does not match output width"
            )
        value_cache[cache_key] = (aggregated, expected)
    all_values, expected = value_cache[cache_key]
    if local_output_index >= all_values.shape[2]:
        raise ValueError("Requested TreeSHAP output is outside the fitted model")
    return (
        all_values[:, :, local_output_index],
        float(expected[local_output_index]),
    )


def true_pipeline_shap(
    predictor,
    X: pd.DataFrame,
    shap_cache: Optional[
        MutableMapping[Tuple[object, ...], Tuple[np.ndarray, np.ndarray]]
    ] = None,
) -> Tuple[np.ndarray, float, float]:
    """Return original-feature SHAP, expected value, and additivity error."""
    if predictor.source_model == "affine_calibration":
        calibrated = predictor.primary_model
        base_values, base_expected = _extract_tree_shap(
            calibrated.base_estimator_, X, predictor.target_index, cache=shap_cache
        )
        slope = float(calibrated.calibration_slopes_[predictor.target_index])
        intercept = float(calibrated.calibration_intercepts_[predictor.target_index])
        values = base_values * slope
        expected = base_expected * slope + intercept
    elif predictor.source_model == "half_multioutput_half_targetwise_et":
        primary_values, primary_expected = _extract_tree_shap(
            predictor.primary_model, X, predictor.target_index, cache=shap_cache
        )
        secondary_values, secondary_expected = _extract_tree_shap(
            predictor.secondary_model, X, 0, cache=shap_cache
        )
        values = 0.5 * primary_values + 0.5 * secondary_values
        expected = 0.5 * primary_expected + 0.5 * secondary_expected
    elif predictor.source_model == "fixed_scale_1.25":
        values, expected = _extract_tree_shap(
            predictor.primary_model, X, predictor.target_index, cache=shap_cache
        )
        values = values * predictor.prediction_scale
        expected = expected * predictor.prediction_scale
    else:
        values, expected = _extract_tree_shap(
            predictor.primary_model, X, predictor.target_index, cache=shap_cache
        )
    prediction = predictor.predict(X)
    additivity_error = float(np.max(np.abs(expected + values.sum(axis=1) - prediction)))
    return values, float(expected), additivity_error


def _group_permutation(
    predictor,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    styles_valid: np.ndarray,
    communities: pd.DataFrame,
    repeat: int,
    fold: int,
    target: str,
    permutations: int,
    seed: int,
    style_minimum_group_n: int,
) -> pd.DataFrame:
    baseline_prediction = predictor.predict(X_valid)
    baseline = regression_metrics(y_valid, baseline_prediction)
    style_groups = _permutation_groups(
        styles_valid, "within_style", style_minimum_group_n
    )
    rows: List[Dict[str, object]] = []
    for community_id, group in communities.groupby("community"):
        features = [feature for feature in group["feature"] if feature in X_valid.columns]
        if not features:
            continue
        blocks = []
        for permutation in range(permutations):
            rng = np.random.default_rng(
                int(seed)
                + int(repeat) * 1_000_003
                + int(fold) * 100_003
                + int(community_id) * 101
                + int(permutation)
            )
            source = grouped_permutation_source_indices(style_groups, rng)
            block = X_valid.copy()
            block.loc[:, features] = X_valid.iloc[source][features].to_numpy()
            blocks.append(block)
        batch = pd.concat(blocks, axis=0, ignore_index=True)
        predictions = predictor.predict(batch).reshape(permutations, len(X_valid))
        for permutation, prediction in enumerate(predictions):
            permuted = regression_metrics(y_valid, prediction)
            rows.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "target": target,
                    "source_model": predictor.source_model,
                    "community": int(community_id),
                    "community_size": int(len(features)),
                    "community_features": "|".join(features),
                    "permutation": int(permutation),
                    "baseline_rmse": baseline["rmse"],
                    "permuted_rmse": permuted["rmse"],
                    "delta_rmse": permuted["rmse"] - baseline["rmse"],
                    "relative_delta_rmse": (
                        (permuted["rmse"] - baseline["rmse"]) / baseline["rmse"]
                        if baseline["rmse"] > 0
                        else np.nan
                    ),
                    "delta_r2_loss": baseline["r2"] - permuted["r2"],
                    "delta_spearman_loss": baseline["spearman"]
                    - permuted["spearman"],
                }
            )
    return pd.DataFrame(rows)


def _shap_stability(shap_values: pd.DataFrame, top_k: int) -> pd.DataFrame:
    fold_level = (
        shap_values.groupby(
            ["target", "grade", "source_model", "feature", "repeat", "fold"],
            as_index=False,
        )
        .agg(
            mean_abs_shap=("abs_shap_value", "mean"),
            direction_rho=("feature_value", lambda values: np.nan),
        )
    )
    # Calculate feature-vs-SHAP direction without relying on groupby side effects.
    directions = (
        shap_values.groupby(
            ["target", "feature", "repeat", "fold"], as_index=False
        )
        .apply(
            lambda group: pd.Series(
                {
                    "direction_rho": safe_spearman(
                        group["feature_value"], group["shap_value"]
                    )
                }
            )
        )
        .reset_index(drop=True)
    )
    fold_level = fold_level.drop(columns="direction_rho").merge(
        directions, on=["target", "feature", "repeat", "fold"], validate="one_to_one"
    )
    fold_level["shap_rank"] = fold_level.groupby(
        ["target", "repeat", "fold"]
    )["mean_abs_shap"].rank(method="min", ascending=False)

    rows = []
    for keys, group in fold_level.groupby(
        ["target", "grade", "source_model", "feature"], sort=False
    ):
        target, grade, source_model, feature = keys
        signs = np.sign(group["direction_rho"].dropna().to_numpy())
        signs = signs[signs != 0]
        positive_share = float((signs > 0).mean()) if len(signs) else np.nan
        finite_direction = group["direction_rho"].dropna()
        rows.append(
            {
                "target": target,
                "grade": grade,
                "source_model": source_model,
                "feature": feature,
                "mean_abs_shap": float(group["mean_abs_shap"].mean()),
                "median_abs_shap": float(group["mean_abs_shap"].median()),
                "median_shap_rank": float(group["shap_rank"].median()),
                f"top{top_k}_shap_frequency": float(
                    (group["shap_rank"] <= top_k).mean()
                ),
                "median_shap_direction_rho": (
                    float(finite_direction.median())
                    if len(finite_direction)
                    else np.nan
                ),
                "shap_direction_consistency": (
                    max(positive_share, 1.0 - positive_share)
                    if np.isfinite(positive_share)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "mean_abs_shap"], ascending=[True, False]
    )


def _method_agreement(
    shap_values: pd.DataFrame,
    raw_permutation: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    shap_fold = (
        shap_values.groupby(["target", "repeat", "fold", "feature"], as_index=False)[
            "abs_shap_value"
        ]
        .mean()
    )
    selected_permutation = raw_permutation[
        raw_permutation["mode"] == "within_style"
    ]
    if "mean_delta_rmse" in selected_permutation.columns:
        permutation_fold = selected_permutation[
            ["target", "repeat", "fold", "feature", "mean_delta_rmse"]
        ].rename(columns={"mean_delta_rmse": "delta_rmse"})
    else:
        permutation_fold = (
            selected_permutation.groupby(
                ["target", "repeat", "fold", "feature"], as_index=False
            )["delta_rmse"]
            .mean()
        )
    rows = []
    keys = shap_fold[["target", "repeat", "fold"]].drop_duplicates()
    for target, repeat, fold in keys.itertuples(index=False, name=None):
        shap_top = set(
            shap_fold[
                (shap_fold["target"] == target)
                & (shap_fold["repeat"] == repeat)
                & (shap_fold["fold"] == fold)
            ]
            .nlargest(top_k, "abs_shap_value")["feature"]
        )
        permutation_top = set(
            permutation_fold[
                (permutation_fold["target"] == target)
                & (permutation_fold["repeat"] == repeat)
                & (permutation_fold["fold"] == fold)
            ]
            .nlargest(top_k, "delta_rmse")["feature"]
        )
        union = shap_top | permutation_top
        rows.append(
            {
                "target": target,
                "repeat": int(repeat),
                "fold": int(fold),
                f"permutation_shap_top{top_k}_jaccard": (
                    float(len(shap_top & permutation_top) / len(union)) if union else 1.0
                ),
                "shared_features": "|".join(sorted(shap_top & permutation_top)),
            }
        )
    return pd.DataFrame(rows)


def _style_heterogeneity(
    shap_values: pd.DataFrame,
    shap_stability: pd.DataFrame,
    permutation_stability: pd.DataFrame,
    top_k: int,
    minimum_n: int,
    min_abs_global_rho: float,
    min_abs_style_rho: float,
) -> pd.DataFrame:
    rows = []
    for target in shap_values["target"].unique():
        shap_top = set(
            shap_stability[shap_stability["target"] == target]
            .nlargest(top_k, "mean_abs_shap")["feature"]
        )
        permutation_top = set(
            permutation_stability[
                (permutation_stability["target"] == target)
                & (permutation_stability["mode"] == "within_style")
            ]
            .nlargest(top_k, "mean_delta_rmse")["feature"]
        )
        for feature in sorted(shap_top | permutation_top):
            feature_rows = shap_values[
                (shap_values["target"] == target)
                & (shap_values["feature"] == feature)
            ]
            global_rho = safe_spearman(
                feature_rows["feature_value"], feature_rows["shap_value"]
            )
            for style, style_rows in feature_rows.groupby("style"):
                sample_n = (
                    int(style_rows["beer"].nunique())
                    if "beer" in style_rows.columns
                    else int(len(style_rows))
                )
                if sample_n < minimum_n:
                    continue
                style_rho = safe_spearman(
                    style_rows["feature_value"], style_rows["shap_value"]
                )
                reversal = bool(
                    np.isfinite(global_rho)
                    and np.isfinite(style_rho)
                    and abs(global_rho) >= min_abs_global_rho
                    and abs(style_rho) >= min_abs_style_rho
                    and np.sign(global_rho) != np.sign(style_rho)
                )
                rows.append(
                    {
                        "target": target,
                        "feature": feature,
                        "style": style,
                        "n": sample_n,
                        "global_shap_direction_rho": float(global_rho),
                        "style_shap_direction_rho": float(style_rho),
                        "direction_reversal": reversal,
                    }
                )
    return pd.DataFrame(rows)


def run_advanced_pilot(project_root: Path, config_path: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = (root / config["base_run"]).resolve()
    output = (root / config["output_root"]).resolve()
    if (root / "problem2").resolve() not in output.parents:
        raise ValueError("Question 2 output must stay under problem2")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    data = load_problem1_data(_problem1_config(root))
    folds = pd.read_csv(root / "problem1/freezes/problem1_v1/base_run_metadata/folds.csv")
    tuning = pd.read_csv(
        root / "problem1/freezes/problem1_v1/base_run_metadata/tuning_trials.csv"
    )
    frozen = pd.read_parquet(
        root / "problem1/freezes/problem1_v1/champion_artifacts/oof_predictions.parquet"
    )
    registry = pd.read_csv(root / "problem1/freezes/problem1_v1/source_registry.csv").set_index(
        "target"
    )
    configured_targets = config["targets"]
    targets = (
        list(map(str, data.main_target_names))
        if configured_targets == "all_main"
        else list(map(str, configured_targets))
    )
    network_nodes_path = config.get("network_nodes_path")
    if network_nodes_path:
        candidate_path = Path(str(network_nodes_path))
        if not candidate_path.is_absolute():
            candidate_path = root / candidate_path
        network_nodes = pd.read_csv(candidate_path)
    else:
        network_nodes = pd.read_csv(base / "network_nodes.csv")
    network_variant = config.get("network_variant")
    if network_variant is not None:
        if "variant" not in network_nodes.columns:
            raise ValueError("network_variant requested but node table has no variant column")
        network_nodes = network_nodes[
            network_nodes["variant"] == str(network_variant)
        ].copy()
    fold_level_path = base / "importance_fold_level.parquet"
    raw_permutation = pd.read_parquet(
        fold_level_path
        if fold_level_path.exists()
        else base / "oof_permutation_importance.parquet"
    )
    permutation_stability = pd.read_csv(base / "importance_stability.csv")

    shap_rows: List[pd.DataFrame] = []
    group_rows: List[pd.DataFrame] = []
    audits: List[Dict[str, object]] = []
    for repeat in map(int, config["repeats"]):
        for fold in map(int, config["outer_folds"]):
            fold_model_cache = {}
            communities = network_nodes[
                (network_nodes["repeat"] == repeat) & (network_nodes["fold"] == fold)
            ][["feature", "community"]]
            if len(communities) != len(data.feature_names):
                raise ValueError(
                    f"Expected {len(data.feature_names)} network nodes for "
                    f"repeat={repeat}, fold={fold}; got {len(communities)}"
                )
            for target in targets:
                predictor, _, valid_index = fit_representative_target_predictor(
                    data,
                    folds,
                    tuning,
                    target,
                    repeat,
                    fold,
                    model_cache=fold_model_cache,
                )
                X_valid = data.X.iloc[valid_index].copy()
                y_valid = data.y.iloc[valid_index][target].to_numpy(dtype=float)
                beers = data.ids.iloc[valid_index].to_numpy(dtype=int)
                styles = data.styles.iloc[valid_index].astype(str).to_numpy()
                frozen_rows = _align_frozen_rows(frozen, target, repeat, fold)
                reconstruction_error = float(
                    np.max(
                        np.abs(
                            predictor.predict(X_valid)
                            - frozen_rows["y_pred"].to_numpy(dtype=float)
                        )
                    )
                )
                if reconstruction_error > float(config["reconstruction_tolerance"]):
                    raise ValueError("Frozen prediction reconstruction failed")
                values, expected, additivity_error = true_pipeline_shap(
                    predictor, X_valid
                )
                if additivity_error > 1e-8:
                    raise ValueError(
                        f"TreeSHAP additivity failed for {target} r{repeat}f{fold}: {additivity_error}"
                    )
                long = pd.DataFrame(
                    {
                        "repeat": np.repeat(repeat, len(X_valid) * len(data.feature_names)),
                        "fold": np.repeat(fold, len(X_valid) * len(data.feature_names)),
                        "target": np.repeat(target, len(X_valid) * len(data.feature_names)),
                        "grade": np.repeat(
                            str(registry.loc[target, "grade"]),
                            len(X_valid) * len(data.feature_names),
                        ),
                        "source_model": np.repeat(
                            predictor.source_model,
                            len(X_valid) * len(data.feature_names),
                        ),
                        "beer": np.repeat(beers, len(data.feature_names)),
                        "style": np.repeat(styles, len(data.feature_names)),
                        "feature": np.tile(data.feature_names, len(X_valid)),
                        "feature_value": X_valid.to_numpy().reshape(-1),
                        "shap_value": values.reshape(-1),
                        "abs_shap_value": np.abs(values).reshape(-1),
                    }
                )
                shap_rows.append(long)
                group_rows.append(
                    _group_permutation(
                        predictor,
                        X_valid,
                        y_valid,
                        styles,
                        communities,
                        repeat,
                        fold,
                        target,
                        int(config["group_permutations"]),
                        int(config["group_permutation_seed"]),
                        int(config["style_minimum_group_n"]),
                    )
                )
                audits.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "target": target,
                        "source_model": predictor.source_model,
                        "expected_value": expected,
                        "shap_additivity_max_abs_error": additivity_error,
                        "reconstruction_max_abs_error": reconstruction_error,
                    }
                )

    shap_values = pd.concat(shap_rows, ignore_index=True)
    group_importance = pd.concat(group_rows, ignore_index=True)
    shap_stability = _shap_stability(shap_values, int(config["top_k"]))
    agreement = _method_agreement(
        shap_values, raw_permutation, int(config["top_k"])
    )
    style = _style_heterogeneity(
        shap_values,
        shap_stability,
        permutation_stability,
        int(config["top_k"]),
        int(config["style_report_minimum_n"]),
        float(config["direction_reversal_min_abs_global_rho"]),
        float(config["direction_reversal_min_abs_style_rho"]),
    )
    group_summary = (
        group_importance.groupby(
            ["target", "repeat", "fold", "community", "community_size", "community_features"],
            as_index=False,
        )
        .agg(
            mean_delta_rmse=("delta_rmse", "mean"),
            positive_frequency=("delta_rmse", lambda values: float((values > 0).mean())),
            mean_relative_delta_rmse=("relative_delta_rmse", "mean"),
        )
    )

    shap_values.to_parquet(output / "oof_shap_values.parquet", index=False)
    shap_stability.to_csv(output / "shap_stability.csv", index=False)
    group_importance.to_parquet(output / "oof_group_permutation_importance.parquet", index=False)
    group_summary.to_csv(output / "group_importance_summary.csv", index=False)
    agreement.to_csv(output / "permutation_shap_agreement.csv", index=False)
    style.to_csv(output / "style_direction_heterogeneity.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "shap_reconstruction_audit.csv", index=False)

    top_k = int(config["top_k"])
    result = {
        "status": "PASS",
        "targets": targets,
        "n_shap_rows": int(len(shap_values)),
        "n_group_permutation_rows": int(len(group_importance)),
        "max_shap_additivity_error": float(
            pd.DataFrame(audits)["shap_additivity_max_abs_error"].max()
        ),
        f"median_permutation_shap_top{top_k}_jaccard": float(
            agreement[f"permutation_shap_top{top_k}_jaccard"].median()
        ),
        "direction_reversal_count": int(style["direction_reversal"].sum()),
        "formal_top3_generated": False,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result

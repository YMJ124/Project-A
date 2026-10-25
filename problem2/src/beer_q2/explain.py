"""Held-out permutation explanations for representative frozen Q1 models.

The main permutation is conditional on validation-fold style groups.  Small
style cells are pooled according to a predeclared rule; unconditional
permutation is retained as a sensitivity analysis.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error, r2_score

from .reconstruct import (
    _align_frozen_rows,
    fit_representative_target_predictor,
)


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman rho without emitting warnings for constant/empty inputs."""
    frame = pd.DataFrame(
        {
            "left": np.asarray(left, dtype=float),
            "right": np.asarray(right, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return float("nan")
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"], method="spearman"))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "spearman": safe_spearman(truth, prediction),
    }


def batched_regression_metrics(
    y_true: np.ndarray, predictions: np.ndarray
) -> Dict[str, np.ndarray]:
    """Vectorized metrics for rows of permutation predictions."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    values = np.asarray(predictions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(truth):
        raise ValueError("Prediction width must match y_true")
    residual = values - truth.reshape(1, -1)
    rmse = np.sqrt(np.mean(residual * residual, axis=1))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = (
        1.0 - np.sum(residual * residual, axis=1) / denominator
        if denominator > 0
        else np.full(values.shape[0], np.nan)
    )
    truth_rank = rankdata(truth, method="average")
    truth_centered = truth_rank - truth_rank.mean()
    truth_norm = float(np.sqrt(np.sum(truth_centered * truth_centered)))
    prediction_rank = rankdata(values, method="average", axis=1)
    prediction_centered = prediction_rank - prediction_rank.mean(axis=1, keepdims=True)
    prediction_norm = np.sqrt(np.sum(prediction_centered * prediction_centered, axis=1))
    denominator_rho = prediction_norm * truth_norm
    numerator_rho = prediction_centered @ truth_centered
    spearman = np.divide(
        numerator_rho,
        denominator_rho,
        out=np.full(values.shape[0], np.nan, dtype=float),
        where=denominator_rho > 0,
    )
    return {"rmse": rmse, "r2": np.asarray(r2), "spearman": spearman}


def _permutation_groups(
    styles: Sequence[str], mode: str, minimum_group_n: int
) -> np.ndarray:
    style_array = np.asarray(styles, dtype=object)
    if mode == "unconditional":
        return np.repeat("__all__", len(style_array)).astype(object)
    if mode != "within_style":
        raise ValueError(f"Unknown permutation mode: {mode}")
    counts = pd.Series(style_array).value_counts()
    return np.asarray(
        [style if int(counts[style]) >= minimum_group_n else "__pooled_rare__" for style in style_array],
        dtype=object,
    )


def permute_values_within_groups(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = np.asarray(values).copy()
    for group in pd.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) >= 2:
            result[positions] = result[rng.permutation(positions)]
    return result


def explain_fold_target(
    data,
    fold_table: pd.DataFrame,
    tuning: pd.DataFrame,
    frozen_oof: pd.DataFrame,
    target: str,
    grade: str,
    repeat: int,
    fold: int,
    modes: Iterable[str],
    permutations: int,
    permutation_seed: int,
    style_minimum_group_n: int,
    tolerance: float,
    model_cache=None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Fit one true target pipeline and explain only its outer validation fold."""
    predictor, train_index, valid_index = fit_representative_target_predictor(
        data,
        fold_table,
        tuning,
        target,
        repeat,
        fold,
        model_cache=model_cache,
    )
    X_valid = data.X.iloc[valid_index].copy()
    y_valid = data.y.iloc[valid_index][target].to_numpy(dtype=float)
    styles_valid = data.styles.iloc[valid_index].astype(str).to_numpy()
    beers_valid = data.ids.iloc[valid_index].to_numpy(dtype=int)
    baseline_prediction = predictor.predict(X_valid)

    frozen_rows = _align_frozen_rows(frozen_oof, target, repeat, fold)
    if not np.array_equal(beers_valid, frozen_rows["beer"].to_numpy(dtype=int)):
        raise ValueError(f"Frozen OOF beer order mismatch for {target}, r{repeat}f{fold}")
    reconstruction_error = baseline_prediction - frozen_rows["y_pred"].to_numpy(dtype=float)
    max_abs_error = float(np.max(np.abs(reconstruction_error)))
    if max_abs_error > tolerance:
        raise ValueError(
            f"Frozen prediction mismatch for {target}, r{repeat}f{fold}: {max_abs_error}"
        )

    baseline = regression_metrics(y_valid, baseline_prediction)
    rows: List[Dict[str, object]] = []
    for mode_index, mode in enumerate(modes):
        groups = _permutation_groups(styles_valid, mode, style_minimum_group_n)
        for feature_index, feature in enumerate(data.feature_names):
            feature_values = X_valid.iloc[:, feature_index].to_numpy()
            blocks = []
            for permutation in range(permutations):
                seed = (
                    int(permutation_seed)
                    + int(repeat) * 1_000_003
                    + int(fold) * 100_003
                    + int(mode_index) * 10_007
                    + int(feature_index) * 101
                    + int(permutation)
                )
                rng = np.random.default_rng(seed)
                block = X_valid.copy()
                block.iloc[:, feature_index] = permute_values_within_groups(
                    feature_values, groups, rng
                )
                blocks.append(block)
            batch = pd.concat(blocks, axis=0, ignore_index=True)
            batch_prediction = predictor.predict(batch).reshape(permutations, len(X_valid))
            direction_rho = safe_spearman(feature_values, y_valid)
            for permutation, prediction in enumerate(batch_prediction):
                permuted = regression_metrics(y_valid, prediction)
                rows.append(
                    {
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "target": target,
                        "grade": grade,
                        "source_model": predictor.source_model,
                        "mode": mode,
                        "feature": feature,
                        "permutation": int(permutation),
                        "n_train": int(len(train_index)),
                        "n_valid": int(len(valid_index)),
                        "baseline_rmse": baseline["rmse"],
                        "baseline_r2": baseline["r2"],
                        "baseline_spearman": baseline["spearman"],
                        "permuted_rmse": permuted["rmse"],
                        "permuted_r2": permuted["r2"],
                        "permuted_spearman": permuted["spearman"],
                        "delta_rmse": permuted["rmse"] - baseline["rmse"],
                        "relative_delta_rmse": (
                            (permuted["rmse"] - baseline["rmse"]) / baseline["rmse"]
                            if baseline["rmse"] > 0
                            else np.nan
                        ),
                        "delta_r2_loss": baseline["r2"] - permuted["r2"],
                        "delta_spearman_loss": baseline["spearman"] - permuted["spearman"],
                        "heldout_feature_target_spearman": direction_rho,
                    }
                )
    audit = {
        "repeat": int(repeat),
        "fold": int(fold),
        "target": target,
        "grade": grade,
        "source_model": predictor.source_model,
        "n_train": int(len(train_index)),
        "n_valid": int(len(valid_index)),
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "baseline_spearman": baseline["spearman"],
        "reconstruction_max_abs_error": max_abs_error,
        "reconstruction_passed": bool(max_abs_error <= tolerance),
    }
    return pd.DataFrame(rows), audit


def summarize_importance(
    raw: pd.DataFrame,
    primary_mode: str,
    top_k: Sequence[int],
    top10_frequency_minimum: float,
    positive_importance_frequency_minimum: float,
    direction_consistency_minimum: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate permutation rows without promoting the pilot to formal Top-3."""
    fold_level = importance_fold_level(raw)
    return summarize_fold_importance(
        fold_level,
        primary_mode,
        top_k,
        top10_frequency_minimum,
        positive_importance_frequency_minimum,
        direction_consistency_minimum,
    )


def importance_fold_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw permutation rows to one row per target-feature outer fold."""
    return (
        raw.groupby(
            ["target", "grade", "source_model", "mode", "feature", "repeat", "fold"],
            as_index=False,
        )
        .agg(
            mean_delta_rmse=("delta_rmse", "mean"),
            median_delta_rmse=("delta_rmse", "median"),
            mean_relative_delta_rmse=("relative_delta_rmse", "mean"),
            positive_permutation_frequency=("delta_rmse", lambda values: float((values > 0).mean())),
            heldout_feature_target_spearman=("heldout_feature_target_spearman", "first"),
        )
    )


def summarize_fold_importance(
    fold_level: pd.DataFrame,
    primary_mode: str,
    top_k: Sequence[int],
    top10_frequency_minimum: float,
    positive_importance_frequency_minimum: float,
    direction_consistency_minimum: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate pre-collapsed fold rows across frozen folds and repeats."""
    fold_level = fold_level.copy()
    fold_level["rank"] = fold_level.groupby(
        ["target", "mode", "repeat", "fold"]
    )["mean_delta_rmse"].rank(method="min", ascending=False)

    rows: List[Dict[str, object]] = []
    for keys, group in fold_level.groupby(
        ["target", "grade", "source_model", "mode", "feature"], sort=False
    ):
        target, grade, source_model, mode, feature = keys
        directions = np.sign(group["heldout_feature_target_spearman"].dropna().to_numpy())
        directions = directions[directions != 0]
        if len(directions):
            positive = float((directions > 0).mean())
            direction_consistency = max(positive, 1.0 - positive)
            consensus_direction = "positive" if positive >= 0.5 else "negative"
        else:
            direction_consistency = np.nan
            consensus_direction = "undetermined"
        row: Dict[str, object] = {
            "target": target,
            "grade": grade,
            "source_model": source_model,
            "mode": mode,
            "feature": feature,
            "n_outer_folds": int(len(group)),
            "mean_delta_rmse": float(group["mean_delta_rmse"].mean()),
            "median_delta_rmse": float(group["mean_delta_rmse"].median()),
            "delta_rmse_q25": float(group["mean_delta_rmse"].quantile(0.25)),
            "delta_rmse_q75": float(group["mean_delta_rmse"].quantile(0.75)),
            "mean_relative_delta_rmse": float(group["mean_relative_delta_rmse"].mean()),
            "median_rank": float(group["rank"].median()),
            "rank_q25": float(group["rank"].quantile(0.25)),
            "rank_q75": float(group["rank"].quantile(0.75)),
            "positive_fold_frequency": float((group["mean_delta_rmse"] > 0).mean()),
            "positive_permutation_frequency": float(
                group["positive_permutation_frequency"].mean()
            ),
            "direction_consistency": float(direction_consistency),
            "consensus_direction": consensus_direction,
            "median_heldout_feature_target_spearman": float(
                group["heldout_feature_target_spearman"].median()
            ),
        }
        for k in top_k:
            row[f"top{k}_frequency"] = float((group["rank"] <= int(k)).mean())
        row["passes_pilot_stability"] = bool(
            mode == primary_mode
            and row.get("top10_frequency", 0.0) >= top10_frequency_minimum
            and row["positive_permutation_frequency"] >= positive_importance_frequency_minimum
            and np.isfinite(row["direction_consistency"])
            and row["direction_consistency"] >= direction_consistency_minimum
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["target", "mode", "mean_delta_rmse"], ascending=[True, True, False]
    ).reset_index(drop=True)
    pilot_view = (
        summary[summary["mode"] == primary_mode]
        .sort_values(["target", "mean_delta_rmse"], ascending=[True, False])
        .groupby("target", group_keys=False)
        .head(10)
        .copy()
    )
    pilot_view.insert(
        0,
        "pilot_rank",
        pilot_view.groupby("target")["mean_delta_rmse"].rank(method="first", ascending=False).astype(int),
    )
    pilot_view["formal_top3_status"] = "NOT_EVALUATED_PILOT_ONLY"
    return summary, pilot_view.reset_index(drop=True)

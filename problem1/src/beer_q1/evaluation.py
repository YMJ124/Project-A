"""OOF metrics, uncertainty summaries, and AGENTS.md stage gates.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    value = spearmanr(y_true, y_pred).correlation
    return 0.0 if not np.isfinite(value) else float(value)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = y_true - y_pred
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
        "residual_mean": float(np.mean(residual)),
        "residual_pred_spearman": safe_spearman(y_pred, residual),
    }


def metrics_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["repeat", "target", "model"]
    for keys, group in oof.groupby(group_cols, sort=True):
        repeat, target, model = keys
        row: Dict[str, Any] = {
            "repeat": int(repeat),
            "target": target,
            "model": model,
            "n": int(len(group)),
        }
        row.update(regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse",
        "mae",
        "r2",
        "spearman",
        "residual_mean",
        "residual_pred_spearman",
    ]
    grouped = metrics.groupby(["target", "model"], sort=True)[metric_cols]
    frames = []
    for statistic, values in (
        ("mean", grouped.mean()),
        ("std", grouped.std(ddof=1).fillna(0.0)),
        ("q25", grouped.quantile(0.25)),
        ("median", grouped.median()),
        ("q75", grouped.quantile(0.75)),
    ):
        renamed = values.rename(columns={column: f"{column}_{statistic}" for column in metric_cols})
        frames.append(renamed)
    result = pd.concat(frames, axis=1).reset_index()
    repeat_count = metrics.groupby(["target", "model"]).size().rename("n_repeats")
    return result.merge(repeat_count.reset_index(), on=["target", "model"])


def select_models_and_grade(
    metrics: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    if "nested_selected" in set(summary["model"]):
        # This track was selected using only inner folds for each outer split.
        selected = summary[summary["model"] == "nested_selected"].copy()
        selected = selected.rename(columns={"model": "selected_model"}).reset_index(drop=True)
    else:
        candidates = summary[~summary["model"].isin(["global_mean", "style_mean"])].copy()
        if candidates.empty:
            raise ValueError("At least one non-baseline model is needed for selection")
        best_index = candidates.groupby("target")["rmse_mean"].idxmin()
        selected = candidates.loc[best_index].copy().reset_index(drop=True)
        selected = selected.rename(columns={"model": "selected_model"})

    baseline_columns = ["target", "model", "rmse_mean"]
    baselines = summary[baseline_columns].pivot(index="target", columns="model", values="rmse_mean")
    selected = selected.merge(baselines.reset_index(), on="target", how="left")
    selected["rmse_improvement_vs_global"] = 1.0 - (
        selected["rmse_mean"] / selected["global_mean"]
    )
    selected["rmse_improvement_vs_style"] = 1.0 - (
        selected["rmse_mean"] / selected["style_mean"]
    )

    selected_repeat = metrics.merge(
        selected[["target", "selected_model"]],
        left_on=["target", "model"],
        right_on=["target", "selected_model"],
    )
    global_repeat = metrics[metrics["model"] == "global_mean"][["target", "repeat", "rmse"]]
    global_repeat = global_repeat.rename(columns={"rmse": "global_rmse"})
    repeat_compare = selected_repeat.merge(global_repeat, on=["target", "repeat"])
    repeat_stats = repeat_compare.groupby("target").agg(
        repeat_win_rate=("rmse", lambda values: float("nan")),
        positive_direction_rate=("spearman", lambda values: float(np.mean(values > 0))),
    )
    # Compute win rate with aligned rows; a named-aggregation lambda cannot see global_rmse.
    win_rate = repeat_compare.groupby("target").apply(
        lambda frame: float(np.mean(frame["rmse"] < frame["global_rmse"]))
    )
    repeat_stats["repeat_win_rate"] = win_rate
    selected = selected.merge(repeat_stats.reset_index(), on="target", how="left")

    residual_ok = (
        selected["residual_mean_mean"].abs() <= 0.10
    ) & (selected["residual_pred_spearman_mean"].abs() < 0.20)
    grade_a = (
        (selected["r2_mean"] >= 0.20)
        & (selected["spearman_mean"] >= 0.45)
        & (selected["rmse_improvement_vs_global"] >= 0.10)
        & (selected["repeat_win_rate"] >= 0.80)
        & residual_ok
    )
    grade_b = (
        (selected["r2_mean"] >= 0.05)
        & (selected["spearman_mean"] >= 0.25)
        & (selected["rmse_improvement_vs_global"] >= 0.05)
    )
    selected["grade"] = np.select([grade_a, grade_b], ["A", "B"], default="C")
    selected["residual_proxy_pass"] = residual_ok
    return selected.sort_values("target").reset_index(drop=True)


def evaluate_stage_gate(selection: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    values = {
        "median_r2": float(selection["r2_mean"].median()),
        "median_spearman": float(selection["spearman_mean"].median()),
        "median_rmse": float(selection["rmse_mean"].median()),
        "median_improvement_vs_global": float(
            selection["rmse_improvement_vs_global"].median()
        ),
        "median_improvement_vs_style": float(
            selection["rmse_improvement_vs_style"].median()
        ),
        "positive_r2_share": float(np.mean(selection["r2_mean"] > 0)),
        "majority_positive_direction_share": float(
            np.mean(selection["positive_direction_rate"] >= 0.50)
        ),
        "grade_a_count": int((selection["grade"] == "A").sum()),
        "grade_ab_count": int(selection["grade"].isin(["A", "B"]).sum()),
        "n_targets": int(len(selection)),
    }
    checks = [
        ("median_r2", values["median_r2"], ">=", 0.15, values["median_r2"] >= 0.15),
        ("median_spearman", values["median_spearman"], ">=", 0.35, values["median_spearman"] >= 0.35),
        ("median_rmse", values["median_rmse"], "<=", 0.92, values["median_rmse"] <= 0.92),
        (
            "median_improvement_vs_global",
            values["median_improvement_vs_global"],
            ">=",
            0.08,
            values["median_improvement_vs_global"] >= 0.08,
        ),
        (
            "median_improvement_vs_style",
            values["median_improvement_vs_style"],
            ">=",
            0.05,
            values["median_improvement_vs_style"] >= 0.05,
        ),
        (
            "positive_r2_share",
            values["positive_r2_share"],
            ">=",
            0.70,
            values["positive_r2_share"] >= 0.70,
        ),
        (
            "majority_positive_direction_share",
            values["majority_positive_direction_share"],
            ">=",
            0.80,
            values["majority_positive_direction_share"] >= 0.80,
        ),
        ("grade_a_count", values["grade_a_count"], ">=", 18, values["grade_a_count"] >= 18),
        ("grade_ab_count", values["grade_ab_count"], ">=", 24, values["grade_ab_count"] >= 24),
    ]
    table = pd.DataFrame(
        checks, columns=["criterion", "observed", "operator", "threshold", "passed"]
    )
    report: Dict[str, Any] = dict(values)
    report["all_automated_checks_passed"] = bool(table["passed"].all())
    report["status"] = "PASS" if report["all_automated_checks_passed"] else "HOLD"
    report["note"] = (
        "A grades use an automated residual proxy; residual plots still require human review."
    )
    return table, report


def build_uncertainty(oof: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    chosen = oof.merge(
        selection[["target", "selected_model"]],
        left_on=["target", "model"],
        right_on=["target", "selected_model"],
    )
    return (
        chosen.groupby(["beer", "style", "target", "y_true"], as_index=False)["y_pred"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "oof_prediction_mean",
                "std": "oof_prediction_std",
                "min": "oof_prediction_min",
                "max": "oof_prediction_max",
                "count": "n_repeats",
            }
        )
    )


def bootstrap_metric_intervals(
    metrics: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 20261024
) -> pd.DataFrame:
    """Bootstrap repeat-level metric means for stability intervals."""
    metric_names = ["rmse", "mae", "r2", "spearman"]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for (target, model), group in metrics.groupby(["target", "model"], sort=True):
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            if len(values) == 1:
                bootstrap_means = values.copy()
            else:
                sampled = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
                bootstrap_means = values[sampled].mean(axis=1)
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                    "n_repeats": int(len(values)),
                    "n_bootstrap": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def style_error_tables(
    oof: pd.DataFrame, model: str = "nested_selected", min_style_n: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute repeat-wise and summarized target errors within each beer style."""
    chosen = oof[oof["model"] == model]
    rows: List[Dict[str, Any]] = []
    for (repeat, target, style), group in chosen.groupby(
        ["repeat", "target", "style"], sort=True
    ):
        residual = group["y_true"].to_numpy() - group["y_pred"].to_numpy()
        rows.append(
            {
                "repeat": int(repeat),
                "target": target,
                "style": style,
                "n_samples": int(len(group)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias_true_minus_pred": float(np.mean(residual)),
                "formal_report": bool(len(group) >= min_style_n),
            }
        )
    by_repeat = pd.DataFrame(rows)
    summary = (
        by_repeat.groupby(["target", "style", "n_samples", "formal_report"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias_true_minus_pred", "mean"),
            bias_std=("bias_true_minus_pred", "std"),
            n_repeats=("repeat", "nunique"),
        )
        .fillna({"rmse_std": 0.0, "bias_std": 0.0})
    )
    return by_repeat, summary


def save_residual_diagnostics(
    oof: pd.DataFrame, output_path: Path, model: str = "nested_selected"
) -> None:
    """Save one residual-vs-prediction panel per target for human review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = oof[oof["model"] == model]
    averaged = (
        chosen.groupby(["beer", "style", "target"], as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    )
    targets = sorted(averaged["target"].unique())
    n_columns = 5
    n_rows = int(np.ceil(len(targets) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(3.2 * n_columns, 2.8 * n_rows), squeeze=False
    )
    for axis, target in zip(axes.flat, targets):
        group = averaged[averaged["target"] == target]
        residual = group["y_true"] - group["y_pred"]
        axis.scatter(group["y_pred"], residual, s=10, alpha=0.55)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(target, fontsize=8)
        axis.set_xlabel("OOF prediction", fontsize=7)
        axis.set_ylabel("true - prediction", fontsize=7)
        axis.tick_params(labelsize=6)
    for axis in axes.flat[len(targets) :]:
        axis.set_visible(False)
    figure.suptitle("Nested OOF residual diagnostics", fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

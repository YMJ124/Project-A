"""Evaluate the predeclared F_hops_all 1.25 scale on untouched OOF folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_et_gbr_hybrid import KEYS, evaluate_variant
from beer_q1.evaluation import metrics_from_oof


SOURCE = Path("artifacts/problem1/q1_fixed_scale_confirmation_control_v1")
OUTPUT = Path("artifacts/problem1/q1_fixed_scale_confirmation_v1")
TARGET = "F_hops_all"
SCALE = 1.25
MODEL = "extra_trees_fixed_scale"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = pd.read_parquet(SOURCE / "oof_predictions.parquet")
    base = source[source["model"] == "extra_trees"].copy()
    baselines = source[source["model"].isin(["global_mean", "style_mean"])].copy()
    scaled = base.copy()
    mask = scaled["target"] == TARGET
    scaled.loc[mask, "y_pred"] *= SCALE
    scaled["model"] = MODEL
    scaled["source_model"] = np.where(mask, "fixed_scale_1.25", "extra_trees")
    scaled.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)

    gate = evaluate_variant(baselines, scaled, OUTPUT, "all", range(7))
    base_metrics = metrics_from_oof(base[base["target"] == TARGET])
    scaled_metrics = metrics_from_oof(scaled[scaled["target"] == TARGET])
    paired = base_metrics.merge(
        scaled_metrics,
        on=["repeat", "target"],
        suffixes=("_base", "_scaled"),
        validate="one_to_one",
    )
    paired["rmse_gain"] = (
        paired["rmse_base"] - paired["rmse_scaled"]
    ) / paired["rmse_base"]
    paired["delta_spearman"] = (
        paired["spearman_scaled"] - paired["spearman_base"]
    )
    paired["residual_abs_reduction"] = (
        paired["residual_pred_spearman_base"].abs()
        - paired["residual_pred_spearman_scaled"].abs()
    )
    paired.to_csv(OUTPUT / "paired_target_metrics.csv", index=False)
    selection = pd.read_csv(OUTPUT / "all_model_selection.csv")
    target_result = selection[selection["target"] == TARGET].iloc[0]
    decision = {
        "source_run": SOURCE.name,
        "target": TARGET,
        "fixed_scale": SCALE,
        "base_seed": 20311124,
        "mean_rmse_gain": float(paired["rmse_gain"].mean()),
        "rmse_wins_out_of_7": int((paired["rmse_gain"] > 0).sum()),
        "maximum_absolute_spearman_delta": float(
            paired["delta_spearman"].abs().max()
        ),
        "residual_improvement_wins_out_of_7": int(
            (paired["residual_abs_reduction"] > 0).sum()
        ),
        "residual_pred_spearman_mean": float(
            target_result["residual_pred_spearman_mean"]
        ),
        "confirmed_grade": target_result["grade"],
        "passed": bool(
            paired["rmse_gain"].mean() >= 0.01
            and (paired["rmse_gain"] > 0).sum() >= 5
            and paired["delta_spearman"].abs().max() <= 1e-12
            and abs(target_result["residual_pred_spearman_mean"]) < 0.20
            and target_result["grade"] == "A"
        ),
        "stage_gate": gate,
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

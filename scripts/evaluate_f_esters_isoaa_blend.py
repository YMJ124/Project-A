"""Confirm the fixed 50/50 F_esters_isoaa ET blend on held-out seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_et_gbr_hybrid import KEYS, evaluate_variant
from beer_q1.evaluation import metrics_from_oof


ROOT = Path("artifacts/problem1")
BASE_RUN = ROOT / "q1_fixed_scale_confirmation_control_v1"
TARGETWISE_RUN = ROOT / "q1_f_esters_isoaa_blend_targetwise_v1"
OUTPUT = ROOT / "q1_f_esters_isoaa_blend_confirmation_v1"
TARGET = "F_esters_isoaa"
WEIGHT = 0.5
MODEL = "f_esters_isoaa_half_blend"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base_all = pd.read_parquet(BASE_RUN / "oof_predictions.parquet")
    targetwise_all = pd.read_parquet(TARGETWISE_RUN / "oof_predictions.parquet")
    base = base_all[base_all["model"] == "extra_trees"].copy()
    baselines = base_all[base_all["model"].isin(["global_mean", "style_mean"])].copy()
    targetwise = targetwise_all[
        (targetwise_all["model"] == "extra_trees_targetwise")
        & (targetwise_all["target"] == TARGET)
    ].copy()

    hybrid = base.copy()
    mask = hybrid["target"] == TARGET
    target_index = pd.MultiIndex.from_frame(hybrid.loc[mask, KEYS])
    targetwise = targetwise.set_index(KEYS).loc[target_index]
    if not (hybrid.loc[mask, "y_true"].to_numpy() == targetwise["y_true"].to_numpy()).all():
        raise AssertionError("Truth mismatch between blend sources")
    hybrid.loc[mask, "y_pred"] = (
        (1.0 - WEIGHT) * hybrid.loc[mask, "y_pred"].to_numpy()
        + WEIGHT * targetwise["y_pred"].to_numpy()
    )
    hybrid["model"] = MODEL
    hybrid["source_model"] = "fold_tuned_extra_trees"
    hybrid.loc[mask, "source_model"] = "half_multioutput_half_targetwise_et"
    hybrid.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)

    gate = evaluate_variant(baselines, hybrid, OUTPUT, "all", range(7))
    base_metrics = metrics_from_oof(base[base["target"] == TARGET])
    hybrid_metrics = metrics_from_oof(hybrid[hybrid["target"] == TARGET])
    paired = base_metrics.merge(
        hybrid_metrics,
        on=["repeat", "target"],
        suffixes=("_base", "_blend"),
        validate="one_to_one",
    )
    paired["rmse_gain"] = (
        paired["rmse_base"] - paired["rmse_blend"]
    ) / paired["rmse_base"]
    paired["delta_spearman"] = paired["spearman_blend"] - paired["spearman_base"]
    paired.to_csv(OUTPUT / "paired_target_metrics.csv", index=False)
    selected = pd.read_csv(OUTPUT / "all_model_selection.csv")
    target_result = selected[selected["target"] == TARGET].iloc[0]
    decision = {
        "target": TARGET,
        "base_seed": 20311124,
        "multioutput_weight": 0.5,
        "targetwise_weight": 0.5,
        "mean_rmse_gain": float(paired["rmse_gain"].mean()),
        "rmse_wins_out_of_7": int((paired["rmse_gain"] > 0).sum()),
        "median_spearman_delta": float(paired["delta_spearman"].median()),
        "spearman_nonnegative_out_of_7": int(
            (paired["delta_spearman"] >= 0).sum()
        ),
        "r2_mean": float(target_result["r2_mean"]),
        "spearman_mean": float(target_result["spearman_mean"]),
        "rmse_improvement_vs_global": float(
            target_result["rmse_improvement_vs_global"]
        ),
        "confirmed_grade": target_result["grade"],
        "passed": bool(
            paired["rmse_gain"].mean() >= 0.005
            and (paired["rmse_gain"] > 0).sum() >= 5
            and paired["delta_spearman"].median() >= 0
            and target_result["grade"] == "B"
        ),
        "stage_gate": gate,
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

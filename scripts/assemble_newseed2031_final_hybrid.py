"""Assemble the predeclared 12-replacement Problem-1 hybrid on 20311124 folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from analyze_et_gbr_hybrid import KEYS, evaluate_variant
from beer_q1.evaluation import (
    build_uncertainty,
    metrics_from_oof,
    save_residual_diagnostics,
    style_error_tables,
)


ROOT = Path("artifacts/problem1")
OUTPUT = ROOT / "q1_newseed2031_final_hybrid_v1"
MODEL = "final_specialist_hybrid"
BASE_RUN = "q1_fixed_scale_confirmation_control_v1"

DIRECT: Dict[str, Tuple[str, str, str]] = {
    "A_esters_isoaa": ("q1_newseed2031_gbr_specialist_v1", "gradient_boosting", "gradient_boosting"),
    "F_esters_all": ("q1_newseed2031_targetwise_et_specialist_v1", "extra_trees_targetwise", "extra_trees_targetwise"),
    "A_malt_cara": ("q1_newseed2031_selectk30_specialist_v1", "extra_trees_targetwise_selectk", "selectk30"),
    "F_malt_all": ("q1_newseed2031_selectk60_specialist_v1", "extra_trees_targetwise_selectk", "selectk60"),
    "F_esters_fruity": ("q1_newseed2031_selectk60_specialist_v1", "extra_trees_targetwise_selectk", "selectk60"),
    "A_malt_all": ("q1_newseed2031_selectk120_specialist_v1", "extra_trees_targetwise_selectk", "selectk120"),
    "F_hops_noble": ("q1_newseed2031_selectk120_specialist_v1", "extra_trees_targetwise_selectk", "selectk120"),
    "F_malt_cara": ("q1_newseed2031_selectk120_specialist_v1", "extra_trees_targetwise_selectk", "selectk120"),
    "A_hops_all": ("q1_newseed2031_affine_specialists_v1", "extra_trees_affine_calibrated", "affine_calibration"),
    "F_esters_ethac": ("q1_newseed2031_affine_specialists_v1", "extra_trees_affine_calibrated", "affine_calibration"),
}


def read_target(run: str, model: str, target: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / run / "oof_predictions.parquet")
    frame = frame[(frame["model"] == model) & (frame["target"] == target)].copy()
    if len(frame) != 250 * 7 or frame[KEYS].duplicated().any():
        raise AssertionError(f"Invalid source rows: {run}/{model}/{target}")
    return frame.set_index(KEYS)


def replace(hybrid: pd.DataFrame, target: str, source: pd.DataFrame, label: str) -> None:
    mask = hybrid["target"] == target
    index = pd.MultiIndex.from_frame(hybrid.loc[mask, KEYS])
    aligned = source.loc[index]
    if not np.allclose(hybrid.loc[mask, "y_true"], aligned["y_true"]):
        raise AssertionError(f"Truth mismatch for {target}")
    hybrid.loc[mask, "y_pred"] = aligned["y_pred"].to_numpy()
    hybrid.loc[mask, "source_model"] = label


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base_all = pd.read_parquet(ROOT / BASE_RUN / "oof_predictions.parquet")
    base = base_all[base_all["model"] == "extra_trees"].copy()
    baselines = base_all[base_all["model"].isin(["global_mean", "style_mean"])].copy()
    hybrid = base[KEYS + ["y_true", "y_pred"]].copy()
    hybrid["source_model"] = "fold_tuned_extra_trees"

    for target, (run, model, label) in DIRECT.items():
        replace(hybrid, target, read_target(run, model, target), label)

    hops_mask = hybrid["target"] == "F_hops_all"
    hybrid.loc[hops_mask, "y_pred"] *= 1.25
    hybrid.loc[hops_mask, "source_model"] = "fixed_scale_1.25"

    isoaa = read_target(
        "q1_f_esters_isoaa_blend_targetwise_v1",
        "extra_trees_targetwise",
        "F_esters_isoaa",
    )
    isoaa_mask = hybrid["target"] == "F_esters_isoaa"
    isoaa_index = pd.MultiIndex.from_frame(hybrid.loc[isoaa_mask, KEYS])
    aligned_isoaa = isoaa.loc[isoaa_index]
    hybrid.loc[isoaa_mask, "y_pred"] = 0.5 * (
        hybrid.loc[isoaa_mask, "y_pred"].to_numpy()
        + aligned_isoaa["y_pred"].to_numpy()
    )
    hybrid.loc[isoaa_mask, "source_model"] = "half_multioutput_half_targetwise_et"

    hybrid["model"] = MODEL
    hybrid = hybrid[KEYS + ["model", "source_model", "y_true", "y_pred"]]
    hybrid.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)

    base_eval = base[KEYS + ["model", "y_true", "y_pred"]]
    base_gate = evaluate_variant(baselines, base_eval, OUTPUT, "base", range(7))
    hybrid_gate = evaluate_variant(baselines, hybrid, OUTPUT, "all", range(7))
    selection = pd.read_csv(OUTPUT / "all_model_selection.csv")
    comparison = pd.concat([baselines, hybrid], ignore_index=True)
    build_uncertainty(comparison, selection).to_csv(OUTPUT / "uncertainty.csv", index=False)
    style_repeat, style_summary = style_error_tables(comparison, model=MODEL)
    style_repeat.to_csv(OUTPUT / "style_metrics_by_repeat.csv", index=False)
    style_summary.to_csv(OUTPUT / "style_metrics_summary.csv", index=False)
    save_residual_diagnostics(comparison, OUTPUT / "residual_diagnostics.png", model=MODEL)

    base_metrics = metrics_from_oof(base)
    hybrid_metrics = metrics_from_oof(hybrid)
    effects = base_metrics.merge(
        hybrid_metrics,
        on=["repeat", "target"],
        suffixes=("_base", "_hybrid"),
        validate="one_to_one",
    )
    effects["rmse_gain"] = (effects["rmse_base"] - effects["rmse_hybrid"]) / effects["rmse_base"]
    effects["delta_spearman"] = effects["spearman_hybrid"] - effects["spearman_base"]
    effects.to_csv(OUTPUT / "replacement_effects_by_repeat.csv", index=False)

    protocol = {
        "base_seed": 20311124,
        "base_run": BASE_RUN,
        "replacement_count": 12,
        "direct_replacements": {target: label for target, (_, _, label) in DIRECT.items()},
        "fixed_scale": {"F_hops_all": 1.25},
        "fixed_blend": {"F_esters_isoaa": {"multioutput": 0.5, "targetwise": 0.5}},
        "base_stage_gate": base_gate,
        "hybrid_stage_gate": hybrid_gate,
        "selection_note": "The complete replacement map was frozen before assembly.",
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

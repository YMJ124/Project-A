"""Assemble the fixed Problem-1 specialist hybrid on 20301124 OOF folds.

The eight previously confirmed specialists are carried forward unchanged, and
the two affine-calibrated targets passed the predeclared confirmation and tuned
ET challenge.  This script does not select targets from the 20301124 results;
it only evaluates the already fixed ten-target replacement map on common folds.
"""

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
OUTPUT = ROOT / "q1_newseed2030_calibrated_specialist_hybrid_v1"
MODEL_NAME = "calibrated_specialist_hybrid"
REPEATS = list(range(7))

SOURCES: Dict[str, Tuple[str, str, str]] = {
    "A_esters_isoaa": (
        "q1_newseed2030_gbr_specialist_v1",
        "gradient_boosting",
        "gradient_boosting",
    ),
    "F_esters_all": (
        "q1_newseed2030_targetwise_et_specialist_v1",
        "extra_trees_targetwise",
        "extra_trees_targetwise",
    ),
    "A_malt_cara": (
        "q1_newseed2030_selectk30_specialist_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k30",
    ),
    "F_malt_all": (
        "q1_newseed2030_selectk60_specialist_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k60",
    ),
    "F_esters_fruity": (
        "q1_newseed2030_selectk60_specialist_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k60",
    ),
    "A_malt_all": (
        "q1_newseed2030_selectk120_specialist_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k120",
    ),
    "F_hops_noble": (
        "q1_newseed2030_selectk120_specialist_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k120",
    ),
    "F_malt_cara": (
        "q1_newseed2030_selectk120_specialist_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k120",
    ),
    "A_hops_all": (
        "q1_affine_calibration_confirmation_v1",
        "extra_trees_affine_calibrated",
        "cross_fitted_affine_calibration",
    ),
    "F_esters_ethac": (
        "q1_affine_calibration_confirmation_v1",
        "extra_trees_affine_calibrated",
        "cross_fitted_affine_calibration",
    ),
}


def _read_target(run: str, model: str, target: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / run / "oof_predictions.parquet")
    frame = frame[(frame["model"] == model) & (frame["target"] == target)].copy()
    expected = 250 * 7
    if len(frame) != expected:
        raise AssertionError(
            f"{run}/{model}/{target}: expected {expected} rows, got {len(frame)}"
        )
    if frame[KEYS].duplicated().any():
        raise AssertionError(f"Duplicate OOF keys in {run}/{model}/{target}")
    return frame


def _replacement_effects(base: pd.DataFrame, hybrid: pd.DataFrame) -> pd.DataFrame:
    base_metrics = metrics_from_oof(base)
    hybrid_metrics = metrics_from_oof(hybrid)
    paired = base_metrics.merge(
        hybrid_metrics,
        on=["repeat", "target"],
        suffixes=("_base", "_hybrid"),
        validate="one_to_one",
    )
    paired["rmse_gain"] = (
        paired["rmse_base"] - paired["rmse_hybrid"]
    ) / paired["rmse_base"]
    paired["delta_spearman"] = (
        paired["spearman_hybrid"] - paired["spearman_base"]
    )
    replaced = paired[paired["target"].isin(SOURCES)].copy()
    summary = replaced.groupby("target", as_index=False).agg(
        mean_rmse_gain=("rmse_gain", "mean"),
        median_rmse_gain=("rmse_gain", "median"),
        rmse_wins_out_of_7=("rmse_gain", lambda x: int((x > 0).sum())),
        median_spearman_delta=("delta_spearman", "median"),
        spearman_nonnegative_out_of_7=(
            "delta_spearman",
            lambda x: int((x >= 0).sum()),
        ),
    )
    summary["source_model"] = summary["target"].map(
        {target: details[2] for target, details in SOURCES.items()}
    )
    return summary.sort_values("mean_rmse_gain", ascending=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base_run = ROOT / "q1_et_affine_confirmation_control_v1"
    base_all = pd.read_parquet(base_run / "oof_predictions.parquet")
    base = base_all[base_all["model"] == "extra_trees"].copy()
    baselines = base_all[base_all["model"].isin(["global_mean", "style_mean"])].copy()

    expected = 250 * 30 * 7
    if len(base) != expected:
        raise AssertionError(f"Expected {expected} base rows, got {len(base)}")
    hybrid = base[KEYS + ["y_true", "y_pred"]].copy()
    hybrid["source_model"] = "fold_tuned_extra_trees"

    for target, (run, model, label) in SOURCES.items():
        mask = hybrid["target"] == target
        target_index = pd.MultiIndex.from_frame(hybrid.loc[mask, KEYS])
        source = _read_target(run, model, target).set_index(KEYS).loc[target_index]
        if not np.allclose(hybrid.loc[mask, "y_true"], source["y_true"]):
            raise AssertionError(f"Truth mismatch for {target}")
        hybrid.loc[mask, "y_pred"] = source["y_pred"].to_numpy()
        hybrid.loc[mask, "source_model"] = label

    hybrid["model"] = MODEL_NAME
    hybrid = hybrid[KEYS + ["model", "source_model", "y_true", "y_pred"]]
    hybrid.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)

    base_for_eval = base[KEYS + ["model", "y_true", "y_pred"]].copy()
    base_gate = evaluate_variant(baselines, base_for_eval, OUTPUT, "base", REPEATS)
    hybrid_gate = evaluate_variant(baselines, hybrid, OUTPUT, "all", REPEATS)

    selection = pd.read_csv(OUTPUT / "all_model_selection.csv")
    comparison = pd.concat([baselines, hybrid], ignore_index=True)
    build_uncertainty(comparison, selection).to_csv(
        OUTPUT / "uncertainty.csv", index=False
    )
    style_repeat, style_summary = style_error_tables(comparison, model=MODEL_NAME)
    style_repeat.to_csv(OUTPUT / "style_metrics_by_repeat.csv", index=False)
    style_summary.to_csv(OUTPUT / "style_metrics_summary.csv", index=False)
    save_residual_diagnostics(
        comparison, OUTPUT / "residual_diagnostics.png", model=MODEL_NAME
    )
    effects = _replacement_effects(base, hybrid)
    effects.to_csv(OUTPUT / "replacement_effects.csv", index=False)

    protocol = {
        "base_run": "q1_et_affine_confirmation_control_v1",
        "outer_repeats": REPEATS,
        "base_seed": 20301124,
        "replacement_map_fixed_before_assembly": {
            target: label for target, (_, _, label) in SOURCES.items()
        },
        "selection_note": (
            "No source was selected from the 20301124 total-model results. "
            "Eight specialists were already confirmed on prior folds; the two "
            "calibration targets passed predeclared confirmation and tuned-ET checks."
        ),
        "base_stage_gate": base_gate,
        "hybrid_stage_gate": hybrid_gate,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

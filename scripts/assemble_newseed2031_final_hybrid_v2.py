"""Add independently confirmed A_esters_all GBR to the locked 2031 hybrid."""

from __future__ import annotations

import json
from pathlib import Path

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
OLD = ROOT / "q1_newseed2031_final_hybrid_v1"
SOURCE = ROOT / "q1_a_esters_all_gbr_2031_source_v1"
BASE_RUN = ROOT / "q1_fixed_scale_confirmation_control_v1"
OUTPUT = ROOT / "q1_newseed2031_final_hybrid_v2"
MODEL = "final_specialist_hybrid_v2"
TARGET = "A_esters_all"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    old = pd.read_parquet(OLD / "oof_predictions.parquet").copy()
    source_all = pd.read_parquet(SOURCE / "oof_predictions.parquet")
    source = source_all[
        (source_all["model"] == "gradient_boosting")
        & (source_all["target"] == TARGET)
    ].copy()
    if len(source) != 250 * 7 or source[KEYS].duplicated().any():
        raise AssertionError("Invalid A_esters_all GBR source")

    hybrid = old.copy()
    mask = hybrid["target"] == TARGET
    index = pd.MultiIndex.from_frame(hybrid.loc[mask, KEYS])
    aligned = source.set_index(KEYS).loc[index]
    if not np.allclose(hybrid.loc[mask, "y_true"], aligned["y_true"]):
        raise AssertionError("A_esters_all truth mismatch")
    hybrid.loc[mask, "y_pred"] = aligned["y_pred"].to_numpy()
    hybrid.loc[mask, "source_model"] = "gradient_boosting"
    hybrid["model"] = MODEL
    hybrid.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)

    base_all = pd.read_parquet(BASE_RUN / "oof_predictions.parquet")
    base = base_all[base_all["model"] == "extra_trees"].copy()
    baselines = base_all[base_all["model"].isin(["global_mean", "style_mean"])].copy()
    base_gate = evaluate_variant(
        baselines, base[KEYS + ["model", "y_true", "y_pred"]], OUTPUT, "base", range(7)
    )
    hybrid_gate = evaluate_variant(baselines, hybrid, OUTPUT, "all", range(7))

    selection = pd.read_csv(OUTPUT / "all_model_selection.csv")
    comparison = pd.concat([baselines, hybrid], ignore_index=True)
    build_uncertainty(comparison, selection).to_csv(OUTPUT / "uncertainty.csv", index=False)
    style_repeat, style_summary = style_error_tables(comparison, model=MODEL)
    style_repeat.to_csv(OUTPUT / "style_metrics_by_repeat.csv", index=False)
    style_summary.to_csv(OUTPUT / "style_metrics_summary.csv", index=False)
    save_residual_diagnostics(comparison, OUTPUT / "residual_diagnostics.png", model=MODEL)

    old_metrics = metrics_from_oof(old)
    new_metrics = metrics_from_oof(hybrid)
    effects = old_metrics.merge(
        new_metrics,
        on=["repeat", "target"],
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    effects["rmse_gain"] = (effects["rmse_v1"] - effects["rmse_v2"]) / effects["rmse_v1"]
    effects["delta_spearman"] = effects["spearman_v2"] - effects["spearman_v1"]
    effects.to_csv(OUTPUT / "replacement_effects_vs_v1.csv", index=False)

    protocol = {
        "base_seed": 20311124,
        "parent_hybrid": OLD.name,
        "new_replacement": {TARGET: "fixed_gradient_boosting"},
        "replacement_count": 13,
        "independent_confirmation": "q1_a_esters_all_gbr_confirmation_v1 (base seed 20351124)",
        "selection_note": "Only the independently confirmed locked source was added; no 2031 target/model selection was performed.",
        "base_stage_gate": base_gate,
        "hybrid_stage_gate": hybrid_gate,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

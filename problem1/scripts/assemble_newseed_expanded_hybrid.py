"""Assemble the fully confirmed seven-repeat Problem-1 specialist hybrid.

Every source prediction uses the same 20281124 seven-repeat outer folds.  The
base is fold-tuned multi-output Extra Trees.  Two previously confirmed target
specialists and six fresh-seed-confirmed fold-local SelectKBest specialists are
substituted without inspecting any other target from the confirmation runs.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
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
OUTPUT = ROOT / "q1_newseed_expanded_specialist_hybrid_v1"
MODEL_NAME = "expanded_specialist_hybrid"
REPEATS = list(range(7))

SOURCES: Dict[str, Tuple[str, str, str]] = {
    "A_esters_isoaa": (
        "q1_gbr_newseed_final_control_v1",
        "gradient_boosting",
        "gradient_boosting",
    ),
    "F_esters_all": (
        "q1_targetwise_et_newseed_final_control_v1",
        "extra_trees_targetwise",
        "extra_trees_targetwise",
    ),
    "A_malt_cara": (
        "q1_targetwise_selectk30_confirmation_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k30",
    ),
    "F_malt_all": (
        "q1_targetwise_selectk60_confirmation_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k60",
    ),
    "F_esters_fruity": (
        "q1_targetwise_selectk60_confirmation_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k60",
    ),
    "A_malt_all": (
        "q1_targetwise_selectk120_confirmation_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k120",
    ),
    "F_hops_noble": (
        "q1_targetwise_selectk120_confirmation_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k120",
    ),
    "F_malt_cara": (
        "q1_targetwise_selectk120_confirmation_v1",
        "extra_trees_targetwise_selectk",
        "extra_trees_targetwise_selectk_k120",
    ),
}

NOMINATED_SELECTK = {
    "A_malt_cara": 30,
    "F_malt_all": 60,
    "F_esters_fruity": 60,
    "A_malt_all": 120,
    "F_hops_noble": 120,
    "F_malt_cara": 120,
    "A_esters_all": 120,
    "A_hops_woody": 120,
}


def _read_model(run: str, model: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / run / "oof_predictions.parquet")
    frame = frame[frame["model"] == model].copy()
    expected = 250 * 30 * 7
    if len(frame) != expected:
        raise AssertionError(f"{run}/{model}: expected {expected} rows, got {len(frame)}")
    return frame


def _selector_confirmation(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cache: Dict[int, pd.DataFrame] = {}
    for target, k in NOMINATED_SELECTK.items():
        if k not in cache:
            cache[k] = _read_model(
                f"q1_targetwise_selectk{k}_confirmation_v1",
                "extra_trees_targetwise_selectk",
            )
        base_metrics = metrics_from_oof(base[base["target"] == target])
        new_metrics = metrics_from_oof(cache[k][cache[k]["target"] == target])
        paired = base_metrics.merge(
            new_metrics,
            on=["repeat", "target"],
            suffixes=("_base", "_new"),
            validate="one_to_one",
        )
        gain = (paired["rmse_base"] - paired["rmse_new"]) / paired["rmse_base"]
        delta_rho = paired["spearman_new"] - paired["spearman_base"]
        passed = bool(
            gain.mean() >= 0.01
            and (gain > 0).sum() >= 5
            and delta_rho.median() >= 0
        )
        # The C-grade woody target had the additional requirement that it must
        # become at least B; its 0.67% gain already fails the common rule.
        if target == "A_hops_woody":
            passed = False
        rows.append(
            {
                "target": target,
                "k": k,
                "mean_rmse_gain": float(gain.mean()),
                "median_rmse_gain": float(gain.median()),
                "rmse_wins_out_of_7": int((gain > 0).sum()),
                "median_spearman_delta": float(delta_rho.median()),
                "spearman_nonnegative_out_of_7": int((delta_rho >= 0).sum()),
                "adopted": passed,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rmse_gain", ascending=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    et_oof = pd.read_parquet(
        ROOT / "q1_et_newseed_candidate_control_v1" / "oof_predictions.parquet"
    )
    base = et_oof[et_oof["model"] == "extra_trees"].copy()
    baselines = et_oof[et_oof["model"].isin(["global_mean", "style_mean"])].copy()

    hybrid = base[KEYS + ["y_true", "y_pred"]].copy()
    hybrid["source_model"] = "fold_tuned_extra_trees"
    source_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for target, (run, model, label) in SOURCES.items():
        cache_key = (run, model)
        if cache_key not in source_cache:
            source_cache[cache_key] = _read_model(run, model).set_index(KEYS)
        mask = hybrid["target"] == target
        target_keys = pd.MultiIndex.from_frame(hybrid.loc[mask, KEYS])
        source = source_cache[cache_key].loc[target_keys]
        if not np.allclose(hybrid.loc[mask, "y_true"], source["y_true"]):
            raise AssertionError(f"Truth mismatch for {target}")
        hybrid.loc[mask, "y_pred"] = source["y_pred"].to_numpy()
        hybrid.loc[mask, "source_model"] = label

    hybrid["model"] = MODEL_NAME
    hybrid = hybrid[KEYS + ["model", "source_model", "y_true", "y_pred"]]
    hybrid.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)

    gate = evaluate_variant(baselines, hybrid, OUTPUT, "all", REPEATS)
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

    confirmation = _selector_confirmation(base)
    confirmation.to_csv(OUTPUT / "feature_selection_confirmation.csv", index=False)
    protocol = {
        "base_run": "q1_et_newseed_candidate_control_v1",
        "base_model": "fold-tuned multi-output Extra Trees",
        "outer_repeats": REPEATS,
        "base_seed": 20281124,
        "confirmed_replacements": {
            target: label for target, (_, _, label) in SOURCES.items()
        },
        "selectk_rule": {
            "mean_rmse_gain_minimum": 0.01,
            "minimum_rmse_wins_out_of_7": 5,
            "median_spearman_delta_minimum": 0.0,
            "no_grade_downgrade": True,
            "A_hops_woody_additional_requirement": "must improve C to B or A",
        },
        "rejected_selectk_targets": ["A_esters_all", "A_hops_woody"],
        "leakage_note": (
            "All feature ranking was fitted inside each training fold. No "
            "Attachment-8 importance, external labels, or style feature was used."
        ),
        "stage_gate": gate,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

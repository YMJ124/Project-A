"""Assemble the confirmed target-specialist champion and its OOF evidence.

The base model is tuned multi-output Extra Trees. A_esters_isoaa is replaced by
fixed target-wise Gradient Boosting, and F_esters_all by fixed target-wise Extra
Trees. Both replacements were nominated on repeats 0-2 and accepted only after
predeclared checks on repeats 3-9.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from analyze_et_gbr_hybrid import (
    CONFIRMATION_REPEATS,
    DISCOVERY_REPEATS,
    KEYS,
    evaluate_variant,
    paired_predictions,
)
from beer_q1.evaluation import (
    build_uncertainty,
    save_residual_diagnostics,
    style_error_tables,
)


MODEL_NAME = "confirmed_specialist_hybrid"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _targetwise_confirmation_table(
    shared_fixed_metrics: pd.DataFrame,
    targetwise_metrics: pd.DataFrame,
    tuned_et_metrics: pd.DataFrame,
) -> pd.DataFrame:
    shared = shared_fixed_metrics[
        shared_fixed_metrics["model"] == "extra_trees"
    ].copy()
    targetwise = targetwise_metrics[
        targetwise_metrics["model"] == "extra_trees_targetwise"
    ].copy()
    tuned = tuned_et_metrics[tuned_et_metrics["model"] == "extra_trees"].copy()
    paired = shared.merge(
        targetwise,
        on=["repeat", "target"],
        suffixes=("_shared_fixed", "_targetwise"),
        validate="one_to_one",
    )
    paired = paired.merge(
        tuned[["repeat", "target", "rmse", "spearman", "r2"]].rename(
            columns={
                "rmse": "rmse_tuned_et",
                "spearman": "spearman_tuned_et",
                "r2": "r2_tuned_et",
            }
        ),
        on=["repeat", "target"],
        validate="one_to_one",
    )
    paired["rmse_gain_vs_shared_fixed"] = (
        paired["rmse_shared_fixed"] - paired["rmse_targetwise"]
    ) / paired["rmse_shared_fixed"]
    paired["delta_spearman_vs_shared_fixed"] = (
        paired["spearman_targetwise"] - paired["spearman_shared_fixed"]
    )
    paired["rmse_gain_vs_tuned_et"] = (
        paired["rmse_tuned_et"] - paired["rmse_targetwise"]
    ) / paired["rmse_tuned_et"]
    paired["delta_spearman_vs_tuned_et"] = (
        paired["spearman_targetwise"] - paired["spearman_tuned_et"]
    )
    paired["evidence_slice"] = np.where(
        paired["repeat"].isin(DISCOVERY_REPEATS), "discovery", "confirmation"
    )
    return paired[paired["target"] == "F_esters_all"].sort_values("repeat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--et-run",
        type=Path,
        default=Path("artifacts/problem1/q1_et_confirmation_v1"),
    )
    parser.add_argument(
        "--gbr-run",
        type=Path,
        default=Path("artifacts/problem1/q1_gbr_confirmation_v1"),
    )
    parser.add_argument(
        "--targetwise-run",
        type=Path,
        default=Path("artifacts/problem1/q1_targetwise_et_confirmation_v1"),
    )
    parser.add_argument(
        "--shared-fixed-run",
        type=Path,
        default=Path("artifacts/problem1/q1_shared_et_fixed_confirmation_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/problem1/q1_confirmed_specialist_hybrid_v1"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    et_oof = pd.read_parquet(args.et_run / "oof_predictions.parquet")
    gbr_oof = pd.read_parquet(args.gbr_run / "oof_predictions.parquet")
    targetwise_oof = pd.read_parquet(args.targetwise_run / "oof_predictions.parquet")
    paired = paired_predictions(et_oof, gbr_oof)

    targetwise = targetwise_oof[
        targetwise_oof["model"] == "extra_trees_targetwise"
    ][KEYS + ["y_true", "y_pred"]].rename(
        columns={"y_true": "y_true_targetwise", "y_pred": "y_pred_targetwise"}
    )
    hybrid = paired.merge(targetwise, on=KEYS, validate="one_to_one")
    if not np.allclose(hybrid["y_true_et"], hybrid["y_true_targetwise"]):
        raise AssertionError("Target-wise ET truth values differ from base ET")

    hybrid["y_true"] = hybrid["y_true_et"]
    hybrid["y_pred"] = hybrid["y_pred_et"]
    hybrid["source_model"] = "extra_trees"
    use_gbr = hybrid["target"] == "A_esters_isoaa"
    use_targetwise = hybrid["target"] == "F_esters_all"
    hybrid.loc[use_gbr, "y_pred"] = hybrid.loc[use_gbr, "y_pred_gbr"]
    hybrid.loc[use_gbr, "source_model"] = "gradient_boosting"
    hybrid.loc[use_targetwise, "y_pred"] = hybrid.loc[
        use_targetwise, "y_pred_targetwise"
    ]
    hybrid.loc[use_targetwise, "source_model"] = "extra_trees_targetwise"
    hybrid["model"] = MODEL_NAME
    hybrid = hybrid[KEYS + ["model", "source_model", "y_true", "y_pred"]]
    hybrid.to_parquet(args.output / "oof_predictions.parquet", index=False)

    baselines = et_oof[et_oof["model"].isin(["global_mean", "style_mean"])].copy()
    gates = {
        "all": evaluate_variant(
            baselines,
            hybrid,
            args.output,
            "all",
            DISCOVERY_REPEATS + CONFIRMATION_REPEATS,
        ),
        "confirmation": evaluate_variant(
            baselines,
            hybrid,
            args.output,
            "confirmation",
            CONFIRMATION_REPEATS,
        ),
    }

    selection = pd.read_csv(args.output / "all_model_selection.csv")
    comparison_oof = pd.concat([baselines, hybrid], ignore_index=True)
    build_uncertainty(comparison_oof, selection).to_csv(
        args.output / "uncertainty.csv", index=False
    )
    style_repeat, style_summary = style_error_tables(comparison_oof, model=MODEL_NAME)
    style_repeat.to_csv(args.output / "style_metrics_by_repeat.csv", index=False)
    style_summary.to_csv(args.output / "style_metrics_summary.csv", index=False)
    save_residual_diagnostics(
        comparison_oof,
        args.output / "residual_diagnostics.png",
        model=MODEL_NAME,
    )

    targetwise_metrics = pd.read_csv(args.targetwise_run / "metrics_by_repeat.csv")
    shared_fixed_metrics = pd.read_csv(args.shared_fixed_run / "metrics_by_repeat.csv")
    tuned_et_metrics = pd.read_csv(args.et_run / "metrics_by_repeat.csv")
    targetwise_pair = _targetwise_confirmation_table(
        shared_fixed_metrics, targetwise_metrics, tuned_et_metrics
    )
    targetwise_pair.to_csv(
        args.output / "f_esters_all_targetwise_confirmation.csv", index=False
    )
    confirm = targetwise_pair[targetwise_pair["repeat"].isin(CONFIRMATION_REPEATS)]
    acceptance = {
        "mean_rmse_gain_vs_shared_fixed": float(
            confirm["rmse_gain_vs_shared_fixed"].mean()
        ),
        "rmse_wins_vs_shared_fixed": int(
            (confirm["rmse_gain_vs_shared_fixed"] > 0).sum()
        ),
        "median_spearman_delta_vs_shared_fixed": float(
            confirm["delta_spearman_vs_shared_fixed"].median()
        ),
        "mean_rmse_gain_vs_tuned_et": float(
            confirm["rmse_gain_vs_tuned_et"].mean()
        ),
        "rmse_wins_vs_tuned_et": int((confirm["rmse_gain_vs_tuned_et"] > 0).sum()),
        "median_spearman_delta_vs_tuned_et": float(
            confirm["delta_spearman_vs_tuned_et"].median()
        ),
        "predeclared_rule_passed": bool(
            (confirm["rmse_gain_vs_shared_fixed"].mean() >= 0.02)
            and ((confirm["rmse_gain_vs_shared_fixed"] > 0).sum() >= 6)
            and (confirm["delta_spearman_vs_shared_fixed"].median() >= 0)
        ),
    }
    protocol = {
        "base_model": "fold-tuned multi-output Extra Trees",
        "confirmed_replacements": {
            "A_esters_isoaa": "fixed target-wise Gradient Boosting",
            "F_esters_all": "fixed target-wise Extra Trees",
        },
        "discovery_repeats": DISCOVERY_REPEATS,
        "confirmation_repeats": CONFIRMATION_REPEATS,
        "f_esters_all_predeclared_rule": {
            "mean_rmse_gain_minimum": 0.02,
            "minimum_rmse_wins_out_of_7": 6,
            "median_spearman_delta_minimum": 0.0,
        },
        "f_esters_all_acceptance": acceptance,
        "leakage_note": (
            "No external labels or style inputs were used. Only targets nominated "
            "on repeats 0-2 were evaluated for adoption on repeats 3-9."
        ),
        "stage_gates": gates,
    }
    _write_json(args.output / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

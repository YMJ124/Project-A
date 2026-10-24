"""Build reproducible ET/GBR paired-confirmation and hybrid artifacts.

Repeats 0-2 are the discovery slice used to nominate target-level challengers.
Repeats 3-9 are untouched confirmation evidence.  The strict champion replaces
only A_esters_isoaa; the two-target variant remains an exploratory candidate.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from beer_q1.evaluation import (
    evaluate_stage_gate,
    metrics_from_oof,
    select_models_and_grade,
    summarize_metrics,
)


KEYS = ["beer", "style", "repeat", "fold", "target"]
DISCOVERY_REPEATS = [0, 1, 2]
CONFIRMATION_REPEATS = [3, 4, 5, 6, 7, 8, 9]
STRICT_TARGETS = ["A_esters_isoaa"]
EXPLORATORY_TARGETS = ["A_esters_isoaa", "A_esters_all"]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _python_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def paired_predictions(et_oof: pd.DataFrame, gbr_oof: pd.DataFrame) -> pd.DataFrame:
    et = et_oof[et_oof["model"] == "extra_trees"].copy()
    gbr = gbr_oof[gbr_oof["model"] == "gradient_boosting"].copy()
    paired = et.merge(
        gbr[KEYS + ["y_true", "y_pred"]],
        on=KEYS,
        suffixes=("_et", "_gbr"),
        validate="one_to_one",
    )
    if not np.allclose(paired["y_true_et"], paired["y_true_gbr"]):
        raise AssertionError("ET and GBR truth values differ on matched OOF rows")
    expected = 250 * 30 * 10
    if len(paired) != expected:
        raise AssertionError(f"Expected {expected} paired rows, got {len(paired)}")
    return paired


def paired_metric_table(et_oof: pd.DataFrame, gbr_oof: pd.DataFrame) -> pd.DataFrame:
    et_metrics = metrics_from_oof(et_oof[et_oof["model"] == "extra_trees"])
    gbr_metrics = metrics_from_oof(
        gbr_oof[gbr_oof["model"] == "gradient_boosting"]
    )
    paired = et_metrics.merge(
        gbr_metrics,
        on=["repeat", "target"],
        suffixes=("_et", "_gbr"),
        validate="one_to_one",
    )
    paired["rmse_gain_fraction"] = (
        paired["rmse_et"] - paired["rmse_gbr"]
    ) / paired["rmse_et"]
    paired["delta_r2"] = paired["r2_gbr"] - paired["r2_et"]
    paired["delta_spearman"] = paired["spearman_gbr"] - paired["spearman_et"]
    paired["evidence_slice"] = np.where(
        paired["repeat"].isin(DISCOVERY_REPEATS), "discovery", "confirmation"
    )
    return paired.sort_values(["target", "repeat"]).reset_index(drop=True)


def make_hybrid(
    paired: pd.DataFrame, targets: Sequence[str], model_name: str
) -> pd.DataFrame:
    hybrid = paired[KEYS + ["y_true_et", "y_pred_et", "y_pred_gbr"]].copy()
    use_gbr = hybrid["target"].isin(targets)
    hybrid["y_true"] = hybrid.pop("y_true_et")
    hybrid["y_pred"] = np.where(use_gbr, hybrid.pop("y_pred_gbr"), hybrid.pop("y_pred_et"))
    hybrid["model"] = model_name
    hybrid["source_model"] = np.where(use_gbr, "gradient_boosting", "extra_trees")
    return hybrid


def evaluate_variant(
    baseline_oof: pd.DataFrame,
    model_oof: pd.DataFrame,
    output_dir: Path,
    label: str,
    repeats: Iterable[int],
) -> Dict[str, Any]:
    repeats = list(repeats)
    scoped_model = model_oof[model_oof["repeat"].isin(repeats)].copy()
    scoped_baselines = baseline_oof[baseline_oof["repeat"].isin(repeats)].copy()
    comparison = pd.concat([scoped_baselines, scoped_model], ignore_index=True)
    metrics = metrics_from_oof(comparison)
    summary = summarize_metrics(metrics)
    selection = select_models_and_grade(metrics, summary)
    gate, report = evaluate_stage_gate(selection)

    metrics.to_csv(output_dir / f"{label}_metrics_by_repeat.csv", index=False)
    summary.to_csv(output_dir / f"{label}_metrics_summary.csv", index=False)
    selection.to_csv(output_dir / f"{label}_model_selection.csv", index=False)
    gate.to_csv(output_dir / f"{label}_stage_gate.csv", index=False)
    _write_json(
        output_dir / f"{label}_stage_gate.json",
        {key: _python_value(value) for key, value in report.items()},
    )
    return {key: _python_value(value) for key, value in report.items()}


def candidate_summary(paired_metrics: pd.DataFrame, target: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for label, repeats in (
        ("discovery", DISCOVERY_REPEATS),
        ("confirmation", CONFIRMATION_REPEATS),
        ("all", DISCOVERY_REPEATS + CONFIRMATION_REPEATS),
    ):
        group = paired_metrics[
            (paired_metrics["target"] == target)
            & (paired_metrics["repeat"].isin(repeats))
        ]
        result[label] = {
            "n_repeats": int(len(group)),
            "mean_rmse_gain_fraction": float(group["rmse_gain_fraction"].mean()),
            "median_rmse_gain_fraction": float(group["rmse_gain_fraction"].median()),
            "rmse_win_rate": float(np.mean(group["rmse_gain_fraction"] > 0)),
            "mean_delta_spearman": float(group["delta_spearman"].mean()),
            "median_delta_spearman": float(group["delta_spearman"].median()),
            "spearman_nonnegative_rate": float(np.mean(group["delta_spearman"] >= 0)),
            "mean_delta_r2": float(group["delta_r2"].mean()),
        }
    return result


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
        "--output",
        type=Path,
        default=Path("artifacts/problem1/q1_et_gbr_hybrid_analysis_v1"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    et_oof = pd.read_parquet(args.et_run / "oof_predictions.parquet")
    gbr_oof = pd.read_parquet(args.gbr_run / "oof_predictions.parquet")
    paired = paired_predictions(et_oof, gbr_oof)
    paired_metrics = paired_metric_table(et_oof, gbr_oof)
    paired_metrics.to_csv(args.output / "paired_target_metrics.csv", index=False)

    baseline_oof = et_oof[et_oof["model"].isin(["global_mean", "style_mean"])].copy()
    strict = make_hybrid(paired, STRICT_TARGETS, "hybrid_strict")
    exploratory = make_hybrid(
        paired, EXPLORATORY_TARGETS, "hybrid_two_target_exploratory"
    )
    strict.to_parquet(args.output / "hybrid_strict_oof.parquet", index=False)
    exploratory.to_parquet(
        args.output / "hybrid_two_target_exploratory_oof.parquet", index=False
    )

    gates = {
        "strict_all": evaluate_variant(
            baseline_oof,
            strict,
            args.output,
            "strict_all",
            DISCOVERY_REPEATS + CONFIRMATION_REPEATS,
        ),
        "strict_confirmation": evaluate_variant(
            baseline_oof,
            strict,
            args.output,
            "strict_confirmation",
            CONFIRMATION_REPEATS,
        ),
        "exploratory_all": evaluate_variant(
            baseline_oof,
            exploratory,
            args.output,
            "exploratory_all",
            DISCOVERY_REPEATS + CONFIRMATION_REPEATS,
        ),
        "exploratory_confirmation": evaluate_variant(
            baseline_oof,
            exploratory,
            args.output,
            "exploratory_confirmation",
            CONFIRMATION_REPEATS,
        ),
    }
    protocol = {
        "et_run": str(args.et_run.resolve()),
        "gbr_run": str(args.gbr_run.resolve()),
        "discovery_repeats": DISCOVERY_REPEATS,
        "confirmation_repeats": CONFIRMATION_REPEATS,
        "strict_targets": STRICT_TARGETS,
        "exploratory_targets": EXPLORATORY_TARGETS,
        "strict_decision": (
            "Adopt A_esters_isoaa GBR challenger: confirmation RMSE wins 7/7, "
            "mean gain >=3%, and Spearman improves 7/7."
        ),
        "exploratory_decision": (
            "Keep A_esters_all as conditional: confirmation RMSE wins 7/7, but "
            "mean gain is just below 3% and Spearman improves in only 5/7 repeats."
        ),
        "leakage_note": (
            "No external labels or style features were used. Repeats 3-9 were not "
            "used to nominate the challenger targets."
        ),
    }
    _write_json(args.output / "protocol.json", protocol)
    summary = {
        "candidate_targets": {
            target: candidate_summary(paired_metrics, target)
            for target in EXPLORATORY_TARGETS
        },
        "stage_gates": gates,
    }
    _write_json(args.output / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

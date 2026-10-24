"""Compare the predeclared threshold-mechanism candidates on identical folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from beer_q1.evaluation import metrics_from_oof


ROOT = Path("artifacts/problem1")
OUTPUT = ROOT / "q1_threshold_mechanism_screening_analysis_v1"
RUNS = {
    "raw_targetwise_et": "q1_threshold_mechanism_control_v1",
    "mechanism_ridge": "q1_threshold_mechanism_ridge_v1",
    "mechanism_augmented_et": "q1_threshold_mechanism_augmented_et_v1",
    "mechanism_residual_et": "q1_threshold_mechanism_residual_et_v1",
}
CONTROL = "raw_targetwise_et"


def _load_candidate(label: str, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = ROOT / run_name
    oof = pd.read_parquet(run / "oof_predictions.parquet")
    selected = oof[oof["model"] == "nested_selected"].copy()
    selected["model"] = label
    grades = pd.read_csv(run / "model_selection.csv")[
        ["target", "grade", "rmse_mean", "r2_mean", "spearman_mean"]
    ].copy()
    grades.insert(0, "candidate", label)
    return selected, grades


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    reference_folds = None
    predictions = []
    grades = []
    for label, run_name in RUNS.items():
        folds = pd.read_csv(ROOT / run_name / "folds.csv")
        if reference_folds is None:
            reference_folds = folds
        else:
            pd.testing.assert_frame_equal(reference_folds, folds)
        candidate_oof, candidate_grades = _load_candidate(label, run_name)
        predictions.append(candidate_oof)
        grades.append(candidate_grades)

    all_oof = pd.concat(predictions, ignore_index=True)
    all_oof.to_parquet(OUTPUT / "candidate_oof_predictions.parquet", index=False)
    metrics = metrics_from_oof(all_oof)
    metrics.to_csv(OUTPUT / "metrics_by_repeat.csv", index=False)
    grades_frame = pd.concat(grades, ignore_index=True)
    grades_frame.to_csv(OUTPUT / "candidate_grades.csv", index=False)

    control = metrics[metrics["model"] == CONTROL].rename(
        columns={column: f"{column}_control" for column in metrics.columns if column not in ["repeat", "target"]}
    )
    challengers = metrics[metrics["model"] != CONTROL].copy()
    paired = challengers.merge(control, on=["repeat", "target"], validate="many_to_one")
    paired["candidate"] = paired["model"]
    paired["rmse_gain"] = (
        paired["rmse_control"] - paired["rmse"]
    ) / paired["rmse_control"]
    paired["delta_r2"] = paired["r2"] - paired["r2_control"]
    paired["delta_spearman"] = paired["spearman"] - paired["spearman_control"]
    paired["residual_abs_reduction"] = (
        paired["residual_pred_spearman_control"].abs()
        - paired["residual_pred_spearman"].abs()
    )
    paired.to_csv(OUTPUT / "paired_metrics_by_repeat.csv", index=False)

    summary = (
        paired.groupby(["candidate", "target"], as_index=False)
        .agg(
            mean_rmse_gain=("rmse_gain", "mean"),
            rmse_wins=("rmse_gain", lambda values: int((values > 0).sum())),
            mean_delta_r2=("delta_r2", "mean"),
            median_delta_spearman=("delta_spearman", "median"),
            mean_delta_spearman=("delta_spearman", "mean"),
            mean_residual_abs_reduction=("residual_abs_reduction", "mean"),
        )
    )
    grade_lookup = grades_frame.pivot(index="target", columns="candidate", values="grade")
    summary["control_grade"] = summary["target"].map(grade_lookup[CONTROL])
    summary["candidate_grade"] = [
        grade_lookup.loc[target, candidate]
        for candidate, target in zip(summary["candidate"], summary["target"])
    ]
    grade_order = {"C": 0, "B": 1, "A": 2}
    summary["grade_not_lower"] = [
        grade_order[candidate] >= grade_order[control]
        for candidate, control in zip(
            summary["candidate_grade"], summary["control_grade"]
        )
    ]
    summary["passed_discovery"] = (
        (summary["mean_rmse_gain"] >= 0.01)
        & (summary["rmse_wins"] >= 2)
        & (summary["median_delta_spearman"] >= 0.0)
        & summary["grade_not_lower"]
    )
    summary["large_uplift"] = (
        (summary["mean_rmse_gain"] >= 0.03)
        | (
            (summary["candidate_grade"] == "A")
            & (summary["control_grade"] != "A")
        )
    )
    summary = summary.sort_values(
        ["target", "mean_rmse_gain"], ascending=[True, False]
    )
    summary.to_csv(OUTPUT / "candidate_comparison.csv", index=False)

    overall = (
        summary.groupby("candidate", as_index=False)
        .agg(
            median_target_rmse_gain=("mean_rmse_gain", "median"),
            passed_targets=("passed_discovery", "sum"),
            large_uplift_targets=("large_uplift", "sum"),
        )
        .sort_values("median_target_rmse_gain", ascending=False)
    )
    overall.to_csv(OUTPUT / "overall_comparison.csv", index=False)

    accepted = summary[summary["passed_discovery"]]
    best_by_target = {}
    for target, group in accepted.groupby("target"):
        best = group.sort_values("mean_rmse_gain", ascending=False).iloc[0]
        best_by_target[target] = {
            "candidate": best["candidate"],
            "mean_rmse_gain": float(best["mean_rmse_gain"]),
            "rmse_wins_out_of_3": int(best["rmse_wins"]),
            "median_delta_spearman": float(best["median_delta_spearman"]),
            "candidate_grade": best["candidate_grade"],
            "large_uplift": bool(best["large_uplift"]),
        }
    decision = {
        "protocol": "problem1/docs/problem1_threshold_mechanism_protocol_v1.md",
        "base_seed": 20461124,
        "targets": ["A_hops_citrus", "F_hops_citrus"],
        "acceptance": {
            "mean_rmse_gain_minimum": 0.01,
            "rmse_wins_minimum_out_of_3": 2,
            "median_spearman_delta_minimum": 0.0,
            "grade_must_not_drop": True,
        },
        "large_uplift_definition": "mean RMSE gain >=3% or stable promotion to A",
        "best_passing_candidate_by_target": best_by_target,
        "any_large_uplift": bool(summary["large_uplift"].any()),
        "note": (
            "Discovery evidence only. A passing target requires seven untouched "
            "repeats and comparison with the current champion before adoption."
        ),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

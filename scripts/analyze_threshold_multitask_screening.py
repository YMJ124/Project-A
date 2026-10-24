"""Evaluate the predeclared mechanism-plus-A/F-sharing experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beer_q1.evaluation import metrics_from_oof


ROOT = Path("artifacts/problem1")
OUTPUT = ROOT / "q1_threshold_multitask_screening_analysis_v1"
RUNS = {
    "raw_targetwise_et": "q1_threshold_multitask_targetwise_control_v1",
    "raw_shared_et": "q1_threshold_multitask_shared_control_v1",
    "mechanism_shared_et": "q1_threshold_multitask_mechanism_v1",
}
CANDIDATE = "mechanism_shared_et"
CONTROLS = ["raw_targetwise_et", "raw_shared_et"]


def _selected_oof(label: str, run_name: str) -> pd.DataFrame:
    oof = pd.read_parquet(ROOT / run_name / "oof_predictions.parquet")
    selected = oof[oof["model"] == "nested_selected"].copy()
    selected["model"] = label
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    reference_folds = None
    predictions = []
    grade_rows = []
    for label, run_name in RUNS.items():
        folds = pd.read_csv(ROOT / run_name / "folds.csv")
        if reference_folds is None:
            reference_folds = folds
        else:
            pd.testing.assert_frame_equal(reference_folds, folds)
        predictions.append(_selected_oof(label, run_name))
        selection = pd.read_csv(ROOT / run_name / "model_selection.csv")
        selection = selection[
            ["target", "grade", "rmse_mean", "r2_mean", "spearman_mean"]
        ].copy()
        selection.insert(0, "candidate", label)
        grade_rows.append(selection)

    oof = pd.concat(predictions, ignore_index=True)
    metrics = metrics_from_oof(oof)
    metrics.to_csv(OUTPUT / "metrics_by_repeat.csv", index=False)
    grades = pd.concat(grade_rows, ignore_index=True)
    grades.to_csv(OUTPUT / "candidate_grades.csv", index=False)

    candidate = metrics[metrics["model"] == CANDIDATE].copy()
    comparisons = []
    for control_name in CONTROLS:
        control = metrics[metrics["model"] == control_name].copy()
        paired = candidate.merge(
            control,
            on=["repeat", "target"],
            suffixes=("_candidate", "_control"),
            validate="one_to_one",
        )
        paired["control"] = control_name
        paired["rmse_gain"] = (
            paired["rmse_control"] - paired["rmse_candidate"]
        ) / paired["rmse_control"]
        paired["delta_r2"] = paired["r2_candidate"] - paired["r2_control"]
        paired["delta_spearman"] = (
            paired["spearman_candidate"] - paired["spearman_control"]
        )
        comparisons.append(paired)
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(OUTPUT / "paired_metrics_by_repeat.csv", index=False)

    summary = (
        paired.groupby(["control", "target"], as_index=False)
        .agg(
            mean_rmse_gain=("rmse_gain", "mean"),
            rmse_wins=("rmse_gain", lambda values: int((values > 0).sum())),
            mean_delta_r2=("delta_r2", "mean"),
            median_delta_spearman=("delta_spearman", "median"),
        )
    )
    grade_lookup = grades.pivot(index="target", columns="candidate", values="grade")
    grade_order = {"C": 0, "B": 1, "A": 2}
    summary["control_grade"] = [
        grade_lookup.loc[target, control]
        for control, target in zip(summary["control"], summary["target"])
    ]
    summary["candidate_grade"] = summary["target"].map(grade_lookup[CANDIDATE])
    summary["grade_not_lower"] = [
        grade_order[candidate_grade] >= grade_order[control_grade]
        for candidate_grade, control_grade in zip(
            summary["candidate_grade"], summary["control_grade"]
        )
    ]
    summary["passed_against_control"] = (
        (summary["mean_rmse_gain"] >= 0.01)
        & (summary["rmse_wins"] >= 2)
        & (summary["median_delta_spearman"] >= 0.0)
        & summary["grade_not_lower"]
    )
    summary.to_csv(OUTPUT / "comparison_by_control.csv", index=False)

    target_decisions = {}
    for target, group in summary.groupby("target"):
        passed = bool(group["passed_against_control"].all())
        minimum_gain = float(group["mean_rmse_gain"].min())
        candidate_grade = str(group["candidate_grade"].iloc[0])
        control_grades = group["control_grade"].tolist()
        large_uplift = bool(
            minimum_gain >= 0.03
            or (candidate_grade == "A" and any(grade != "A" for grade in control_grades))
        )
        target_decisions[target] = {
            "passed_both_controls": passed,
            "minimum_mean_rmse_gain": minimum_gain,
            "candidate_grade": candidate_grade,
            "large_uplift": large_uplift,
        }
    decision = {
        "protocol": "docs/problem1_threshold_mechanism_protocol_v2.md",
        "base_seed": 20471124,
        "candidate": CANDIDATE,
        "controls": CONTROLS,
        "target_decisions": target_decisions,
        "advance_to_confirmation": any(
            value["passed_both_controls"] for value in target_decisions.values()
        ),
        "any_large_uplift": any(
            value["large_uplift"] for value in target_decisions.values()
        ),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

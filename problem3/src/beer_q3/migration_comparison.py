"""Paired decision audit for a sparse migration candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .smoke import _resolve, _sha256


KEY = ["repeat", "fold", "source_beer"]


def _load_run(root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(root / "migration_selected_metrics.csv")
    sensory = pd.read_csv(root / "migration_sensory_metrics.csv")
    sensory = sensory.loc[sensory["target_scope"] == "ab_only"].copy()
    if metrics.duplicated(KEY).any() or sensory.duplicated(KEY).any():
        raise ValueError(f"Migration run has duplicate paired keys: {root}")
    return manifest, metrics, sensory


def compare_sparse_migration(project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_root = _resolve(project_root, config["interfaces"]["baseline_root"])
    candidate_root = _resolve(project_root, config["interfaces"]["candidate_root"])
    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    baseline_manifest, baseline, baseline_sensory = _load_run(baseline_root)
    candidate_manifest, candidate, candidate_sensory = _load_run(candidate_root)
    baseline_keys = baseline.loc[:, KEY].sort_values(KEY).reset_index(drop=True)
    candidate_keys = candidate.loc[:, KEY].sort_values(KEY).reset_index(drop=True)
    paired_keys_match = baseline_keys.equals(candidate_keys)
    if not paired_keys_match:
        raise ValueError("Baseline and sparse candidate do not cover identical OOF migrations")
    paired = baseline.merge(candidate, on=KEY, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    sensory = baseline_sensory.merge(
        candidate_sensory, on=KEY, suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    thresholds = config["selection"]

    def ratio(column: str) -> float:
        denominator = float(paired[f"{column}_baseline"].mean())
        numerator = float(paired[f"{column}_candidate"].mean())
        return numerator / denominator if abs(denominator) > 1e-12 else np.nan

    style_retention = ratio("style_probability_gain")
    chemistry_retention = ratio("target_distance_improvement")
    fields_ratio = ratio("changed_fields")
    modules_ratio = ratio("changed_modules")
    l1_ratio = ratio("standardized_l1")
    l2_ratio = ratio("standardized_l2")
    baseline_sensory_mean = float(sensory["sensory_distance_improvement_baseline"].mean())
    candidate_sensory_mean = float(sensory["sensory_distance_improvement_candidate"].mean())
    sensory_retention = (
        candidate_sensory_mean / baseline_sensory_mean
        if baseline_sensory_mean > 1e-12
        else np.nan
    )
    baseline_failures = int((sensory["sensory_distance_improvement_baseline"] <= 0.0).sum())
    candidate_failures = int((sensory["sensory_distance_improvement_candidate"] <= 0.0).sum())
    repeat_cost = (
        paired.groupby("repeat", as_index=False)
        .agg(
            fields_baseline=("changed_fields_baseline", "mean"),
            fields_candidate=("changed_fields_candidate", "mean"),
            modules_baseline=("changed_modules_baseline", "mean"),
            modules_candidate=("changed_modules_candidate", "mean"),
        )
    )
    repeat_cost["both_costs_reduced"] = (
        (repeat_cost["fields_candidate"] < repeat_cost["fields_baseline"] - 1e-12)
        & (repeat_cost["modules_candidate"] < repeat_cost["modules_baseline"] - 1e-12)
    )
    repeat_cost_wins = int(repeat_cost["both_costs_reduced"].sum())
    candidate_methods_sparse = bool(
        candidate["method"].astype(str).str.startswith("module_counterfactual_").all()
    )
    conditions = {
        "paired_oof_keys_match": paired_keys_match,
        "baseline_absolute_confirmation_passed": bool(baseline_manifest["criteria_passed"]),
        "candidate_absolute_confirmation_passed": bool(candidate_manifest["criteria_passed"]),
        "candidate_uses_only_module_counterfactuals": candidate_methods_sparse,
        "style_gain_retention_pass": style_retention
        >= float(thresholds["minimum_style_gain_retention"]),
        "chemistry_gain_retention_pass": chemistry_retention
        >= float(thresholds["minimum_chemistry_gain_retention"]),
        "ab_sensory_mean_retention_pass": bool(
            np.isfinite(sensory_retention)
            and sensory_retention
            >= float(thresholds["minimum_ab_sensory_mean_improvement_retention"])
        ),
        "ab_sensory_failure_count_protected": candidate_failures
        <= baseline_failures + int(thresholds["maximum_additional_ab_sensory_failures"]),
        "changed_fields_reduction_pass": fields_ratio
        <= float(thresholds["maximum_changed_fields_ratio"]),
        "changed_modules_reduction_pass": modules_ratio
        <= float(thresholds["maximum_changed_modules_ratio"]),
        "standardized_l1_protected": l1_ratio
        <= float(thresholds["maximum_standardized_l1_ratio"]),
        "standardized_l2_protected": l2_ratio
        <= float(thresholds["maximum_standardized_l2_ratio"]),
        "repeat_cost_reduction_wins_pass": repeat_cost_wins
        >= int(thresholds["minimum_repeats_with_both_field_and_module_reduction"]),
    }
    passed = bool(all(conditions.values()))
    status = "SPARSE_MIGRATION_CONFIRMED" if passed else "SPARSE_MIGRATION_REJECTED"
    statistics = {
        "paired_oof_rows": int(len(paired)),
        "style_gain_retention": style_retention,
        "chemistry_gain_retention": chemistry_retention,
        "ab_sensory_mean_improvement_retention": sensory_retention,
        "baseline_ab_sensory_failures": baseline_failures,
        "candidate_ab_sensory_failures": candidate_failures,
        "changed_fields_ratio": fields_ratio,
        "changed_modules_ratio": modules_ratio,
        "standardized_l1_ratio": l1_ratio,
        "standardized_l2_ratio": l2_ratio,
        "repeat_cost_reduction_wins": repeat_cost_wins,
        "baseline_mean_changed_fields": float(paired["changed_fields_baseline"].mean()),
        "candidate_mean_changed_fields": float(paired["changed_fields_candidate"].mean()),
        "baseline_mean_changed_modules": float(paired["changed_modules_baseline"].mean()),
        "candidate_mean_changed_modules": float(paired["changed_modules_candidate"].mean()),
    }
    paired.to_csv(output / "paired_migration_metrics.csv", index=False)
    sensory.to_csv(output / "paired_ab_sensory_metrics.csv", index=False)
    repeat_cost.to_csv(output / "repeat_cost_comparison.csv", index=False)
    pd.DataFrame([{**{"status": status, "criteria_passed": passed}, **conditions}]).to_csv(
        output / "selection_audit.csv", index=False
    )
    manifest = {
        "status": status,
        "criteria_passed": passed,
        "selection_conditions": conditions,
        "statistics": statistics,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "config_sha256": _sha256(config_path),
        "external_test_data_used": False,
        "q1_status_inherited": "HOLD",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "decision.md").write_text(
        "# Sparse migration paired confirmation\n\n"
        f"- Status: `{status}`.\n"
        f"- Paired OOF migrations: {len(paired)}.\n"
        f"- Style/chemical gain retained: {style_retention:.2%}/{chemistry_retention:.2%}.\n"
        f"- A/B sensory mean improvement retained: {sensory_retention:.2%}.\n"
        f"- Changed fields ratio: {fields_ratio:.2%}; modules ratio: {modules_ratio:.2%}.\n"
        f"- Standardized L1/L2 ratio: {l1_ratio:.2%}/{l2_ratio:.2%}.\n"
        f"- Repeats reducing both field and module cost: {repeat_cost_wins}/{repeat_cost['repeat'].nunique()}.\n",
        encoding="utf-8",
    )
    return manifest

"""Repeated outer-fold confirmation for Pils/Lager to Tripel migration."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
import yaml

from beer_q1.data import load_problem1_data
from beer_q2.reconstruct import (
    fit_representative_target_predictor,
    predict_frozen_targets,
)

from .derived import load_formula_config
from .evaluation import fit_style_classifier
from .migration import build_migration_candidates
from .smoke import _git_commit, _resolve, _sha256, _write_parquet_atomic


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _robust_target_scale(targets: pd.DataFrame) -> pd.Series:
    scale = (targets.quantile(0.75) - targets.quantile(0.25)) / 1.349
    fallback = targets.std(ddof=0)
    return scale.where(scale > 1e-12, fallback).where(lambda x: x > 1e-12, 1.0)


def _fold_positions(
    ids: pd.Series,
    folds: pd.DataFrame,
    repeat: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return immutable-order train/validation positions for one outer fold."""
    rows = folds.loc[folds["repeat"] == int(repeat)].copy()
    if len(rows) != len(ids) or rows["beer"].duplicated().any():
        raise ValueError(f"Repeat {repeat} does not cover each beer exactly once")
    if set(rows["beer"].astype(int)) != set(ids.astype(int)):
        raise ValueError(f"Repeat {repeat} and the data have different beer sets")
    assignment = rows.set_index(rows["beer"].astype(int))["fold"]
    fold_by_row = ids.astype(int).map(assignment)
    if fold_by_row.isna().any():
        raise ValueError("Fold assignment lost one or more beers")
    valid = fold_by_row.to_numpy(dtype=int) == int(fold)
    if not valid.any() or valid.all():
        raise ValueError(f"Invalid outer split repeat={repeat}, fold={fold}")
    seeds = rows["seed"].drop_duplicates()
    if len(seeds) != 1:
        raise ValueError(f"Repeat {repeat} does not have one fixed seed")
    return np.flatnonzero(~valid), np.flatnonzero(valid), int(seeds.iloc[0])


def _decorate(frame: pd.DataFrame, repeat: int, fold: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "fold", int(fold))
    result.insert(0, "repeat", int(repeat))
    if "candidate_id" in result:
        result["candidate_id"] = result["candidate_id"].map(
            lambda value: f"r{repeat}_f{fold}_{value}"
        )
    return result


def _sensory_audit(
    *,
    data,
    folds: pd.DataFrame,
    tuning: pd.DataFrame,
    registry: pd.DataFrame,
    X_source: pd.DataFrame,
    X_migrated: pd.DataFrame,
    source_ids: list[int],
    candidate_ids: list[str],
    train_index: np.ndarray,
    repeat: int,
    fold: int,
    target_style: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache: dict[tuple[object, ...], object] = {}
    predictors = []
    for target in data.target_names:
        predictor, fitted_train, fitted_valid = fit_representative_target_predictor(
            data,
            folds,
            tuning,
            target,
            repeat,
            fold,
            model_cache=cache,
            n_jobs=n_jobs,
        )
        if not np.array_equal(np.sort(fitted_train), np.sort(train_index)):
            raise AssertionError("Frozen Q1 predictor used a different outer-training partition")
        if np.intersect1d(fitted_train, fitted_valid).size:
            raise AssertionError("Frozen Q1 train and validation positions overlap")
        predictors.append(predictor)

    before = pd.DataFrame(
        predict_frozen_targets(predictors, X_source), columns=data.target_names
    )
    after = pd.DataFrame(
        predict_frozen_targets(predictors, X_migrated), columns=data.target_names
    )
    y_train = data.y.iloc[train_index].reset_index(drop=True)
    styles_train = data.styles.iloc[train_index].astype(str).reset_index(drop=True)
    target_rows = styles_train.to_numpy() == str(target_style)
    if target_rows.sum() < 2:
        raise ValueError(f"Outer training fold has insufficient {target_style} sensory rows")
    center = y_train.loc[target_rows].median()
    scale = _robust_target_scale(y_train)
    grades = dict(zip(registry["target"], registry["grade"]))

    metric_rows: list[dict[str, Any]] = []
    scopes = {
        "all_30": list(data.target_names),
        "ab_only": [target for target in data.target_names if grades[target] in {"A", "B"}],
    }
    for scope, targets in scopes.items():
        target_center = center.loc[targets].to_numpy(dtype=float)
        denominator = scale.loc[targets].to_numpy(dtype=float)
        before_distance = np.sqrt(
            np.mean(((before.loc[:, targets].to_numpy() - target_center) / denominator) ** 2, axis=1)
        )
        after_distance = np.sqrt(
            np.mean(((after.loc[:, targets].to_numpy() - target_center) / denominator) ** 2, axis=1)
        )
        for position, source_id in enumerate(source_ids):
            metric_rows.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "source_beer": int(source_id),
                    "candidate_id": candidate_ids[position],
                    "target_scope": scope,
                    "target_count": len(targets),
                    "sensory_distance_before": float(before_distance[position]),
                    "sensory_distance_after": float(after_distance[position]),
                    "sensory_distance_improvement": float(
                        before_distance[position] - after_distance[position]
                    ),
                    "sensory_target_improved": bool(
                        after_distance[position] < before_distance[position] - 1e-12
                    ),
                }
            )

    profile_rows: list[dict[str, Any]] = []
    for position, source_id in enumerate(source_ids):
        for target in data.target_names:
            target_scale = float(scale[target])
            target_center = float(center[target])
            for role, values in (("before", before), ("after", after)):
                prediction = float(values.loc[position, target])
                profile_rows.append(
                    {
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "source_beer": int(source_id),
                        "candidate_id": candidate_ids[position],
                        "role": role,
                        "target": target,
                        "confidence_grade": grades[target],
                        "primary_ab_target": grades[target] in {"A", "B"},
                        "standardized_prediction": prediction,
                        "training_fold_tripel_center": target_center,
                        "training_fold_target_scale": target_scale,
                        "relative_reconstructed_scale": float(
                            np.clip(3.0 + (prediction - target_center) / target_scale, 0.0, 6.0)
                        ),
                        "scale_interpretation": "relative_reconstructed_scale_not_raw_human_score",
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)


def _run_fold(
    *,
    data,
    folds: pd.DataFrame,
    tuning: pd.DataFrame,
    registry: pd.DataFrame,
    actionability: pd.DataFrame,
    defects: pd.DataFrame,
    formula_config: Mapping[str, Any],
    migration_config: Mapping[str, Any],
    repeat: int,
    fold: int,
    include_sensory: bool,
    n_jobs: int,
) -> dict[str, pd.DataFrame]:
    train_index, valid_index, repeat_seed = _fold_positions(
        data.ids, folds, repeat, fold
    )
    X_train = data.X.iloc[train_index].reset_index(drop=True)
    styles_train = data.styles.iloc[train_index].reset_index(drop=True)
    X_valid = data.X.iloc[valid_index].reset_index(drop=True)
    styles_valid = data.styles.iloc[valid_index].reset_index(drop=True)
    ids_valid = data.ids.iloc[valid_index].astype(int).reset_index(drop=True)
    source_style = str(migration_config["source_style"])
    target_style = str(migration_config["target_style"])
    source_mask = styles_valid.astype(str).to_numpy() == source_style
    X_source = X_valid.loc[source_mask].reset_index(drop=True)
    source_ids = ids_valid.loc[source_mask].astype(int).tolist()
    if not source_ids:
        raise ValueError(f"No validation {source_style} rows for repeat={repeat}, fold={fold}")

    classifier = fit_style_classifier(
        X_train,
        styles_train,
        seed=repeat_seed + int(fold) * 10007,
    )
    migration = build_migration_candidates(
        X_train,
        styles_train,
        data.ids.iloc[train_index].astype(int).tolist(),
        X_source,
        source_ids,
        actionability,
        defects,
        formula_config,
        classifier,
        source_style=source_style,
        target_style=target_style,
        interpolation_budgets=[float(value) for value in migration_config["interpolation_budgets"]],
        module_budgets=[int(value) for value in migration_config["module_budgets"]],
        n_components=int(migration_config["n_components"]),
        coral_shrinkage=float(migration_config["coral_shrinkage"]),
        sinkhorn_epsilon=float(migration_config["sinkhorn_epsilon"]),
        sinkhorn_iterations=int(migration_config["sinkhorn_iterations"]),
        selection_gain_fraction=float(migration_config["selection_gain_fraction"]),
        eligible_method_prefixes=migration_config.get("eligible_method_prefixes"),
        range_policy=str(
            migration_config.get("range_policy", "strict_outer_training_minmax_v1")
        ),
    )
    candidates = _decorate(migration.candidates, repeat, fold)
    pareto = _decorate(migration.pareto, repeat, fold)
    selected = _decorate(migration.selected, repeat, fold)
    selected_metrics = pareto.loc[pareto["selected"]].copy()
    coverage = pd.DataFrame({"source_beer": source_ids})
    coverage["repeat"] = int(repeat)
    coverage["fold"] = int(fold)
    coverage["selected"] = coverage["source_beer"].isin(
        selected["source_beer"].astype(int)
    )
    coverage["selection_status"] = np.where(
        coverage["selected"], "SELECTED", "NO_SAFE_ELIGIBLE_CANDIDATE"
    )
    if len(selected) != len(selected_metrics):
        raise AssertionError("Selected chemistry and metric row counts differ")
    selected = selected.sort_values("source_beer").reset_index(drop=True)
    selected_metrics = selected_metrics.sort_values("source_beer").reset_index(drop=True)
    if selected["source_beer"].astype(int).tolist() != selected_metrics["source_beer"].astype(int).tolist():
        raise AssertionError("Selected chemistry and metric rows lost alignment")
    selected_source_ids = selected["source_beer"].astype(int).tolist()
    source_by_id = X_source.copy()
    source_by_id.index = source_ids
    selected_source = source_by_id.loc[selected_source_ids].reset_index(drop=True)
    selected_chemistry = selected.loc[:, list(data.feature_names)].reset_index(drop=True)

    if include_sensory and len(selected):
        sensory_metrics, sensory_profiles = _sensory_audit(
            data=data,
            folds=folds,
            tuning=tuning,
            registry=registry,
            X_source=selected_source,
            X_migrated=selected_chemistry,
            source_ids=selected_source_ids,
            candidate_ids=selected["candidate_id"].astype(str).tolist(),
            train_index=train_index,
            repeat=repeat,
            fold=fold,
            target_style=target_style,
            n_jobs=n_jobs,
        )
    else:
        sensory_metrics = pd.DataFrame(
            columns=[
                "repeat", "fold", "source_beer", "candidate_id", "target_scope",
                "target_count", "sensory_distance_before", "sensory_distance_after",
                "sensory_distance_improvement", "sensory_target_improved",
            ]
        )
        sensory_profiles = pd.DataFrame(
            columns=[
                "repeat", "fold", "source_beer", "candidate_id", "role", "target",
                "confidence_grade", "primary_ab_target", "standardized_prediction",
                "training_fold_tripel_center", "training_fold_target_scale",
                "relative_reconstructed_scale", "scale_interpretation",
            ]
        )
    return {
        "candidates": candidates,
        "pareto": pareto,
        "selected": selected,
        "selected_metrics": selected_metrics,
        "coverage": coverage,
        "sensory_metrics": sensory_metrics,
        "sensory_profiles": sensory_profiles,
    }


def _aggregate_decision(
    selected: pd.DataFrame,
    sensory: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    expected_rows: int,
) -> tuple[dict[str, bool], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    thresholds = config["selection"]
    repeat_summary = (
        selected.groupby("repeat", as_index=False)
        .agg(
            n_sources=("source_beer", "size"),
            mean_style_probability_gain=("style_probability_gain", "mean"),
            mean_target_distance_improvement=("target_distance_improvement", "mean"),
            mean_standardized_l1=("standardized_l1", "mean"),
            mean_standardized_l2=("standardized_l2", "mean"),
            mean_changed_fields=("changed_fields", "mean"),
            mean_changed_modules=("changed_modules", "mean"),
        )
    )
    source_summary = (
        selected.groupby("source_beer", as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            median_style_probability_gain=("style_probability_gain", "median"),
            median_target_distance_improvement=("target_distance_improvement", "median"),
            median_standardized_l1=("standardized_l1", "median"),
            median_standardized_l2=("standardized_l2", "median"),
            median_changed_fields=("changed_fields", "median"),
            median_changed_modules=("changed_modules", "median"),
        )
    )
    style_rate = float((selected["style_probability_gain"] > 1e-12).mean())
    chemistry_rate = float((selected["target_distance_improvement"] >= -1e-12).mean())
    style_repeat_wins = int((repeat_summary["mean_style_probability_gain"] > 1e-12).sum())
    chemistry_repeat_wins = int(
        (repeat_summary["mean_target_distance_improvement"] >= -1e-12).sum()
    )
    style_source_wins = int((source_summary["median_style_probability_gain"] > 1e-12).sum())
    chemistry_source_wins = int(
        (source_summary["median_target_distance_improvement"] >= -1e-12).sum()
    )
    ab = sensory.loc[sensory["target_scope"] == "ab_only"].copy()
    if len(ab):
        ab_rate = float(ab["sensory_target_improved"].astype(bool).mean())
        ab_repeat = ab.groupby("repeat")["sensory_distance_improvement"].mean()
        ab_source = ab.groupby("source_beer")["sensory_distance_improvement"].median()
        ab_repeat_wins = int((ab_repeat > 1e-12).sum())
        ab_source_wins = int((ab_source > 1e-12).sum())
        repeat_summary = repeat_summary.merge(
            ab.rename(columns={"sensory_distance_improvement": "ab_sensory_improvement"})
            .groupby("repeat", as_index=False)["ab_sensory_improvement"]
            .mean(),
            on="repeat",
            how="left",
            validate="one_to_one",
        )
        source_summary = source_summary.merge(
            ab.rename(columns={"sensory_distance_improvement": "ab_sensory_improvement"})
            .groupby("source_beer", as_index=False)["ab_sensory_improvement"]
            .median(),
            on="source_beer",
            how="left",
            validate="one_to_one",
        )
    else:
        ab_rate = float("nan")
        ab_repeat_wins = 0
        ab_source_wins = 0

    conditions = {
        "complete_expected_oof_rows": len(selected) == int(expected_rows),
        "all_selected_hard_constraints_pass": bool(selected["hard_constraints_pass"].all()),
        "zero_exact_training_copies": bool((~selected["exact_training_copy"]).all()),
        "zero_dangerous_near_neighbors": bool((~selected["near_neighbor_danger"]).all()),
        "style_positive_row_rate_pass": style_rate
        >= float(thresholds["minimum_row_positive_rate_style"]),
        "chemistry_positive_row_rate_pass": chemistry_rate
        >= float(thresholds["minimum_row_positive_rate_chemistry"]),
        "style_repeat_mean_wins_pass": style_repeat_wins
        >= int(thresholds["minimum_repeat_mean_wins_style"]),
        "chemistry_repeat_mean_wins_pass": chemistry_repeat_wins
        >= int(thresholds["minimum_repeat_mean_wins_chemistry"]),
        "style_source_median_wins_pass": style_source_wins
        >= int(thresholds["minimum_source_median_wins_style"]),
        "chemistry_source_median_wins_pass": chemistry_source_wins
        >= int(thresholds["minimum_source_median_wins_chemistry"]),
        "ab_sensory_complete": len(ab) == int(expected_rows),
        "ab_sensory_positive_row_rate_pass": bool(
            np.isfinite(ab_rate)
            and ab_rate >= float(thresholds["minimum_row_positive_rate_ab_sensory"])
        ),
        "ab_sensory_repeat_mean_wins_pass": ab_repeat_wins
        >= int(thresholds["minimum_repeat_mean_wins_ab_sensory"]),
        "ab_sensory_source_median_wins_pass": ab_source_wins
        >= int(thresholds["minimum_source_median_wins_ab_sensory"]),
    }
    statistics = {
        "expected_oof_rows": int(expected_rows),
        "observed_oof_rows": int(len(selected)),
        "style_positive_row_rate": style_rate,
        "chemistry_positive_row_rate": chemistry_rate,
        "style_repeat_mean_wins": style_repeat_wins,
        "chemistry_repeat_mean_wins": chemistry_repeat_wins,
        "style_source_median_wins": style_source_wins,
        "chemistry_source_median_wins": chemistry_source_wins,
        "ab_sensory_positive_row_rate": ab_rate,
        "ab_sensory_repeat_mean_wins": ab_repeat_wins,
        "ab_sensory_source_median_wins": ab_source_wins,
        "mean_standardized_l1": float(selected["standardized_l1"].mean()),
        "mean_standardized_l2": float(selected["standardized_l2"].mean()),
        "mean_changed_fields": float(selected["changed_fields"].mean()),
        "mean_changed_modules": float(selected["changed_modules"].mean()),
    }
    return conditions, repeat_summary, source_summary, statistics


def run_migration_confirmation(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Run or resume the locked repeated outer-fold migration confirmation."""
    started = time.perf_counter()
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    supported_range_policies = {
        "strict_outer_training_minmax_v1",
        "training_or_paired_source_envelope_no_further_excursion",
    }
    observed_range_policy = config["migration"].get("range_policy")
    if observed_range_policy is not None and observed_range_policy not in supported_range_policies:
        raise ValueError(
            f"Unsupported migration range policy: {observed_range_policy}"
        )
    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    subrun_root = output / "subruns"
    subrun_root.mkdir(parents=True, exist_ok=True)

    q1_config = {
        "data": {
            "chemicals_csv": str(_resolve(project_root, config["data"]["chemicals_csv"])),
            "sensory_csv": str(_resolve(project_root, config["data"]["sensory_csv"])),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": "main",
        },
        "runtime": {},
    }
    data = load_problem1_data(q1_config)
    folds_path = _resolve(project_root, config["interfaces"]["folds_csv"])
    tuning_path = _resolve(project_root, config["interfaces"]["q1_tuning_csv"])
    registry_path = _resolve(project_root, config["interfaces"]["q1_source_registry_csv"])
    actionability_path = _resolve(
        project_root, config["interfaces"]["actionability_registry_csv"]
    )
    defects_path = _resolve(project_root, config["interfaces"]["defect_registry_csv"])
    formula_path = _resolve(project_root, config["interfaces"]["derived_formula_yaml"])
    folds = pd.read_csv(folds_path)
    tuning = pd.read_csv(tuning_path)
    registry = pd.read_csv(registry_path)
    actionability = pd.read_csv(actionability_path)
    defects = pd.read_csv(defects_path)
    formula_config = load_formula_config(formula_path)
    repeats = [int(value) for value in config["evaluation"]["repeats"]]
    fold_values = [int(value) for value in config["evaluation"]["folds"]]
    include_sensory = bool(config["evaluation"]["include_frozen_q1_sensory"])
    n_jobs = int(config["runtime"].get("n_jobs", -1))

    identity = {
        "config_sha256": _sha256(config_path),
        "folds_sha256": _sha256(folds_path),
        "q1_tuning_sha256": _sha256(tuning_path),
        "q1_source_registry_sha256": _sha256(registry_path),
        "actionability_sha256": _sha256(actionability_path),
        "defects_sha256": _sha256(defects_path),
        "formula_sha256": _sha256(formula_path),
        "fold_frame_sha256": _frame_sha256(folds),
        "runner_source_sha256": _sha256(Path(__file__)),
        "migration_source_sha256": _sha256(Path(__file__).with_name("migration.py")),
    }
    completed: list[list[int]] = []
    resumed: list[list[int]] = []
    for repeat in repeats:
        for fold in fold_values:
            subrun = subrun_root / f"repeat_{repeat}_fold_{fold}"
            manifest_path = subrun / "manifest.json"
            required = [
                subrun / "candidates.parquet",
                subrun / "pareto.csv",
                subrun / "selected.parquet",
                subrun / "selected_metrics.csv",
                subrun / "selection_coverage.csv",
            ]
            if include_sensory:
                required.extend([subrun / "sensory_metrics.csv", subrun / "sensory_profiles.parquet"])
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing.get("identity") == identity and all(path.exists() for path in required):
                    resumed.append([repeat, fold])
                    print(f"resumed repeat={repeat} fold={fold}", flush=True)
                    continue
            subrun.mkdir(parents=True, exist_ok=True)
            result = _run_fold(
                data=data,
                folds=folds,
                tuning=tuning,
                registry=registry,
                actionability=actionability,
                defects=defects,
                formula_config=formula_config,
                migration_config=config["migration"],
                repeat=repeat,
                fold=fold,
                include_sensory=include_sensory,
                n_jobs=n_jobs,
            )
            _write_parquet_atomic(result["candidates"], subrun / "candidates.parquet")
            result["pareto"].to_csv(subrun / "pareto.csv", index=False)
            _write_parquet_atomic(result["selected"], subrun / "selected.parquet")
            result["selected_metrics"].to_csv(subrun / "selected_metrics.csv", index=False)
            result["coverage"].to_csv(subrun / "selection_coverage.csv", index=False)
            if include_sensory:
                result["sensory_metrics"].to_csv(subrun / "sensory_metrics.csv", index=False)
                _write_parquet_atomic(
                    result["sensory_profiles"], subrun / "sensory_profiles.parquet"
                )
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "repeat": repeat,
                        "fold": fold,
                        "source_rows": int(len(result["coverage"])),
                        "selected_rows": int(len(result["selected"])),
                        "identity": identity,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed.append([repeat, fold])
            print(f"completed repeat={repeat} fold={fold}", flush=True)

    subruns = [subrun_root / f"repeat_{r}_fold_{f}" for r in repeats for f in fold_values]
    candidates = pd.concat(
        [pd.read_parquet(path / "candidates.parquet") for path in subruns], ignore_index=True
    )
    pareto = pd.concat([pd.read_csv(path / "pareto.csv") for path in subruns], ignore_index=True)
    selected_samples = pd.concat(
        [pd.read_parquet(path / "selected.parquet") for path in subruns], ignore_index=True
    )
    selected_metrics = pd.concat(
        [pd.read_csv(path / "selected_metrics.csv") for path in subruns], ignore_index=True
    )
    coverage = pd.concat(
        [pd.read_csv(path / "selection_coverage.csv") for path in subruns], ignore_index=True
    )
    if include_sensory:
        sensory_metrics = pd.concat(
            [pd.read_csv(path / "sensory_metrics.csv") for path in subruns], ignore_index=True
        )
        sensory_profiles = pd.concat(
            [pd.read_parquet(path / "sensory_profiles.parquet") for path in subruns],
            ignore_index=True,
        )
    else:
        sensory_metrics = pd.DataFrame()
        sensory_profiles = pd.DataFrame()

    source_style = str(config["migration"]["source_style"])
    expected_rows = 0
    for repeat in repeats:
        rows = folds.loc[folds["repeat"] == repeat]
        valid_beers = set(rows.loc[rows["fold"].isin(fold_values), "beer"].astype(int))
        expected_rows += int(
            (
                data.ids.astype(int).isin(valid_beers)
                & data.styles.astype(str).eq(source_style)
            ).sum()
        )
    conditions, repeat_summary, source_summary, statistics = _aggregate_decision(
        selected_metrics,
        sensory_metrics,
        config,
        expected_rows=expected_rows,
    )
    passed = bool(all(conditions.values()))
    status = (
        "MIGRATION_CONFIRMATION_PASSED_WITH_Q1_LIMITATIONS"
        if passed
        else "MIGRATION_CONFIRMATION_REJECTED"
    )
    method_summary = (
        pareto.loc[pareto["champion_eligible_method"].astype(bool)]
        .groupby(["method", "budget"], as_index=False)
        .agg(
            rows=("candidate_id", "size"),
            selection_rate=("selected", "mean"),
            style_probability_gain_mean=("style_probability_gain", "mean"),
            target_distance_improvement_mean=("target_distance_improvement", "mean"),
            standardized_l1_mean=("standardized_l1", "mean"),
            near_neighbor_danger_rate=("near_neighbor_danger", "mean"),
        )
    )

    _write_parquet_atomic(candidates, output / "migration_candidates.parquet")
    pareto.to_csv(output / "migration_pareto.csv", index=False)
    _write_parquet_atomic(selected_samples, output / "migration_selected.parquet")
    selected_metrics.to_csv(output / "migration_selected_metrics.csv", index=False)
    coverage.to_csv(output / "migration_selection_coverage.csv", index=False)
    method_summary.to_csv(output / "migration_method_summary.csv", index=False)
    repeat_summary.to_csv(output / "migration_repeat_summary.csv", index=False)
    source_summary.to_csv(output / "migration_source_summary.csv", index=False)
    if include_sensory:
        sensory_metrics.to_csv(output / "migration_sensory_metrics.csv", index=False)
        _write_parquet_atomic(sensory_profiles, output / "migration_sensory_profiles.parquet")
    (output / "selection_audit.csv").write_text(
        pd.DataFrame([{**{"status": status, "criteria_passed": passed}, **conditions}]).to_csv(
            index=False
        ),
        encoding="utf-8",
    )
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "status": status,
        "criteria_passed": passed,
        "selection_conditions": conditions,
        "statistics": statistics,
        "repeats": repeats,
        "folds": fold_values,
        "completed_subruns": completed,
        "resumed_subruns": resumed,
        "outer_fold_isolation": True,
        "external_test_data_used": False,
        "q1_status_inherited": "HOLD",
        "q1_sensory_role": "fold_isolated_secondary_confirmation",
        "q1_fresh_repeat_base_parameter_rule": "frozen_fallback_not_retuned",
        "elapsed_seconds": float(time.perf_counter() - started),
        "identity": identity,
        "git_commit_at_run": _git_commit(project_root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "decision.md").write_text(
        "# Pils/Lager to Tripel repeated outer-fold confirmation\n\n"
        f"- Status: `{status}`.\n"
        f"- OOF migrations: {statistics['observed_oof_rows']}/{statistics['expected_oof_rows']}.\n"
        f"- Positive style gain rate: {statistics['style_positive_row_rate']:.3%}; repeat wins: {statistics['style_repeat_mean_wins']}/{len(repeats)}.\n"
        f"- Positive chemical target-distance rate: {statistics['chemistry_positive_row_rate']:.3%}; repeat wins: {statistics['chemistry_repeat_mean_wins']}/{len(repeats)}.\n"
        f"- Positive frozen-Q1 A/B sensory target-distance rate: {statistics['ab_sensory_positive_row_rate']:.3%}; repeat wins: {statistics['ab_sensory_repeat_mean_wins']}/{len(repeats)}.\n"
        f"- Mean standardized L1/L2: {statistics['mean_standardized_l1']:.4f}/{statistics['mean_standardized_l2']:.4f}.\n"
        f"- Mean changed fields/modules: {statistics['mean_changed_fields']:.2f}/{statistics['mean_changed_modules']:.2f}.\n\n"
        "All transforms, migration mechanisms, classifier fitting and frozen-Q1 fitting are outer-training-fold only. Public external test data are not used in this confirmation. Q1 remains HOLD even if the migration criteria pass.\n",
        encoding="utf-8",
    )
    return manifest

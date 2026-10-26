"""Fold-isolated Q1 sensory evaluation for a chemistry-confirmed generator."""

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
from scipy.stats import wasserstein_distance

from beer_q1.data import load_problem1_data
from beer_q2.reconstruct import (
    fit_representative_target_predictor,
    predict_frozen_targets,
)

from .smoke import _git_commit, _resolve, _sha256, _write_parquet_atomic


def _frame_sha256(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _robust_scale(values: pd.DataFrame) -> pd.Series:
    iqr = values.quantile(0.75) - values.quantile(0.25)
    scale = iqr / 1.349
    fallback = values.std(ddof=1).replace(0.0, np.nan)
    return scale.where(scale > 1e-12, fallback).fillna(1.0)


def _relative_scale(prediction: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(3.0 + (prediction - center) / np.maximum(scale, 1e-12), 0.0, 6.0)


def _reference_rows(
    training_predictions: pd.DataFrame,
    training_styles: pd.Series,
    target_style: str,
    desired_size: int,
    target_scale: pd.Series,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    pool = training_predictions.loc[
        training_styles.astype(str).to_numpy() == str(target_style)
    ].reset_index(drop=True)
    subset_size = min(int(desired_size), len(pool) // 2)
    if subset_size < 2:
        raise ValueError(f"Insufficient sensory reference support for {target_style}")
    rng = np.random.default_rng(seed)
    rows = []
    for resample in range(int(n_resamples)):
        order = rng.permutation(len(pool))
        left = pool.iloc[order[:subset_size]]
        right = pool.iloc[order[subset_size : 2 * subset_size]]
        for target in pool.columns:
            distance = wasserstein_distance(left[target], right[target]) / float(
                target_scale[target]
            )
            rows.append(
                {
                    "target_style": target_style,
                    "target": target,
                    "resample": resample,
                    "subset_size": subset_size,
                    "normalized_wasserstein": float(distance),
                }
            )
    return pd.DataFrame(rows)


def _profile_rows(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    role: str,
    grades: Mapping[str, str],
    target_center: pd.Series,
    target_scale: pd.Series,
    residual_quantiles: pd.DataFrame,
) -> pd.DataFrame:
    prediction = predictions.copy()
    prediction.insert(0, "sample_id", metadata["sample_id"].to_numpy())
    prediction.insert(1, "target_style", metadata["target_style"].to_numpy())
    long = prediction.melt(
        id_vars=["sample_id", "target_style"],
        var_name="target",
        value_name="standardized_prediction",
    )
    long["role"] = role
    long["confidence_grade"] = long["target"].map(grades)
    long["primary_ab_target"] = long["confidence_grade"].isin(["A", "B"])
    long = long.merge(residual_quantiles, on="target", how="left", validate="many_to_one")
    long["standardized_prediction_lower"] = (
        long["standardized_prediction"] + long["residual_q025"]
    )
    long["standardized_prediction_upper"] = (
        long["standardized_prediction"] + long["residual_q975"]
    )
    centers = long["target"].map(target_center).to_numpy(dtype=float)
    scales = long["target"].map(target_scale).to_numpy(dtype=float)
    for source, destination in (
        ("standardized_prediction", "relative_reconstructed_scale"),
        ("standardized_prediction_lower", "relative_reconstructed_scale_lower"),
        ("standardized_prediction_upper", "relative_reconstructed_scale_upper"),
    ):
        long[destination] = _relative_scale(
            long[source].to_numpy(dtype=float), centers, scales
        )
    return long


def _run_fold(
    *,
    data,
    folds: pd.DataFrame,
    tuning: pd.DataFrame,
    grades: Mapping[str, str],
    residual_quantiles: pd.DataFrame,
    candidate_samples: pd.DataFrame,
    baseline_samples: pd.DataFrame,
    candidate_method: str,
    baseline_method: str,
    target_styles: list[str],
    repeat: int,
    fold: int,
    n_reference: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows = folds.loc[folds["repeat"] == repeat]
    valid_beers = set(repeat_rows.loc[repeat_rows["fold"] == fold, "beer"].astype(int))
    valid_mask = data.ids.astype(int).isin(valid_beers).to_numpy()
    train_mask = ~valid_mask
    X_train = data.X.loc[train_mask].reset_index(drop=True)
    y_train = data.y.loc[train_mask].reset_index(drop=True)
    style_train = data.styles.loc[train_mask].reset_index(drop=True)
    X_valid = data.X.loc[valid_mask].reset_index(drop=True)
    style_valid = data.styles.loc[valid_mask].reset_index(drop=True)
    ids_valid = data.ids.loc[valid_mask].reset_index(drop=True)

    cache: dict[tuple[object, ...], object] = {}
    predictors = []
    for target in data.target_names:
        predictor, _, _ = fit_representative_target_predictor(
            data,
            folds,
            tuning,
            target,
            repeat,
            fold,
            model_cache=cache,
            n_jobs=n_jobs,
        )
        predictors.append(predictor)
    train_prediction = pd.DataFrame(
        predict_frozen_targets(predictors, X_train), columns=data.target_names
    )
    valid_prediction = pd.DataFrame(
        predict_frozen_targets(predictors, X_valid), columns=data.target_names
    )
    target_scale = _robust_scale(y_train)
    target_center = y_train.median()

    metric_rows = []
    reference_frames = []
    profile_frames = []
    for style_index, style in enumerate(target_styles):
        valid_style_mask = style_valid.astype(str).to_numpy() == style
        real_prediction = valid_prediction.loc[valid_style_mask].reset_index(drop=True)
        real_meta = pd.DataFrame(
            {
                "sample_id": [f"real_beer_{int(value)}" for value in ids_valid.loc[valid_style_mask]],
                "target_style": style,
            }
        )
        if real_prediction.empty:
            raise ValueError(f"No validation sensory rows for {repeat}/{fold}/{style}")
        reference = _reference_rows(
            train_prediction,
            style_train,
            style,
            len(real_prediction),
            target_scale,
            n_resamples=n_reference,
            seed=int(repeat_rows["seed"].iloc[0]) + fold * 10007 + style_index * 2027,
        )
        reference.insert(0, "fold", fold)
        reference.insert(0, "repeat", repeat)
        reference_frames.append(reference)
        lines = (
            reference.groupby("target")["normalized_wasserstein"]
            .agg(reference_median="median", reference_q95=lambda x: x.quantile(0.95))
        )

        for role, samples, method in (
            ("candidate_generated", candidate_samples, candidate_method),
            ("baseline_generated", baseline_samples, baseline_method),
        ):
            selected = samples.loc[
                (samples["repeat"] == repeat)
                & (samples["fold"] == fold)
                & (samples["target_style"].astype(str) == style)
                & (samples["method"].astype(str) == method)
            ].reset_index(drop=True)
            if len(selected) != len(real_prediction):
                raise ValueError(
                    f"Sample-count mismatch for {role}/{repeat}/{fold}/{style}: "
                    f"{len(selected)} versus {len(real_prediction)}"
                )
            generated_prediction = pd.DataFrame(
                predict_frozen_targets(
                    predictors, selected.loc[:, list(data.feature_names)]
                ),
                columns=data.target_names,
            )
            for target in data.target_names:
                value = wasserstein_distance(
                    generated_prediction[target], real_prediction[target]
                ) / float(target_scale[target])
                q95 = float(lines.loc[target, "reference_q95"])
                gap = max(0.0, (value - q95) / (abs(q95) + 1e-12))
                metric_rows.append(
                    {
                        "method": method,
                        "role": role,
                        "repeat": repeat,
                        "fold": fold,
                        "target_style": style,
                        "target": target,
                        "confidence_grade": grades[target],
                        "primary_ab_target": grades[target] in {"A", "B"},
                        "normalized_wasserstein": float(value),
                        "reference_median": float(lines.loc[target, "reference_median"]),
                        "reference_q95": q95,
                        "relative_gap": float(gap),
                        "status": "PASS" if gap == 0.0 else "NEAR" if gap < 0.05 else "FAIL",
                    }
                )
            meta = selected.loc[:, ["sample_id", "target_style"]]
            profile_frames.append(
                _profile_rows(
                    generated_prediction,
                    meta,
                    role,
                    grades,
                    target_center,
                    target_scale,
                    residual_quantiles,
                )
            )
        profile_frames.append(
            _profile_rows(
                real_prediction,
                real_meta,
                "validation_real",
                grades,
                target_center,
                target_scale,
                residual_quantiles,
            )
        )

    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles.insert(0, "fold", fold)
    profiles.insert(0, "repeat", repeat)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(reference_frames, ignore_index=True),
        profiles,
    )


def run_sensory_evaluation(project_root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
    folds = pd.read_csv(_resolve(project_root, config["interfaces"]["folds_csv"]))
    tuning = pd.read_csv(_resolve(project_root, config["interfaces"]["q1_tuning_csv"]))
    registry = pd.read_csv(_resolve(project_root, config["interfaces"]["q1_source_registry_csv"]))
    grades = dict(zip(registry["target"], registry["grade"]))
    oof = pd.read_parquet(_resolve(project_root, config["interfaces"]["q1_oof_predictions_parquet"]))
    oof = oof.loc[oof["model"] == "final_specialist_hybrid_v2"].copy()
    oof["residual"] = oof["y_true"] - oof["y_pred"]
    residual_quantiles = (
        oof.groupby("target")["residual"]
        .agg(
            residual_q025=lambda x: x.quantile(0.025),
            residual_q975=lambda x: x.quantile(0.975),
        )
        .reset_index()
    )

    candidate_root = _resolve(project_root, config["interfaces"]["candidate_root"])
    baseline_root = _resolve(project_root, config["interfaces"]["baseline_root"])
    candidate_samples = pd.read_parquet(candidate_root / "generated_samples.parquet")
    baseline_samples = pd.read_parquet(baseline_root / "generated_samples.parquet")
    candidate_method = str(config["methods"]["candidate"])
    baseline_method = str(config["methods"]["baseline"])
    repeats = [int(value) for value in config["evaluation"]["repeats"]]
    fold_values = [int(value) for value in config["evaluation"]["folds"]]
    target_styles = [str(value) for value in config["styles"]["generation_targets"]]
    n_reference = int(config["evaluation"]["real_real_resamples"])
    n_jobs = int(config["runtime"].get("n_jobs", -1))

    identity = {
        "config_sha256": _sha256(config_path),
        "candidate_samples_sha256": _sha256(candidate_root / "generated_samples.parquet"),
        "baseline_samples_sha256": _sha256(baseline_root / "generated_samples.parquet"),
        "folds_sha256": _frame_sha256(folds),
        "source_registry_sha256": _frame_sha256(registry),
    }
    completed = []
    resumed = []
    for repeat in repeats:
        for fold in fold_values:
            subrun = subrun_root / f"repeat_{repeat}_fold_{fold}"
            manifest_path = subrun / "manifest.json"
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                required = [subrun / "metrics.csv", subrun / "reference.csv", subrun / "profiles.parquet"]
                if existing.get("identity") == identity and all(path.exists() for path in required):
                    resumed.append([repeat, fold])
                    continue
            subrun.mkdir(parents=True, exist_ok=True)
            metrics, reference, profiles = _run_fold(
                data=data,
                folds=folds,
                tuning=tuning,
                grades=grades,
                residual_quantiles=residual_quantiles,
                candidate_samples=candidate_samples,
                baseline_samples=baseline_samples,
                candidate_method=candidate_method,
                baseline_method=baseline_method,
                target_styles=target_styles,
                repeat=repeat,
                fold=fold,
                n_reference=n_reference,
                n_jobs=n_jobs,
            )
            metrics.to_csv(subrun / "metrics.csv", index=False)
            reference.to_csv(subrun / "reference.csv", index=False)
            _write_parquet_atomic(profiles, subrun / "profiles.parquet")
            manifest_path.write_text(
                json.dumps(
                    {"status": "COMPLETE", "repeat": repeat, "fold": fold, "identity": identity},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed.append([repeat, fold])

    subruns = [subrun_root / f"repeat_{r}_fold_{f}" for r in repeats for f in fold_values]
    metrics = pd.concat([pd.read_csv(path / "metrics.csv") for path in subruns], ignore_index=True)
    references = pd.concat([pd.read_csv(path / "reference.csv") for path in subruns], ignore_index=True)
    profiles = pd.concat([pd.read_parquet(path / "profiles.parquet") for path in subruns], ignore_index=True)
    primary = metrics.loc[metrics["primary_ab_target"]].copy()
    summary = (
        primary.groupby(["method", "role"], as_index=False)
        .agg(
            n_rows=("target", "size"),
            pass_rate=("status", lambda x: x.eq("PASS").mean()),
            mean_normalized_wasserstein=("normalized_wasserstein", "mean"),
            median_normalized_wasserstein=("normalized_wasserstein", "median"),
        )
    )
    style_summary = (
        primary.groupby(["method", "role", "target_style"], as_index=False)
        .agg(mean_normalized_wasserstein=("normalized_wasserstein", "mean"))
    )
    repeat_summary = (
        primary.groupby(["method", "role", "repeat"], as_index=False)
        .agg(mean_normalized_wasserstein=("normalized_wasserstein", "mean"))
    )
    indexed = summary.set_index("role")
    candidate = indexed.loc["candidate_generated"]
    baseline = indexed.loc["baseline_generated"]
    style_pivot = style_summary.pivot(index="target_style", columns="role", values="mean_normalized_wasserstein")
    style_pivot["candidate_to_baseline_ratio"] = (
        style_pivot["candidate_generated"] / style_pivot["baseline_generated"]
    )
    repeat_pivot = repeat_summary.pivot(
        index="repeat", columns="role", values="mean_normalized_wasserstein"
    )
    repeat_pivot["candidate_improved"] = (
        repeat_pivot["candidate_generated"]
        < repeat_pivot["baseline_generated"] - 1e-12
    )
    required_improving_styles = [
        str(value)
        for value in config.get("selection", {}).get(
            "required_improving_styles", []
        )
    ]
    missing_required_styles = sorted(
        set(required_improving_styles).difference(style_pivot.index.astype(str))
    )
    if missing_required_styles:
        raise ValueError(
            f"Required improving styles are absent: {missing_required_styles}"
        )
    minimum_repeat_wins = int(
        config.get("selection", {}).get("minimum_repeat_wins", 0)
    )
    conditions = {
        "complete_finite_primary_schema": bool(
            len(primary) == len(repeats) * len(fold_values) * len(target_styles) * 24 * 2
            and np.isfinite(primary["normalized_wasserstein"]).all()
        ),
        "candidate_primary_row_count_expected": bool(
            len(primary.loc[primary["role"] == "candidate_generated"])
            == len(repeats) * len(fold_values) * len(target_styles) * 24
        ),
        "pass_rate_protected": bool(candidate["pass_rate"] >= baseline["pass_rate"] - 1e-12),
        "mean_distance_protected": bool(
            candidate["mean_normalized_wasserstein"]
            <= baseline["mean_normalized_wasserstein"] + 1e-12
        ),
        "all_styles_within_five_percent": bool(
            (style_pivot["candidate_to_baseline_ratio"] <= 1.05 + 1e-12).all()
        ),
        "required_styles_improved": bool(
            (
                style_pivot.loc[
                    required_improving_styles, "candidate_to_baseline_ratio"
                ]
                <= 1.0 + 1e-12
            ).all()
        ) if required_improving_styles else True,
        "repeat_wins_sufficient": bool(
            int(repeat_pivot["candidate_improved"].sum()) >= minimum_repeat_wins
        ),
        "c_targets_audit_only": bool(
            not metrics.loc[~metrics["primary_ab_target"], "confidence_grade"].isin(["A", "B"]).any()
        ),
    }
    passed = bool(all(conditions.values()))
    audit_only = bool(config.get("runtime", {}).get("audit_only", False))
    status = (
        "SENSORY_EXPLORATORY_AUDIT_COMPLETE"
        if audit_only
        else "SENSORY_CONFIRMATION_PASSED"
        if passed
        else "SENSORY_CONFIRMATION_REJECTED"
    )

    metrics.to_csv(output / "sensory_metrics_by_fold.csv", index=False)
    references.to_csv(output / "sensory_real_real_reference.csv", index=False)
    summary.to_csv(output / "sensory_summary.csv", index=False)
    style_pivot.reset_index().to_csv(output / "sensory_style_summary.csv", index=False)
    repeat_pivot.reset_index().to_csv(
        output / "sensory_repeat_summary.csv", index=False
    )
    _write_parquet_atomic(profiles, output / "sensory_profiles.parquet")
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    audit = pd.DataFrame([conditions])
    audit.insert(0, "criteria_passed", passed)
    audit.insert(0, "audit_only", audit_only)
    audit.insert(0, "status", status)
    audit.to_csv(output / "selection_audit.csv", index=False)
    manifest = {
        "status": status,
        "candidate_method": candidate_method,
        "baseline_method": baseline_method,
        "repeats": repeats,
        "folds": fold_values,
        "completed_subruns": completed,
        "resumed_subruns": resumed,
        "selection_conditions": conditions,
        "criteria_passed": passed,
        "audit_only": audit_only,
        "champion_eligible": bool(passed and not audit_only),
        "required_improving_styles": required_improving_styles,
        "minimum_repeat_wins": minimum_repeat_wins,
        "q1_status_inherited": "HOLD",
        "external_test_data_status": "cohorts_locked_not_scored",
        "migration_status": "not_run",
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
    decision_title = (
        "Q3 sensory exploratory audit" if audit_only else "Q3 sensory confirmation decision"
    )
    interpretation = (
        "- This run is decision-support only because the upstream chemistry pre-registration did not pass; it cannot confer champion status.\n"
        if audit_only
        else ""
    )
    (output / "decision.md").write_text(
        f"# {decision_title}\n\n"
        f"- Status: `{status}`.\n"
        f"- Frozen sensory criteria passed: `{passed}`.\n"
        f"- Candidate A/B PASS rate: {candidate['pass_rate']:.6f}; comparator: {baseline['pass_rate']:.6f}.\n"
        f"- Candidate mean normalized Wasserstein: {candidate['mean_normalized_wasserstein']:.6f}; comparator: {baseline['mean_normalized_wasserstein']:.6f}.\n"
        f"- Worst style candidate/comparator distance ratio: {style_pivot['candidate_to_baseline_ratio'].max():.6f}.\n"
        f"{interpretation}"
        "- Six C-grade targets remain audit-only; Q1 remains HOLD and migration is not run.\n",
        encoding="utf-8",
    )
    return manifest

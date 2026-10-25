"""Resumable shared-prediction runner for formal Q2 permutation evidence.

The frozen model is unchanged. Speed comes from evaluating each shared fitted
model once per permutation batch, vectorizing metrics across permutations, and
streaming one parquet row group at a time instead of retaining raw results in
memory.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import yaml

from beer_q1.data import load_problem1_data

from .experiment import _q1_data_config, _resolve, _validate_output_boundary
from .explain import (
    _permutation_groups,
    batched_regression_metrics,
    permute_values_within_groups,
    regression_metrics,
    safe_spearman,
    summarize_fold_importance,
)
from .preflight import sha256_file
from .reconstruct import (
    _align_frozen_rows,
    fit_representative_target_predictor,
    predict_frozen_targets,
)


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _permutation_depths(config: Dict[str, Any]) -> Dict[str, int]:
    modes = list(map(str, config["explanation"]["modes"]))
    configured = config["explanation"].get("permutations_by_mode")
    if configured is None:
        depth = int(config["explanation"]["permutations"])
        return {mode: depth for mode in modes}
    result = {mode: int(configured[mode]) for mode in modes}
    if any(value <= 0 for value in result.values()):
        raise ValueError("Every permutation mode needs a positive depth")
    return result


def _open_stream(path: Path):
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    state: Dict[str, Any] = {"writer": None, "temporary": temporary, "final": path}

    def write(frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if state["writer"] is None:
            state["writer"] = pq.ParquetWriter(
                temporary, table.schema, compression="zstd"
            )
        state["writer"].write_table(table)

    def close() -> None:
        if state["writer"] is None:
            raise ValueError("Permutation stream produced no rows")
        state["writer"].close()
        temporary.replace(path)

    return write, close


def _run_fold(
    data,
    folds: pd.DataFrame,
    tuning: pd.DataFrame,
    frozen: pd.DataFrame,
    registry: pd.DataFrame,
    targets: Sequence[str],
    repeat: int,
    fold: int,
    config: Dict[str, Any],
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    started = time.perf_counter()
    cache = {}
    predictors = []
    train_index = valid_index = None
    n_jobs = int(config["runtime"].get("n_jobs", 1))
    for target in targets:
        predictor, current_train, current_valid = fit_representative_target_predictor(
            data,
            folds,
            tuning,
            target,
            repeat,
            fold,
            model_cache=cache,
            n_jobs=n_jobs,
        )
        if train_index is None:
            train_index, valid_index = current_train, current_valid
        elif not (
            np.array_equal(train_index, current_train)
            and np.array_equal(valid_index, current_valid)
        ):
            raise ValueError("Targets did not resolve to the same frozen outer fold")
        predictors.append(predictor)
    assert train_index is not None and valid_index is not None

    X_valid = data.X.iloc[valid_index].copy()
    y_valid = data.y.iloc[valid_index][list(targets)].to_numpy(dtype=float)
    styles = data.styles.iloc[valid_index].astype(str).to_numpy()
    beers = data.ids.iloc[valid_index].to_numpy(dtype=int)
    baseline_prediction = predict_frozen_targets(predictors, X_valid)
    baseline_metrics = [
        regression_metrics(y_valid[:, index], baseline_prediction[:, index])
        for index in range(len(targets))
    ]
    grades = [str(registry.loc[target, "grade"]) for target in targets]
    sources = [predictor.source_model for predictor in predictors]

    audits = []
    tolerance = float(config["explanation"]["reconstruction_tolerance"])
    for index, target in enumerate(targets):
        frozen_rows = _align_frozen_rows(frozen, target, repeat, fold)
        if not np.array_equal(beers, frozen_rows["beer"].to_numpy(dtype=int)):
            raise ValueError(f"Frozen beer order mismatch for {target} r{repeat}f{fold}")
        error = baseline_prediction[:, index] - frozen_rows["y_pred"].to_numpy(
            dtype=float
        )
        max_error = float(np.max(np.abs(error)))
        if max_error > tolerance:
            raise ValueError(
                f"Frozen reconstruction failed for {target} r{repeat}f{fold}: {max_error}"
            )
        audits.append(
            {
                "repeat": repeat,
                "fold": fold,
                "target": target,
                "grade": grades[index],
                "source_model": sources[index],
                "n_train": int(len(train_index)),
                "n_valid": int(len(valid_index)),
                "baseline_rmse": baseline_metrics[index]["rmse"],
                "baseline_r2": baseline_metrics[index]["r2"],
                "baseline_spearman": baseline_metrics[index]["spearman"],
                "reconstruction_max_abs_error": max_error,
                "reconstruction_passed": True,
            }
        )

    write_chunk, close_stream = _open_stream(raw_path)
    fold_rows: List[Dict[str, Any]] = []
    depths = _permutation_depths(config)
    base_values = X_valid.to_numpy(copy=True)
    columns = list(X_valid.columns)
    seed_base = int(config["explanation"]["permutation_seed"])
    minimum_group_n = int(config["explanation"]["style_minimum_group_n"])
    total_features = len(columns)
    rows_written = 0

    for mode_index, mode in enumerate(config["explanation"]["modes"]):
        mode = str(mode)
        permutations = depths[mode]
        groups = _permutation_groups(styles, mode, minimum_group_n)
        for feature_index, feature in enumerate(columns):
            feature_values = base_values[:, feature_index]
            blocks = []
            for permutation in range(permutations):
                seed = (
                    seed_base
                    + repeat * 1_000_003
                    + fold * 100_003
                    + mode_index * 10_007
                    + feature_index * 101
                    + permutation
                )
                block = base_values.copy()
                block[:, feature_index] = permute_values_within_groups(
                    feature_values, groups, np.random.default_rng(seed)
                )
                blocks.append(block)
            batch = pd.DataFrame(np.concatenate(blocks, axis=0), columns=columns)
            predictions = predict_frozen_targets(predictors, batch).reshape(
                permutations, len(X_valid), len(targets)
            )
            directions = np.asarray(
                [
                    safe_spearman(feature_values, y_valid[:, target_index])
                    for target_index in range(len(targets))
                ],
                dtype=float,
            )
            metric_arrays = {
                "rmse": np.empty((len(targets), permutations), dtype=float),
                "r2": np.empty((len(targets), permutations), dtype=float),
                "spearman": np.empty((len(targets), permutations), dtype=float),
            }
            for target_index in range(len(targets)):
                metrics = batched_regression_metrics(
                    y_valid[:, target_index], predictions[:, :, target_index]
                )
                for name in metric_arrays:
                    metric_arrays[name][target_index] = metrics[name]

            baseline_rmse = np.asarray(
                [item["rmse"] for item in baseline_metrics], dtype=float
            )
            baseline_r2 = np.asarray(
                [item["r2"] for item in baseline_metrics], dtype=float
            )
            baseline_spearman = np.asarray(
                [item["spearman"] for item in baseline_metrics], dtype=float
            )
            delta_rmse = metric_arrays["rmse"] - baseline_rmse[:, None]
            relative_delta = np.divide(
                delta_rmse,
                baseline_rmse[:, None],
                out=np.full_like(delta_rmse, np.nan),
                where=baseline_rmse[:, None] > 0,
            )
            chunk = pd.DataFrame(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "target": np.repeat(np.asarray(targets, dtype=object), permutations),
                    "grade": np.repeat(np.asarray(grades, dtype=object), permutations),
                    "source_model": np.repeat(
                        np.asarray(sources, dtype=object), permutations
                    ),
                    "mode": mode,
                    "feature": feature,
                    "permutation": np.tile(np.arange(permutations), len(targets)),
                    "n_train": int(len(train_index)),
                    "n_valid": int(len(valid_index)),
                    "baseline_rmse": np.repeat(baseline_rmse, permutations),
                    "baseline_r2": np.repeat(baseline_r2, permutations),
                    "baseline_spearman": np.repeat(
                        baseline_spearman, permutations
                    ),
                    "permuted_rmse": metric_arrays["rmse"].reshape(-1),
                    "permuted_r2": metric_arrays["r2"].reshape(-1),
                    "permuted_spearman": metric_arrays["spearman"].reshape(-1),
                    "delta_rmse": delta_rmse.reshape(-1),
                    "relative_delta_rmse": relative_delta.reshape(-1),
                    "delta_r2_loss": (
                        baseline_r2[:, None] - metric_arrays["r2"]
                    ).reshape(-1),
                    "delta_spearman_loss": (
                        baseline_spearman[:, None] - metric_arrays["spearman"]
                    ).reshape(-1),
                    "heldout_feature_target_spearman": np.repeat(
                        directions, permutations
                    ),
                }
            )
            write_chunk(chunk)
            rows_written += len(chunk)
            for target_index, target in enumerate(targets):
                fold_rows.append(
                    {
                        "target": target,
                        "grade": grades[target_index],
                        "source_model": sources[target_index],
                        "mode": mode,
                        "feature": feature,
                        "repeat": repeat,
                        "fold": fold,
                        "mean_delta_rmse": float(delta_rmse[target_index].mean()),
                        "median_delta_rmse": float(
                            np.median(delta_rmse[target_index])
                        ),
                        "mean_relative_delta_rmse": float(
                            relative_delta[target_index].mean()
                        ),
                        "positive_permutation_frequency": float(
                            (delta_rmse[target_index] > 0).mean()
                        ),
                        "heldout_feature_target_spearman": directions[target_index],
                    }
                )
            if (feature_index + 1) % 25 == 0 or feature_index + 1 == total_features:
                print(
                    f"repeat={repeat} fold={fold} mode={mode}: "
                    f"{feature_index + 1}/{total_features} features",
                    flush=True,
                )
    close_stream()
    elapsed = time.perf_counter() - started
    diagnostics = {
        "repeat": repeat,
        "fold": fold,
        "elapsed_seconds": elapsed,
        "raw_rows": rows_written,
        "unique_fitted_models": len(cache),
        "configured_targets": len(targets),
        "permutations_by_mode": depths,
    }
    return pd.DataFrame(fold_rows), pd.DataFrame(audits), diagnostics


def run_formal_fast(project_root: Path, config_path: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = _resolve(root, config["runtime"]["output_root"])
    _validate_output_boundary(root, output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "config.resolved.yaml"
    if resolved_path.exists():
        existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Existing resumable output uses a different configuration")
    else:
        resolved_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    data = load_problem1_data(_q1_data_config(root, config))
    folds_path = _resolve(root, config["frozen_q1"]["folds"])
    tuning_path = _resolve(root, config["frozen_q1"]["tuning_trials"])
    oof_path = _resolve(root, config["frozen_q1"]["oof_predictions"])
    registry_path = _resolve(root, config["frozen_q1"]["source_registry"])
    folds = pd.read_csv(folds_path)
    tuning = pd.read_csv(tuning_path)
    frozen = pd.read_parquet(oof_path)
    registry = pd.read_csv(registry_path).set_index("target")
    configured_targets = config["experiment"]["targets"]
    targets = (
        list(map(str, data.main_target_names))
        if configured_targets == "all_main"
        else list(map(str, configured_targets))
    )
    repeats = list(map(int, config["experiment"]["repeats"]))
    outer_folds = list(map(int, config["experiment"]["outer_folds"]))

    raw_dir = output / "oof_permutation_importance_parts"
    fold_dir = output / "importance_fold_level_parts"
    audit_dir = output / "reconstruction_audit_parts"
    raw_dir.mkdir(exist_ok=True)
    fold_dir.mkdir(exist_ok=True)
    audit_dir.mkdir(exist_ok=True)
    diagnostics_path = output / "fold_runtime.json"
    diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.exists()
        else {}
    )

    for repeat in repeats:
        for fold in outer_folds:
            key = f"r{repeat}_f{fold}"
            raw_path = raw_dir / f"{key}.parquet"
            fold_path = fold_dir / f"{key}.parquet"
            audit_path = audit_dir / f"{key}.parquet"
            if raw_path.exists() and fold_path.exists() and audit_path.exists():
                print(f"Skipping completed {key}", flush=True)
                continue
            fold_level, audit, fold_diagnostics = _run_fold(
                data,
                folds,
                tuning,
                frozen,
                registry,
                targets,
                repeat,
                fold,
                config,
                raw_path,
            )
            fold_level.to_parquet(fold_path.with_suffix(".parquet.partial"), index=False)
            fold_path.with_suffix(".parquet.partial").replace(fold_path)
            audit.to_parquet(audit_path.with_suffix(".parquet.partial"), index=False)
            audit_path.with_suffix(".parquet.partial").replace(audit_path)
            diagnostics[key] = fold_diagnostics
            _write_json_atomic(diagnostics_path, diagnostics)

    expected_keys = {f"r{repeat}_f{fold}" for repeat in repeats for fold in outer_folds}
    completed_keys = {
        path.stem for path in raw_dir.glob("r*_f*.parquet")
        if (fold_dir / path.name).exists() and (audit_dir / path.name).exists()
    }
    missing = sorted(expected_keys - completed_keys)
    if missing:
        raise ValueError(f"Formal chunk remains incomplete: {missing}")

    fold_level = pd.concat(
        [pd.read_parquet(fold_dir / f"{key}.parquet") for key in sorted(expected_keys)],
        ignore_index=True,
    )
    reconstruction = pd.concat(
        [pd.read_parquet(audit_dir / f"{key}.parquet") for key in sorted(expected_keys)],
        ignore_index=True,
    )
    stability, view = summarize_fold_importance(
        fold_level,
        primary_mode=str(config["explanation"]["primary_mode"]),
        top_k=config["explanation"]["top_k"],
        top10_frequency_minimum=float(
            config["stability"]["top10_frequency_minimum"]
        ),
        positive_importance_frequency_minimum=float(
            config["stability"]["positive_importance_frequency_minimum"]
        ),
        direction_consistency_minimum=float(
            config["stability"]["direction_consistency_minimum"]
        ),
    )
    fold_level.to_parquet(output / "importance_fold_level.parquet", index=False)
    reconstruction.to_csv(output / "reconstruction_audit.csv", index=False)
    stability.to_csv(output / "importance_stability.csv", index=False)
    view.to_csv(output / "pilot_top10_view.csv", index=False)
    folds[folds["repeat"].isin(repeats)].to_csv(output / "folds.csv", index=False)

    depths = _permutation_depths(config)
    expected_rows = int(
        len(targets)
        * len(outer_folds)
        * len(data.feature_names)
        * sum(depths.values())
        * len(repeats)
    )
    actual_rows = int(sum(int(diagnostics[key]["raw_rows"]) for key in expected_keys))
    audit = {
        "status": "PASS",
        "run_name": config["experiment"]["run_name"],
        "scope": config["experiment"].get("scope", "formal_explanation_chunk"),
        "targets": targets,
        "repeats": repeats,
        "outer_folds": outer_folds,
        "n_samples": int(len(data.X)),
        "n_features": int(len(data.feature_names)),
        "permutations_by_mode": depths,
        "expected_raw_rows": expected_rows,
        "actual_raw_rows": actual_rows,
        "all_reconstructions_passed": bool(
            reconstruction["reconstruction_passed"].all()
        ),
        "max_reconstruction_abs_error": float(
            reconstruction["reconstruction_max_abs_error"].max()
        ),
        "resumable_fold_parts": True,
        "shared_model_prediction": True,
        "formal_top3_generated": False,
        "input_sha256": {
            "chemicals_csv": sha256_file(_resolve(root, config["data"]["chemicals_csv"])),
            "sensory_csv": sha256_file(_resolve(root, config["data"]["sensory_csv"])),
            "folds": sha256_file(folds_path),
            "frozen_oof": sha256_file(oof_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": pa.__version__,
        },
    }
    if actual_rows != expected_rows:
        raise ValueError(f"Raw row count mismatch: {actual_rows} != {expected_rows}")
    _write_json_atomic(output / "data_audit.json", audit)
    return audit

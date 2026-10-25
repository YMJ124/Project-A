"""Resumable formal TreeSHAP and correlated-community explanation runner.

Each frozen outer fold is an atomic checkpoint. Raw SHAP values and community
permutation rows are streamed to fold-level Parquet files, while compact fold
summaries support final stability and method-agreement calculations without
loading the full raw explanation table into memory.
"""

from __future__ import annotations

import gc
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
import shap
import sklearn
import yaml

from beer_q1.data import load_problem1_data

from .advanced import _style_heterogeneity, grouped_permutation_source_indices, true_pipeline_shap
from .experiment import _q1_data_config, _resolve, _validate_output_boundary
from .explain import _permutation_groups, batched_regression_metrics, safe_spearman
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


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


class _ParquetStream:
    def __init__(self, path: Path):
        self.final = path
        self.temporary = path.with_suffix(path.suffix + ".partial")
        if self.temporary.exists():
            self.temporary.unlink()
        self.writer = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary, table.schema, compression="zstd"
            )
        self.writer.write_table(table)
        self.rows += len(frame)

    def finish(self) -> None:
        if self.writer is None:
            raise ValueError(f"Stream produced no rows: {self.final}")
        self.writer.close()
        self.writer = None
        self.temporary.replace(self.final)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _load_network_nodes(root: Path, config: Dict[str, Any]) -> pd.DataFrame:
    configured = config.get("network_nodes_paths")
    if configured is None:
        configured = [config["network_nodes_path"]]
    frames = [pd.read_csv(_resolve(root, str(path))) for path in configured]
    nodes = pd.concat(frames, ignore_index=True)
    variant = config.get("network_variant")
    if variant is not None:
        if "variant" not in nodes.columns:
            raise ValueError("network_variant requested but node table has no variant")
        nodes = nodes[nodes["variant"] == str(variant)].copy()
    duplicated = nodes.duplicated(["repeat", "fold", "feature"])
    if duplicated.any():
        raise ValueError("Network node inputs contain duplicate repeat-fold features")
    return nodes


def _shap_fold_frame(
    values: np.ndarray,
    X_valid: pd.DataFrame,
    target: str,
    grade: str,
    source_model: str,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    rows = []
    for feature_index, feature in enumerate(X_valid.columns):
        rows.append(
            {
                "target": target,
                "grade": grade,
                "source_model": source_model,
                "feature": feature,
                "repeat": repeat,
                "fold": fold,
                "mean_abs_shap": float(np.mean(np.abs(values[:, feature_index]))),
                "direction_rho": safe_spearman(
                    X_valid.iloc[:, feature_index], values[:, feature_index]
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_fold(
    data,
    folds: pd.DataFrame,
    tuning: pd.DataFrame,
    frozen: pd.DataFrame,
    registry: pd.DataFrame,
    network_nodes: pd.DataFrame,
    targets: Sequence[str],
    repeat: int,
    fold: int,
    config: Dict[str, Any],
    shap_path: Path,
    group_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    started = time.perf_counter()
    model_cache: Dict[tuple, object] = {}
    predictors = []
    train_index = valid_index = None
    n_jobs = int(config.get("n_jobs", 1))
    for target in targets:
        predictor, current_train, current_valid = fit_representative_target_predictor(
            data,
            folds,
            tuning,
            target,
            repeat,
            fold,
            model_cache=model_cache,
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
    grades = [str(registry.loc[target, "grade"]) for target in targets]
    predictions = predict_frozen_targets(predictors, X_valid)
    tolerance = float(config["reconstruction_tolerance"])

    reconstruction_errors = []
    for target_index, target in enumerate(targets):
        frozen_rows = _align_frozen_rows(frozen, target, repeat, fold)
        if not np.array_equal(beers, frozen_rows["beer"].to_numpy(dtype=int)):
            raise ValueError(f"Frozen beer order mismatch for {target} r{repeat}f{fold}")
        error = float(
            np.max(
                np.abs(
                    predictions[:, target_index]
                    - frozen_rows["y_pred"].to_numpy(dtype=float)
                )
            )
        )
        if error > tolerance:
            raise ValueError(
                f"Frozen reconstruction failed for {target} r{repeat}f{fold}: {error}"
            )
        reconstruction_errors.append(error)

    shap_stream = _ParquetStream(shap_path)
    group_stream = _ParquetStream(group_path)
    shap_cache: Dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    fold_frames: List[pd.DataFrame] = []
    audits: List[Dict[str, Any]] = []
    try:
        feature_values = X_valid.to_numpy()
        n_features = len(data.feature_names)
        for target_index, (target, predictor) in enumerate(zip(targets, predictors)):
            values, expected, additivity_error = true_pipeline_shap(
                predictor, X_valid, shap_cache=shap_cache
            )
            if additivity_error > 1e-8:
                raise ValueError(
                    f"TreeSHAP additivity failed for {target} r{repeat}f{fold}: "
                    f"{additivity_error}"
                )
            raw = pd.DataFrame(
                {
                    "repeat": np.repeat(repeat, len(X_valid) * n_features),
                    "fold": np.repeat(fold, len(X_valid) * n_features),
                    "target": np.repeat(target, len(X_valid) * n_features),
                    "grade": np.repeat(grades[target_index], len(X_valid) * n_features),
                    "source_model": np.repeat(
                        predictor.source_model, len(X_valid) * n_features
                    ),
                    "beer": np.repeat(beers, n_features),
                    "style": np.repeat(styles, n_features),
                    "feature": np.tile(data.feature_names, len(X_valid)),
                    "feature_value": feature_values.reshape(-1),
                    "shap_value": values.reshape(-1),
                    "abs_shap_value": np.abs(values).reshape(-1),
                }
            )
            shap_stream.write(raw)
            fold_frames.append(
                _shap_fold_frame(
                    values,
                    X_valid,
                    target,
                    grades[target_index],
                    predictor.source_model,
                    repeat,
                    fold,
                )
            )
            audits.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "target": target,
                    "grade": grades[target_index],
                    "source_model": predictor.source_model,
                    "expected_value": expected,
                    "shap_additivity_max_abs_error": additivity_error,
                    "reconstruction_max_abs_error": reconstruction_errors[
                        target_index
                    ],
                }
            )

        communities = network_nodes[
            (network_nodes["repeat"] == repeat)
            & (network_nodes["fold"] == fold)
        ][["feature", "community"]]
        if len(communities) != len(data.feature_names):
            raise ValueError(
                f"Expected {len(data.feature_names)} nodes for r{repeat}f{fold}; "
                f"got {len(communities)}"
            )
        baseline_metrics = [
            batched_regression_metrics(
                y_valid[:, target_index], predictions[:, target_index]
            )
            for target_index in range(len(targets))
        ]
        baseline_rmse = np.asarray(
            [item["rmse"][0] for item in baseline_metrics], dtype=float
        )
        baseline_r2 = np.asarray(
            [item["r2"][0] for item in baseline_metrics], dtype=float
        )
        baseline_spearman = np.asarray(
            [item["spearman"][0] for item in baseline_metrics], dtype=float
        )
        style_groups = _permutation_groups(
            styles, "within_style", int(config["style_minimum_group_n"])
        )
        permutations = int(config["group_permutations"])
        seed_base = int(config["group_permutation_seed"])
        base_values = X_valid.to_numpy(copy=True)
        columns = list(X_valid.columns)
        feature_to_index = {feature: index for index, feature in enumerate(columns)}
        for community_index, (community, group) in enumerate(
            communities.groupby("community", sort=True)
        ):
            features = [str(feature) for feature in group["feature"]]
            indices = [feature_to_index[feature] for feature in features]
            blocks = []
            for permutation in range(permutations):
                rng = np.random.default_rng(
                    seed_base
                    + repeat * 1_000_003
                    + fold * 100_003
                    + int(community) * 101
                    + permutation
                )
                source = grouped_permutation_source_indices(style_groups, rng)
                block = base_values.copy()
                block[:, indices] = base_values[source][:, indices]
                blocks.append(block)
            batch = pd.DataFrame(np.concatenate(blocks, axis=0), columns=columns)
            community_predictions = predict_frozen_targets(
                predictors, batch
            ).reshape(permutations, len(X_valid), len(targets))
            rows = []
            for target_index, target in enumerate(targets):
                metrics = batched_regression_metrics(
                    y_valid[:, target_index],
                    community_predictions[:, :, target_index],
                )
                delta_rmse = metrics["rmse"] - baseline_rmse[target_index]
                for permutation in range(permutations):
                    rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "target": target,
                            "source_model": predictors[target_index].source_model,
                            "community": int(community),
                            "community_size": len(features),
                            "community_features": "|".join(features),
                            "permutation": permutation,
                            "baseline_rmse": baseline_rmse[target_index],
                            "permuted_rmse": metrics["rmse"][permutation],
                            "delta_rmse": delta_rmse[permutation],
                            "relative_delta_rmse": (
                                delta_rmse[permutation]
                                / baseline_rmse[target_index]
                                if baseline_rmse[target_index] > 0
                                else np.nan
                            ),
                            "delta_r2_loss": baseline_r2[target_index]
                            - metrics["r2"][permutation],
                            "delta_spearman_loss": baseline_spearman[target_index]
                            - metrics["spearman"][permutation],
                        }
                    )
            group_stream.write(pd.DataFrame(rows))
            if (community_index + 1) % 10 == 0:
                print(
                    f"repeat={repeat} fold={fold}: "
                    f"{community_index + 1}/{communities['community'].nunique()} communities",
                    flush=True,
                )
        shap_stream.finish()
        group_stream.finish()
    except Exception:
        shap_stream.abort()
        group_stream.abort()
        raise

    elapsed = time.perf_counter() - started
    diagnostics = {
        "repeat": repeat,
        "fold": fold,
        "elapsed_seconds": elapsed,
        "n_valid": len(valid_index),
        "shap_rows": shap_stream.rows,
        "group_rows": group_stream.rows,
        "unique_fitted_models": len(model_cache),
        "unique_shap_explanations": len(shap_cache),
        "communities": int(communities["community"].nunique()),
    }
    return pd.concat(fold_frames, ignore_index=True), pd.DataFrame(audits), diagnostics


def _shap_stability_from_fold(fold_level: pd.DataFrame, top_k: int) -> pd.DataFrame:
    frame = fold_level.copy()
    frame["shap_rank"] = frame.groupby(["target", "repeat", "fold"])[
        "mean_abs_shap"
    ].rank(method="min", ascending=False)
    rows = []
    for keys, group in frame.groupby(
        ["target", "grade", "source_model", "feature"], sort=False
    ):
        target, grade, source_model, feature = keys
        signs = np.sign(group["direction_rho"].dropna().to_numpy())
        signs = signs[signs != 0]
        positive_share = float((signs > 0).mean()) if len(signs) else np.nan
        finite_direction = group["direction_rho"].dropna()
        rows.append(
            {
                "target": target,
                "grade": grade,
                "source_model": source_model,
                "feature": feature,
                "mean_abs_shap": float(group["mean_abs_shap"].mean()),
                "median_abs_shap": float(group["mean_abs_shap"].median()),
                "median_shap_rank": float(group["shap_rank"].median()),
                f"top{top_k}_shap_frequency": float(
                    (group["shap_rank"] <= top_k).mean()
                ),
                "median_shap_direction_rho": (
                    float(finite_direction.median())
                    if len(finite_direction)
                    else np.nan
                ),
                "shap_direction_consistency": (
                    max(positive_share, 1.0 - positive_share)
                    if np.isfinite(positive_share)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "mean_abs_shap"], ascending=[True, False]
    )


def _method_agreement_from_fold(
    shap_fold: pd.DataFrame, permutation_fold: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    permutation = permutation_fold[permutation_fold["mode"] == "within_style"]
    rows = []
    for target, repeat, fold in shap_fold[
        ["target", "repeat", "fold"]
    ].drop_duplicates().itertuples(index=False, name=None):
        shap_top = set(
            shap_fold[
                (shap_fold["target"] == target)
                & (shap_fold["repeat"] == repeat)
                & (shap_fold["fold"] == fold)
            ].nlargest(top_k, "mean_abs_shap")["feature"]
        )
        permutation_top = set(
            permutation[
                (permutation["target"] == target)
                & (permutation["repeat"] == repeat)
                & (permutation["fold"] == fold)
            ].nlargest(top_k, "mean_delta_rmse")["feature"]
        )
        union = shap_top | permutation_top
        rows.append(
            {
                "target": target,
                "repeat": int(repeat),
                "fold": int(fold),
                f"permutation_shap_top{top_k}_jaccard": (
                    len(shap_top & permutation_top) / len(union) if union else 1.0
                ),
                "shared_features": "|".join(sorted(shap_top & permutation_top)),
            }
        )
    return pd.DataFrame(rows)


def _selected_style_rows(
    shap_paths: Sequence[Path],
    shap_stability: pd.DataFrame,
    permutation_stability: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    selected: Dict[str, set] = {}
    for target in shap_stability["target"].unique():
        shap_top = set(
            shap_stability[shap_stability["target"] == target]
            .nlargest(top_k, "mean_abs_shap")["feature"]
        )
        permutation_top = set(
            permutation_stability[
                (permutation_stability["target"] == target)
                & (permutation_stability["mode"] == "within_style")
            ].nlargest(top_k, "mean_delta_rmse")["feature"]
        )
        selected[str(target)] = shap_top | permutation_top
    retained = []
    columns = [
        "repeat",
        "fold",
        "target",
        "beer",
        "style",
        "feature",
        "feature_value",
        "shap_value",
    ]
    for path in shap_paths:
        frame = pd.read_parquet(path, columns=columns)
        mask = np.zeros(len(frame), dtype=bool)
        for target, features in selected.items():
            mask |= (frame["target"].to_numpy() == target) & frame[
                "feature"
            ].isin(features).to_numpy()
        retained.append(frame.loc[mask])
    return pd.concat(retained, ignore_index=True)


def run_formal_advanced_fast(
    project_root: Path, config_path: Path
) -> Dict[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = _resolve(root, config["output_root"])
    _validate_output_boundary(root, output)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "config.resolved.yaml"
    if resolved.exists():
        existing = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Existing resumable output uses a different configuration")
    else:
        resolved.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    data = load_problem1_data(_q1_data_config(root, config))
    folds_path = root / "problem1/freezes/problem1_v1/base_run_metadata/folds.csv"
    tuning_path = root / "problem1/freezes/problem1_v1/base_run_metadata/tuning_trials.csv"
    frozen_path = root / "problem1/freezes/problem1_v1/champion_artifacts/oof_predictions.parquet"
    registry_path = root / "problem1/freezes/problem1_v1/source_registry.csv"
    folds = pd.read_csv(folds_path)
    tuning = pd.read_csv(tuning_path)
    frozen = pd.read_parquet(frozen_path)
    registry = pd.read_csv(registry_path).set_index("target")
    network_nodes = _load_network_nodes(root, config)
    configured_targets = config["targets"]
    targets = (
        list(map(str, data.main_target_names))
        if configured_targets == "all_main"
        else list(map(str, configured_targets))
    )
    repeats = list(map(int, config["repeats"]))
    outer_folds = list(map(int, config["outer_folds"]))

    shap_dir = output / "oof_shap_values_parts"
    group_dir = output / "oof_group_permutation_parts"
    fold_dir = output / "shap_fold_level_parts"
    audit_dir = output / "shap_reconstruction_audit_parts"
    for directory in [shap_dir, group_dir, fold_dir, audit_dir]:
        directory.mkdir(exist_ok=True)
    runtime_path = output / "fold_runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {}
    )

    expected_keys = {
        f"r{repeat}_f{fold}" for repeat in repeats for fold in outer_folds
    }
    for repeat in repeats:
        for fold in outer_folds:
            key = f"r{repeat}_f{fold}"
            shap_path = shap_dir / f"{key}.parquet"
            group_path = group_dir / f"{key}.parquet"
            fold_path = fold_dir / f"{key}.parquet"
            audit_path = audit_dir / f"{key}.parquet"
            paths = [shap_path, group_path, fold_path, audit_path]
            if all(path.exists() for path in paths):
                print(f"Skipping completed {key}", flush=True)
                continue
            for path in paths:
                if path.exists():
                    path.unlink()
                partial = path.with_suffix(path.suffix + ".partial")
                if partial.exists():
                    partial.unlink()
            fold_level, audit, diagnostics = _run_fold(
                data,
                folds,
                tuning,
                frozen,
                registry,
                network_nodes,
                targets,
                repeat,
                fold,
                config,
                shap_path,
                group_path,
            )
            _write_parquet_atomic(fold_level, fold_path)
            _write_parquet_atomic(audit, audit_path)
            runtime[key] = diagnostics
            _write_json_atomic(runtime_path, runtime)
            gc.collect()

    completed = {
        path.stem
        for path in shap_dir.glob("r*_f*.parquet")
        if (group_dir / path.name).exists()
        and (fold_dir / path.name).exists()
        and (audit_dir / path.name).exists()
    }
    missing = sorted(expected_keys - completed)
    if missing:
        raise ValueError(f"Advanced formal chunk remains incomplete: {missing}")

    ordered_keys = sorted(expected_keys)
    shap_paths = [shap_dir / f"{key}.parquet" for key in ordered_keys]
    shap_fold = pd.concat(
        [pd.read_parquet(fold_dir / f"{key}.parquet") for key in ordered_keys],
        ignore_index=True,
    )
    group_importance = pd.concat(
        [pd.read_parquet(group_dir / f"{key}.parquet") for key in ordered_keys],
        ignore_index=True,
    )
    audits = pd.concat(
        [pd.read_parquet(audit_dir / f"{key}.parquet") for key in ordered_keys],
        ignore_index=True,
    )
    top_k = int(config["top_k"])
    shap_stability = _shap_stability_from_fold(shap_fold, top_k)
    base = _resolve(root, config["base_run"])
    permutation_fold = pd.read_parquet(base / "importance_fold_level.parquet")
    permutation_stability = pd.read_csv(base / "importance_stability.csv")
    agreement = _method_agreement_from_fold(shap_fold, permutation_fold, top_k)
    selected_shap = _selected_style_rows(
        shap_paths, shap_stability, permutation_stability, top_k
    )
    style = _style_heterogeneity(
        selected_shap,
        shap_stability,
        permutation_stability,
        top_k,
        int(config["style_report_minimum_n"]),
        float(config["direction_reversal_min_abs_global_rho"]),
        float(config["direction_reversal_min_abs_style_rho"]),
    )
    group_summary = (
        group_importance.groupby(
            [
                "target",
                "repeat",
                "fold",
                "community",
                "community_size",
                "community_features",
            ],
            as_index=False,
        )
        .agg(
            mean_delta_rmse=("delta_rmse", "mean"),
            positive_frequency=(
                "delta_rmse",
                lambda values: float((values > 0).mean()),
            ),
            mean_relative_delta_rmse=("relative_delta_rmse", "mean"),
        )
    )

    shap_fold.to_parquet(output / "shap_fold_level.parquet", index=False)
    shap_stability.to_csv(output / "shap_stability.csv", index=False)
    group_importance.to_parquet(
        output / "oof_group_permutation_importance.parquet", index=False
    )
    group_summary.to_csv(output / "group_importance_summary.csv", index=False)
    agreement.to_csv(output / "permutation_shap_agreement.csv", index=False)
    style.to_csv(output / "style_direction_heterogeneity.csv", index=False)
    audits.to_csv(output / "shap_reconstruction_audit.csv", index=False)

    selected_validation_rows = int(
        len(
            folds[
                folds["repeat"].isin(repeats)
                & folds["fold"].isin(outer_folds)
            ]
        )
    )
    expected_shap_rows = int(
        len(targets) * selected_validation_rows * len(data.feature_names)
    )
    actual_shap_rows = int(sum(runtime[key]["shap_rows"] for key in expected_keys))
    selected_network = network_nodes[
        network_nodes["repeat"].isin(repeats)
        & network_nodes["fold"].isin(outer_folds)
    ]
    expected_group_rows = int(
        selected_network.groupby(["repeat", "fold"])["community"].nunique().sum()
        * len(targets)
        * int(config["group_permutations"])
    )
    expected_audit_rows = len(targets) * len(repeats) * len(outer_folds)
    result = {
        "status": "PASS",
        "targets": targets,
        "repeats": repeats,
        "outer_folds": outer_folds,
        "expected_shap_rows": expected_shap_rows,
        "actual_shap_rows": actual_shap_rows,
        "expected_group_permutation_rows": expected_group_rows,
        "n_group_permutation_rows": int(len(group_importance)),
        "expected_audit_rows": expected_audit_rows,
        "actual_audit_rows": int(len(audits)),
        "max_shap_additivity_error": float(
            audits["shap_additivity_max_abs_error"].max()
        ),
        "max_reconstruction_error": float(
            audits["reconstruction_max_abs_error"].max()
        ),
        f"median_permutation_shap_top{top_k}_jaccard": float(
            agreement[f"permutation_shap_top{top_k}_jaccard"].median()
        ),
        "direction_reversal_count": int(style["direction_reversal"].sum()),
        "raw_shap_stored_by_fold": True,
        "resumable_fold_parts": True,
        "formal_top3_generated": False,
        "input_sha256": {
            "folds": sha256_file(folds_path),
            "frozen_oof": sha256_file(frozen_path),
            "source_registry": sha256_file(registry_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "pyarrow": pa.__version__,
        },
    }
    if actual_shap_rows != expected_shap_rows:
        raise ValueError(
            f"SHAP row count mismatch: {actual_shap_rows} != {expected_shap_rows}"
        )
    if len(group_importance) != expected_group_rows:
        raise ValueError(
            "Group-permutation row count mismatch: "
            f"{len(group_importance)} != {expected_group_rows}"
        )
    if len(audits) != expected_audit_rows:
        raise ValueError(
            f"Audit row count mismatch: {len(audits)} != {expected_audit_rows}"
        )
    _write_json_atomic(output / "summary.json", result)
    return result

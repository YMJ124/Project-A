"""Repeated nested-CV experiment runner.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
import yaml

from .data import Problem1Data, load_problem1_data
from .evaluation import (
    bootstrap_metric_intervals,
    build_uncertainty,
    evaluate_stage_gate,
    metrics_from_oof,
    save_residual_diagnostics,
    select_models_and_grade,
    style_error_tables,
    summarize_metrics,
)
from .folds import build_repeated_folds
from .models import (
    BASELINE_MODELS,
    build_model,
    ensure_2d_prediction,
    predict_global_mean,
    predict_style_mean,
    tune_model_one_se,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_event(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _validate_fixed_mode(
    candidate_models: List[str], grids: Dict[str, Any], enabled: bool
) -> None:
    """Prevent a no-inner-CV run from masquerading as model selection."""
    if not enabled:
        return
    if len(candidate_models) != 1:
        raise ValueError(
            "fixed_without_inner_cv requires exactly one non-baseline model; "
            "otherwise family selection would have no honest inner-CV evidence"
        )
    model_name = candidate_models[0]
    if len(grids.get(model_name, [])) != 1:
        raise ValueError(
            "fixed_without_inner_cv requires exactly one parameter candidate "
            f"for {model_name}; got {len(grids.get(model_name, []))}"
        )


def _predictions_to_long(
    data: Problem1Data,
    valid_index: np.ndarray,
    prediction: np.ndarray,
    repeat: int,
    fold: int,
    model_name: str,
) -> pd.DataFrame:
    n_valid, n_targets = prediction.shape
    if n_valid != len(valid_index) or n_targets != len(data.target_names):
        raise ValueError("Prediction dimensions do not match validation data")
    base = pd.DataFrame(
        {
            "beer": np.repeat(data.ids.iloc[valid_index].to_numpy(), n_targets),
            "style": np.repeat(data.styles.iloc[valid_index].to_numpy(), n_targets),
            "repeat": repeat,
            "fold": fold,
            "target": np.tile(data.target_names, n_valid),
            "model": model_name,
            "y_true": data.y.iloc[valid_index].to_numpy().reshape(-1),
            "y_pred": prediction.reshape(-1),
        }
    )
    return base


def run_experiment(
    config: Dict[str, Any], run_id: Optional[str] = None
) -> Path:
    start = time.monotonic()
    run_id = run_id or datetime.now().strftime("q1_%Y%m%d_%H%M%S")
    output_dir = Path(config["runtime"]["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    progress_path = output_dir / "progress.jsonl"

    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    data = load_problem1_data(config)
    _write_json(output_dir / "data_audit.json", data.audit)
    pd.DataFrame({"feature": data.feature_names}).to_csv(
        output_dir / "feature_list.csv", index=False
    )
    pd.DataFrame({"target": data.target_names}).to_csv(
        output_dir / "target_list.csv", index=False
    )

    validation = config["validation"]
    folds = build_repeated_folds(
        ids=data.ids,
        styles=data.styles,
        n_splits=int(validation["outer_splits"]),
        n_repeats=int(validation["outer_repeats"]),
        base_seed=int(validation["base_seed"]),
    )
    folds.to_csv(output_dir / "folds.csv", index=False)

    model_config = config["models"]
    enabled_models = list(model_config["enabled"])
    missing_baselines = BASELINE_MODELS.difference(enabled_models)
    if missing_baselines:
        raise ValueError(f"Required baselines missing: {sorted(missing_baselines)}")

    candidate_models = [name for name in enabled_models if name not in BASELINE_MODELS]
    fixed_without_inner_cv = bool(model_config.get("fixed_without_inner_cv", False))

    n_jobs = int(config["runtime"].get("n_jobs", 1))
    grids = model_config.get("grids", {})
    _validate_fixed_mode(candidate_models, grids, fixed_without_inner_cv)
    oof_frames: List[pd.DataFrame] = []
    tuning_frames: List[pd.DataFrame] = []
    family_selection_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    print(
        f"[{run_id}] samples={len(data.ids)} targets={len(data.target_names)} "
        f"repeats={validation['outer_repeats']} folds={validation['outer_splits']} "
        f"models={enabled_models}",
        flush=True,
    )

    for repeat in range(int(validation["outer_repeats"])):
        repeat_rows = folds[folds["repeat"] == repeat].sort_values("beer")
        outer_fold = repeat_rows["fold"].to_numpy()
        repeat_seed = int(repeat_rows["seed"].iloc[0])
        for fold in range(int(validation["outer_splits"])):
            fold_started = time.monotonic()
            fold_frame_start = len(oof_frames)
            fold_tuning_start = len(tuning_frames)
            fold_selection_start = len(family_selection_rows)
            print(
                f"[{run_id}] repeat {repeat + 1}/{validation['outer_repeats']} "
                f"fold {fold + 1}/{validation['outer_splits']}",
                flush=True,
            )
            train_index = np.flatnonzero(outer_fold != fold)
            valid_index = np.flatnonzero(outer_fold == fold)
            X_train = data.X.iloc[train_index]
            X_valid = data.X.iloc[valid_index]
            y_train = data.y.iloc[train_index]
            styles_train = data.styles.iloc[train_index]
            styles_valid = data.styles.iloc[valid_index]

            baseline_predictions = {
                "global_mean": predict_global_mean(y_train, len(valid_index)),
                "style_mean": predict_style_mean(y_train, styles_train, styles_valid),
            }
            for model_name, prediction in baseline_predictions.items():
                oof_frames.append(
                    _predictions_to_long(
                        data, valid_index, prediction, repeat, fold, model_name
                    )
                )

            fold_predictions: Dict[str, np.ndarray] = {}
            fold_inner_scores: Dict[str, pd.Series] = {}
            for model_offset, model_name in enumerate(enabled_models):
                if model_name in BASELINE_MODELS:
                    continue
                model_started = time.monotonic()
                seed = repeat_seed + fold * 10007 + model_offset * 503
                candidates = grids.get(model_name, [])
                if fixed_without_inner_cv:
                    selected_params = dict(candidates[0])
                    trials = pd.DataFrame(
                        [
                            {
                                "candidate_index": 0,
                                "params_json": json.dumps(selected_params, sort_keys=True),
                                "inner_fold": -1,
                                "score_scope": "fixed",
                                "target": "__all__",
                                "rmse": np.nan,
                                "selected": True,
                                "selection_threshold": np.nan,
                                "selected_mean_rmse": np.nan,
                                "selected_se": np.nan,
                                "selection_method": "fixed_without_inner_cv",
                            }
                        ]
                    )
                    # There is only one eligible family, so no comparative inner
                    # score is needed or claimed. The outer fold remains untouched.
                    fold_inner_scores[model_name] = pd.Series(
                        0.0, index=data.target_names, dtype=float
                    )
                else:
                    selected_params, trials = tune_model_one_se(
                        name=model_name,
                        candidates=candidates,
                        X=X_train.reset_index(drop=True),
                        y=y_train.reset_index(drop=True),
                        styles=styles_train.reset_index(drop=True),
                        inner_splits=int(validation["inner_splits"]),
                        seed=seed,
                        n_jobs=n_jobs,
                    )
                trials.insert(0, "model", model_name)
                trials.insert(0, "outer_fold", fold)
                trials.insert(0, "repeat", repeat)
                tuning_frames.append(trials)
                if not fixed_without_inner_cv:
                    selected_target_trials = trials[
                        trials["selected"] & (trials["score_scope"] == "target")
                    ]
                    fold_inner_scores[model_name] = selected_target_trials.groupby(
                        "target"
                    )["rmse"].mean()

                model = build_model(model_name, selected_params, seed, n_jobs)
                model.fit(X_train, y_train)
                if hasattr(model, "calibration_slopes_"):
                    for target_index, target in enumerate(data.target_names):
                        calibration_rows.append(
                            {
                                "repeat": repeat,
                                "outer_fold": fold,
                                "target": target,
                                "model": model_name,
                                "raw_slope": float(
                                    model.calibration_raw_slopes_[target_index]
                                ),
                                "shrunk_bounded_slope": float(
                                    model.calibration_slopes_[target_index]
                                ),
                                "intercept": float(
                                    model.calibration_intercepts_[target_index]
                                ),
                            }
                        )
                prediction = ensure_2d_prediction(
                    model.predict(X_valid), n_targets=len(data.target_names)
                )
                fold_predictions[model_name] = prediction
                oof_frames.append(
                    _predictions_to_long(
                        data, valid_index, prediction, repeat, fold, model_name
                    )
                )
                if config["runtime"].get("save_models", False):
                    model_dir = output_dir / "models"
                    model_dir.mkdir(exist_ok=True)
                    joblib.dump(model, model_dir / f"{model_name}_r{repeat}_f{fold}.joblib")
                print(
                    f"  {model_name}: {time.monotonic() - model_started:.1f}s "
                    f"params={selected_params}",
                    flush=True,
                )

            # Honest target-wise family selection: choose with inner folds only,
            # then evaluate the chosen family's already-held-out outer prediction.
            nested_prediction = np.empty((len(valid_index), len(data.target_names)))
            for target_index, target in enumerate(data.target_names):
                if fixed_without_inner_cv:
                    selected_family = candidate_models[0]
                    inner_scores: Dict[str, Optional[float]] = {selected_family: None}
                    selected_inner_rmse = None
                    selection_mode = "fixed_without_inner_cv"
                else:
                    inner_scores = {
                        name: float(fold_inner_scores[name].loc[target])
                        for name in candidate_models
                    }
                    selected_family = min(inner_scores, key=inner_scores.get)
                    selected_inner_rmse = inner_scores[selected_family]
                    selection_mode = "inner_cv_family_selection"
                nested_prediction[:, target_index] = fold_predictions[selected_family][
                    :, target_index
                ]
                family_selection_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": fold,
                        "target": target,
                        "selected_family": selected_family,
                        "inner_rmse": selected_inner_rmse,
                        "all_inner_rmse_json": json.dumps(inner_scores, sort_keys=True),
                        "selection_mode": selection_mode,
                    }
                )
            oof_frames.append(
                _predictions_to_long(
                    data,
                    valid_index,
                    nested_prediction,
                    repeat,
                    fold,
                    "nested_selected",
                )
            )
            pd.concat(oof_frames[fold_frame_start:], ignore_index=True).to_parquet(
                checkpoint_dir / f"oof_r{repeat}_f{fold}.parquet", index=False
            )
            if len(tuning_frames) > fold_tuning_start:
                pd.concat(tuning_frames[fold_tuning_start:], ignore_index=True).to_csv(
                    checkpoint_dir / f"tuning_r{repeat}_f{fold}.csv", index=False
                )
            pd.DataFrame(family_selection_rows[fold_selection_start:]).to_csv(
                checkpoint_dir / f"selection_r{repeat}_f{fold}.csv", index=False
            )
            fold_event = {
                "event": "outer_fold_complete",
                "repeat": repeat,
                "fold": fold,
                "elapsed_seconds": time.monotonic() - fold_started,
            }
            _append_event(progress_path, fold_event)
            print(
                f"  outer fold complete: {fold_event['elapsed_seconds']:.1f}s",
                flush=True,
            )

    oof = pd.concat(oof_frames, ignore_index=True)
    expected_per_model = len(data.ids) * len(data.target_names) * int(
        validation["outer_repeats"]
    )
    actual_counts = oof.groupby("model").size()
    if not (actual_counts == expected_per_model).all():
        raise AssertionError(
            f"Incomplete OOF predictions: expected {expected_per_model}; "
            f"got {actual_counts.to_dict()}"
        )
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(
            output_dir / "tuning_trials.csv", index=False
        )
    family_selection = pd.DataFrame(family_selection_rows)
    family_selection.to_csv(output_dir / "model_family_selection.csv", index=False)

    metrics = metrics_from_oof(oof)
    summary = summarize_metrics(metrics)
    selection = select_models_and_grade(metrics, summary)
    family_frequency = (
        family_selection.groupby(["target", "selected_family"])
        .size()
        .rename("selection_count")
        .reset_index()
    )
    family_frequency["selection_share"] = family_frequency["selection_count"] / (
        int(validation["outer_repeats"]) * int(validation["outer_splits"])
    )
    dominant_index = family_frequency.groupby("target")["selection_count"].idxmax()
    dominant = family_frequency.loc[dominant_index].rename(
        columns={
            "selected_family": "dominant_selected_family",
            "selection_share": "dominant_family_share",
        }
    )
    family_frequency.to_csv(output_dir / "model_family_frequency.csv", index=False)
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            output_dir / "calibration_parameters.csv", index=False
        )
    selection = selection.merge(
        dominant[["target", "dominant_selected_family", "dominant_family_share"]],
        on="target",
        how="left",
    )
    gate_table, gate_report = evaluate_stage_gate(selection)
    uncertainty = build_uncertainty(oof, selection)
    metric_intervals = bootstrap_metric_intervals(
        metrics, seed=int(validation["base_seed"])
    )
    style_by_repeat, style_summary = style_error_tables(oof)

    metrics.to_csv(output_dir / "metrics_by_repeat.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    selection.to_csv(output_dir / "model_selection.csv", index=False)
    gate_table.to_csv(output_dir / "stage_gate.csv", index=False)
    _write_json(output_dir / "stage_gate.json", gate_report)
    uncertainty.to_csv(output_dir / "uncertainty.csv", index=False)
    metric_intervals.to_csv(output_dir / "metrics_bootstrap_ci.csv", index=False)
    style_by_repeat.to_csv(output_dir / "style_metrics_by_repeat.csv", index=False)
    style_summary.to_csv(output_dir / "style_metrics_summary.csv", index=False)
    save_residual_diagnostics(oof, output_dir / "residual_diagnostics.png")

    elapsed = time.monotonic() - start
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "models": enabled_models,
        "n_targets": len(data.target_names),
        "outer_repeats": int(validation["outer_repeats"]),
        "outer_splits": int(validation["outer_splits"]),
        "inner_splits": int(validation["inner_splits"]),
        "stage_gate_status": gate_report["status"],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _append_event(progress_path, {"event": "run_complete", **manifest})
    print(
        f"[{run_id}] complete in {elapsed:.1f}s; stage gate={gate_report['status']}",
        flush=True,
    )
    return output_dir

"""Reconstruct a small sample of the frozen Question 1 champion.

The purpose is interface validation for Question 2 explanations. This is not
model selection and it does not change the Question 1 champion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from beer_q1.data import load_problem1_data
from beer_q1.models import build_model, ensure_2d_prediction


DEFAULT_TARGETS = ["A_hops_all", "F_esters_isoaa", "A_hops_woody"]
MODEL_OFFSET = 2  # global_mean, style_mean, then the candidate model
BASE_PARAMS_FALLBACK = {
    "n_estimators": 250,
    "max_features": 0.5,
    "min_samples_leaf": 5,
}
AFFINE_PARAMS = {
    "n_estimators": 250,
    "max_features": 0.5,
    "min_samples_leaf": 5,
    "random_state_override": 20311124,
    "calibration_splits": 3,
    "identity_weight": 0.5,
    "slope_min": 0.8,
    "slope_max": 1.5,
    "clip_to_training_range": False,
}
TARGETWISE_PARAMS = {
    "n_estimators": 250,
    "max_features": 0.5,
    "min_samples_leaf": 5,
}


def _problem1_config(root: Path) -> Dict[str, object]:
    return {
        "data": {
            "chemicals_csv": str(root / "2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"),
            "sensory_csv": str(root / "2026年MathorCup大数据挑战赛-赛道A初赛/附件/4.csv"),
            "id_column": "beer",
            "style_column": "tasting_category_fine",
            "forbidden_input_columns": ["beer", "beer_id", "tasting_category_fine"],
            "target_scope": "main",
        },
        "runtime": {},
    }


def _base_params(tuning: pd.DataFrame, repeat: int, fold: int) -> Dict[str, object]:
    rows = tuning[
        (tuning["repeat"] == repeat)
        & (tuning["outer_fold"] == fold)
        & (tuning["model"] == "extra_trees")
        & (tuning["score_scope"] == "aggregate")
        & (tuning["selected"] == True)  # noqa: E712 - pandas boolean column
    ]
    unique_params = rows["params_json"].drop_duplicates().tolist()
    if len(unique_params) != 1:
        return dict(BASE_PARAMS_FALLBACK)
    return json.loads(str(unique_params[0]))


def _align_frozen_rows(
    frozen: pd.DataFrame, target: str, repeat: int, fold: int
) -> pd.DataFrame:
    result = frozen[
        (frozen["model"] == "final_specialist_hybrid_v2")
        & (frozen["target"] == target)
        & (frozen["repeat"] == repeat)
        & (frozen["fold"] == fold)
    ].copy()
    if result.empty:
        raise ValueError(f"No frozen OOF rows for {target}, repeat={repeat}, fold={fold}")
    return result.sort_values("beer").reset_index(drop=True)


def _predict_fold(
    data,
    fold_table: pd.DataFrame,
    tuning: pd.DataFrame,
    target: str,
    repeat: int,
    fold: int,
) -> np.ndarray:
    repeat_rows = fold_table[fold_table["repeat"] == repeat].sort_values("beer")
    repeat_seed = int(repeat_rows["seed"].iloc[0])
    valid_beers = repeat_rows.loc[repeat_rows["fold"] == fold, "beer"].to_numpy()
    train_beers = repeat_rows.loc[repeat_rows["fold"] != fold, "beer"].to_numpy()
    beer_to_position = {int(beer): i for i, beer in enumerate(data.ids.to_numpy())}
    valid_index = np.array([beer_to_position[int(beer)] for beer in valid_beers], dtype=int)
    train_index = np.array([beer_to_position[int(beer)] for beer in train_beers], dtype=int)
    X_train, X_valid = data.X.iloc[train_index], data.X.iloc[valid_index]
    y_train = data.y.iloc[train_index]
    seed = repeat_seed + fold * 10007 + MODEL_OFFSET * 503

    if target == "A_hops_all":
        model = build_model("extra_trees_affine_calibrated", AFFINE_PARAMS, seed, 1)
        model.fit(X_train, y_train)
        target_index = data.target_names.index(target)
        return ensure_2d_prediction(model.predict(X_valid), len(data.target_names))[:, target_index]

    base = build_model("extra_trees", _base_params(tuning, repeat, fold), seed, 1)
    base.fit(X_train, y_train)
    base_pred = ensure_2d_prediction(base.predict(X_valid), len(data.target_names))
    target_index = data.target_names.index(target)
    if target == "A_hops_woody":
        return base_pred[:, target_index]

    if target == "F_esters_isoaa":
        targetwise = build_model("extra_trees_targetwise", TARGETWISE_PARAMS, seed, 1)
        targetwise.fit(X_train, y_train[[target]])
        targetwise_pred = ensure_2d_prediction(targetwise.predict(X_valid), 1)[:, 0]
        return 0.5 * base_pred[:, target_index] + 0.5 * targetwise_pred

    raise ValueError(f"Unsupported demonstration target: {target}")


def reconstruct_and_compare(
    project_root: Path,
    targets: Sequence[str] = DEFAULT_TARGETS,
    repeats: Iterable[int] = range(7),
    tolerance: float = 1e-9,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    root = project_root.resolve()
    config = _problem1_config(root)
    data = load_problem1_data(config)
    fold_table = pd.read_csv(root / "problem1/freezes/problem1_v1/base_run_metadata/folds.csv")
    tuning = pd.read_csv(root / "problem1/freezes/problem1_v1/base_run_metadata/tuning_trials.csv")
    frozen = pd.read_parquet(root / "problem1/freezes/problem1_v1/champion_artifacts/oof_predictions.parquet")
    targets = list(targets)
    unknown = sorted(set(targets).difference(data.target_names))
    if unknown:
        raise ValueError(f"Unknown demonstration targets: {unknown}")

    rows: List[Dict[str, object]] = []
    for repeat in repeats:
        for fold in range(3):
            for target in targets:
                frozen_rows = _align_frozen_rows(frozen, target, int(repeat), fold)
                prediction = _predict_fold(data, fold_table, tuning, target, int(repeat), fold)
                if len(prediction) != len(frozen_rows):
                    raise ValueError(f"Validation row count mismatch for {target}, r{repeat}f{fold}")
                expected_source = str(frozen_rows["source_model"].iloc[0])
                error = prediction - frozen_rows["y_pred"].to_numpy(dtype=float)
                rows.append(
                    {
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "target": target,
                        "n_valid": int(len(prediction)),
                        "frozen_source_model": expected_source,
                        "max_abs_error": float(np.max(np.abs(error))),
                        "mean_abs_error": float(np.mean(np.abs(error))),
                        "rmse_between_predictions": float(np.sqrt(np.mean(error**2))),
                        "passed_tolerance": bool(np.max(np.abs(error)) <= tolerance),
                    }
                )
    result = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "reconstruction_comparison.csv", index=False)
        summary = {
            "targets": targets,
            "repeats": list(map(int, repeats)),
            "tolerance": tolerance,
            "all_passed": bool(result["passed_tolerance"].all()),
            "max_abs_error": float(result["max_abs_error"].max()),
            "mean_abs_error": float(result["mean_abs_error"].mean()),
            "rows": int(len(result)),
        }
        (output_dir / "reconstruction_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result

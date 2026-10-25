"""Reconstruct a small sample of the frozen Question 1 champion.

The purpose is interface validation for Question 2 explanations. This is not
model selection and it does not change the Question 1 champion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

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
GBR_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 1,
    "min_samples_leaf": 8,
}
SELECTK_PARAMS = {
    "n_estimators": 250,
    "max_features": 0.5,
    "min_samples_leaf": 5,
}

# This map is the executable form of the immutable problem1_v1 source registry.
# A test asserts exact agreement with the frozen CSV so a future registry change
# cannot silently leave the reconstruction logic stale.
FROZEN_SOURCE_BY_TARGET = {
    "A_esters_all": "gradient_boosting",
    "A_esters_ethac": "fold_tuned_extra_trees",
    "A_esters_flower": "fold_tuned_extra_trees",
    "A_esters_fruity": "fold_tuned_extra_trees",
    "A_esters_isoaa": "gradient_boosting",
    "A_hops_all": "affine_calibration",
    "A_hops_citrus": "fold_tuned_extra_trees",
    "A_hops_noble": "fold_tuned_extra_trees",
    "A_hops_tropical": "fold_tuned_extra_trees",
    "A_hops_woody": "fold_tuned_extra_trees",
    "A_malt_all": "selectk120",
    "A_malt_bread": "fold_tuned_extra_trees",
    "A_malt_burn": "fold_tuned_extra_trees",
    "A_malt_cara": "selectk30",
    "A_malt_grain": "fold_tuned_extra_trees",
    "F_esters_all": "extra_trees_targetwise",
    "F_esters_ethac": "affine_calibration",
    "F_esters_flower": "fold_tuned_extra_trees",
    "F_esters_fruity": "selectk60",
    "F_esters_isoaa": "half_multioutput_half_targetwise_et",
    "F_hops_all": "fixed_scale_1.25",
    "F_hops_citrus": "fold_tuned_extra_trees",
    "F_hops_noble": "selectk120",
    "F_hops_tropical": "fold_tuned_extra_trees",
    "F_hops_woody": "fold_tuned_extra_trees",
    "F_malt_all": "selectk60",
    "F_malt_bread": "fold_tuned_extra_trees",
    "F_malt_burn": "fold_tuned_extra_trees",
    "F_malt_cara": "selectk120",
    "F_malt_grain": "fold_tuned_extra_trees",
}


@dataclass
class FrozenTargetPredictor:
    """A fitted representative Q1 target pipeline with one prediction API."""

    target: str
    target_index: int
    n_targets: int
    source_model: str
    primary_model: object
    secondary_model: Optional[object] = None
    prediction_scale: float = 1.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        primary = ensure_2d_prediction(
            self.primary_model.predict(X),
            self.n_targets,
        )[:, self.target_index]
        if self.source_model != "half_multioutput_half_targetwise_et":
            return self.prediction_scale * primary
        if self.secondary_model is None:
            raise ValueError("Blend predictor is missing its targetwise component")
        secondary = ensure_2d_prediction(self.secondary_model.predict(X), 1)[:, 0]
        return 0.5 * primary + 0.5 * secondary


def predict_frozen_targets(
    predictors: Sequence[FrozenTargetPredictor], X: pd.DataFrame
) -> np.ndarray:
    """Predict many targets while evaluating each shared fitted model once.

    Several frozen targets use the same fitted multi-output Extra Trees model.
    Calling ``FrozenTargetPredictor.predict`` target by target would repeat the
    identical tree traversal and then discard all but one output column.
    """
    model_outputs: Dict[int, np.ndarray] = {}

    def output(model: object) -> np.ndarray:
        key = id(model)
        if key not in model_outputs:
            values = np.asarray(model.predict(X), dtype=float)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            model_outputs[key] = values
        return model_outputs[key]

    columns = []
    for predictor in predictors:
        primary = output(predictor.primary_model)[:, predictor.target_index]
        if predictor.source_model == "half_multioutput_half_targetwise_et":
            if predictor.secondary_model is None:
                raise ValueError("Blend predictor is missing its targetwise component")
            secondary = output(predictor.secondary_model)[:, 0]
            columns.append(0.5 * primary + 0.5 * secondary)
        else:
            columns.append(predictor.prediction_scale * primary)
    return np.column_stack(columns)


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


def fold_positions(data, fold_table: pd.DataFrame, repeat: int, fold: int):
    """Return train/validation row positions in the immutable attachment order."""
    repeat_rows = fold_table[fold_table["repeat"] == repeat].sort_values("beer")
    if repeat_rows.empty:
        raise ValueError(f"No frozen fold rows for repeat={repeat}")
    repeat_seed = int(repeat_rows["seed"].iloc[0])
    valid_beers = repeat_rows.loc[repeat_rows["fold"] == fold, "beer"].to_numpy()
    train_beers = repeat_rows.loc[repeat_rows["fold"] != fold, "beer"].to_numpy()
    beer_to_position = {int(beer): i for i, beer in enumerate(data.ids.to_numpy())}
    valid_index = np.array([beer_to_position[int(beer)] for beer in valid_beers], dtype=int)
    train_index = np.array([beer_to_position[int(beer)] for beer in train_beers], dtype=int)
    return train_index, valid_index, repeat_seed


def fit_representative_target_predictor(
    data,
    fold_table: pd.DataFrame,
    tuning: pd.DataFrame,
    target: str,
    repeat: int,
    fold: int,
    model_cache: Optional[MutableMapping[Tuple[object, ...], object]] = None,
    n_jobs: int = 1,
) -> tuple[FrozenTargetPredictor, np.ndarray, np.ndarray]:
    """Fit the frozen problem1_v1 source for one target on an outer train fold."""
    train_index, valid_index, repeat_seed = fold_positions(data, fold_table, repeat, fold)
    X_train = data.X.iloc[train_index]
    y_train = data.y.iloc[train_index]
    seed = repeat_seed + fold * 10007 + MODEL_OFFSET * 503
    global_target_index = data.target_names.index(target)
    source_model = FROZEN_SOURCE_BY_TARGET.get(target)
    if source_model is None:
        raise ValueError(f"Target is absent from the frozen source registry: {target}")
    cache = model_cache if model_cache is not None else {}

    def fitted(
        key: Tuple[object, ...], model_name: str, params: Dict[str, object], y
    ):
        if key not in cache:
            model = build_model(model_name, params, seed, n_jobs)
            model.fit(X_train, y)
            cache[key] = model
        return cache[key]

    if source_model == "affine_calibration":
        model = fitted(
            ("affine_calibration", repeat, fold),
            "extra_trees_affine_calibrated",
            AFFINE_PARAMS,
            y_train,
        )
        predictor = FrozenTargetPredictor(
            target=target,
            target_index=global_target_index,
            n_targets=len(data.target_names),
            source_model="affine_calibration",
            primary_model=model,
        )
        return predictor, train_index, valid_index

    if source_model == "gradient_boosting":
        model = fitted(
            ("gradient_boosting", repeat, fold, target),
            "gradient_boosting",
            GBR_PARAMS,
            y_train[[target]],
        )
        predictor = FrozenTargetPredictor(
            target=target,
            target_index=0,
            n_targets=1,
            source_model=source_model,
            primary_model=model,
        )
        return predictor, train_index, valid_index

    if source_model.startswith("selectk"):
        k = int(source_model.removeprefix("selectk"))
        params = {"k": k, **SELECTK_PARAMS}
        model = fitted(
            (source_model, repeat, fold, target),
            "extra_trees_targetwise_selectk",
            params,
            y_train[[target]],
        )
        predictor = FrozenTargetPredictor(
            target=target,
            target_index=0,
            n_targets=1,
            source_model=source_model,
            primary_model=model,
        )
        return predictor, train_index, valid_index

    if source_model == "extra_trees_targetwise":
        model = fitted(
            ("extra_trees_targetwise", repeat, fold, target),
            "extra_trees_targetwise",
            TARGETWISE_PARAMS,
            y_train[[target]],
        )
        predictor = FrozenTargetPredictor(
            target=target,
            target_index=0,
            n_targets=1,
            source_model=source_model,
            primary_model=model,
        )
        return predictor, train_index, valid_index

    base = fitted(
        ("fold_tuned_extra_trees", repeat, fold),
        "extra_trees",
        _base_params(tuning, repeat, fold),
        y_train,
    )
    if source_model in {"fold_tuned_extra_trees", "fixed_scale_1.25"}:
        predictor = FrozenTargetPredictor(
            target=target,
            target_index=global_target_index,
            n_targets=len(data.target_names),
            source_model=source_model,
            primary_model=base,
            prediction_scale=1.25 if source_model == "fixed_scale_1.25" else 1.0,
        )
        return predictor, train_index, valid_index

    if source_model == "half_multioutput_half_targetwise_et":
        targetwise = fitted(
            ("blend_targetwise", repeat, fold, target),
            "extra_trees_targetwise",
            TARGETWISE_PARAMS,
            y_train[[target]],
        )
        predictor = FrozenTargetPredictor(
            target=target,
            target_index=global_target_index,
            n_targets=len(data.target_names),
            source_model=source_model,
            primary_model=base,
            secondary_model=targetwise,
        )
        return predictor, train_index, valid_index

    raise ValueError(f"Unsupported frozen source model for {target}: {source_model}")


def _predict_fold(
    data,
    fold_table: pd.DataFrame,
    tuning: pd.DataFrame,
    target: str,
    repeat: int,
    fold: int,
    model_cache: Optional[MutableMapping[Tuple[object, ...], object]] = None,
) -> np.ndarray:
    predictor, _, valid_index = fit_representative_target_predictor(
        data, fold_table, tuning, target, repeat, fold, model_cache=model_cache
    )
    return predictor.predict(data.X.iloc[valid_index])


def reconstruct_and_compare(
    project_root: Path,
    targets: Optional[Sequence[str]] = DEFAULT_TARGETS,
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
    targets = list(data.target_names if targets is None else targets)
    unknown = sorted(set(targets).difference(data.target_names))
    if unknown:
        raise ValueError(f"Unknown demonstration targets: {unknown}")

    rows: List[Dict[str, object]] = []
    for repeat in repeats:
        for fold in range(3):
            model_cache: Dict[Tuple[object, ...], object] = {}
            for target in targets:
                frozen_rows = _align_frozen_rows(frozen, target, int(repeat), fold)
                prediction = _predict_fold(
                    data,
                    fold_table,
                    tuning,
                    target,
                    int(repeat),
                    fold,
                    model_cache=model_cache,
                )
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

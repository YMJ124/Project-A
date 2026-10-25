"""Full-data inference adapter for the immutable Question 1 v1 source map.

This module creates a deployment model after model selection has finished. It
does not generate new validation evidence: all reported generalization metrics
remain the frozen repeated OOF results from Question 1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from beer_q1.data import load_problem1_data
from beer_q1.models import build_model, ensure_2d_prediction
from beer_q2.reconstruct import (
    AFFINE_PARAMS,
    GBR_PARAMS,
    SELECTK_PARAMS,
    TARGETWISE_PARAMS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modal_base_params(tuning: pd.DataFrame) -> Dict[str, Any]:
    """Choose the most frequent outer-fold selected ET configuration."""
    selected = tuning[
        (tuning["model"] == "extra_trees")
        & (tuning["score_scope"] == "aggregate")
        & (tuning["selected"] == True)  # noqa: E712 - pandas boolean column
    ][["repeat", "outer_fold", "params_json"]].drop_duplicates()
    if len(selected) != 21:
        raise ValueError(f"Expected 21 selected outer-fold configurations, got {len(selected)}")
    counts = selected["params_json"].value_counts()
    if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
        raise ValueError("Selected Extra Trees configurations do not have a unique mode")
    return json.loads(str(counts.index[0]))


@dataclass
class TargetComponent:
    target: str
    source_model: str
    primary_model: Any
    primary_index: int
    primary_outputs: int
    secondary_model: Any = None
    prediction_scale: float = 1.0


@dataclass
class FullDataQ1Predictor:
    """One stable prediction API for all 30 frozen Q1 target sources."""

    feature_names: Sequence[str]
    target_names: Sequence[str]
    components: Sequence[TargetComponent]
    training_input_sha256: str
    training_target_sha256: str

    def _validated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Q1 inference requires a pandas DataFrame with named columns")
        missing = sorted(set(self.feature_names).difference(X.columns))
        extra = sorted(set(X.columns).difference(self.feature_names))
        if missing or extra:
            raise ValueError(f"Q1 feature schema mismatch; missing={missing}, extra={extra}")
        return X.loc[:, list(self.feature_names)]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._validated_features(X)
        model_outputs: Dict[int, np.ndarray] = {}

        def output(model: Any) -> np.ndarray:
            key = id(model)
            if key not in model_outputs:
                values = np.asarray(model.predict(features), dtype=float)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                model_outputs[key] = values
            return model_outputs[key]

        predictions = []
        for component in self.components:
            primary = output(component.primary_model)[:, component.primary_index]
            if component.source_model == "half_multioutput_half_targetwise_et":
                if component.secondary_model is None:
                    raise ValueError(f"Missing blend component for {component.target}")
                secondary = output(component.secondary_model)[:, 0]
                prediction = 0.5 * primary + 0.5 * secondary
            else:
                prediction = component.prediction_scale * primary
            predictions.append(prediction)
        result = np.column_stack(predictions)
        if result.shape != (len(features), len(self.target_names)) or not np.isfinite(result).all():
            raise ValueError("Q1 inference produced an invalid prediction matrix")
        return pd.DataFrame(result, index=X.index, columns=list(self.target_names))


def _path(root: Path, config: Mapping[str, Any], section: str, key: str) -> Path:
    value = Path(config[section][key])
    return value if value.is_absolute() else root / value


def fit_full_data_predictor(project_root: Path, config_path: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    q1_config = {
        "data": {
            "chemicals_csv": str(_path(project_root, config, "data", "chemicals_csv")),
            "sensory_csv": str(_path(project_root, config, "data", "sensory_csv")),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": "main",
        },
        "runtime": {},
    }
    data = load_problem1_data(q1_config)
    registry_path = _path(project_root, config, "frozen_q1", "source_registry")
    tuning_path = _path(project_root, config, "frozen_q1", "tuning_trials")
    gate_path = _path(project_root, config, "frozen_q1", "stage_gate")
    registry = pd.read_csv(registry_path)
    tuning = pd.read_csv(tuning_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if set(registry["target"]) != set(data.target_names) or registry["target"].nunique() != 30:
        raise ValueError("Frozen source registry does not cover the 30-target data contract")
    source_by_target = dict(zip(registry["target"], registry["source_model"]))

    seed = int(config["fit"]["seed"])
    n_jobs = int(config["fit"]["n_jobs"])
    base_params = modal_base_params(tuning)
    models: Dict[str, Any] = {}

    def fitted(key: str, model_name: str, params: Dict[str, Any], y: pd.DataFrame) -> Any:
        if key not in models:
            model = build_model(model_name, params, seed, n_jobs)
            model.fit(data.X, y)
            models[key] = model
        return models[key]

    components = []
    for target_index, target in enumerate(data.target_names):
        source = source_by_target[target]
        if source == "affine_calibration":
            primary = fitted("affine_calibration", "extra_trees_affine_calibrated", AFFINE_PARAMS, data.y)
            component = TargetComponent(target, source, primary, target_index, 30)
        elif source == "gradient_boosting":
            primary = fitted(f"gbr:{target}", "gradient_boosting", GBR_PARAMS, data.y[[target]])
            component = TargetComponent(target, source, primary, 0, 1)
        elif source.startswith("selectk"):
            k = int(source.removeprefix("selectk"))
            primary = fitted(
                f"{source}:{target}",
                "extra_trees_targetwise_selectk",
                {"k": k, **SELECTK_PARAMS},
                data.y[[target]],
            )
            component = TargetComponent(target, source, primary, 0, 1)
        elif source == "extra_trees_targetwise":
            primary = fitted(
                f"targetwise:{target}",
                "extra_trees_targetwise",
                TARGETWISE_PARAMS,
                data.y[[target]],
            )
            component = TargetComponent(target, source, primary, 0, 1)
        elif source in {"fold_tuned_extra_trees", "fixed_scale_1.25"}:
            primary = fitted("base", "extra_trees", base_params, data.y)
            component = TargetComponent(
                target,
                source,
                primary,
                target_index,
                30,
                prediction_scale=1.25 if source == "fixed_scale_1.25" else 1.0,
            )
        elif source == "half_multioutput_half_targetwise_et":
            primary = fitted("base", "extra_trees", base_params, data.y)
            secondary = fitted(
                f"blend_targetwise:{target}",
                "extra_trees_targetwise",
                TARGETWISE_PARAMS,
                data.y[[target]],
            )
            component = TargetComponent(target, source, primary, target_index, 30, secondary)
        else:
            raise ValueError(f"Unsupported frozen Q1 source for {target}: {source}")
        components.append(component)

    predictor = FullDataQ1Predictor(
        feature_names=tuple(data.feature_names),
        target_names=tuple(data.target_names),
        components=tuple(components),
        training_input_sha256=data.audit["chemicals_sha256"],
        training_target_sha256=data.audit["sensory_sha256"],
    )
    training_prediction = predictor.predict(data.X)
    training_rmse = np.sqrt(np.mean((training_prediction.to_numpy() - data.y.to_numpy()) ** 2, axis=0))

    output = _path(project_root, config, "runtime", "output_root")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "q1_full_predictor.joblib"
    joblib.dump(predictor, model_path, compress=3)
    manifest = {
        "status": "PASS",
        "scope": "full_data_deployment_only_not_generalization_evidence",
        "q1_frozen_stage_status": gate.get("status"),
        "n_training_samples": len(data.X),
        "n_features": len(data.feature_names),
        "n_targets": len(data.target_names),
        "target_order": data.target_names,
        "source_model_counts": {
            str(key): int(value) for key, value in registry["source_model"].value_counts().items()
        },
        "base_params": base_params,
        "base_parameter_rule": config["fit"]["base_parameter_rule"],
        "seed": seed,
        "training_input_sha256": data.audit["chemicals_sha256"],
        "training_target_sha256": data.audit["sensory_sha256"],
        "source_registry_sha256": sha256_file(registry_path),
        "tuning_trials_sha256": sha256_file(tuning_path),
        "model_sha256": sha256_file(model_path),
        "training_rmse_median_diagnostic_only": float(np.median(training_rmse)),
        "generalization_metrics_source": "problem1/freezes/problem1_v1/champion_artifacts/all_stage_gate.json",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "target": data.target_names,
            "source_model": [source_by_target[target] for target in data.target_names],
            "training_rmse_diagnostic_only": training_rmse,
        }
    ).to_csv(output / "training_fit_diagnostics.csv", index=False)
    return manifest


def load_full_data_predictor(model_path: Path) -> FullDataQ1Predictor:
    predictor = joblib.load(model_path)
    if not isinstance(predictor, FullDataQ1Predictor):
        raise TypeError("Artifact is not a FullDataQ1Predictor")
    return predictor

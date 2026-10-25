import json
from pathlib import Path

import numpy as np
import pandas as pd

from beer_q3.q1_adapter import FullDataQ1Predictor, TargetComponent, modal_base_params


class _ConstantModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return np.tile(self.values, (len(X), 1))


def test_modal_base_params_uses_one_vote_per_outer_fold():
    params_a = json.dumps({"n_estimators": 250, "max_features": 0.5})
    params_b = json.dumps({"n_estimators": 400, "max_features": "sqrt"})
    rows = []
    for repeat in range(7):
        for fold in range(3):
            params = params_a if repeat * 3 + fold < 11 else params_b
            for inner in range(2):
                rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": fold,
                        "model": "extra_trees",
                        "score_scope": "aggregate",
                        "selected": True,
                        "params_json": params,
                        "inner_fold": inner,
                    }
                )
    assert modal_base_params(pd.DataFrame(rows))["n_estimators"] == 250


def test_full_predictor_validates_schema_and_applies_scale_and_blend():
    shared = _ConstantModel([1.0, 2.0])
    targetwise = _ConstantModel([4.0])
    predictor = FullDataQ1Predictor(
        feature_names=("a", "b"),
        target_names=("first", "second"),
        components=(
            TargetComponent("first", "fixed_scale_1.25", shared, 0, 2, prediction_scale=1.25),
            TargetComponent(
                "second",
                "half_multioutput_half_targetwise_et",
                shared,
                1,
                2,
                secondary_model=targetwise,
            ),
        ),
        training_input_sha256="x",
        training_target_sha256="y",
    )
    result = predictor.predict(pd.DataFrame({"b": [2.0], "a": [1.0]}))
    assert result.iloc[0].tolist() == [1.25, 3.0]


def test_saved_full_predictor_artifact_if_present():
    project_root = Path(__file__).resolve().parents[2]
    output = project_root / "problem3/artifacts/interfaces/q1_full_predictor_v1"
    if not (output / "manifest.json").exists():
        return
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["n_features"] == 231
    assert manifest["n_targets"] == 30

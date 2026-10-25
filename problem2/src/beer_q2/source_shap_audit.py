"""TreeSHAP integrity audit across every frozen Q1 source architecture.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from beer_q1.data import load_problem1_data

from .advanced import true_pipeline_shap
from .reconstruct import (
    FROZEN_SOURCE_BY_TARGET,
    _align_frozen_rows,
    _problem1_config,
    fit_representative_target_predictor,
)


def run_source_shap_audit(
    project_root: Path,
    repeat: int,
    fold: int,
    output_dir: Path,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    root = project_root.resolve()
    output = output_dir.resolve()
    if (root / "problem2").resolve() not in output.parents:
        raise ValueError("Question 2 output must stay under problem2")
    output.mkdir(parents=True, exist_ok=False)

    data = load_problem1_data(_problem1_config(root))
    folds = pd.read_csv(
        root / "problem1/freezes/problem1_v1/base_run_metadata/folds.csv"
    )
    tuning = pd.read_csv(
        root / "problem1/freezes/problem1_v1/base_run_metadata/tuning_trials.csv"
    )
    frozen = pd.read_parquet(
        root / "problem1/freezes/problem1_v1/champion_artifacts/oof_predictions.parquet"
    )
    representatives = {}
    for target in data.target_names:
        source = FROZEN_SOURCE_BY_TARGET[target]
        representatives.setdefault(source, target)

    cache: Dict[Tuple[object, ...], object] = {}
    rows: List[Dict[str, Any]] = []
    for source_model, target in representatives.items():
        predictor, _, valid_index = fit_representative_target_predictor(
            data,
            folds,
            tuning,
            target,
            int(repeat),
            int(fold),
            model_cache=cache,
        )
        X_valid = data.X.iloc[valid_index]
        prediction = predictor.predict(X_valid)
        frozen_rows = _align_frozen_rows(frozen, target, int(repeat), int(fold))
        reconstruction_error = float(
            np.max(np.abs(prediction - frozen_rows["y_pred"].to_numpy(dtype=float)))
        )
        values, expected, additivity_error = true_pipeline_shap(predictor, X_valid)
        nonzero_features = int((np.abs(values).max(axis=0) > 0).sum())
        expected_nonzero_maximum = (
            int(source_model.removeprefix("selectk"))
            if source_model.startswith("selectk")
            else len(data.feature_names)
        )
        rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "target": target,
                "source_model": source_model,
                "n_valid": int(len(X_valid)),
                "expected_value": expected,
                "reconstruction_max_abs_error": reconstruction_error,
                "shap_additivity_max_abs_error": additivity_error,
                "nonzero_source_features": nonzero_features,
                "expected_nonzero_feature_maximum": expected_nonzero_maximum,
                "selectk_zero_fill_passed": bool(
                    not source_model.startswith("selectk")
                    or nonzero_features <= expected_nonzero_maximum
                ),
                "passed": bool(
                    reconstruction_error <= 1e-9
                    and additivity_error <= tolerance
                    and (
                        not source_model.startswith("selectk")
                        or nonzero_features <= expected_nonzero_maximum
                    )
                ),
            }
        )
    audit = pd.DataFrame(rows).sort_values("source_model")
    audit.to_csv(output / "source_shap_audit.csv", index=False)
    summary = {
        "status": "PASS" if audit["passed"].all() else "FAIL",
        "repeat": int(repeat),
        "fold": int(fold),
        "source_architectures": int(len(audit)),
        "passed_architectures": int(audit["passed"].sum()),
        "max_reconstruction_abs_error": float(
            audit["reconstruction_max_abs_error"].max()
        ),
        "max_shap_additivity_abs_error": float(
            audit["shap_additivity_max_abs_error"].max()
        ),
        "all_selectk_zero_fill_passed": bool(
            audit["selectk_zero_fill_passed"].all()
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary

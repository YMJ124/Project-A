"""Reproduce all already-fixed specialist sources on 20311124 folds."""

from __future__ import annotations

from pathlib import Path

from beer_q1.config import load_config, resolve_paths
from beer_q1.runner import run_experiment


ROOT = Path(__file__).resolve().parents[2]
SEED = 20311124

SPECIALISTS = [
    (
        "problem1/configs/problem1_newseed2030_gbr_specialist.yaml",
        "q1_newseed2031_gbr_specialist_v1",
    ),
    (
        "problem1/configs/problem1_newseed2030_targetwise_et_specialist.yaml",
        "q1_newseed2031_targetwise_et_specialist_v1",
    ),
    (
        "problem1/configs/problem1_newseed2030_selectk30_specialist.yaml",
        "q1_newseed2031_selectk30_specialist_v1",
    ),
    (
        "problem1/configs/problem1_newseed2030_selectk60_specialist.yaml",
        "q1_newseed2031_selectk60_specialist_v1",
    ),
    (
        "problem1/configs/problem1_newseed2030_selectk120_specialist.yaml",
        "q1_newseed2031_selectk120_specialist_v1",
    ),
]


def _resolved(path: str):
    config = resolve_paths(load_config(ROOT / path), ROOT)
    config["validation"]["base_seed"] = SEED
    return config


def main() -> None:
    for path, run_id in SPECIALISTS:
        run_experiment(_resolved(path), run_id=run_id)

    affine = _resolved("problem1/configs/problem1_affine_calibration_confirmation.yaml")
    affine["models"]["fixed_without_inner_cv"] = True
    affine["models"]["enabled"] = [
        "global_mean",
        "style_mean",
        "extra_trees_affine_calibrated",
    ]
    params = affine["models"]["grids"]["extra_trees_affine_calibrated"][0]
    params["random_state_override"] = SEED
    run_experiment(affine, run_id="q1_newseed2031_affine_specialists_v1")


if __name__ == "__main__":
    main()

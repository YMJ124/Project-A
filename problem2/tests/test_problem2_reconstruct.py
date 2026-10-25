from pathlib import Path

import pandas as pd

from beer_q2.reconstruct import FROZEN_SOURCE_BY_TARGET, reconstruct_and_compare


def test_executable_source_map_matches_frozen_registry():
    project_root = Path(__file__).resolve().parents[2]
    registry = pd.read_csv(
        project_root / "problem1/freezes/problem1_v1/source_registry.csv"
    )
    observed = dict(zip(registry["target"], registry["source_model"]))
    assert FROZEN_SOURCE_BY_TARGET == observed


def test_representative_q1_reconstruction_one_repeat(tmp_path):
    result = reconstruct_and_compare(
        Path(__file__).resolve().parents[2],
        targets=["A_hops_all", "F_esters_isoaa", "A_hops_woody"],
        repeats=range(1),
        output_dir=tmp_path,
    )
    assert len(result) == 9
    assert result["passed_tolerance"].all()
    assert result["max_abs_error"].max() < 1e-9

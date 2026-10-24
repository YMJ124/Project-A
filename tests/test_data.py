"""Data contract regression tests.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from pathlib import Path

import pytest

from beer_q1.config import load_config, resolve_paths
from beer_q1.data import load_problem1_data


ROOT = Path(__file__).resolve().parents[1]


def test_official_attachments_match_problem1_contract() -> None:
    config = resolve_paths(load_config(ROOT / "configs/problem1_screening.yaml"), ROOT)
    data = load_problem1_data(config)
    assert data.X.shape == (250, 231)
    assert data.y.shape == (250, 30)
    assert len(data.main_target_names) == 30
    assert len(data.appendix_target_names) == 20
    assert data.ids.is_unique
    assert data.audit["missing_values_total"] == 73
    assert data.audit["join_key"] == "beer"
    assert "beer_id" not in data.feature_names
    assert "tasting_category_fine" not in data.feature_names


def test_explicit_target_names_select_only_declared_main_targets() -> None:
    config = resolve_paths(load_config(ROOT / "configs/problem1_screening.yaml"), ROOT)
    config["data"]["target_names"] = ["A_hops_all", "F_esters_ethac"]
    data = load_problem1_data(config)
    assert data.target_names == ["A_hops_all", "F_esters_ethac"]
    assert data.y.columns.tolist() == ["A_hops_all", "F_esters_ethac"]
    assert data.audit["n_selected_targets"] == 2


def test_explicit_target_names_reject_unknown_target() -> None:
    config = resolve_paths(load_config(ROOT / "configs/problem1_screening.yaml"), ROOT)
    config["data"]["target_names"] = ["not_a_sensory_target"]
    with pytest.raises(ValueError, match="outside target_scope"):
        load_problem1_data(config)

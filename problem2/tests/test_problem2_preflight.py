from pathlib import Path

import pandas as pd

from beer_q2.preflight import build_field_dictionary


def test_field_dictionary_preserves_order_and_aliases():
    fields = ["acetaldehyde", "CS2", "GFS", "esters_sum"]
    nodes = ["Acetaldehyde", "Carbon disulfide", "Sugars", "Total ester compounds"]
    result = build_field_dictionary(fields, nodes)
    assert result["position_1based"].tolist() == [1, 2, 3, 4]
    assert result.loc[1, "mapping_status"] == "alias_confirmed"
    assert result.loc[2, "mapping_status"] == "alias_confirmed"
    assert result.loc[3, "mapping_status"] == "alias_confirmed"


def test_reconstruction_registry_covers_30_main_targets():
    path = Path(__file__).resolve().parents[1] / "configs/problem2_q1_reconstruction_registry.csv"
    registry = pd.read_csv(path)
    assert len(registry) == 30
    assert registry["target"].nunique() == 30
    assert set(registry["grade"]) == {"A", "B", "C"}
    assert registry["fit_boundary"].eq("train_fold_only").all()

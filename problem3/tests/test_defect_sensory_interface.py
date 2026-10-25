from pathlib import Path

import pandas as pd


def test_frozen_defect_sensory_interface_records_stale_hops_limit():
    root = Path(__file__).resolve().parents[1] / "interfaces/defect_sensory_oof_v1"
    selection = pd.read_csv(root / "model_selection.csv").set_index("target")
    assert set(selection.index) == {"dms", "diacetyl", "X4vg", "t2n", "stale_hops"}
    assert selection.loc["X4vg", "grade"] == "A"
    assert selection.loc["stale_hops", "grade"] == "C"
    assert selection.loc["stale_hops", "rmse_improvement_vs_global"] < 0.05

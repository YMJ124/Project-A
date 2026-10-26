from pathlib import Path

import pandas as pd

from beer_q3.derived import audit_derived_formulas, load_formula_config, recompute_derived_features


def test_locked_formulas_reproduce_observed_aggregates(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source_config = root / "problem3/configs/derived_feature_formulas_v1.yaml"
    config_text = source_config.read_text(encoding="utf-8").replace(
        'output_root: "problem3/interfaces/derived_feature_formulas_v1"',
        f'output_root: "{tmp_path.as_posix()}"',
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    chemicals = root / "2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    summary = audit_derived_formulas(root, config_path, chemicals)
    assert summary["status"] == "PASS"
    assert summary["component_count"] == 21
    assert summary["maximum_absolute_error"] < 5e-8
    assert summary["uses_sensory_targets"] is False


def test_recomputation_overwrites_all_four_read_only_fields():
    root = Path(__file__).resolve().parents[2]
    config = load_formula_config(root / "problem3/configs/derived_feature_formulas_v1.yaml")
    chemicals = pd.read_csv(
        root / "2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    ).iloc[:5]
    computed = recompute_derived_features(chemicals, config)
    assert list(computed.columns) == [
        "esters_sum",
        "aroma_hops_sum",
        "acids_sum",
        "sulfur_sum",
    ]
    assert computed.notna().all().all()

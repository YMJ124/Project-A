from pathlib import Path

import pandas as pd

from beer_q3.external_audit import build_external_cohorts, freeze_external_audit


def test_locked_public_cohort_counts_and_missing_profiles(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "problem3/configs/external_test_data_v1.yaml"
    config = config_path.read_text(encoding="utf-8").replace(
        'output_root: "problem3/interfaces/external_test_data_v1"',
        f'output_root: "{tmp_path.as_posix()}"',
    )
    temporary = tmp_path / "config.yaml"
    temporary.write_text(config, encoding="utf-8")
    summary = freeze_external_audit(project_root, temporary)
    assert summary["status"] == "PASS_EXTERNAL_COHORTS_LOCKED_NOT_SCORED"
    assert summary["cohort_counts"] == {
        "Kriek_strict": 16,
        "Lambic_main": 147,
        "Pils_Lager_broad": 1050,
        "Pils_Lager_narrow": 350,
        "Tripel": 50,
    }
    assert summary["missing_descriptor_profiles"]["Tripel"] == 0
    assert summary["missing_descriptor_profiles"]["Kriek_strict"] == 0
    assert summary["missing_descriptor_profiles"]["Lambic_main"] == 11
    assert summary["missing_descriptor_profiles"]["Pils_Lager_broad"] == 7
    assert summary["external_scores_computed"] is False


def test_kriek_match_requires_a_complete_word():
    columns = {
        "key": [1, 2],
        "Name": ["Kriek sample", "Kriekish sample"],
        "Description": ["", ""],
        "Style": ["Lambic - Fruit", "Lambic - Fruit"],
    }
    columns.update({name: [1, 1] for name in (
        "Astringency", "Body", "Alcohol", "Bitter", "Sweet", "Sour",
        "Salty", "Fruits", "Hoppy", "Spices", "Malty"
    )})
    cohorts = build_external_cohorts(pd.DataFrame(columns))
    assert cohorts.loc[cohorts["cohort"] == "Kriek_strict", "key"].tolist() == [1]
    assert cohorts.loc[cohorts["cohort"] == "Lambic_main", "key"].tolist() == [2]

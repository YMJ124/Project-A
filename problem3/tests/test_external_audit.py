from pathlib import Path

import numpy as np
import pandas as pd

from beer_q3.external_audit import (
    build_external_cohorts,
    freeze_external_audit,
    score_external_anchor_audit,
    transform_public_descriptor_profiles,
)


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


def test_external_anchor_scoring_uses_public_beer_bootstrap_only():
    descriptor_names = [
        "Astringency", "Body", "Alcohol", "Bitter", "Sweet", "Sour",
        "Salty", "Fruits", "Hoppy", "Spices", "Malty",
    ]
    public = pd.DataFrame(
        {
            "key": range(1, 7),
            "Name": [f"b{i}" for i in range(6)],
            "Description": [""] * 6,
            "Style": ["Tripel", "Tripel", "Lambic - Fruit", "Lambic - Fruit", "Lambic - Gueuze", "Lambic - Gueuze"],
        }
    )
    for descriptor in descriptor_names:
        public[descriptor] = 1.0
    public.loc[:1, ["Fruits", "Malty"]] = [8.0, 5.0]
    public.loc[2:3, ["Fruits", "Hoppy"]] = [9.0, 1.0]
    public.loc[4:5, ["Fruits", "Hoppy"]] = [5.0, 2.0]
    cohorts = pd.DataFrame(
        {
            "cohort": ["Tripel", "Tripel", "Kriek_strict", "Kriek_strict", "Lambic_main", "Lambic_main"],
            "key": range(1, 7),
            "eligible_for_external_anchor_audit": [True] * 6,
        }
    )
    mapping = pd.DataFrame(
        {
            "external_descriptor": ["Malty", "Hoppy", "Fruits"],
            "q1_targets": ["A_malt_all;F_malt_all", "A_hops_all;F_hops_all", "A_esters_all;F_esters_all;A_esters_fruity;F_esters_fruity"],
        }
    )
    targets = sorted({value for values in mapping["q1_targets"] for value in values.split(";")})
    training = pd.DataFrame({target: np.linspace(-1, 1, 12) for target in targets})
    rows = []
    style_values = {"Tripel": 1.0, "Kriek": 0.5, "Lambic": -0.5}
    for style, value in style_values.items():
        for target in targets:
            rows.append(
                {
                    "sample_id": style,
                    "target_style": style,
                    "target": target,
                    "standardized_prediction": value,
                }
            )
    result = score_external_anchor_audit(
        public,
        cohorts,
        mapping,
        pd.DataFrame(rows),
        training,
        bootstrap_resamples=10,
        seed=7,
    )
    assert len(result["scorecard"]) == 3
    assert len(result["bootstrap"]) == 30
    assert result["summary"]["used_for_model_selection"] is False
    transformed = transform_public_descriptor_profiles(public)
    assert np.allclose(transformed.sum(axis=1), 1.0)

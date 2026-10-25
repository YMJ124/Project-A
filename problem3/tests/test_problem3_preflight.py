from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from beer_q3.preflight import build_style_support, classify_transform, run_preflight


def test_classify_transform_identifies_identity_and_log10():
    calibrated = pd.Series([0.1, 1.0, 10.0, 100.0])
    identity = classify_transform(calibrated, calibrated)
    log10 = classify_transform(pd.Series(np.log10(calibrated)), calibrated)
    assert identity["model_scale"] == "identity"
    assert identity["inverse_transform"] == "identity"
    assert log10["model_scale"] == "log10"
    assert log10["inverse_transform"] == "pow10"


def test_style_support_marks_generation_and_transfer_roles():
    styles = pd.Series(["Tripel"] * 12 + ["Kriek"] * 10 + ["Pils/Lager"] * 11)
    result = build_style_support(
        styles,
        generation_targets=["Tripel", "Kriek"],
        transfer_source="Pils/Lager",
        transfer_target="Tripel",
        minimum_required=10,
    ).set_index("style")
    assert result.loc["Tripel", "support_pass"]
    assert result.loc["Kriek", "support_pass"]
    assert "transfer_target" in result.loc["Tripel", "roles"]
    assert result.loc["Pils/Lager", "roles"] == "transfer_source"


def test_real_preflight_creates_auditable_contract(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "problem3/configs/problem3_preflight_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["runtime"]["output_root"] = str(tmp_path / "preflight")
    temporary_config = tmp_path / "config.yaml"
    temporary_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    summary = run_preflight(project_root, temporary_config)
    assert summary["status"] == "PASS_WITH_OPEN_INTERFACES"
    assert summary["n_samples"] == 250
    assert summary["n_features"] == 231
    assert summary["q2_status"] == "READY"
    assert summary["q1_full_data_predictor_available"] is False
    assert summary["validation_repeats"] == 7
    assert summary["validation_folds_per_repeat"] == 3
    assert summary["defect_chemical_constraints_ready"] == 4
    registry = pd.read_csv(tmp_path / "preflight/defect_constraint_registry.csv")
    assert set(registry["defect"]) == {
        "DMS",
        "Diacetyl",
        "4VG",
        "trans-2-nonenal",
        "Stale hops",
    }
    assert registry.loc[registry["defect"] == "DMS", "model_scale"].iloc[0] == "log10"
    assert registry.loc[registry["defect"] == "4VG", "model_scale"].iloc[0] == "log10"
    assert registry.loc[registry["defect"] == "Diacetyl", "threshold_used"].iloc[0] == 0.09
    assert (tmp_path / "preflight/validation_splits.csv").exists()

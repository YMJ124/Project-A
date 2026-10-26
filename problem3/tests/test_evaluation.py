import numpy as np
import pandas as pd

from beer_q3.baselines import empirical_resample
from beer_q3.evaluation import (
    DERIVED_FEATURES,
    FoldGeometry,
    build_real_real_reference,
    defect_violation_mask,
    evaluate_hard_gates,
    exact_copy_mask,
    map_relative_0_6,
)


def _toy_features(n=12):
    rng = np.random.default_rng(17)
    columns = ["x", "y", *DERIVED_FEATURES]
    return pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)


def test_fold_geometry_uses_training_only_and_rejects_schema_drift():
    train = _toy_features()
    geometry = FoldGeometry.fit(train)
    assert geometry.feature_names == tuple(train.columns)
    transformed = geometry.transform(train.iloc[:2])
    assert transformed.shape == (2, train.shape[1])
    try:
        geometry.transform(train.drop(columns="x"))
    except ValueError as error:
        assert "missing=['x']" in str(error)
    else:
        raise AssertionError("Schema drift was not rejected")


def test_empirical_reference_is_detected_as_exact_training_copy():
    train = _toy_features()
    styles = pd.Series(["Tripel"] * 6 + ["Kriek"] * 6)
    batch = empirical_resample(
        train,
        styles,
        np.arange(len(train)),
        "Tripel",
        4,
        seed=5,
    )
    assert exact_copy_mask(batch.chemicals, train).all()
    assert batch.derived_expected.equals(batch.chemicals.loc[:, list(DERIVED_FEATURES)])


def test_real_real_reference_uses_disjoint_fold_training_subsets():
    train = _toy_features(16)
    styles = pd.Series(["Tripel"] * 8 + ["Kriek"] * 8)
    geometry = FoldGeometry.fit(train)
    reference = build_real_real_reference(
        train,
        styles,
        "Tripel",
        desired_size=5,
        geometry=geometry,
        n_resamples=4,
        seed=9,
    )
    assert set(reference["subset_size"]) == {4}
    assert reference["resample"].nunique() == 4
    assert "wasserstein_median" in set(reference["metric"])


def test_relative_scale_is_monotone_and_clipped():
    training = pd.DataFrame({"a": [-1.0, 0.0, 1.0, 2.0]})
    predictions = pd.DataFrame({"a": [-100.0, 0.0, 100.0]})
    continuous, integer = map_relative_0_6(predictions, training)
    assert continuous["a"].tolist() == sorted(continuous["a"].tolist())
    assert continuous["a"].min() == 0.0
    assert continuous["a"].max() == 6.0
    assert set(integer["a"]).issubset(set(range(7)))


def test_illegal_defect_sample_triggers_absolute_and_relative_checks():
    generated = pd.DataFrame({"DMS": [0.0, 1.0], "t2n": [0.1, 0.3]})
    source = pd.DataFrame({"DMS": [0.0, 0.0], "t2n": [0.2, 0.2]})
    registry = pd.DataFrame(
        [
            {
                "defect": "DMS",
                "chemical_field": "DMS",
                "chemical_field_present": True,
                "inverse_transform": "pow10",
                "constraint_mode": "absolute_excess_nonincrease",
                "threshold_used": 5.0,
            },
            {
                "defect": "trans-2-nonenal",
                "chemical_field": "t2n",
                "chemical_field_present": True,
                "inverse_transform": "identity",
                "constraint_mode": "relative_concentration_nonincrease",
                "threshold_used": np.nan,
            },
        ]
    )
    failed, details = defect_violation_mask(generated, registry, source=source)
    assert failed.tolist() == [False, True]
    assert details.set_index("defect").loc["DMS", "violation_count"] == 1
    assert details.set_index("defect").loc["trans-2-nonenal", "violation_count"] == 1


def test_transfer_allows_existing_absolute_excess_but_not_worsening():
    generated = pd.DataFrame({"DMS": [1.0, 1.1]})
    source = pd.DataFrame({"DMS": [1.0, 1.0]})
    registry = pd.DataFrame(
        [
            {
                "defect": "DMS",
                "chemical_field": "DMS",
                "chemical_field_present": True,
                "inverse_transform": "pow10",
                "constraint_mode": "absolute_excess_nonincrease",
                "threshold_used": 5.0,
            }
        ]
    )
    failed, details = defect_violation_mask(generated, registry, source=source)
    assert failed.tolist() == [False, True]
    assert details.loc[0, "policy"] == "source_absolute_excess_nonincrease"


def test_near_neighbor_gate_is_calibrated_against_real_validation_rate():
    training = _toy_features(20)
    geometry = FoldGeometry.fit(training)
    validation = training.iloc[:4].copy() + 1e-6
    generated = validation.copy()
    generated.index = pd.Index([f"g{i}" for i in range(len(generated))])
    expected = generated.loc[:, list(DERIVED_FEATURES)].copy()
    empty_registry = pd.DataFrame(
        columns=[
            "chemical_field_present",
            "chemical_field",
            "constraint_mode",
            "defect",
            "inverse_transform",
            "threshold_used",
        ]
    )
    gates, _, _ = evaluate_hard_gates(
        generated,
        training,
        validation,
        geometry,
        empty_registry,
        target_style_training=training.iloc[:8],
        derived_expected=expected,
    )
    near = gates.set_index("gate").loc["near_neighbor_rate"]
    assert near["reference_failure_rate"] == 1.0
    assert near["allowed_failure_rate"] == 1.0
    assert bool(near["pass"])

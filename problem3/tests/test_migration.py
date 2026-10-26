import numpy as np
import pandas as pd

from beer_q3.evaluation import DERIVED_FEATURES, FoldGeometry, fit_style_classifier
from beer_q3.closure import _migration_sensory_target_audit
from beer_q3.migration import (
    build_migration_candidates,
    project_transfer_constraints,
    source_aware_range_mask,
)


def _toy_contract():
    base = ["a", "b", "DMS", "diacetyl", "X4VG", "t2n"]
    columns = [*base, *DERIVED_FEATURES]
    pils = pd.DataFrame(
        {
            "a": np.linspace(1.0, 2.0, 8),
            "b": np.linspace(0.5, 1.5, 8),
            "DMS": np.linspace(0.1, 0.3, 8),
            "diacetyl": np.linspace(0.05, 0.15, 8),
            "X4VG": np.linspace(0.05, 0.12, 8),
            "t2n": np.linspace(0.1, 0.2, 8),
        }
    )
    tripel = pils.copy()
    tripel["a"] += 2.0
    tripel["b"] += 1.0
    tripel["DMS"] += 0.05
    tripel["diacetyl"] += 0.02
    tripel["X4VG"] += 0.03
    tripel["t2n"] += 0.02
    training = pd.concat([pils, tripel], ignore_index=True)
    for feature in DERIVED_FEATURES:
        training[feature] = np.log10(training["a"])
    training = training.loc[:, columns]
    styles = pd.Series(["Pils/Lager"] * len(pils) + ["Tripel"] * len(tripel))
    actionability = pd.DataFrame(
        {
            "feature": columns,
            "generator_decision_variable": [True] * len(base) + [False] * 4,
            "consensus_module": ["M1", "M1", "M2", "M2", "M3", "M3", "D1", "D2", "D3", "D4"],
            "q2_candidate_count": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        }
    )
    defects = pd.DataFrame(
        [
            {
                "defect": name,
                "chemical_field": feature,
                "chemical_field_present": True,
                "inverse_transform": "identity",
                "constraint_mode": "relative_concentration_nonincrease" if name == "t2n" else "absolute_excess_nonincrease",
                "threshold_used": np.nan if name == "t2n" else 0.4,
            }
            for name, feature in [("DMS", "DMS"), ("Diacetyl", "diacetyl"), ("4VG", "X4VG"), ("t2n", "t2n")]
        ]
    )
    formulas = {
        "derived_features": {
            feature: [{"feature": "a", "threshold": 1.0, "inverse_transform": "identity"}]
            for feature in DERIVED_FEATURES
        }
    }
    return training, styles, actionability, defects, formulas


def test_transfer_projection_preserves_context_and_recomputes_derived():
    training, _, actionability, defects, formulas = _toy_contract()
    geometry = FoldGeometry.fit(training)
    source = training.iloc[[0]].copy()
    candidate = source.copy()
    candidate["a"] = 3.0
    candidate["t2n"] = 0.9
    projected, audit = project_transfer_constraints(
        candidate, source, geometry, defects, formulas, actionability
    )
    assert projected.loc[0, "t2n"] == source.loc[0, "t2n"]
    assert np.isclose(projected.loc[0, "esters_sum"], np.log10(projected.loc[0, "a"]))
    assert audit.set_index("defect").loc["t2n", "projected_rows"] == 1


def test_four_migration_mechanisms_produce_one_selection_per_source():
    training, styles, actionability, defects, formulas = _toy_contract()
    classifier = fit_style_classifier(training, styles, seed=3)
    source = training.iloc[:2].copy()
    result = build_migration_candidates(
        training,
        styles,
        np.arange(len(training)),
        source,
        [101, 102],
        actionability,
        defects,
        formulas,
        classifier,
        interpolation_budgets=[0.5, 1.0],
        module_budgets=[2],
        n_components=3,
        sinkhorn_iterations=50,
    )
    assert {"pca_centroid_shift", "shrink_coral", "sinkhorn_barycentric"}.issubset(
        set(result.metrics["method"])
    )
    assert any(value.startswith("module_counterfactual") for value in result.metrics["method"])
    assert len(result.selected) == 2
    assert result.pareto.groupby("source_beer")["selected"].sum().eq(1).all()
    assert result.selected["style_probability_gain"].notna().all()
    assert (result.selected["style_probability_gain"] > 0).all()
    assert (result.selected["target_distance_improvement"] >= 0).all()
    assert (~result.selected["near_neighbor_danger"]).all()


def test_migration_selection_can_be_restricted_to_module_counterfactuals():
    training, styles, actionability, defects, formulas = _toy_contract()
    classifier = fit_style_classifier(training, styles, seed=4)
    result = build_migration_candidates(
        training,
        styles,
        np.arange(len(training)),
        training.iloc[:2].copy(),
        [201, 202],
        actionability,
        defects,
        formulas,
        classifier,
        interpolation_budgets=[0.5, 1.0],
        module_budgets=[2],
        n_components=3,
        sinkhorn_iterations=50,
        eligible_method_prefixes=["module_counterfactual_"],
    )
    assert result.selected["method"].str.startswith("module_counterfactual_").all()
    latent = result.metrics["method"].isin(
        ["pca_centroid_shift", "shrink_coral", "sinkhorn_barycentric"]
    )
    assert (~result.metrics.loc[latent, "champion_eligible_method"]).all()


def test_migration_sensory_target_audit_reports_ab_and_all_scopes():
    training_targets = pd.DataFrame(
        {
            "A_good": [-1.0, -0.5, 1.0, 1.5],
            "F_weak": [-1.0, -0.5, 1.0, 1.5],
        }
    )
    styles = pd.Series(["Pils/Lager", "Pils/Lager", "Tripel", "Tripel"])
    before = pd.DataFrame({"A_good": [-0.8], "F_weak": [-0.8]})
    after = pd.DataFrame({"A_good": [0.8], "F_weak": [0.8]})
    audit = _migration_sensory_target_audit(
        before,
        after,
        training_targets,
        styles,
        [101],
        {"A_good": "A", "F_weak": "C"},
        target_style="Tripel",
    )
    assert set(audit["target_scope"]) == {"all_30", "ab_only"}
    assert audit["sensory_target_improved"].all()
    assert audit.set_index("target_scope").loc["ab_only", "target_count"] == 1


def test_source_aware_range_allows_existing_extreme_but_not_worsening():
    training, _, _, _, _ = _toy_contract()
    geometry = FoldGeometry.fit(training)
    source = training.iloc[[0]].copy()
    source.loc[source.index[0], "a"] = geometry.maximum["a"] + 2.0
    unchanged = source.copy()
    worsened = source.copy()
    worsened.loc[worsened.index[0], "a"] += 0.1
    assert source_aware_range_mask(unchanged, source, geometry).tolist() == [True]
    assert source_aware_range_mask(worsened, source, geometry).tolist() == [False]

"""Full-data engineering closure for Question 3.

This runner freezes deployable outputs from the current comprehensive/Pareto
choice.  It is not a substitute for outer-fold confirmation and therefore
keeps the inherited Q1 and generator limitations explicit in every manifest.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import pairwise_distances

from beer_q1.data import load_problem1_data

from .baselines import rank_stratified_skeptic_copula
from .derived import load_formula_config, recompute_derived_features
from .evaluation import (
    DERIVED_FEATURES,
    FoldGeometry,
    defect_violation_mask,
    exact_copy_mask,
    fit_style_classifier,
)
from .external_audit import (
    score_external_anchor_audit,
    score_external_migration_direction,
)
from .migration import build_migration_candidates
from .q1_adapter import load_full_data_predictor
from .smoke import _git_commit, _resolve, _sha256, _write_parquet_atomic


def _robust_target_scale(targets: pd.DataFrame) -> pd.Series:
    scale = (targets.quantile(0.75) - targets.quantile(0.25)) / 1.349
    fallback = targets.std(ddof=0)
    return scale.where(scale > 1e-12, fallback).where(lambda x: x > 1e-12, 1.0)


def _residual_quantiles(oof: pd.DataFrame) -> pd.DataFrame:
    selected = oof.loc[oof["model"] == "final_specialist_hybrid_v2"].copy()
    if selected.empty:
        raise ValueError("Frozen Q1 OOF artifact lacks final_specialist_hybrid_v2")
    selected["residual"] = selected["y_true"] - selected["y_pred"]
    return (
        selected.groupby("target")["residual"]
        .agg(
            residual_q025=lambda x: x.quantile(0.025),
            residual_q975=lambda x: x.quantile(0.975),
        )
        .reset_index()
    )


def build_deployment_sensory_profiles(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    training_targets: pd.DataFrame,
    grades: Mapping[str, str],
    residual_quantiles: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    """Create standardized, uncertainty and relative-0--6 long-form output."""
    required = {"sample_id", "target_style"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Sensory metadata are missing columns: {missing}")
    if list(predictions.columns) != list(training_targets.columns):
        raise ValueError("Prediction and training-target schemas differ")
    frame = predictions.copy().reset_index(drop=True)
    frame.insert(0, "target_style", metadata["target_style"].to_numpy())
    frame.insert(0, "sample_id", metadata["sample_id"].astype(str).to_numpy())
    long = frame.melt(
        id_vars=["sample_id", "target_style"],
        var_name="target",
        value_name="standardized_prediction",
    )
    for optional in ["source_beer", "method", "budget"]:
        if optional in metadata.columns:
            values = metadata.set_index("sample_id")[optional]
            long[optional] = long["sample_id"].map(values)
    long["role"] = role
    long["confidence_grade"] = long["target"].map(grades)
    long["primary_ab_target"] = long["confidence_grade"].isin(["A", "B"])
    long = long.merge(residual_quantiles, on="target", how="left", validate="many_to_one")
    long["standardized_prediction_lower"] = (
        long["standardized_prediction"] + long["residual_q025"]
    )
    long["standardized_prediction_upper"] = (
        long["standardized_prediction"] + long["residual_q975"]
    )
    center = training_targets.median()
    scale = _robust_target_scale(training_targets)
    target_center = long["target"].map(center).to_numpy(dtype=float)
    target_scale = long["target"].map(scale).to_numpy(dtype=float)
    for source, destination in [
        ("standardized_prediction", "relative_reconstructed_scale"),
        ("standardized_prediction_lower", "relative_reconstructed_scale_lower"),
        ("standardized_prediction_upper", "relative_reconstructed_scale_upper"),
    ]:
        long[destination] = np.clip(
            3.0
            + (long[source].to_numpy(dtype=float) - target_center)
            / np.maximum(target_scale, 1e-12),
            0.0,
            6.0,
        )
    long["relative_reconstructed_grade"] = np.rint(
        long["relative_reconstructed_scale"]
    ).astype(int)
    long["scale_interpretation"] = "relative_reconstructed_scale_not_raw_human_score"
    return long


def _deployment_generator_gates(
    generated: pd.DataFrame,
    training: pd.DataFrame,
    target_training: pd.DataFrame,
    geometry: FoldGeometry,
    defects: pd.DataFrame,
    formula_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chemistry = geometry.validate_schema(generated)
    finite = np.isfinite(chemistry.to_numpy(dtype=float)).all(axis=1)
    within = chemistry.ge(geometry.minimum - 1e-12).all(axis=1).to_numpy() & chemistry.le(
        geometry.maximum + 1e-12
    ).all(axis=1).to_numpy()
    expected = recompute_derived_features(chemistry, formula_config)
    derived = np.isclose(
        chemistry.loc[:, list(DERIVED_FEATURES)].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        atol=1e-10,
        rtol=0.0,
    ).all(axis=1)
    defect_failed, _ = defect_violation_mask(
        chemistry,
        defects,
        target_style_training=target_training,
    )
    copied = exact_copy_mask(chemistry, geometry.impute(training))
    train_z = geometry.transform(training)
    pairwise = pairwise_distances(train_z)
    np.fill_diagonal(pairwise, np.inf)
    danger_line = float(np.quantile(pairwise.min(axis=1), 0.05))
    nearest = pairwise_distances(geometry.transform(chemistry), train_z).min(axis=1)
    near = nearest < danger_line - 1e-12
    trace = pd.DataFrame(
        {
            "numeric_valid": finite,
            "within_training_minmax": within,
            "derived_fields_verified": derived,
            "chemical_defects_pass": ~defect_failed,
            "exact_training_copy": copied,
            "nearest_training_distance": nearest,
            "near_neighbor_danger": near,
        },
        index=chemistry.index,
    )
    rows = [
        ("numeric_validity", finite.all(), (~finite).sum()),
        ("physical_training_range", within.all(), (~within).sum()),
        ("derived_field_recomputation", derived.all(), (~derived).sum()),
        ("chemical_defect_constraints", (~defect_failed).all(), defect_failed.sum()),
        ("exact_copy_rate", (~copied).all(), copied.sum()),
        ("near_neighbor_q05", (~near).all(), near.sum()),
        ("end_to_end_output", True, 0),
    ]
    gates = pd.DataFrame(rows, columns=["gate", "pass", "failure_count"])
    gates["n_samples"] = len(chemistry)
    gates["all_hard_gates_pass"] = bool(gates["pass"].all())
    return gates, trace


def _migration_anchor_change(
    profiles: pd.DataFrame,
    anchor_mapping: pd.DataFrame,
    training_targets: pd.DataFrame,
) -> pd.DataFrame:
    center = training_targets.median()
    scale = _robust_target_scale(training_targets)
    rows = profiles.copy()
    rows["scaled"] = (
        rows["standardized_prediction"] - rows["target"].map(center)
    ) / rows["target"].map(scale)
    output = []
    for anchor in anchor_mapping.itertuples(index=False):
        targets = str(anchor.q1_targets).split(";")
        subset = rows.loc[rows["target"].isin(targets)]
        means = subset.groupby("role")["scaled"].mean()
        output.append(
            {
                "anchor": str(anchor.external_descriptor),
                "before_mean": float(means.loc["migration_before"]),
                "after_mean": float(means.loc["migration_after"]),
            }
        )
    return pd.DataFrame(output)


def _migration_sensory_target_audit(
    before: pd.DataFrame,
    after: pd.DataFrame,
    training_targets: pd.DataFrame,
    training_styles: pd.Series,
    source_ids: list[int],
    grades: Mapping[str, str],
    *,
    target_style: str,
) -> pd.DataFrame:
    """Audit whether frozen-Q1 profiles move toward the observed target style.

    This is deliberately descriptive: the full-data Q1 predictor and target
    centroid are deployment interfaces, not outer-fold generalization proof.
    """
    if len(before) != len(after) or len(before) != len(source_ids):
        raise ValueError("Migration sensory audit rows are not aligned")
    target_mask = training_styles.astype(str).to_numpy() == str(target_style)
    if not target_mask.any():
        raise ValueError(f"No observed sensory rows for target style {target_style}")
    center = training_targets.loc[target_mask].median()
    scale = _robust_target_scale(training_targets)
    rows: list[dict[str, Any]] = []
    for scope, columns in {
        "all_30": list(training_targets.columns),
        "ab_only": [name for name in training_targets.columns if grades.get(name) in {"A", "B"}],
    }.items():
        if not columns:
            continue
        denominator = scale.loc[columns].to_numpy(dtype=float)
        target = center.loc[columns].to_numpy(dtype=float)
        before_values = before.loc[:, columns].to_numpy(dtype=float)
        after_values = after.loc[:, columns].to_numpy(dtype=float)
        before_distance = np.sqrt(np.mean(((before_values - target) / denominator) ** 2, axis=1))
        after_distance = np.sqrt(np.mean(((after_values - target) / denominator) ** 2, axis=1))
        for position, source_id in enumerate(source_ids):
            rows.append(
                {
                    "source_beer": int(source_id),
                    "target_style": str(target_style),
                    "target_scope": scope,
                    "target_count": len(columns),
                    "sensory_distance_before": float(before_distance[position]),
                    "sensory_distance_after": float(after_distance[position]),
                    "sensory_distance_improvement": float(
                        before_distance[position] - after_distance[position]
                    ),
                    "sensory_target_improved": bool(
                        after_distance[position] < before_distance[position] - 1e-12
                    ),
                    "evidence_role": "full_data_frozen_q1_descriptive_audit",
                }
            )
    return pd.DataFrame(rows)


def run_engineering_closure(project_root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)

    q1_config = {
        "data": {
            "chemicals_csv": str(_resolve(project_root, config["data"]["chemicals_csv"])),
            "sensory_csv": str(_resolve(project_root, config["data"]["sensory_csv"])),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": "main",
        },
        "runtime": {},
    }
    data = load_problem1_data(q1_config)
    defects = pd.read_csv(_resolve(project_root, config["interfaces"]["defect_registry_csv"]))
    actionability = pd.read_csv(
        _resolve(project_root, config["interfaces"]["actionability_registry_csv"])
    )
    formula_config = load_formula_config(
        _resolve(project_root, config["interfaces"]["derived_formula_yaml"])
    )
    registry = pd.read_csv(_resolve(project_root, config["interfaces"]["q1_source_registry_csv"]))
    grades = dict(zip(registry["target"], registry["grade"]))
    oof = pd.read_parquet(_resolve(project_root, config["interfaces"]["q1_oof_predictions_parquet"]))
    residuals = _residual_quantiles(oof)
    predictor = load_full_data_predictor(
        _resolve(project_root, config["interfaces"]["q1_full_predictor_joblib"])
    )
    geometry = FoldGeometry.fit(data.X)
    classifier = fit_style_classifier(
        data.X,
        data.styles,
        seed=int(config["runtime"]["seed"]),
    )

    generation_frames = []
    generation_trace_frames = []
    generation_gate_frames = []
    generation_profile_frames = []
    for style_index, style in enumerate(config["generation"]["target_styles"]):
        style = str(style)
        count = int((data.styles.astype(str) == style).sum())
        target_training = data.X.loc[data.styles.astype(str) == style].reset_index(drop=True)
        batch = rank_stratified_skeptic_copula(
            data.X,
            data.styles,
            style,
            count,
            geometry,
            defects,
            formula_config,
            seed=int(config["runtime"]["seed"]) + style_index * 1009,
            prior_strength=float(config["generation"]["prior_strength"]),
            skeptic_blend_weight=float(config["generation"]["skeptic_blend_weight"]),
            minimum_training_distance_quantile=float(
                config["generation"]["minimum_training_distance_quantile"]
            ),
        )
        gates, trace = _deployment_generator_gates(
            batch.chemicals,
            data.X,
            target_training,
            geometry,
            defects,
            formula_config,
        )
        method = "rank_stratified_skeptic_blend0.5_prior5_distance_q0.05"
        metadata = pd.DataFrame(
            {
                "run_name": config["runtime"]["run_name"],
                "method": method,
                "target_style": style,
                "sample_id": batch.chemicals.index.astype(str),
            }
        )
        generation_frames.append(
            pd.concat([metadata, batch.chemicals.reset_index(drop=True)], axis=1)
        )
        trace.insert(0, "sample_id", batch.chemicals.index.astype(str))
        trace.insert(0, "target_style", style)
        trace.insert(0, "method", method)
        generation_trace_frames.append(trace.reset_index(drop=True))
        gates.insert(0, "target_style", style)
        gates.insert(0, "method", method)
        generation_gate_frames.append(gates)
        predictions = predictor.predict(batch.chemicals)
        generation_profile_frames.append(
            build_deployment_sensory_profiles(
                predictions,
                metadata,
                data.y,
                grades,
                residuals,
                role="generated",
            )
        )

    generated = pd.concat(generation_frames, ignore_index=True)
    generation_traces = pd.concat(generation_trace_frames, ignore_index=True)
    generation_gates = pd.concat(generation_gate_frames, ignore_index=True)
    generated_profiles = pd.concat(generation_profile_frames, ignore_index=True)

    source_mask = data.styles.astype(str) == str(config["migration"]["source_style"])
    migration_source = data.X.loc[source_mask].reset_index(drop=True)
    migration_source_ids = data.ids.loc[source_mask].astype(int).tolist()
    migration = build_migration_candidates(
        data.X,
        data.styles,
        data.ids,
        migration_source,
        migration_source_ids,
        actionability,
        defects,
        formula_config,
        classifier,
        source_style=str(config["migration"]["source_style"]),
        target_style=str(config["migration"]["target_style"]),
        interpolation_budgets=[float(value) for value in config["migration"]["interpolation_budgets"]],
        module_budgets=[int(value) for value in config["migration"]["module_budgets"]],
        n_components=int(config["migration"]["n_components"]),
        coral_shrinkage=float(config["migration"]["coral_shrinkage"]),
        sinkhorn_epsilon=float(config["migration"]["sinkhorn_epsilon"]),
        sinkhorn_iterations=int(config["migration"]["sinkhorn_iterations"]),
        selection_gain_fraction=float(config["migration"]["selection_gain_fraction"]),
    )
    selected_chemistry = migration.selected.loc[:, list(data.feature_names)]
    selected_ids = migration.selected["source_beer"].astype(int).tolist()
    source_by_id = data.X.copy()
    source_by_id.index = data.ids.astype(int)
    selected_source = source_by_id.loc[selected_ids].reset_index(drop=True)
    before_predictions = predictor.predict(selected_source)
    after_predictions = predictor.predict(selected_chemistry)
    migration_metadata = migration.selected.loc[
        :, ["candidate_id", "source_beer", "method", "budget"]
    ].rename(columns={"candidate_id": "sample_id"})
    migration_metadata["target_style"] = str(config["migration"]["target_style"])
    before_profiles = build_deployment_sensory_profiles(
        before_predictions,
        migration_metadata,
        data.y,
        grades,
        residuals,
        role="migration_before",
    )
    after_profiles = build_deployment_sensory_profiles(
        after_predictions,
        migration_metadata,
        data.y,
        grades,
        residuals,
        role="migration_after",
    )
    migration_profiles = pd.concat([before_profiles, after_profiles], ignore_index=True)
    migration_sensory_audit = _migration_sensory_target_audit(
        before_predictions,
        after_predictions,
        data.y,
        data.styles,
        selected_ids,
        grades,
        target_style=str(config["migration"]["target_style"]),
    )

    public_data = pd.read_csv(_resolve(project_root, config["external"]["beer_csv"]))
    cohort_registry = pd.read_csv(
        _resolve(project_root, config["external"]["cohort_registry_csv"])
    )
    anchor_mapping = pd.read_csv(
        _resolve(project_root, config["external"]["anchor_mapping_csv"])
    )
    external = score_external_anchor_audit(
        public_data,
        cohort_registry,
        anchor_mapping,
        generated_profiles,
        data.y,
        bootstrap_resamples=int(config["external"]["bootstrap_resamples"]),
        seed=int(config["external"]["bootstrap_seed"]),
    )
    migration_anchor_change = _migration_anchor_change(
        migration_profiles, anchor_mapping, data.y
    )
    external_migration = pd.concat(
        [
            score_external_migration_direction(
                public_data,
                cohort_registry,
                migration_anchor_change,
                source_cohort=cohort,
            )
            for cohort in ["Pils_Lager_broad", "Pils_Lager_narrow"]
        ],
        ignore_index=True,
    )

    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_parquet_atomic(generated, output / "generated_samples.parquet")
    _write_parquet_atomic(generation_traces, output / "sample_trace.parquet")
    generation_gates.to_csv(output / "hard_gates.csv", index=False)
    _write_parquet_atomic(generated_profiles, output / "sensory_profiles.parquet")
    _write_parquet_atomic(migration.candidates, output / "migration_candidates.parquet")
    migration.metrics.to_csv(output / "migration_metrics.csv", index=False)
    migration.pareto.to_csv(output / "migration_pareto.csv", index=False)
    _write_parquet_atomic(migration.selected, output / "migration_selected.parquet")
    _write_parquet_atomic(migration_profiles, output / "migration_sensory_profiles.parquet")
    migration_sensory_audit.to_csv(output / "migration_sensory_target_audit.csv", index=False)
    migration_anchor_change.to_csv(output / "migration_anchor_change.csv", index=False)
    external["scorecard"].to_csv(output / "external_anchor_scorecard.csv", index=False)
    external["bootstrap"].to_csv(output / "external_anchor_bootstrap.csv", index=False)
    external["public_profiles"].to_csv(output / "external_public_profiles.csv", index=False)
    external["internal_profiles"].to_csv(output / "external_internal_profiles.csv", index=False)
    external_migration.to_csv(output / "external_migration_direction.csv", index=False)
    (output / "external_summary.json").write_text(
        json.dumps(external["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected_metrics = migration.pareto.loc[migration.pareto["selected"]]
    generation_hard_pass = bool(generation_gates["pass"].all())
    migration_complete = len(migration.selected) == len(migration_source)
    positive_migrations = int((selected_metrics["style_probability_gain"] > 0.0).sum())
    chemistry_improving_migrations = int(
        (selected_metrics["target_distance_improvement"] >= 0.0).sum()
    )
    near_neighbor_safe_migrations = int((~selected_metrics["near_neighbor_danger"]).sum())
    migration_hard_pass = bool(
        migration_complete
        and positive_migrations == len(migration_source)
        and chemistry_improving_migrations == len(migration_source)
        and near_neighbor_safe_migrations == len(migration_source)
        and selected_metrics["hard_constraints_pass"].all()
    )
    ab_sensory = migration_sensory_audit.loc[
        migration_sensory_audit["target_scope"] == "ab_only"
    ]
    sensory_improved = int(ab_sensory["sensory_target_improved"].sum())
    manifest = {
        "status": "ENGINEERING_CLOSURE_COMPLETE_WITH_LIMITATIONS"
        if generation_hard_pass and migration_hard_pass
        else "ENGINEERING_CLOSURE_HOLD",
        "scope": "full_data_deployment_and_external_directional_audit_not_generalization_evidence",
        "run_name": config["runtime"]["run_name"],
        "current_generator_role": "best_comprehensive_engineering_pareto_candidate_not_formal_champion",
        "q1_status_inherited": "HOLD",
        "q2_status_inherited": "READY_WITH_LIMITATIONS",
        "generator_formal_status_inherited": "HOLD_NO_GENERATOR_CHAMPION",
        "generation_hard_gates_all_pass": generation_hard_pass,
        "generated_samples": int(len(generated)),
        "migration_source_samples": int(len(migration_source)),
        "migration_selected_samples": int(len(migration.selected)),
        "migration_positive_style_probability_gain_samples": positive_migrations,
        "migration_positive_target_distance_improvement_samples": chemistry_improving_migrations,
        "migration_near_neighbor_safe_samples": near_neighbor_safe_migrations,
        "migration_all_sources_have_selection": migration_complete,
        "migration_hard_gates_all_pass": migration_hard_pass,
        "migration_ab_sensory_target_improved_samples": sensory_improved,
        "migration_ab_sensory_target_audit_samples": int(len(ab_sensory)),
        "migration_sensory_evidence_role": "full_data_frozen_q1_descriptive_audit_not_hard_gate",
        "external_test_data_status": external["summary"]["status"],
        "external_used_for_model_selection": False,
        "external_evidence_role": "discovery_only_after_post_audit_engineering_correction",
        "future_optimization_rule": "external_results_are_discovery_only; new independent evidence required",
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit_at_run": _git_commit(project_root),
        "config_sha256": _sha256(config_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "decision.md").write_text(
        "# Q3 engineering end-to-end closure v1\n\n"
        f"- Status: `{manifest['status']}`.\n"
        f"- Generated target-style samples: {len(generated)}; all deployment hard gates pass: `{generation_hard_pass}`.\n"
        f"- Pils/Lager sources with a selected Tripel migration: {len(migration.selected)}/{len(migration_source)}.\n"
        f"- Selected migrations with positive target-style probability gain: {positive_migrations}/{len(migration_source)}.\n"
        f"- Selected migrations closer to the Tripel chemical center: {chemistry_improving_migrations}/{len(migration_source)}.\n"
        f"- Selected migrations outside the near-copy danger zone: {near_neighbor_safe_migrations}/{len(migration_source)}.\n"
        f"- Frozen-Q1 A/B sensory profiles closer to the observed Tripel center: {sensory_improved}/{len(ab_sensory)} (descriptive, not a hard gate).\n"
        f"- External three-anchor audit: `{external['summary']['status']}`; it is secondary evidence and was not used for model selection.\n\n"
        "This artifact closes the engineering path from chemistry generation/migration through Q2-aware constraints, frozen Q1 sensory prediction, relative reconstructed 0--6 reporting, and the public-data directional audit. The external results are now discovery-only because an internal hard-gate correction was made after the first audit; any future external confirmation needs independent evidence. It does not upgrade Q1 or the generator to READY. The current generator remains the best comprehensive engineering/Pareto choice with its predeclared Q2 hard-count limitation.\n",
        encoding="utf-8",
    )
    return manifest

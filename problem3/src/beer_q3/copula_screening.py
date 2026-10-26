"""Discovery screening for one mechanism: stronger style conditioning in copulas."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.optimize import linear_sum_assignment

from beer_q1.data import load_problem1_data
from beer_q2.reconstruct import (
    fit_representative_target_predictor,
    predict_frozen_targets,
)

from .baselines import (
    GeneratedBatch,
    factor_rank_gaussian_copula,
    moment_preserving_mixture_rank_gaussian_copula,
    module_block_rank_gaussian_copula,
    module_correlation_rank_gaussian_copula,
    rank_gaussian_copula,
    rank_stratified_skeptic_copula,
)
from .derived import load_formula_config
from .discovery import build_method_scorecard, summarize_discovery_metrics
from .evaluation import (
    FoldGeometry,
    attach_reference_targets,
    evaluate_generated_vs_real,
    evaluate_hard_gates,
    fit_style_classifier,
    summarize_reference,
)
from .smoke import _git_commit, _resolve, _sha256, _write_parquet_atomic


def _method_name(
    prior_strength: float,
    prefix: str = "rank_gaussian_copula",
) -> str:
    return f"{prefix}_prior_{prior_strength:g}"


def _module_method_name(prior_strength: float, module_weight: float) -> str:
    return (
        f"module_block_rank_gaussian_copula_prior_{prior_strength:g}"
        f"_weight_{module_weight:g}"
    )


def _module_correlation_method_name(
    prior_strength: float, module_weight: float
) -> str:
    return (
        f"module_correlation_rank_gaussian_copula_prior_{prior_strength:g}"
        f"_weight_{module_weight:g}"
    )


def _decoupled_method_name(
    mean_prior_strength: float, covariance_prior_strength: float
) -> str:
    return (
        f"rank_gaussian_copula_mean_prior_{mean_prior_strength:g}"
        f"_covariance_prior_{covariance_prior_strength:g}"
    )


def _factor_method_name(
    prior_strength: float,
    n_factors: int,
    minimum_distance_quantile: Optional[float] = None,
    poet_threshold_constant: Optional[float] = None,
) -> str:
    base = (
        f"factor_rank_gaussian_copula_prior_{prior_strength:g}"
        f"_factors_{int(n_factors)}"
    )
    if poet_threshold_constant is not None:
        base = f"{base}_poet_c{poet_threshold_constant:g}"
    if minimum_distance_quantile is not None:
        base = f"{base}_distance_q{minimum_distance_quantile:g}"
    return base


def _skeptic_method_name(
    prior_strength: float,
    minimum_distance_quantile: Optional[float] = None,
    blend_weight: float = 1.0,
    sensory_anchor_pool_multiplier: Optional[int] = None,
) -> str:
    base = f"skeptic_spearman_rank_gaussian_copula_prior_{prior_strength:g}"
    if float(blend_weight) < 1.0:
        base = f"{base}_blend_{blend_weight:g}"
    if minimum_distance_quantile is not None:
        base = f"{base}_distance_q{minimum_distance_quantile:g}"
    if sensory_anchor_pool_multiplier is not None:
        base = f"{base}_sensory_anchor{int(sensory_anchor_pool_multiplier)}"
    return base


def _moment_mixture_method_name(
    prior_strength: float,
    blend_weight: float,
    n_components: int,
    n_factors: int,
    between_variance_cap: float,
    minimum_distance_quantile: Optional[float],
) -> str:
    base = (
        f"moment_preserving_mixture{int(n_components)}_pc{int(n_factors)}"
        f"_cap{between_variance_cap:g}_skeptic_blend{blend_weight:g}"
        f"_prior{prior_strength:g}"
    )
    if minimum_distance_quantile is not None:
        base = f"{base}_distance_q{minimum_distance_quantile:g}"
    return base


def _rank_stratified_method_name(
    prior_strength: float,
    blend_weight: float,
    minimum_distance_quantile: Optional[float],
) -> str:
    base = (
        f"rank_stratified_skeptic_blend{blend_weight:g}"
        f"_prior{prior_strength:g}"
    )
    if minimum_distance_quantile is not None:
        base = f"{base}_distance_q{minimum_distance_quantile:g}"
    return base


def _metric_pass_rate(distances: pd.DataFrame, method: str, metric: str) -> float:
    rows = distances.loc[
        (distances["method"] == method) & (distances["metric"] == metric)
    ]
    return float(rows["status"].eq("PASS").mean()) if len(rows) else np.nan


def build_selection_audit(
    scorecard: pd.DataFrame,
    distances: pd.DataFrame,
    *,
    baseline_method: str,
    candidate_methods: list[str],
    minimum_style_hit_gain: float,
    maximum_structure_pass_rate_drop: float,
    minimum_q2_edge_strength_pass_rate: float = 0.0,
    minimum_q2_edge_strength_gain: float = -1.0,
) -> pd.DataFrame:
    """Apply the predeclared protection and improvement conditions."""
    indexed = scorecard.set_index("method")
    baseline = indexed.loc[baseline_method]
    baseline_structure = _metric_pass_rate(
        distances, baseline_method, "spearman_structure_rmse"
    )
    baseline_edges = _metric_pass_rate(
        distances, baseline_method, "q2_edge_strength_mae"
    )
    rows = []
    for method in candidate_methods:
        row = indexed.loc[method]
        structure = _metric_pass_rate(distances, method, "spearman_structure_rmse")
        edges = _metric_pass_rate(distances, method, "q2_edge_strength_mae")
        conditions = {
            "all_hard_gates": bool(row["all_hard_gates_all_cells"]),
            "wasserstein_core_protected": bool(
                row["wasserstein_core_pass_rate"]
                >= baseline["wasserstein_core_pass_rate"] - 1e-12
            ),
            "overall_distance_protected": bool(
                row["real_reference_distance_pass_rate"]
                >= baseline["real_reference_distance_pass_rate"] - 1e-12
            ),
            "style_hit_improved": bool(
                row["style_hit_rate_generated_mean"]
                >= baseline["style_hit_rate_generated_mean"]
                + float(minimum_style_hit_gain) - 1e-12
            ),
            "spearman_structure_protected": bool(
                structure
                >= baseline_structure - float(maximum_structure_pass_rate_drop) - 1e-12
            ),
            "q2_edge_strength_protected": bool(
                edges >= baseline_edges - float(maximum_structure_pass_rate_drop) - 1e-12
            ),
            "q2_edge_strength_recovered": bool(
                edges >= float(minimum_q2_edge_strength_pass_rate) - 1e-12
                and edges >= baseline_edges + float(minimum_q2_edge_strength_gain) - 1e-12
            ),
        }
        rows.append(
            {
                "method": method,
                "style_hit_gain": float(
                    row["style_hit_rate_generated_mean"]
                    - baseline["style_hit_rate_generated_mean"]
                ),
                "spearman_structure_pass_rate": structure,
                "q2_edge_strength_pass_rate": edges,
                **conditions,
                "all_predeclared_conditions": bool(all(conditions.values())),
            }
        )
    return pd.DataFrame(rows)


def apply_relaxed_pareto_confirmation(
    audit: pd.DataFrame,
    comparison_scorecard: pd.DataFrame,
    comparison_distances: pd.DataFrame,
    combined_gates: pd.DataFrame,
    combined_metrics: pd.DataFrame,
    *,
    baseline_method: str,
    candidate_methods: list[str],
    minimum_overall_distance_gain: float,
    minimum_wasserstein_core_gain: float,
    maximum_spearman_mean_relative_degradation: float,
    minimum_joint_repeat_wins: int,
    maximum_q2_edge_loss_pass_rate_drop: Optional[float] = None,
    maximum_q2_edge_strength_mean_relative_degradation: Optional[float] = None,
    maximum_q2_edge_loss_mean_relative_degradation: Optional[float] = None,
    maximum_q2_edge_sign_mean_relative_degradation: Optional[float] = None,
) -> pd.DataFrame:
    """Apply the pre-registered bounded-regression Pareto confirmation rules."""
    result = audit.copy()
    score = comparison_scorecard.set_index("method")
    baseline = score.loc[baseline_method]
    baseline_spearman = comparison_distances.loc[
        (comparison_distances["method"] == baseline_method)
        & (comparison_distances["metric"] == "spearman_structure_rmse"),
        "value",
    ].mean()
    baseline_q2_strength = comparison_distances.loc[
        (comparison_distances["method"] == baseline_method)
        & (comparison_distances["metric"] == "q2_edge_strength_mae"),
        "value",
    ].mean()
    baseline_q2_loss = comparison_distances.loc[
        (comparison_distances["method"] == baseline_method)
        & (comparison_distances["metric"] == "q2_edge_loss_rate"),
        "value",
    ].mean()
    baseline_q2_sign = comparison_distances.loc[
        (comparison_distances["method"] == baseline_method)
        & (comparison_distances["metric"] == "q2_edge_sign_agreement"),
        "value",
    ].mean()
    baseline_q2_loss_pass = _metric_pass_rate(
        comparison_distances, baseline_method, "q2_edge_loss_rate"
    )
    rows: dict[str, dict[str, Any]] = {}
    repeats = sorted(comparison_distances["repeat"].unique().astype(int).tolist())
    for method in candidate_methods:
        candidate = score.loc[method]
        candidate_spearman = comparison_distances.loc[
            (comparison_distances["method"] == method)
            & (comparison_distances["metric"] == "spearman_structure_rmse"),
            "value",
        ].mean()
        relative_degradation = float(
            (candidate_spearman - baseline_spearman)
            / max(abs(float(baseline_spearman)), 1e-12)
        )
        candidate_q2_strength = comparison_distances.loc[
            (comparison_distances["method"] == method)
            & (comparison_distances["metric"] == "q2_edge_strength_mae"),
            "value",
        ].mean()
        candidate_q2_loss = comparison_distances.loc[
            (comparison_distances["method"] == method)
            & (comparison_distances["metric"] == "q2_edge_loss_rate"),
            "value",
        ].mean()
        candidate_q2_sign = comparison_distances.loc[
            (comparison_distances["method"] == method)
            & (comparison_distances["metric"] == "q2_edge_sign_agreement"),
            "value",
        ].mean()
        candidate_q2_loss_pass = _metric_pass_rate(
            comparison_distances, method, "q2_edge_loss_rate"
        )
        q2_strength_relative_degradation = float(
            (candidate_q2_strength - baseline_q2_strength)
            / max(abs(float(baseline_q2_strength)), 1e-12)
        )
        q2_loss_relative_degradation = float(
            (candidate_q2_loss - baseline_q2_loss)
            / max(abs(float(baseline_q2_loss)), 1e-12)
        )
        q2_sign_relative_degradation = float(
            (baseline_q2_sign - candidate_q2_sign)
            / max(abs(float(baseline_q2_sign)), 1e-12)
        )
        repeat_rows = []
        for repeat in repeats:
            gates_repeat = combined_gates.loc[combined_gates["repeat"] == repeat]
            distances_repeat = comparison_distances.loc[
                comparison_distances["repeat"] == repeat
            ]
            metrics_repeat = combined_metrics.loc[combined_metrics["repeat"] == repeat]
            repeat_score = build_method_scorecard(
                gates_repeat, distances_repeat, metrics_repeat
            ).set_index("method")
            base_repeat = repeat_score.loc[baseline_method]
            candidate_repeat = repeat_score.loc[method]
            repeat_rows.append(
                {
                    "repeat": repeat,
                    "overall_gain": float(
                        candidate_repeat["real_reference_distance_pass_rate"]
                        - base_repeat["real_reference_distance_pass_rate"]
                    ),
                    "wasserstein_gain": float(
                        candidate_repeat["wasserstein_core_pass_rate"]
                        - base_repeat["wasserstein_core_pass_rate"]
                    ),
                    "style_gain": float(
                        candidate_repeat["style_hit_rate_generated_mean"]
                        - base_repeat["style_hit_rate_generated_mean"]
                    ),
                }
            )
        repeat_frame = pd.DataFrame(repeat_rows)
        joint_repeat_wins = int(
            (
                (repeat_frame["overall_gain"] > 0.0)
                & (repeat_frame["wasserstein_gain"] > 0.0)
                & (repeat_frame["style_gain"] > 0.0)
            ).sum()
        )
        rows[method] = {
            "overall_distance_gain": float(
                candidate["real_reference_distance_pass_rate"]
                - baseline["real_reference_distance_pass_rate"]
            ),
            "wasserstein_core_gain": float(
                candidate["wasserstein_core_pass_rate"]
                - baseline["wasserstein_core_pass_rate"]
            ),
            "spearman_mean_baseline": float(baseline_spearman),
            "spearman_mean_candidate": float(candidate_spearman),
            "spearman_mean_relative_degradation": relative_degradation,
            "q2_edge_strength_mean_baseline": float(baseline_q2_strength),
            "q2_edge_strength_mean_candidate": float(candidate_q2_strength),
            "q2_edge_strength_mean_relative_degradation": q2_strength_relative_degradation,
            "q2_edge_loss_pass_rate_baseline": float(baseline_q2_loss_pass),
            "q2_edge_loss_pass_rate_candidate": float(candidate_q2_loss_pass),
            "q2_edge_loss_mean_baseline": float(baseline_q2_loss),
            "q2_edge_loss_mean_candidate": float(candidate_q2_loss),
            "q2_edge_loss_mean_relative_degradation": q2_loss_relative_degradation,
            "q2_edge_sign_mean_baseline": float(baseline_q2_sign),
            "q2_edge_sign_mean_candidate": float(candidate_q2_sign),
            "q2_edge_sign_mean_relative_degradation": q2_sign_relative_degradation,
            "joint_repeat_wins": joint_repeat_wins,
            "overall_gain_sufficient": bool(
                candidate["real_reference_distance_pass_rate"]
                >= baseline["real_reference_distance_pass_rate"]
                + float(minimum_overall_distance_gain)
                - 1e-12
            ),
            "wasserstein_gain_sufficient": bool(
                candidate["wasserstein_core_pass_rate"]
                >= baseline["wasserstein_core_pass_rate"]
                + float(minimum_wasserstein_core_gain)
                - 1e-12
            ),
            "spearman_mean_within_relative_limit": bool(
                relative_degradation
                <= float(maximum_spearman_mean_relative_degradation) + 1e-12
            ),
            "joint_repeat_wins_sufficient": bool(
                joint_repeat_wins >= int(minimum_joint_repeat_wins)
            ),
        }
        if maximum_q2_edge_loss_pass_rate_drop is not None:
            rows[method]["q2_edge_loss_pass_rate_protected"] = bool(
                candidate_q2_loss_pass
                >= baseline_q2_loss_pass
                - float(maximum_q2_edge_loss_pass_rate_drop)
                - 1e-12
            )
        if maximum_q2_edge_strength_mean_relative_degradation is not None:
            rows[method]["q2_edge_strength_mean_within_relative_limit"] = bool(
                q2_strength_relative_degradation
                <= float(maximum_q2_edge_strength_mean_relative_degradation) + 1e-12
            )
        if maximum_q2_edge_loss_mean_relative_degradation is not None:
            rows[method]["q2_edge_loss_mean_within_relative_limit"] = bool(
                q2_loss_relative_degradation
                <= float(maximum_q2_edge_loss_mean_relative_degradation) + 1e-12
            )
        if maximum_q2_edge_sign_mean_relative_degradation is not None:
            rows[method]["q2_edge_sign_mean_within_relative_limit"] = bool(
                q2_sign_relative_degradation
                <= float(maximum_q2_edge_sign_mean_relative_degradation) + 1e-12
            )
    for column in next(iter(rows.values())):
        result[column] = result["method"].map(
            {method: values[column] for method, values in rows.items()}
        )
    extra_conditions = [
        "overall_gain_sufficient",
        "wasserstein_gain_sufficient",
        "spearman_mean_within_relative_limit",
        "joint_repeat_wins_sufficient",
    ]
    extra_conditions.extend(
        column
        for column in [
            "q2_edge_loss_pass_rate_protected",
            "q2_edge_strength_mean_within_relative_limit",
            "q2_edge_loss_mean_within_relative_limit",
            "q2_edge_sign_mean_within_relative_limit",
        ]
        if column in result.columns
    )
    result["all_predeclared_conditions"] = (
        result["all_predeclared_conditions"]
        & result.loc[:, extra_conditions].all(axis=1)
    )
    return result


def run_copula_screening(project_root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)

    data_paths = {
        key: _resolve(project_root, value)
        for key, value in config["data"].items()
        if key.endswith(("_csv", "_xlsx"))
    }
    q1_config = {
        "data": {
            "chemicals_csv": str(data_paths["chemicals_csv"]),
            "sensory_csv": str(data_paths["sensory_csv"]),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": "main",
        },
        "runtime": {},
    }
    data = load_problem1_data(q1_config)
    fold_path = _resolve(project_root, config["interfaces"]["folds_csv"])
    defect_path = _resolve(project_root, config["interfaces"]["defect_registry_csv"])
    edge_path = _resolve(project_root, config["interfaces"]["consensus_edges_csv"])
    formula_path = _resolve(project_root, config["interfaces"]["derived_formula_yaml"])
    baseline_root = _resolve(project_root, config["interfaces"]["baseline_discovery_root"])
    folds = pd.read_csv(fold_path)
    defects = pd.read_csv(defect_path)
    edges = pd.read_csv(edge_path)
    formulas = load_formula_config(formula_path)
    use_module_blocks = bool(config["screening"].get("use_module_blocks", False))
    use_module_correlations = bool(
        config["screening"].get("use_module_correlations", False)
    )
    use_decoupled_shrinkage = bool(
        config["screening"].get("use_decoupled_shrinkage", False)
    )
    use_factor_copula = bool(config["screening"].get("use_factor_copula", False))
    use_spearman_skeptic = bool(
        config["screening"].get("use_spearman_skeptic", False)
    )
    use_moment_preserving_mixture = bool(
        config["screening"].get("use_moment_preserving_mixture", False)
    )
    use_rank_stratified_sampling = bool(
        config["screening"].get("use_rank_stratified_sampling", False)
    )
    if sum(
        [
            use_module_blocks,
            use_module_correlations,
            use_decoupled_shrinkage,
            use_factor_copula,
            use_spearman_skeptic,
            use_moment_preserving_mixture,
            use_rank_stratified_sampling,
        ]
    ) > 1:
        raise ValueError("Select only one copula mechanism per experiment")
    modules = None
    if use_module_blocks or use_module_correlations:
        module_path = _resolve(project_root, config["interfaces"]["module_registry_csv"])
        modules = pd.read_csv(module_path)
    references = pd.read_csv(baseline_root / "real_real_reference.csv")
    baseline_gates = pd.read_csv(baseline_root / "hard_gates.csv")
    baseline_metrics = pd.read_csv(baseline_root / "metrics_by_fold.csv")
    baseline_distances = pd.read_csv(baseline_root / "distance_to_target.csv")

    repeats = [int(value) for value in config["evaluation"]["repeats"]]
    fold_values = [int(value) for value in config["evaluation"]["folds"]]
    target_styles = [str(value) for value in config["styles"]["generation_targets"]]
    strengths = [float(value) for value in config["screening"]["prior_strengths"]]
    module_weights = [
        float(value) for value in config["screening"].get("module_weights", [1.0])
    ]
    covariance_prior_strengths = [
        float(value)
        for value in config["screening"].get("covariance_prior_strengths", [])
    ]
    if use_decoupled_shrinkage and not covariance_prior_strengths:
        raise ValueError("Decoupled shrinkage requires covariance_prior_strengths")
    factor_counts = [
        int(value) for value in config["screening"].get("factor_counts", [])
    ]
    if use_factor_copula and not factor_counts:
        raise ValueError("Factor copula requires factor_counts")
    factor_distance_quantile = config["screening"].get(
        "factor_minimum_training_distance_quantile"
    )
    if factor_distance_quantile is not None:
        factor_distance_quantile = float(factor_distance_quantile)
    factor_poet_threshold_constant = config["screening"].get(
        "factor_poet_threshold_constant"
    )
    if factor_poet_threshold_constant is not None:
        factor_poet_threshold_constant = float(factor_poet_threshold_constant)
    skeptic_distance_quantile = config["screening"].get(
        "skeptic_minimum_training_distance_quantile"
    )
    if skeptic_distance_quantile is not None:
        skeptic_distance_quantile = float(skeptic_distance_quantile)
    skeptic_blend_weight = float(
        config["screening"].get("skeptic_blend_weight", 1.0)
    )
    mixture_components = int(config["screening"].get("mixture_components", 2))
    mixture_factors = int(config["screening"].get("mixture_factors", 5))
    mixture_between_variance_cap = float(
        config["screening"].get("mixture_between_variance_cap", 0.5)
    )
    mixture_min_component_size = int(
        config["screening"].get("mixture_min_component_size", 2)
    )
    if use_moment_preserving_mixture and mixture_components != 2:
        raise ValueError("Moment-preserving mixture is pre-registered with 2 components")
    use_sensory_anchor_selection = bool(
        config["screening"].get("use_sensory_anchor_selection", False)
    )
    sensory_anchor_pool_multiplier = int(
        config["screening"].get("sensory_anchor_pool_multiplier", 10)
    )
    if use_sensory_anchor_selection and not use_spearman_skeptic:
        raise ValueError("Sensory-anchor selection requires Spearman-SKEPTIC")
    if sensory_anchor_pool_multiplier < 2:
        raise ValueError("sensory_anchor_pool_multiplier must be at least 2")
    tuning = None
    primary_targets: list[str] = []
    if use_sensory_anchor_selection:
        tuning = pd.read_csv(
            _resolve(project_root, config["interfaces"]["q1_tuning_csv"])
        )
        registry = pd.read_csv(
            _resolve(project_root, config["interfaces"]["q1_source_registry_csv"])
        )
        primary_targets = registry.loc[
            registry["grade"].isin(["A", "B"]), "target"
        ].tolist()
        if len(primary_targets) != 24:
            raise ValueError("Sensory-anchor selection requires exactly 24 A/B targets")
    generated_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    defect_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    mixture_diagnostic_rows: list[dict[str, Any]] = []
    sampling_diagnostic_rows: list[dict[str, Any]] = []

    for repeat in repeats:
        selected = folds.loc[
            folds["repeat"] == repeat, ["beer", "repeat", "fold", "seed"]
        ]
        fold_by_beer = selected.set_index("beer")["fold"]
        assignments = data.ids.map(fold_by_beer)
        if assignments.isna().any() or len(selected) != len(data.X):
            raise ValueError(f"Frozen repeat {repeat} does not cover all beers")
        for fold in fold_values:
            train_mask = assignments.to_numpy() != fold
            valid_mask = ~train_mask
            X_train = data.X.loc[train_mask].reset_index(drop=True)
            y_train = data.y.loc[train_mask].reset_index(drop=True)
            X_valid = data.X.loc[valid_mask].reset_index(drop=True)
            style_train = data.styles.loc[train_mask].reset_index(drop=True)
            style_valid = data.styles.loc[valid_mask].reset_index(drop=True)
            geometry = FoldGeometry.fit(X_train)
            seed = int(selected["seed"].iloc[0]) + fold * 10007
            classifier = fit_style_classifier(X_train, style_train, seed=seed)
            q1_predictors = None
            training_sensory = None
            sensory_scale = None
            if use_sensory_anchor_selection:
                model_cache: dict[tuple[object, ...], object] = {}
                q1_predictors = []
                for target in data.target_names:
                    predictor, _, _ = fit_representative_target_predictor(
                        data,
                        folds,
                        tuning,
                        target,
                        repeat,
                        fold,
                        model_cache=model_cache,
                        n_jobs=int(config["runtime"].get("n_jobs", -1)),
                    )
                    q1_predictors.append(predictor)
                training_sensory = pd.DataFrame(
                    predict_frozen_targets(q1_predictors, X_train),
                    columns=data.target_names,
                ).loc[:, primary_targets]
                sensory_iqr = (
                    y_train.loc[:, primary_targets].quantile(0.75)
                    - y_train.loc[:, primary_targets].quantile(0.25)
                ) / 1.349
                sensory_fallback = y_train.loc[:, primary_targets].std(ddof=1)
                sensory_scale = sensory_iqr.where(
                    sensory_iqr > 1e-12, sensory_fallback
                ).fillna(1.0)

            for style_index, style in enumerate(target_styles):
                valid_style_mask = style_valid.astype(str).to_numpy() == style
                validation_style = X_valid.loc[valid_style_mask].reset_index(drop=True)
                target_training = X_train.loc[
                    style_train.astype(str).to_numpy() == style
                ].reset_index(drop=True)
                reference = references.loc[
                    (references["repeat"] == repeat)
                    & (references["fold"] == fold)
                    & (references["target_style"] == style)
                ]
                reference_summary = summarize_reference(reference)
                if validation_style.empty or reference.empty:
                    raise ValueError(f"Missing validation/reference data for {repeat}/{fold}/{style}")

                shared_seed = seed + style_index * 1009 + 307
                for strength in strengths:
                    uses_modules = use_module_blocks or use_module_correlations
                    variants = (
                        module_weights
                        if uses_modules
                        else covariance_prior_strengths
                        if use_decoupled_shrinkage
                        else factor_counts
                        if use_factor_copula
                        else [0.0]
                    )
                    for variant in variants:
                        method = (
                            _module_method_name(strength, variant)
                            if use_module_blocks
                            else _module_correlation_method_name(
                                strength, variant
                            )
                            if use_module_correlations
                            else _decoupled_method_name(strength, variant)
                            if use_decoupled_shrinkage
                            else _factor_method_name(
                                strength,
                                int(variant),
                                factor_distance_quantile,
                                factor_poet_threshold_constant,
                            )
                            if use_factor_copula
                            else _moment_mixture_method_name(
                                strength,
                                skeptic_blend_weight,
                                mixture_components,
                                mixture_factors,
                                mixture_between_variance_cap,
                                skeptic_distance_quantile,
                            )
                            if use_moment_preserving_mixture
                            else _rank_stratified_method_name(
                                strength,
                                skeptic_blend_weight,
                                skeptic_distance_quantile,
                            )
                            if use_rank_stratified_sampling
                            else _skeptic_method_name(
                                strength,
                                skeptic_distance_quantile,
                                skeptic_blend_weight,
                                (
                                    sensory_anchor_pool_multiplier
                                    if use_sensory_anchor_selection
                                    else None
                                ),
                            )
                            if use_spearman_skeptic
                            else _method_name(strength)
                        )
                        if use_module_blocks:
                            batch = module_block_rank_gaussian_copula(
                                X_train,
                                style_train,
                                style,
                                len(validation_style),
                                geometry,
                                defects,
                                formulas,
                                modules,
                                seed=shared_seed,
                                prior_strength=strength,
                                module_weight=variant,
                            )
                        elif use_module_correlations:
                            batch = module_correlation_rank_gaussian_copula(
                                X_train,
                                style_train,
                                style,
                                len(validation_style),
                                geometry,
                                defects,
                                formulas,
                                modules,
                                seed=shared_seed,
                                prior_strength=strength,
                                module_weight=variant,
                            )
                        elif use_factor_copula:
                            batch = factor_rank_gaussian_copula(
                                X_train,
                                style_train,
                                style,
                                len(validation_style),
                                geometry,
                                defects,
                                formulas,
                                seed=shared_seed,
                                prior_strength=strength,
                                n_factors=int(variant),
                                minimum_training_distance_quantile=(
                                    factor_distance_quantile
                                ),
                                poet_threshold_constant=(
                                    factor_poet_threshold_constant
                                ),
                            )
                        elif use_moment_preserving_mixture:
                            batch = moment_preserving_mixture_rank_gaussian_copula(
                                X_train,
                                style_train,
                                style,
                                len(validation_style),
                                geometry,
                                defects,
                                formulas,
                                seed=shared_seed,
                                prior_strength=strength,
                                skeptic_blend_weight=skeptic_blend_weight,
                                n_components=mixture_components,
                                n_factors=mixture_factors,
                                between_variance_cap=mixture_between_variance_cap,
                                min_component_size=mixture_min_component_size,
                                minimum_training_distance_quantile=(
                                    skeptic_distance_quantile
                                ),
                            )
                            mixture_diagnostic_rows.append(
                                {
                                    "method": method,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "target_style": style,
                                    **dict(batch.diagnostics or {}),
                                }
                            )
                        elif use_rank_stratified_sampling:
                            batch = rank_stratified_skeptic_copula(
                                X_train,
                                style_train,
                                style,
                                len(validation_style),
                                geometry,
                                defects,
                                formulas,
                                seed=shared_seed,
                                prior_strength=strength,
                                skeptic_blend_weight=skeptic_blend_weight,
                                minimum_training_distance_quantile=(
                                    skeptic_distance_quantile
                                ),
                            )
                            sampling_diagnostic_rows.append(
                                {
                                    "method": method,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "target_style": style,
                                    **dict(batch.diagnostics or {}),
                                }
                            )
                        elif use_spearman_skeptic:
                            requested_samples = (
                                len(validation_style) * sensory_anchor_pool_multiplier
                                if use_sensory_anchor_selection
                                else len(validation_style)
                            )
                            batch = rank_gaussian_copula(
                                X_train,
                                style_train,
                                style,
                                requested_samples,
                                geometry,
                                defects,
                                formulas,
                                seed=shared_seed,
                                prior_strength=strength,
                                dependence_estimator="spearman_skeptic",
                                minimum_training_distance_quantile=(
                                    skeptic_distance_quantile
                                ),
                                skeptic_blend_weight=skeptic_blend_weight,
                            )
                            if use_sensory_anchor_selection:
                                if q1_predictors is None or training_sensory is None or sensory_scale is None:
                                    raise AssertionError("Missing fold-local sensory selector")
                                pool_sensory = pd.DataFrame(
                                    predict_frozen_targets(q1_predictors, batch.chemicals),
                                    columns=data.target_names,
                                ).loc[:, primary_targets]
                                target_mask = style_train.astype(str).to_numpy() == style
                                anchor_pool = training_sensory.loc[target_mask].reset_index(drop=True)
                                anchor_rng = np.random.default_rng(shared_seed + 7919)
                                anchor_indices = anchor_rng.choice(
                                    len(anchor_pool),
                                    size=len(validation_style),
                                    replace=False,
                                )
                                anchors = anchor_pool.iloc[anchor_indices]
                                pool_z = pool_sensory.to_numpy(dtype=float) / sensory_scale.to_numpy(dtype=float)
                                anchor_z = anchors.to_numpy(dtype=float) / sensory_scale.to_numpy(dtype=float)
                                cost = np.sqrt(
                                    np.mean(
                                        (pool_z[:, None, :] - anchor_z[None, :, :]) ** 2,
                                        axis=2,
                                    )
                                )
                                selected_rows, _ = linear_sum_assignment(cost)
                                selected_rows = np.sort(selected_rows)
                                batch = GeneratedBatch(
                                    chemicals=batch.chemicals.iloc[selected_rows].reset_index(drop=True),
                                    derived_expected=batch.derived_expected.iloc[selected_rows].reset_index(drop=True),
                                    source_training_ids=tuple(
                                        batch.source_training_ids[index]
                                        for index in selected_rows
                                    ),
                                    method="sensory_anchor_selected_skeptic_copula",
                                )
                        else:
                            batch = rank_gaussian_copula(
                                X_train,
                                style_train,
                                style,
                                len(validation_style),
                                geometry,
                                defects,
                                formulas,
                                seed=shared_seed,
                                prior_strength=strength,
                                covariance_prior_strength=(
                                    variant if use_decoupled_shrinkage else None
                                ),
                            )
                        gates, trace, defect_detail = evaluate_hard_gates(
                            batch.chemicals,
                            X_train,
                            validation_style,
                            geometry,
                            defects,
                            target_style_training=target_training,
                            derived_expected=batch.derived_expected,
                        )
                        metrics = evaluate_generated_vs_real(
                            batch.chemicals,
                            validation_style,
                            X_train,
                            style_train,
                            style,
                            geometry,
                            classifier,
                            edges,
                        )
                        metadata = pd.DataFrame(
                            {
                                "run_name": config["runtime"]["run_name"],
                                "method": method,
                                "repeat": repeat,
                                "fold": fold,
                                "target_style": style,
                                "sample_id": [
                                    f"{style}__{method}__r{repeat}f{fold}__{i:04d}"
                                    for i in range(len(batch.chemicals))
                                ],
                                "source_training_beer": batch.source_training_ids,
                            }
                        )
                        generated_frames.append(
                            pd.concat(
                                [metadata, batch.chemicals.reset_index(drop=True)], axis=1
                            )
                        )
                        trace.insert(0, "sample_id", metadata["sample_id"].to_numpy())
                        trace.insert(0, "target_style", style)
                        trace.insert(0, "fold", fold)
                        trace.insert(0, "repeat", repeat)
                        trace.insert(0, "method", method)
                        trace_frames.append(trace.reset_index(drop=True))
                        for frame in (gates, defect_detail, metrics):
                            if "target_style" not in frame:
                                frame.insert(0, "target_style", style)
                            frame.insert(0, "fold", fold)
                            frame.insert(0, "repeat", repeat)
                            frame.insert(0, "method", method)
                        gate_frames.append(gates)
                        defect_frames.append(defect_detail)
                        metric_frames.append(metrics)

    generated = pd.concat(generated_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    gates = pd.concat(gate_frames, ignore_index=True)
    defect_details = pd.concat(defect_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    distances = []
    for (repeat, fold, style), group in metrics.groupby(
        ["repeat", "fold", "target_style"], sort=True
    ):
        reference = references.loc[
            (references["repeat"] == repeat)
            & (references["fold"] == fold)
            & (references["target_style"] == style)
        ]
        distances.append(attach_reference_targets(group, summarize_reference(reference)))
    distances = pd.concat(distances, ignore_index=True)
    metrics_summary = summarize_discovery_metrics(
        metrics,
        bootstrap_resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    scorecard = build_method_scorecard(gates, distances, metrics)

    baseline_method_original = str(
        config["screening"].get("baseline_method_source", "rank_gaussian_copula")
    )
    baseline_method = str(
        config["screening"].get(
            "baseline_method_label",
            _method_name(float(config["screening"]["baseline_prior_strength"])),
        )
    )
    baseline_gates = baseline_gates.loc[
        (baseline_gates["method"] == baseline_method_original)
        & baseline_gates["repeat"].isin(repeats)
        & baseline_gates["fold"].isin(fold_values)
    ].copy()
    baseline_metrics = baseline_metrics.loc[
        (baseline_metrics["method"] == baseline_method_original)
        & baseline_metrics["repeat"].isin(repeats)
        & baseline_metrics["fold"].isin(fold_values)
    ].copy()
    baseline_distances = baseline_distances.loc[
        (baseline_distances["method"] == baseline_method_original)
        & baseline_distances["repeat"].isin(repeats)
        & baseline_distances["fold"].isin(fold_values)
    ].copy()
    for frame in (baseline_gates, baseline_metrics, baseline_distances):
        frame["method"] = baseline_method
    baseline_score = build_method_scorecard(
        baseline_gates, baseline_distances, baseline_metrics
    )
    comparison_scorecard = pd.concat([baseline_score, scorecard], ignore_index=True)
    comparison_distances = pd.concat([baseline_distances, distances], ignore_index=True)
    combined_gates = pd.concat([baseline_gates, gates], ignore_index=True)
    combined_metrics = pd.concat([baseline_metrics, metrics], ignore_index=True)
    candidate_methods = (
        [
            _module_method_name(strength, weight)
            for strength in strengths
            for weight in module_weights
        ]
        if use_module_blocks
        else [
            _module_correlation_method_name(strength, weight)
            for strength in strengths
            for weight in module_weights
        ]
        if use_module_correlations
        else [
            _decoupled_method_name(strength, covariance_strength)
            for strength in strengths
            for covariance_strength in covariance_prior_strengths
        ]
        if use_decoupled_shrinkage
        else [
            _factor_method_name(
                strength,
                factor_count,
                factor_distance_quantile,
                factor_poet_threshold_constant,
            )
            for strength in strengths
            for factor_count in factor_counts
        ]
        if use_factor_copula
        else [
            _moment_mixture_method_name(
                strength,
                skeptic_blend_weight,
                mixture_components,
                mixture_factors,
                mixture_between_variance_cap,
                skeptic_distance_quantile,
            )
            for strength in strengths
        ]
        if use_moment_preserving_mixture
        else [
            _rank_stratified_method_name(
                strength,
                skeptic_blend_weight,
                skeptic_distance_quantile,
            )
            for strength in strengths
        ]
        if use_rank_stratified_sampling
        else [
            _skeptic_method_name(
                strength,
                skeptic_distance_quantile,
                skeptic_blend_weight,
                sensory_anchor_pool_multiplier if use_sensory_anchor_selection else None,
            )
            for strength in strengths
        ]
        if use_spearman_skeptic
        else [_method_name(value) for value in strengths]
    )
    audit = build_selection_audit(
        comparison_scorecard,
        comparison_distances,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
        minimum_style_hit_gain=float(config["selection"]["minimum_style_hit_gain"]),
        maximum_structure_pass_rate_drop=float(
            config["selection"]["maximum_structure_pass_rate_drop"]
        ),
        minimum_q2_edge_strength_pass_rate=float(
            config["selection"].get("minimum_q2_edge_strength_pass_rate", 0.0)
        ),
        minimum_q2_edge_strength_gain=float(
            config["selection"].get("minimum_q2_edge_strength_gain", -1.0)
        ),
    )
    selection_mode = str(config["selection"].get("mode", "strict"))
    if selection_mode == "relaxed_pareto_confirmation":
        audit = apply_relaxed_pareto_confirmation(
            audit,
            comparison_scorecard,
            comparison_distances,
            combined_gates,
            combined_metrics,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            minimum_overall_distance_gain=float(
                config["selection"]["minimum_overall_distance_gain"]
            ),
            minimum_wasserstein_core_gain=float(
                config["selection"]["minimum_wasserstein_core_gain"]
            ),
            maximum_spearman_mean_relative_degradation=float(
                config["selection"][
                    "maximum_spearman_mean_relative_degradation"
                ]
            ),
            minimum_joint_repeat_wins=int(
                config["selection"]["minimum_joint_repeat_wins"]
            ),
            maximum_q2_edge_loss_pass_rate_drop=config["selection"].get(
                "maximum_q2_edge_loss_pass_rate_drop"
            ),
            maximum_q2_edge_strength_mean_relative_degradation=config[
                "selection"
            ].get("maximum_q2_edge_strength_mean_relative_degradation"),
            maximum_q2_edge_loss_mean_relative_degradation=config[
                "selection"
            ].get("maximum_q2_edge_loss_mean_relative_degradation"),
            maximum_q2_edge_sign_mean_relative_degradation=config[
                "selection"
            ].get("maximum_q2_edge_sign_mean_relative_degradation"),
        )
    elif selection_mode != "strict":
        raise ValueError(f"Unsupported selection mode: {selection_mode}")
    passing = set(audit.loc[audit["all_predeclared_conditions"], "method"])
    selected_method = next(
        (method for method in candidate_methods if method in passing), None
    )
    phase = str(config["runtime"].get("phase", "screening"))
    if phase not in {"screening", "confirmation"}:
        raise ValueError(f"Unsupported copula experiment phase: {phase}")
    if phase == "confirmation":
        run_status = (
            "CONFIRMATION_CANDIDATE_PASSED"
            if selected_method
            else "CONFIRMATION_CANDIDATE_REJECTED"
        )
        confirmation_status = "passed" if selected_method else "rejected"
    else:
        run_status = (
            "SCREENING_CANDIDATE_LOCKED"
            if selected_method
            else "HOLD_NO_SCREENING_CANDIDATE"
        )
        confirmation_status = "not_run"

    used_folds = folds.loc[
        folds["repeat"].isin(repeats), ["beer", "repeat", "fold", "seed"]
    ].copy()
    used_folds["style"] = used_folds["beer"].map(dict(zip(data.ids, data.styles)))
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    used_folds.to_csv(output / "folds.csv", index=False)
    _write_parquet_atomic(generated, output / "generated_samples.parquet")
    _write_parquet_atomic(traces, output / "sample_trace.parquet")
    gates.to_csv(output / "hard_gates.csv", index=False)
    defect_details.to_csv(output / "defect_gate_details.csv", index=False)
    references.loc[
        references["repeat"].isin(repeats) & references["fold"].isin(fold_values)
    ].to_csv(output / "real_real_reference.csv", index=False)
    metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    metrics_summary.to_csv(output / "metrics_summary.csv", index=False)
    distances.to_csv(output / "distance_to_target.csv", index=False)
    comparison_scorecard.to_csv(output / "method_scorecard.csv", index=False)
    audit.to_csv(output / "selection_audit.csv", index=False)
    if mixture_diagnostic_rows:
        pd.DataFrame(mixture_diagnostic_rows).to_csv(
            output / "mixture_diagnostics.csv", index=False
        )
    if sampling_diagnostic_rows:
        pd.DataFrame(sampling_diagnostic_rows).to_csv(
            output / "sampling_diagnostics.csv", index=False
        )
    _write_parquet_atomic(
        pd.DataFrame(
            columns=[
                "method", "repeat", "fold", "target_style", "sample_id", "target",
                "standardized_prediction", "relative_reconstructed_scale", "confidence_grade",
            ]
        ),
        output / "sensory_profiles.parquet",
    )

    manifest = {
        "status": run_status,
        "phase": phase,
        "scope": (
            "q2_module_covariance_weight_single_mechanism_discovery_screening"
            if use_module_blocks
            else "q2_module_correlation_weight_single_mechanism_discovery_screening"
            if use_module_correlations
            else "decoupled_mean_covariance_shrinkage_single_mechanism_screening"
            if use_decoupled_shrinkage
            else "rank_gaussian_factor_poet_residual_single_mechanism_screening"
            if use_factor_copula and factor_poet_threshold_constant is not None
            else "rank_gaussian_factor_copula_single_mechanism_screening"
            if use_factor_copula
            else "moment_preserving_latent_mixture_single_mechanism_screening"
            if use_moment_preserving_mixture
            else "rank_stratified_parent_sampling_single_mechanism_screening"
            if use_rank_stratified_sampling
            else "rank_gaussian_spearman_skeptic_single_mechanism_screening"
            if use_spearman_skeptic
            else "copula_prior_strength_single_mechanism_discovery_screening"
        ),
        "run_name": config["runtime"]["run_name"],
        "repeats": repeats,
        "folds": fold_values,
        "prior_strengths": strengths,
        "use_module_blocks": use_module_blocks,
        "use_module_correlations": use_module_correlations,
        "use_decoupled_shrinkage": use_decoupled_shrinkage,
        "use_factor_copula": use_factor_copula,
        "use_spearman_skeptic": use_spearman_skeptic,
        "use_moment_preserving_mixture": use_moment_preserving_mixture,
        "use_rank_stratified_sampling": use_rank_stratified_sampling,
        "use_sensory_anchor_selection": use_sensory_anchor_selection,
        "sensory_anchor_pool_multiplier": (
            sensory_anchor_pool_multiplier if use_sensory_anchor_selection else None
        ),
        "factor_counts": factor_counts if use_factor_copula else [],
        "factor_minimum_training_distance_quantile": (
            factor_distance_quantile if use_factor_copula else None
        ),
        "factor_poet_threshold_constant": (
            factor_poet_threshold_constant if use_factor_copula else None
        ),
        "skeptic_minimum_training_distance_quantile": (
            skeptic_distance_quantile
            if use_spearman_skeptic
            or use_moment_preserving_mixture
            or use_rank_stratified_sampling
            else None
        ),
        "skeptic_blend_weight": (
            skeptic_blend_weight
            if use_spearman_skeptic
            or use_moment_preserving_mixture
            or use_rank_stratified_sampling
            else None
        ),
        "mixture_components": (
            mixture_components if use_moment_preserving_mixture else None
        ),
        "mixture_factors": mixture_factors if use_moment_preserving_mixture else None,
        "mixture_between_variance_cap": (
            mixture_between_variance_cap if use_moment_preserving_mixture else None
        ),
        "mixture_min_component_size": (
            mixture_min_component_size if use_moment_preserving_mixture else None
        ),
        "covariance_prior_strengths": (
            covariance_prior_strengths if use_decoupled_shrinkage else []
        ),
        "module_weights": (
            module_weights if use_module_blocks or use_module_correlations else []
        ),
        "baseline_prior_strength": float(config["screening"]["baseline_prior_strength"]),
        "selection_mode": selection_mode,
        "selected_method": selected_method,
        "confirmation_status": confirmation_status,
        "sensory_status": "not_run",
        "q1_status_inherited": "HOLD",
        "external_test_data_status": "cohorts_locked_not_scored",
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit_at_run": _git_commit(project_root),
        "config_sha256": _sha256(config_path),
        "source_hashes": {
            "copula_screening_py": _sha256(Path(__file__)),
            "baselines_py": _sha256(Path(__file__).with_name("baselines.py")),
            "evaluation_py": _sha256(Path(__file__).with_name("evaluation.py")),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    audit_lines = []
    condition_columns = [
        "all_hard_gates",
        "wasserstein_core_protected",
        "overall_distance_protected",
        "style_hit_improved",
        "spearman_structure_protected",
        "q2_edge_strength_protected",
        "q2_edge_strength_recovered",
        "overall_gain_sufficient",
        "wasserstein_gain_sufficient",
        "spearman_mean_within_relative_limit",
        "joint_repeat_wins_sufficient",
        "q2_edge_loss_pass_rate_protected",
        "q2_edge_strength_mean_within_relative_limit",
        "q2_edge_loss_mean_within_relative_limit",
        "q2_edge_sign_mean_within_relative_limit",
    ]
    for row in audit.itertuples(index=False):
        pareto_note = ""
        if selection_mode == "relaxed_pareto_confirmation":
            failed_conditions = [
                column
                for column in condition_columns
                if column in audit.columns and not bool(getattr(row, column))
            ]
            pareto_note = (
                f"overall gain={row.overall_distance_gain:.3f}, Wasserstein gain="
                f"{row.wasserstein_core_gain:.3f}, joint repeat wins="
                f"{int(row.joint_repeat_wins)}, Spearman mean relative change="
                f"{row.spearman_mean_relative_degradation:.3%}, Q2 strength mean "
                f"relative change={row.q2_edge_strength_mean_relative_degradation:.3%}, "
                f"Q2 loss mean relative change="
                f"{row.q2_edge_loss_mean_relative_degradation:.3%}, failed conditions="
                f"{failed_conditions}, "
            )
        audit_lines.append(
            f"- `{row.method}`: hard={row.all_hard_gates}, style gain="
            f"{row.style_hit_gain:.3f}, Wasserstein protected="
            f"{row.wasserstein_core_protected}, structure protected="
            f"{row.spearman_structure_protected and row.q2_edge_strength_protected}, "
            f"{pareto_note}accepted={row.all_predeclared_conditions}."
        )
    mechanism_note = (
        "Only the Q2 within-module covariance replacement weight changed; "
        "prior strength, folds, seeds, margins, constraints and evaluator stayed fixed."
        if use_module_blocks
        else "Only Q2 within-module correlations changed; the prior=5 mean, latent "
        "variances, empirical margins, folds, seeds, constraints and evaluator stayed fixed."
        if use_module_correlations
        else "Only covariance shrinkage changed; the prior=5 style mean, folds, seeds, "
        "empirical margins, constraints and evaluator stayed fixed."
        if use_decoupled_shrinkage
        else "Only a fixed C=1 POET-style sparse residual covariance was added to "
        "the locked 10-factor, prior-5, q05-distance generator; margins, folds, seeds, "
        "constraints and evaluator stayed fixed."
        if use_factor_copula and factor_poet_threshold_constant is not None
        else "Only the fixed 10-factor generator gained a fold-training q05 minimum-"
        "distance rejection layer; model parameters, margins, folds, seeds, constraints "
        "and evaluator stayed fixed."
        if use_factor_copula and factor_distance_quantile is not None
        else "Only the latent dependence representation changed to shared low-rank "
        "factors plus diagonal residuals; the prior=5 mean and variances, empirical "
        "margins, folds, seeds, constraints and evaluator stayed fixed."
        if use_factor_copula
        else "Only a fixed two-component structure was added in the five leading "
        "parent-whitened directions. Component offsets were capped and the shared "
        "within-component covariance was reduced so the protected half-shrunk "
        "SKEPTIC latent mean and total covariance stayed unchanged; empirical "
        "margins, q05 distance, folds, seeds, constraints and evaluator also stayed fixed."
        if use_moment_preserving_mixture
        else "Only the finite-batch sampling design changed to fixed midpoint normal "
        "scores reordered by the protected parent rank dependence. The fitted "
        "half-shrunk SKEPTIC distribution, empirical margins, q05 distance, folds, "
        "constraints and evaluator stayed fixed; infeasible designs were redrawn as "
        "whole batches rather than cherry-picked."
        if use_rank_stratified_sampling
        else "Only fold-training Q1 A/B sensory-anchor assignment selected samples "
        f"from a fixed {sensory_anchor_pool_multiplier}x pool; the chemical generator, "
        "folds, seeds, constraints and evaluator stayed fixed."
        if use_sensory_anchor_selection
        else "Only the latent correlation estimator changed to the fixed Spearman-"
        f"SKEPTIC/Ledoit-Wolf blend (weight={skeptic_blend_weight:g}); the prior=5 "
        "mean and variances, empirical margins, q05-distance layer, folds, seeds, "
        "constraints and evaluator stayed fixed."
        if use_spearman_skeptic
        else "Only style-conditioning shrinkage changed; folds, seeds, margins, "
        "constraints and evaluator stayed fixed."
    )
    decision_title = (
        "Copula candidate confirmation decision"
        if phase == "confirmation"
        else "Copula shrinkage screening decision"
    )
    eligibility_note = (
        "- A confirmation pass still requires Q1 sensory evaluation before champion eligibility.\n"
        if phase == "confirmation"
        else "- No candidate is a champion before independent confirmation and Q1 sensory evaluation.\n"
    )
    (output / "decision.md").write_text(
        f"# {decision_title}\n\n"
        f"- Status: `{manifest['status']}`.\n"
        f"- Selected {phase} candidate: `{selected_method}`.\n"
        f"- {mechanism_note}\n"
        f"{eligibility_note}\n"
        "## Predeclared audit\n\n" + "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )
    return manifest

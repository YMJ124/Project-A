"""Read-only preflight for Question 3 generation and style transfer.

The preflight fixes upstream identities, audits style support, and converts the
five official defect concepts into explicit machine-readable constraint
statuses. It deliberately does not fit a generator or a sensory model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from beer_q1.data import load_problem1_data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_transform(
    model_values: pd.Series,
    calibrated_values: pd.Series,
    *,
    atol: float = 1e-8,
    minimum_match: float = 0.95,
) -> Dict[str, Any]:
    """Classify identity/log10/ln correspondence using positive calibration pairs."""
    model = pd.to_numeric(model_values, errors="coerce").to_numpy(dtype=float)
    calibrated = pd.to_numeric(calibrated_values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(model) & np.isfinite(calibrated) & (calibrated > 0)
    if not mask.any():
        return {
            "n_positive_pairs": 0,
            "model_scale": "unresolved",
            "inverse_transform": "unresolved",
            "identity_match_fraction": 0.0,
            "log10_match_fraction": 0.0,
            "ln_match_fraction": 0.0,
        }

    observed = model[mask]
    reference = calibrated[mask]
    candidates = {
        "identity": reference,
        "log10": np.log10(reference),
        "ln": np.log(reference),
    }
    fractions = {
        name: float(np.mean(np.isclose(observed, values, atol=atol, rtol=0.0)))
        for name, values in candidates.items()
    }
    winner = max(fractions, key=fractions.get)
    if fractions[winner] < minimum_match:
        scale = "unresolved"
        inverse = "unresolved"
    else:
        scale = winner
        inverse = {"identity": "identity", "log10": "pow10", "ln": "exp"}[winner]
    return {
        "n_positive_pairs": int(mask.sum()),
        "model_scale": scale,
        "inverse_transform": inverse,
        "identity_match_fraction": fractions["identity"],
        "log10_match_fraction": fractions["log10"],
        "ln_match_fraction": fractions["ln"],
    }


def build_style_support(
    styles: pd.Series,
    generation_targets: Sequence[str],
    transfer_source: str,
    transfer_target: str,
    minimum_required: int,
) -> pd.DataFrame:
    counts = styles.value_counts().sort_index()
    requested = set(generation_targets) | {transfer_source, transfer_target}
    rows = []
    for style, count in counts.items():
        roles = []
        if style in generation_targets:
            roles.append("generation_target")
        if style == transfer_source:
            roles.append("transfer_source")
        if style == transfer_target:
            roles.append("transfer_target")
        rows.append(
            {
                "style": str(style),
                "n_samples": int(count),
                "requested": bool(style in requested),
                "roles": ";".join(roles),
                "minimum_required": int(minimum_required) if style in requested else np.nan,
                "support_pass": bool(count >= minimum_required) if style in requested else True,
            }
        )
    missing = sorted(requested.difference(counts.index.astype(str)))
    if missing:
        raise ValueError(f"Requested styles are absent: {missing}")
    return pd.DataFrame(rows)


def _path(project_root: Path, config: Mapping[str, Any], section: str, key: str) -> Path:
    value = Path(config[section][key])
    return value if value.is_absolute() else project_root / value


def _feature_schema(X: pd.DataFrame, quantiles: Iterable[float]) -> pd.DataFrame:
    q_values = list(quantiles)
    rows = []
    for feature in X.columns:
        series = X[feature]
        quantile_values = series.quantile(q_values)
        row: Dict[str, Any] = {
            "feature": feature,
            "missing_count": int(series.isna().sum()),
            "observed_count": int(series.notna().sum()),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "nonnegative_observed": bool((series.dropna() >= 0).all()),
            "constant_observed": bool(series.dropna().nunique() <= 1),
        }
        for quantile, value in zip(q_values, quantile_values):
            row[f"q{int(round(quantile * 100)):02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _sensory_schema(
    all_targets: Sequence[str],
    main_targets: Sequence[str],
    defect_targets: Iterable[str],
) -> pd.DataFrame:
    main = set(main_targets)
    defects = set(defect_targets)
    return pd.DataFrame(
        [
            {
                "target": target,
                "scope": "main_aroma_flavour" if target in main else "appendix",
                "q1_frozen_oof_available": bool(target in main),
                "official_defect_interface": bool(target in defects),
            }
            for target in all_targets
        ]
    )


def _defect_registry(
    config: Mapping[str, Any],
    features_by_beer: pd.DataFrame,
    calibrated_by_beer: pd.DataFrame,
    thresholds: pd.DataFrame,
    sensory_targets: Sequence[str],
) -> pd.DataFrame:
    protocol = config["protocol"]
    rows = []
    for defect in config["defects"]:
        feature = defect.get("chemical_field")
        sensory_target = defect["sensory_target"]
        row: Dict[str, Any] = {
            "defect": defect["name"],
            "chemical_field": feature,
            "sensory_target": sensory_target,
            "threshold_policy": defect["threshold_policy"],
            "sensory_target_present": bool(sensory_target in sensory_targets),
            "q1_frozen_sensory_predictor": bool(sensory_target.startswith(("A_", "F_"))),
        }
        if feature is None:
            row.update(
                {
                    "chemical_field_present": False,
                    "unit": None,
                    "threshold_mean": np.nan,
                    "threshold_median": np.nan,
                    "model_scale": "not_applicable",
                    "inverse_transform": "not_applicable",
                    "constraint_mode": "predicted_relative_nonincrease",
                    "constraint_ready": False,
                    "blocking_reason": "validated_appendix_sensory_predictor_missing",
                }
            )
            rows.append(row)
            continue

        if feature not in features_by_beer.columns or feature not in calibrated_by_beer.columns:
            raise ValueError(f"Defect chemical field is missing: {feature}")
        matches = thresholds[thresholds["field_name_R"].astype(str) == feature]
        if len(matches) != 1:
            raise ValueError(f"Expected one threshold metadata row for {feature}, got {len(matches)}")
        threshold = matches.iloc[0]
        transform = classify_transform(
            features_by_beer[feature],
            calibrated_by_beer[feature],
            atol=float(protocol["transform_match_atol"]),
            minimum_match=float(protocol["transform_match_minimum"]),
        )
        threshold_available = bool(
            pd.notna(threshold["threshold_mean"]) or pd.notna(threshold["threshold_median"])
        )
        available_thresholds = [
            float(value)
            for value in (threshold["threshold_mean"], threshold["threshold_median"])
            if pd.notna(value)
        ]
        threshold_used = min(available_thresholds) if available_thresholds else np.nan
        relative_only = defect["threshold_policy"] == "relative_nonincrease"
        transform_resolved = transform["model_scale"] != "unresolved"
        if threshold_available and transform_resolved:
            constraint_mode = "absolute_excess_nonincrease"
            ready = defect["threshold_policy"] in {
                "conservative_minimum",
                "conservative_minimum_with_style_audit",
            }
            reason = "threshold_policy_not_locked" if not ready else ""
        elif relative_only and transform_resolved:
            constraint_mode = "relative_concentration_nonincrease"
            ready = True
            reason = ""
        else:
            constraint_mode = "unresolved"
            ready = False
            reason = "threshold_or_transform_unresolved"
        row.update(
            {
                "chemical_field_present": True,
                "unit": threshold.get("unit"),
                "threshold_mean": threshold.get("threshold_mean"),
                "threshold_median": threshold.get("threshold_median"),
                "threshold_used": threshold_used,
                **transform,
                "constraint_mode": constraint_mode,
                "constraint_ready": ready,
                "blocking_reason": reason,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_preflight(project_root: Path, config_path: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    q1_config = {
        "data": {
            "chemicals_csv": str(_path(project_root, config, "data", "chemicals_csv")),
            "sensory_csv": str(_path(project_root, config, "data", "sensory_csv")),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": "all",
        },
        "runtime": {},
    }
    data = load_problem1_data(q1_config)
    protocol = config["protocol"]
    expected = {
        "n_samples": int(protocol["expected_samples"]),
        "n_features": int(protocol["expected_features"]),
        "n_styles": int(protocol["expected_styles"]),
        "n_main_targets": int(protocol["expected_main_targets"]),
        "n_all_targets": int(protocol["expected_all_targets"]),
    }
    observed = {
        "n_samples": len(data.X),
        "n_features": len(data.feature_names),
        "n_styles": int(data.styles.nunique()),
        "n_main_targets": len(data.main_target_names),
        "n_all_targets": len(data.target_names),
    }
    mismatches = [f"{key}: expected {value}, got {observed[key]}" for key, value in expected.items() if observed[key] != value]
    if mismatches:
        raise ValueError("Problem 3 core data contract failed:\n- " + "\n- ".join(mismatches))

    network_path = _path(project_root, config, "data", "chemical_network_csv")
    network = pd.read_csv(network_path)
    if network.shape != (231, 232):
        raise ValueError(f"Attachment 2 shape mismatch: {network.shape}")

    metadata_path = _path(project_root, config, "data", "metadata_xlsx")
    calibrated = pd.read_excel(metadata_path, sheet_name="Calibrated chemical dataset")
    thresholds = pd.read_excel(metadata_path, sheet_name="Sensory thresholds")
    if not calibrated["beer"].is_unique:
        raise ValueError("Attachment 8 calibrated dataset has duplicate beer values")
    calibrated = calibrated.set_index("beer").sort_index()
    features = data.X.copy()
    features.index = data.ids.to_numpy()
    features = features.sort_index()
    if not features.index.equals(calibrated.index):
        raise ValueError("Attachment 1 and calibrated metadata beer order/set mismatch")

    styles = config["styles"]
    style_support = build_style_support(
        data.styles,
        styles["generation_targets"],
        styles["transfer_source"],
        styles["transfer_target"],
        int(styles["minimum_required_samples"]),
    )
    requested_support = style_support.loc[style_support["requested"], "support_pass"]
    if not requested_support.all():
        failed = style_support.loc[style_support["requested"] & ~style_support["support_pass"], "style"].tolist()
        raise ValueError(f"Requested styles do not meet minimum support: {failed}")

    interface_paths = {
        key: _path(project_root, config, "interfaces", key)
        for key in config["interfaces"]
    }
    missing_interfaces = [key for key, value in interface_paths.items() if not value.exists()]
    if missing_interfaces:
        raise FileNotFoundError(f"Missing upstream interfaces: {missing_interfaces}")

    q1_gate = json.loads(interface_paths["q1_stage_gate"].read_text(encoding="utf-8"))
    q1_registry = pd.read_csv(interface_paths["q1_source_registry"])
    q1_folds = pd.read_csv(interface_paths["q1_folds"])
    q2_summary = json.loads(interface_paths["q2_summary"].read_text(encoding="utf-8"))
    q2_candidates = pd.read_csv(interface_paths["q2_candidate_catalog"])
    q2_decisions = pd.read_csv(interface_paths["q2_target_decisions"])
    q2_handoff = json.loads(
        interface_paths["q2_handoff_summary"].read_text(encoding="utf-8")
    )
    q2_actionability = pd.read_csv(interface_paths["q2_actionability_registry"])
    defect_selection = pd.read_csv(interface_paths["defect_sensory_selection"])
    defect_gate = json.loads(
        interface_paths["defect_sensory_stage_gate"].read_text(encoding="utf-8")
    )
    if len(q1_registry) != 30 or q1_registry["target"].nunique() != 30:
        raise ValueError("Q1 source registry does not cover 30 unique targets")
    if len(q1_folds) != 250 * 7 or q1_folds[["beer", "repeat"]].duplicated().any():
        raise ValueError("Q1 folds do not provide one assignment per beer and repeat")
    if set(q1_folds["fold"]) != {0, 1, 2} or q1_folds["repeat"].nunique() != 7:
        raise ValueError("Q1 folds do not provide the expected 7 repeats x 3 folds")
    if set(q1_folds["beer"]) != set(data.ids):
        raise ValueError("Q1 fold beer set differs from the Q3 data contract")
    if q2_summary.get("formal_question2_status") != "READY":
        raise ValueError("Q2 upstream status is not READY")
    if len(q2_decisions) != 30 or q2_decisions["target"].nunique() != 30:
        raise ValueError("Q2 target decisions do not cover 30 unique targets")
    if q2_handoff.get("status") != "PASS" or q2_handoff.get("partition_count") != 21:
        raise ValueError("Q2 handoff is absent or does not cover 7 repeats x 3 folds")
    if len(q2_actionability) != 231 or q2_actionability["feature"].nunique() != 231:
        raise ValueError("Q2 actionability handoff does not cover 231 unique features")
    causal_flags = (
        q2_actionability["causal_actionability_claimed"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    if causal_flags.any():
        raise ValueError("Q2 handoff unexpectedly claims causal actionability")
    expected_defects = {"dms", "diacetyl", "X4vg", "t2n", "stale_hops"}
    if set(defect_selection["target"]) != expected_defects:
        raise ValueError("Defect sensory evidence does not cover the five locked targets")
    if defect_gate.get("n_targets") != 5:
        raise ValueError("Defect sensory stage gate has an invalid target count")
    if len(q2_candidates) != q2_handoff.get("q2_candidate_count"):
        raise ValueError("Frozen Q2 catalog and Q3 handoff candidate counts differ")
    unknown_candidates = sorted(set(q2_candidates["representative_feature"]).difference(data.feature_names))
    if unknown_candidates:
        raise ValueError(f"Q2 candidates absent from the chemical schema: {unknown_candidates}")

    defect_targets = [item["sensory_target"] for item in config["defects"]]
    defects = _defect_registry(config, features, calibrated, thresholds, data.target_names)
    defect_evidence = defect_selection[
        ["target", "grade", "r2_mean", "spearman_mean", "rmse_improvement_vs_global"]
    ].rename(
        columns={
            "target": "sensory_target",
            "grade": "sensory_oof_grade",
            "r2_mean": "sensory_oof_r2",
            "spearman_mean": "sensory_oof_spearman",
            "rmse_improvement_vs_global": "sensory_oof_improvement_vs_global",
        }
    )
    defects = defects.merge(defect_evidence, on="sensory_target", how="left", validate="one_to_one")
    if defects["sensory_oof_grade"].isna().any():
        raise ValueError("A defect constraint is missing repeated OOF sensory evidence")
    defects["validated_sensory_predictor"] = defects["sensory_oof_grade"].isin(["A", "B"])
    feature_schema = _feature_schema(data.X, protocol["quantiles"])
    feature_schema["q2_candidate_representative"] = feature_schema["feature"].isin(
        q2_candidates["representative_feature"]
    )
    sensory_schema = _sensory_schema(data.target_names, data.main_target_names, defect_targets)
    sensory_schema = sensory_schema.merge(
        defect_evidence[["sensory_target", "sensory_oof_grade"]].rename(
            columns={"sensory_target": "target"}
        ),
        on="target",
        how="left",
        validate="one_to_one",
    )

    deployment_paths = {
        key: _path(project_root, config, "deployment", key)
        for key in config.get("deployment", {})
    }
    predictor_manifest_path = deployment_paths.get("q1_full_predictor_manifest")
    predictor_model_path = deployment_paths.get("q1_full_predictor_model")
    q1_predictor_available = bool(
        predictor_manifest_path
        and predictor_model_path
        and predictor_manifest_path.is_file()
        and predictor_model_path.is_file()
    )
    predictor_manifest: Dict[str, Any] = {}
    if q1_predictor_available:
        predictor_manifest = json.loads(
            predictor_manifest_path.read_text(encoding="utf-8")
        )
        q1_predictor_available = bool(
            predictor_manifest.get("status") == "PASS"
            and predictor_manifest.get("n_features") == 231
            and predictor_manifest.get("n_targets") == 30
            and predictor_manifest.get("model_sha256")
            == sha256_file(predictor_model_path)
        )

    output = _path(project_root, config, "runtime", "output_root")
    output.mkdir(parents=True, exist_ok=True)
    style_support.to_csv(output / "style_support.csv", index=False)
    feature_schema.to_csv(output / "chemical_feature_schema.csv", index=False)
    sensory_schema.to_csv(output / "sensory_interface_schema.csv", index=False)
    defects.to_csv(output / "defect_constraint_registry.csv", index=False)
    q1_folds.to_csv(output / "validation_splits.csv", index=False)

    input_paths = {
        "chemicals_csv": Path(q1_config["data"]["chemicals_csv"]),
        "sensory_csv": Path(q1_config["data"]["sensory_csv"]),
        "chemical_network_csv": network_path,
        "metadata_xlsx": metadata_path,
        **interface_paths,
        **{
            key: path for key, path in deployment_paths.items() if path.is_file()
        },
    }
    manifest = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in input_paths.items() if path.is_file()}
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    open_interfaces = []
    if not q1_predictor_available:
        open_interfaces.append("q1_full_data_champion_predictor")
    if not bool(
        defects.loc[
            defects["sensory_target"] == "stale_hops", "validated_sensory_predictor"
        ].iloc[0]
    ):
        open_interfaces.append("validated_stale_hops_appendix_predictor")
    summary = {
        "status": "PASS_WITH_OPEN_INTERFACES",
        **observed,
        "requested_styles_pass": True,
        "requested_style_counts": {
            row.style: int(row.n_samples)
            for row in style_support.itertuples()
            if row.requested
        },
        "q1_stage_status": q1_gate.get("status"),
        "q1_main_target_grades": {
            str(key): int(value) for key, value in q1_registry["grade"].value_counts().items()
        },
        "q1_full_data_predictor_available": q1_predictor_available,
        "q1_full_data_predictor_scope": predictor_manifest.get("scope"),
        "validation_repeats": int(q1_folds["repeat"].nunique()),
        "validation_folds_per_repeat": int(q1_folds["fold"].nunique()),
        "q2_status": q2_summary["formal_question2_status"],
        "q2_candidate_count": int(len(q2_candidates)),
        "q2_complete_top3_targets": int((q2_decisions["final_candidate_count"] >= 3).sum()),
        "q2_handoff_status": q2_handoff["status"],
        "q2_consensus_edge_count": int(q2_handoff["consensus_edge_count"]),
        "q2_consensus_module_count": int(q2_handoff["consensus_module_count"]),
        "defect_chemical_constraints_ready": int(defects["constraint_ready"].sum()),
        "defect_sensory_grade_counts": {
            str(key): int(value)
            for key, value in defect_selection["grade"].value_counts().items()
        },
        "validated_defect_sensory_predictors": int(
            defects["validated_sensory_predictor"].sum()
        ),
        "defect_total": int(len(defects)),
        "open_interfaces": open_interfaces,
        "safe_next_steps": [
            "fit_and_evaluate_nonparametric_conditional_generation_baseline",
            "implement_evaluation_on_locked_q1_validation_splits",
            "use_q2_consensus_modules_for_structured_regularization_and_candidate_audit",
            "report_stale_hops_as_exploratory_only",
        ],
        "no_model_training": True,
        "no_generator_selection": True,
    }
    (output / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return summary

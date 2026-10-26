"""Audited recomputation of the four read-only aggregate chemistry fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formula_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(config["derived_features"]) != {
        "esters_sum",
        "aroma_hops_sum",
        "acids_sum",
        "sulfur_sum",
    }:
        raise ValueError("Derived formula registry must cover exactly four locked fields")
    return config


def _physical_values(series: pd.Series, component: Mapping[str, Any]) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    transform = component["inverse_transform"]
    if transform == "pow10":
        physical = np.power(10.0, values)
    elif transform == "identity":
        physical = values.copy()
    else:
        raise ValueError(f"Unsupported inverse transform: {transform}")
    sentinel = component.get("zero_sentinel")
    if sentinel is not None:
        physical[np.isclose(values, float(sentinel), rtol=0.0, atol=1e-10)] = 0.0
    return physical


def recompute_derived_features(
    chemicals: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = pd.DataFrame(index=chemicals.index)
    for derived, components in config["derived_features"].items():
        missing = sorted(
            {component["feature"] for component in components}.difference(chemicals.columns)
        )
        if missing:
            raise ValueError(f"Missing components for {derived}: {missing}")
        oav_sum = np.zeros(len(chemicals), dtype=float)
        valid = np.ones(len(chemicals), dtype=bool)
        for component in components:
            physical = _physical_values(chemicals[component["feature"]], component)
            valid &= np.isfinite(physical)
            oav_sum += physical / float(component["threshold"])
        values = np.full(len(chemicals), np.nan)
        positive = valid & (oav_sum > 0.0)
        values[positive] = np.log10(oav_sum[positive])
        result[derived] = values
    return result


def apply_derived_features(
    chemicals: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = chemicals.copy()
    recomputed = recompute_derived_features(result, config)
    for feature in recomputed.columns:
        result[feature] = recomputed[feature]
    return result


def audit_derived_formulas(
    project_root: Path,
    formula_config_path: Path,
    chemicals_csv: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = load_formula_config(formula_config_path)
    metadata_path = Path(config["source"]["metadata_xlsx"])
    if not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path
    if sha256_file(metadata_path) != config["source"]["expected_sha256"]:
        raise ValueError("Attachment 8 hash differs from the locked formula source")
    thresholds = pd.read_excel(metadata_path, sheet_name="Sensory thresholds")
    threshold_by_feature = thresholds.set_index("field_name_R")["threshold_median"]
    rows = []
    for derived, components in config["derived_features"].items():
        for component in components:
            feature = component["feature"]
            source_threshold = float(threshold_by_feature.loc[feature])
            locked_threshold = float(component["threshold"])
            if not np.isclose(source_threshold, locked_threshold, rtol=0.0, atol=1e-12):
                raise ValueError(f"Threshold mismatch for {feature}")
            rows.append(
                {
                    "derived_feature": derived,
                    "component": feature,
                    "threshold_median": locked_threshold,
                    "inverse_transform": component["inverse_transform"],
                    "zero_sentinel": component.get("zero_sentinel"),
                    "source_sheet": "Sensory thresholds",
                }
            )

    chemicals = pd.read_csv(chemicals_csv)
    if len(chemicals) != int(config["validation"]["expected_rows"]):
        raise ValueError("Chemical row count differs from the formula audit contract")
    computed = recompute_derived_features(chemicals, config)
    tolerance = float(config["validation"]["absolute_tolerance"])
    validation_rows = []
    for feature in computed.columns:
        observed = chemicals[feature]
        mask = observed.notna() & computed[feature].notna()
        error = (computed.loc[mask, feature] - observed.loc[mask]).abs()
        validation_rows.append(
            {
                "derived_feature": feature,
                "observed_rows": int(mask.sum()),
                "missing_observed_rows": int(observed.isna().sum()),
                "maximum_absolute_error": float(error.max()),
                "rmse": float(np.sqrt(np.mean(error.to_numpy() ** 2))),
                "within_tolerance": bool((error <= tolerance).all()),
            }
        )
    validation = pd.DataFrame(validation_rows)
    if not validation["within_tolerance"].all():
        raise ValueError("A derived formula does not reproduce the observed aggregate")

    output = Path(config["runtime"]["output_root"])
    if not output.is_absolute():
        output = project_root / output
    output.mkdir(parents=True, exist_ok=True)
    components_frame = pd.DataFrame(rows)
    components_frame.to_csv(output / "components.csv", index=False)
    validation.to_csv(output / "validation.csv", index=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "formula": config["source"]["formula"],
        "source_metadata_sha256": sha256_file(metadata_path),
        "component_count": int(len(components_frame)),
        "derived_feature_count": int(len(validation)),
        "all_observed_rows_within_tolerance": bool(validation["within_tolerance"].all()),
        "maximum_absolute_error": float(validation["maximum_absolute_error"].max()),
        "uses_sensory_targets": False,
        "role": "deterministic_postprocessing_not_model_selection",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Derived chemistry formulas v1\n\n"
        "The four read-only aggregate fields are the base-10 logarithm of the sum of "
        "component concentrations divided by the locked median sensory thresholds in "
        "Attachment 8. Censored zero observations use the field-specific sentinels "
        "recorded in `components.csv`. The formulas use no sensory response targets.\n",
        encoding="utf-8",
    )
    tracked = [
        output / "README.md",
        output / "components.csv",
        output / "config.resolved.yaml",
        output / "summary.json",
        output / "validation.csv",
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in tracked),
        encoding="utf-8",
    )
    return summary

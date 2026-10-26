"""Lock the public BeerAdvocate cohort before any external model scoring."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


EXPECTED_HASHES = {
    "beer_data_set.csv": "bdfd28f8c75a0da1d86073198aa5341cfe19cabece56827a970915936e0b461a",
    "Beer Descriptors Simplified.xlsx": "161e8a841200f74a8158086192948dee25720576bb5fd28927054da05976f329",
}

DESCRIPTOR_COLUMNS = (
    "Astringency",
    "Body",
    "Alcohol",
    "Bitter",
    "Sweet",
    "Sour",
    "Salty",
    "Fruits",
    "Hoppy",
    "Spices",
    "Malty",
)

NARROW_PILS_LAGER_STYLES = {
    "Pilsner - German",
    "Pilsner - Bohemian / Czech",
    "Lager - American",
    "Lager - European Pale",
    "Lager - Helles",
    "Lager - Light",
    "Lager - Adjunct",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_external_cohorts(data: pd.DataFrame) -> pd.DataFrame:
    required = {"key", "Name", "Description", "Style", *DESCRIPTOR_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Public beer CSV is missing columns: {missing}")
    if not data["key"].is_unique:
        raise ValueError("Public beer key is not unique")

    combined_text = (
        data["Name"].fillna("").astype(str) + " " + data["Description"].fillna("").astype(str)
    )
    kriek = data["Style"].eq("Lambic - Fruit") & combined_text.str.contains(
        re.compile(r"\bkriek\b", flags=re.IGNORECASE), na=False
    )
    lambic = data["Style"].str.startswith("Lambic -", na=False) & ~kriek
    pils_lager_broad = data["Style"].str.contains("Pilsner|Lager", case=False, na=False)
    pils_lager_narrow = data["Style"].isin(NARROW_PILS_LAGER_STYLES)
    descriptor_total = data.loc[:, list(DESCRIPTOR_COLUMNS)].sum(axis=1)

    cohort_masks = {
        "Tripel": data["Style"].eq("Tripel"),
        "Kriek_strict": kriek,
        "Lambic_main": lambic,
        "Pils_Lager_broad": pils_lager_broad,
        "Pils_Lager_narrow": pils_lager_narrow,
    }
    frames = []
    for cohort, mask in cohort_masks.items():
        part = data.loc[mask, ["key", "Name", "Style"]].copy()
        part.insert(0, "cohort", cohort)
        part["descriptor_profile_available"] = descriptor_total.loc[mask].to_numpy() > 0
        part["eligible_for_external_anchor_audit"] = part["descriptor_profile_available"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _audit_dictionary(workbook_path: Path) -> dict[str, Any]:
    workbook = pd.ExcelFile(workbook_path)
    if workbook.sheet_names != ["Mouthfeel", "Taste", "Flavor And Aroma"]:
        raise ValueError(f"Unexpected descriptor dictionary sheets: {workbook.sheet_names}")
    summary = {}
    impact_values = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook_path, sheet_name=sheet)
        impact_columns = [column for column in frame.columns if str(column).startswith("Impact")]
        for column in impact_columns:
            impact_values.extend(frame[column].dropna().astype(float).tolist())
        summary[sheet] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "impact_columns": impact_columns,
        }
    if any(value != 1.0 for value in impact_values):
        raise ValueError("The descriptor dictionary contains a non-unit populated Impact")
    return {
        "sheet_count": len(workbook.sheet_names),
        "sheets": summary,
        "populated_impact_values": len(impact_values),
        "all_populated_impacts_equal_one": True,
    }


def freeze_external_audit(project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    csv_path = _resolve(project_root, config["data"]["beer_csv"])
    dictionary_path = _resolve(project_root, config["data"]["descriptor_xlsx"])
    paths = [csv_path, dictionary_path]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = EXPECTED_HASHES[path.name]
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Public data hash mismatch for {path.name}: {observed}")

    data = pd.read_csv(csv_path)
    if data.shape != (5558, 21):
        raise ValueError(f"Unexpected public beer CSV shape: {data.shape}")
    if data.duplicated().any():
        raise ValueError("Public beer CSV contains duplicate rows")
    cohorts = build_external_cohorts(data)
    counts = cohorts.groupby("cohort").size().to_dict()
    expected_counts = config["protocol"]["expected_cohort_counts"]
    if counts != expected_counts:
        raise ValueError(f"External cohort counts changed: observed={counts}, expected={expected_counts}")

    missing_profiles = (
        cohorts.assign(missing=~cohorts["descriptor_profile_available"])
        .groupby("cohort")["missing"]
        .sum()
        .astype(int)
        .to_dict()
    )
    dictionary = _audit_dictionary(dictionary_path)
    anchor_mapping = pd.DataFrame(config["anchors"])
    if set(anchor_mapping["external_descriptor"]) != {"Malty", "Hoppy", "Fruits"}:
        raise ValueError("The external anchor mapping is not the locked three-anchor contract")

    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(output / "cohort_registry.csv", index=False)
    anchor_mapping.to_csv(output / "anchor_mapping.csv", index=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        path.name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "status": "PASS_EXTERNAL_COHORTS_LOCKED_NOT_SCORED",
        "dataset_rows": int(len(data)),
        "dataset_columns": int(data.shape[1]),
        "unique_keys": int(data["key"].nunique()),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_name_rows": int(data["Name"].isna().sum()),
        "cohort_counts": {key: int(value) for key, value in counts.items()},
        "missing_descriptor_profiles": missing_profiles,
        "descriptor_dictionary": dictionary,
        "anchor_count": int(len(anchor_mapping)),
        "processing_locked": {
            "descriptor_transform": "log1p_then_row_composition",
            "zero_descriptor_total": "missing_profile_excluded_from_anchor_audit",
            "uncertainty_unit": "real_public_beer_bootstrap",
        },
        "used_for_model_selection": False,
        "external_scores_computed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = """# Public external sensory audit interface v1

This interface locks the public cohort membership, input hashes, missing-profile rule, and three sensory anchors before any Question 3 champion is scored.

The public descriptor columns are review word frequencies, not professional panel intensities, chemical concentrations, or original 0--6 labels. They are transformed with `log1p` and row-wise composition only after the generator champion is locked. Rows whose eleven descriptor counts sum to zero remain missing and are excluded from anchor scoring.

This freeze computes no external model score and must not be used for generator tuning, stopping, rejection thresholds, or champion replacement.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    tracked = [
        output / "README.md",
        output / "anchor_mapping.csv",
        output / "cohort_registry.csv",
        output / "config.resolved.yaml",
        output / "input_manifest.json",
        output / "summary.json",
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in tracked]
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return summary

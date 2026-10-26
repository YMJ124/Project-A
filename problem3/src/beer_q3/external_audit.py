"""Lock the public BeerAdvocate cohort before any external model scoring."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
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


STYLE_TO_EXTERNAL_COHORT = {
    "Tripel": "Tripel",
    "Kriek": "Kriek_strict",
    "Lambic": "Lambic_main",
}


def transform_public_descriptor_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the locked log1p plus row-composition public-data transform."""
    missing = sorted(set(DESCRIPTOR_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Public descriptor columns are missing: {missing}")
    values = data.loc[:, list(DESCRIPTOR_COLUMNS)].astype(float)
    if (values < 0.0).any().any():
        raise ValueError("Public descriptor counts cannot be negative")
    transformed = np.log1p(values)
    totals = transformed.sum(axis=1)
    composition = transformed.divide(totals.where(totals > 0.0), axis=0)
    composition.index = data.index
    return composition


def _anchor_targets(anchor_mapping: pd.DataFrame) -> dict[str, list[str]]:
    required = {"external_descriptor", "q1_targets"}
    missing = sorted(required.difference(anchor_mapping.columns))
    if missing:
        raise ValueError(f"Anchor mapping is missing columns: {missing}")
    mapping = {
        str(row.external_descriptor): [
            value for value in str(row.q1_targets).split(";") if value
        ]
        for row in anchor_mapping.itertuples(index=False)
    }
    if set(mapping) != {"Malty", "Hoppy", "Fruits"}:
        raise ValueError("Only the locked Malty/Hoppy/Fruits anchors may be scored")
    return mapping


def build_public_anchor_profiles(
    public_data: pd.DataFrame,
    cohort_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return cohort means and row-level transformed profiles for bootstrapping."""
    composition = transform_public_descriptor_profiles(public_data)
    by_key = public_data[["key"]].join(composition).set_index("key")
    row_profiles: dict[str, pd.DataFrame] = {}
    means = []
    for style, cohort in STYLE_TO_EXTERNAL_COHORT.items():
        keys = cohort_registry.loc[
            (cohort_registry["cohort"] == cohort)
            & cohort_registry["eligible_for_external_anchor_audit"].astype(bool),
            "key",
        ]
        rows = by_key.loc[keys, ["Malty", "Hoppy", "Fruits"]].dropna()
        if rows.empty:
            raise ValueError(f"No eligible public anchor rows for {cohort}")
        row_profiles[style] = rows.reset_index(drop=True)
        mean = rows.mean()
        means.extend(
            {
                "target_style": style,
                "external_cohort": cohort,
                "anchor": anchor,
                "public_mean_composition": float(mean[anchor]),
                "n_public_beers": int(len(rows)),
            }
            for anchor in ["Malty", "Hoppy", "Fruits"]
        )
    return pd.DataFrame(means), row_profiles


def build_internal_anchor_profiles(
    sensory_profiles: pd.DataFrame,
    anchor_mapping: pd.DataFrame,
    training_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Q1 predictions after locked training-distribution scaling."""
    required = {"sample_id", "target_style", "target", "standardized_prediction"}
    missing = sorted(required.difference(sensory_profiles.columns))
    if missing:
        raise ValueError(f"Sensory profiles are missing columns: {missing}")
    mapping = _anchor_targets(anchor_mapping)
    targets = sorted({target for values in mapping.values() for target in values})
    unavailable = sorted(set(targets).difference(training_targets.columns))
    if unavailable:
        raise ValueError(f"Training sensory targets are missing: {unavailable}")
    center = training_targets.loc[:, targets].median()
    scale = (training_targets.loc[:, targets].quantile(0.75) - training_targets.loc[:, targets].quantile(0.25)) / 1.349
    fallback = training_targets.loc[:, targets].std(ddof=0)
    scale = scale.where(scale > 1e-12, fallback).where(lambda x: x > 1e-12, 1.0)
    rows = sensory_profiles.loc[
        sensory_profiles["target_style"].isin(STYLE_TO_EXTERNAL_COHORT)
        & sensory_profiles["target"].isin(targets)
    ].copy()
    rows["scaled_prediction"] = (
        rows["standardized_prediction"] - rows["target"].map(center)
    ) / rows["target"].map(scale)
    sample_rows = []
    for anchor, anchor_targets in mapping.items():
        selected = rows.loc[rows["target"].isin(anchor_targets)]
        grouped = selected.groupby(["sample_id", "target_style"], as_index=False)[
            "scaled_prediction"
        ].mean()
        grouped["anchor"] = anchor
        grouped = grouped.rename(columns={"scaled_prediction": "internal_anchor_value"})
        sample_rows.append(grouped)
    sample_profiles = pd.concat(sample_rows, ignore_index=True)
    means = (
        sample_profiles.groupby(["target_style", "anchor"], as_index=False)
        .agg(
            internal_mean=("internal_anchor_value", "mean"),
            n_generated_samples=("sample_id", "nunique"),
        )
    )
    return means, sample_profiles


def _centered_vector(frame: pd.DataFrame, value_column: str, anchor: str) -> np.ndarray:
    ordered = frame.loc[frame["anchor"] == anchor].set_index("target_style").loc[
        ["Tripel", "Kriek", "Lambic"], value_column
    ].to_numpy(dtype=float)
    return ordered - ordered.mean()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else np.nan


def _direction_agreement(left: np.ndarray, right: np.ndarray) -> tuple[int, int]:
    pairs = [(0, 1), (0, 2), (1, 2)]
    agreement = [
        np.sign(left[a] - left[b]) == np.sign(right[a] - right[b])
        for a, b in pairs
        if abs(left[a] - left[b]) > 1e-12 and abs(right[a] - right[b]) > 1e-12
    ]
    return int(sum(agreement)), int(len(agreement))


def score_external_anchor_audit(
    public_data: pd.DataFrame,
    cohort_registry: pd.DataFrame,
    anchor_mapping: pd.DataFrame,
    sensory_profiles: pd.DataFrame,
    training_targets: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Score the one-time, secondary three-anchor external direction audit.

    Only public-beer rows are bootstrapped.  Generated rows are not treated as
    independent evidence and therefore do not narrow the uncertainty interval.
    """
    public_means, public_rows = build_public_anchor_profiles(public_data, cohort_registry)
    internal_means, internal_rows = build_internal_anchor_profiles(
        sensory_profiles, anchor_mapping, training_targets
    )
    anchors = ["Malty", "Hoppy", "Fruits"]
    observed_rows = []
    for anchor in anchors:
        public_vector = _centered_vector(public_means, "public_mean_composition", anchor)
        internal_vector = _centered_vector(internal_means, "internal_mean", anchor)
        agrees, evaluable = _direction_agreement(public_vector, internal_vector)
        observed_rows.append(
            {
                "anchor": anchor,
                "cosine_similarity": _cosine(public_vector, internal_vector),
                "direction_pairs_agree": agrees,
                "direction_pairs_evaluable": evaluable,
                "direction_agreement_rate": agrees / evaluable if evaluable else np.nan,
            }
        )
    observed = pd.DataFrame(observed_rows)

    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for resample in range(int(bootstrap_resamples)):
        sampled_means = []
        for style, rows in public_rows.items():
            sampled = rows.iloc[rng.integers(0, len(rows), size=len(rows))]
            mean = sampled.mean()
            sampled_means.extend(
                {"target_style": style, "anchor": anchor, "value": float(mean[anchor])}
                for anchor in anchors
            )
        sampled_frame = pd.DataFrame(sampled_means)
        for anchor in anchors:
            public_vector = _centered_vector(sampled_frame, "value", anchor)
            internal_vector = _centered_vector(internal_means, "internal_mean", anchor)
            agrees, evaluable = _direction_agreement(public_vector, internal_vector)
            bootstrap_rows.append(
                {
                    "resample": resample,
                    "anchor": anchor,
                    "cosine_similarity": _cosine(public_vector, internal_vector),
                    "direction_agreement_rate": agrees / evaluable if evaluable else np.nan,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    intervals = (
        bootstrap.groupby("anchor", as_index=False)
        .agg(
            cosine_ci_low=("cosine_similarity", lambda x: x.quantile(0.025)),
            cosine_ci_high=("cosine_similarity", lambda x: x.quantile(0.975)),
            direction_rate_ci_low=("direction_agreement_rate", lambda x: x.quantile(0.025)),
            direction_rate_ci_high=("direction_agreement_rate", lambda x: x.quantile(0.975)),
        )
    )
    scorecard = observed.merge(intervals, on="anchor", how="left", validate="one_to_one")
    public_pivot = public_means.pivot(index="target_style", columns="anchor", values="public_mean_composition").loc[
        ["Tripel", "Kriek", "Lambic"], anchors
    ]
    internal_pivot = internal_means.pivot(index="target_style", columns="anchor", values="internal_mean").loc[
        ["Tripel", "Kriek", "Lambic"], anchors
    ]
    flattened_public = (public_pivot - public_pivot.mean(axis=0)).to_numpy().ravel()
    flattened_internal = (internal_pivot - internal_pivot.mean(axis=0)).to_numpy().ravel()
    summary = {
        "status": "EXTERNAL_DIRECTIONAL_AUDIT_COMPLETE_NOT_A_HARD_GATE",
        "anchors": anchors,
        "overall_centered_cosine_similarity": _cosine(flattened_public, flattened_internal),
        "mean_anchor_cosine_similarity": float(scorecard["cosine_similarity"].mean()),
        "direction_pairs_agree": int(scorecard["direction_pairs_agree"].sum()),
        "direction_pairs_evaluable": int(scorecard["direction_pairs_evaluable"].sum()),
        "uncertainty_unit": "real_public_beer_bootstrap",
        "used_for_model_selection": False,
        "evidence_role": "secondary_directional_audit",
    }
    return {
        "summary": summary,
        "scorecard": scorecard,
        "bootstrap": bootstrap,
        "public_profiles": public_means,
        "internal_profiles": internal_means,
        "internal_sample_profiles": internal_rows,
    }


def score_external_migration_direction(
    public_data: pd.DataFrame,
    cohort_registry: pd.DataFrame,
    before_after_anchors: pd.DataFrame,
    *,
    source_cohort: str = "Pils_Lager_broad",
) -> pd.DataFrame:
    """Compare the migration anchor displacement with public Tripel direction."""
    required = {"anchor", "before_mean", "after_mean"}
    missing = sorted(required.difference(before_after_anchors.columns))
    if missing:
        raise ValueError(f"Migration anchor table is missing columns: {missing}")
    composition = transform_public_descriptor_profiles(public_data)
    by_key = public_data[["key"]].join(composition).set_index("key")
    keys_source = cohort_registry.loc[
        (cohort_registry["cohort"] == source_cohort)
        & cohort_registry["eligible_for_external_anchor_audit"].astype(bool),
        "key",
    ]
    keys_target = cohort_registry.loc[
        (cohort_registry["cohort"] == "Tripel")
        & cohort_registry["eligible_for_external_anchor_audit"].astype(bool),
        "key",
    ]
    anchors = ["Malty", "Hoppy", "Fruits"]
    public_delta = (
        by_key.loc[keys_target, anchors].mean() - by_key.loc[keys_source, anchors].mean()
    )
    internal = before_after_anchors.set_index("anchor").loc[anchors]
    internal_delta = internal["after_mean"] - internal["before_mean"]
    rows = pd.DataFrame(
        {
            "anchor": anchors,
            "public_tripel_minus_source": public_delta.loc[anchors].to_numpy(dtype=float),
            "migration_after_minus_before": internal_delta.loc[anchors].to_numpy(dtype=float),
        }
    )
    rows["direction_agrees"] = np.sign(rows["public_tripel_minus_source"]) == np.sign(
        rows["migration_after_minus_before"]
    )
    rows["vector_cosine_similarity"] = _cosine(
        rows["public_tripel_minus_source"].to_numpy(dtype=float),
        rows["migration_after_minus_before"].to_numpy(dtype=float),
    )
    rows["source_cohort"] = source_cohort
    rows["evidence_role"] = "secondary_directional_audit"
    return rows

"""Question 2 preflight: validate the data contract and frozen Q1 interface.

This module deliberately does not train models, calculate SHAP, or choose
Top-3 compounds. It creates auditable inputs for those later stages.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml

from beer_q1.data import load_problem1_data


KNOWN_ALIASES = {
    "CS2": "Carbon disulfide",
    "DES": "Diethyl sulfide",
    "DMS": "Dimethyl sulfide",
    "H2S": "Hydrogen sulfide",
    "MeSH": "Methanethiol",
    "X4VG": "4-vinylguaiacol",
    "SO2.mg.L.": "Sulfur dioxide",
    "bitterness": "Iso-alpha acids",
    "GFS": "Sugars",
    "CO2.PSI.": "Carbon dioxide",
    "kcalperc": "Caloric value",
    "X2.3.butanedione": "Diacetyl",
    "X2.furanmethanol": "Furfuryl alcohol",
    "dihydroisocaryophyllene.epoxide": "Tobacarol",
    "selin.11.en.4.alpha.ol": "Kongol",
    "esters_sum": "Total ester compounds",
    "aroma_hops_sum": "Total hop aroma compounds",
    "sulfur_sum": "Total sulfur compounds",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_name(value: Any) -> str:
    text = str(value).lower().replace("β", "beta").replace("α", "alpha")
    return re.sub(r"[^a-z0-9]+", "", text)


def _similarity(left: Any, right: Any) -> float:
    return float(difflib.SequenceMatcher(None, _norm_name(left), _norm_name(right)).ratio())


def _load_network(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns[0].lower().startswith("unnamed"):
        frame = frame.set_index(frame.columns[0])
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Chemical network must be square, got {frame.shape}")
    if list(frame.index.astype(str)) != list(frame.columns.astype(str)):
        raise ValueError("Chemical network row/column labels are not identical")
    values = frame.to_numpy(dtype=float)
    if not np.allclose(values, values.T, atol=1e-8, equal_nan=True):
        raise ValueError("Chemical network is not symmetric")
    if not np.allclose(np.diag(values), 1.0, atol=1e-8, equal_nan=True):
        raise ValueError("Chemical network diagonal is not one")
    return frame


def build_field_dictionary(features: Iterable[str], network_nodes: Iterable[str]) -> pd.DataFrame:
    """Build a conservative ordered mapping; never silently reorder nodes."""
    feature_list = list(features)
    node_list = list(map(str, network_nodes))
    if len(feature_list) != len(node_list):
        raise ValueError("1.csv features and 2.csv network nodes have different lengths")
    rows = []
    for position, (feature, node) in enumerate(zip(feature_list, node_list), start=1):
        score = _similarity(feature, node)
        alias_confirmed = KNOWN_ALIASES.get(feature) == node
        if alias_confirmed:
            status = "alias_confirmed"
        elif score >= 0.70:
            status = "position_name_agree"
        elif score >= 0.50:
            status = "position_name_partial"
        else:
            status = "position_order_review"
        rows.append(
            {
                "position_1based": position,
                "field_1_csv": feature,
                "node_2_csv": node,
                "normalized_name_similarity": round(score, 6),
                "alias_confirmed": bool(alias_confirmed),
                "mapping_status": status,
                "mapping_basis": "same_ordinal_position; explicit_alias_or_name_similarity",
            }
        )
    return pd.DataFrame(rows)


def _hash_paths(paths: Iterable[Path]) -> Dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def run_preflight(project_root: Path, config_path: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def path(section: str, key: str) -> Path:
        value = Path(config[section][key])
        return value if value.is_absolute() else project_root / value

    q1_config = {
        "data": {
            "chemicals_csv": str(path("data", "chemicals_csv")),
            "sensory_csv": str(path("data", "sensory_csv")),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": config["data"]["target_scope"],
        },
        "runtime": {},
    }
    problem1 = load_problem1_data(q1_config)
    network = _load_network(path("data", "chemical_network_csv"))
    reference = pd.read_csv(path("data", "chemical_sensory_reference_csv"))
    metadata = pd.ExcelFile(path("data", "metadata_xlsx"))

    freeze_dir = path("frozen_q1", "freeze_dir")
    registry_path = path("frozen_q1", "source_registry")
    local_registry_path = project_root / "configs/problem2_q1_reconstruction_registry.csv"
    source_registry = pd.read_csv(registry_path)
    reconstruction = pd.read_csv(local_registry_path)
    oof = pd.read_parquet(path("frozen_q1", "oof_predictions"))
    folds = pd.read_csv(path("frozen_q1", "folds"))
    target_list = pd.read_csv(freeze_dir / "base_run_metadata/target_list.csv")
    main_targets = list(problem1.main_target_names)

    errors = []
    if len(network) != len(problem1.feature_names):
        errors.append("2.csv node count does not equal the 231 model features")
    if len(reference) != 44:
        errors.append(f"6.csv rows expected 44, got {len(reference)}")
    if set(source_registry["target"]) != set(main_targets):
        errors.append("Frozen source registry does not cover exactly the 30 main Q1 targets")
    if set(reconstruction["target"]) != set(main_targets):
        errors.append("Q2 reconstruction registry does not cover exactly the 30 main targets")
    if len(oof) != 250 * 7 * 30:
        errors.append(f"Frozen OOF row count unexpected: {len(oof)}")
    if oof[["beer", "repeat", "fold", "target"]].duplicated().any():
        errors.append("Frozen OOF contains duplicate beer/repeat/fold/target rows")
    if len(folds) != 250 * 7:
        errors.append(f"Frozen fold row count unexpected: {len(folds)}")
    if list(target_list.iloc[:, 0].astype(str)) != main_targets:
        errors.append("Frozen target list order differs from attachment 4 main-target order")
    if errors:
        raise ValueError("Problem 2 preflight failed:\n- " + "\n- ".join(errors))

    field_dictionary = build_field_dictionary(problem1.feature_names, network.columns)
    registry_match = reconstruction.merge(
        source_registry[["target", "grade", "source_model"]],
        on="target",
        how="left",
        suffixes=("_reconstruction", "_frozen"),
        validate="one_to_one",
    )
    if (registry_match["grade_reconstruction"] != registry_match["grade_frozen"]).any():
        raise ValueError("Reconstruction registry grade disagrees with frozen registry")

    output = path("runtime", "output_root")
    output.mkdir(parents=True, exist_ok=True)
    field_dictionary.to_csv(output / "field_dictionary.csv", index=False)
    registry_match.to_csv(output / "q1_reconstruction_registry_resolved.csv", index=False)

    input_paths = [
        path("data", "chemicals_csv"),
        path("data", "sensory_csv"),
        path("data", "chemical_network_csv"),
        path("data", "chemical_sensory_reference_csv"),
        path("data", "metadata_xlsx"),
    ]
    audit = {
        "status": "PASS",
        "purpose": "problem2_preflight_only",
        "n_samples": int(len(problem1.X)),
        "n_features": int(len(problem1.feature_names)),
        "n_main_targets": int(len(problem1.main_target_names)),
        "n_all_targets": int(len(problem1.main_target_names) + len(problem1.appendix_target_names)),
        "n_styles": int(problem1.styles.nunique()),
        "missing_values_total": int(problem1.X.isna().sum().sum()),
        "network_shape": list(network.shape),
        "network_reference_rows": int(len(reference)),
        "metadata_sheets": metadata.sheet_names,
        "field_mapping_counts": {
            str(k): int(v) for k, v in field_dictionary["mapping_status"].value_counts().items()
        },
        "field_mapping_manual_review": int((field_dictionary["mapping_status"] == "position_order_review").sum()),
        "frozen_oof_rows": int(len(oof)),
        "frozen_fold_rows": int(len(folds)),
        "source_registry_targets": int(len(source_registry)),
        "reconstruction_registry_targets": int(len(reconstruction)),
        "input_sha256": _hash_paths(input_paths),
        "frozen_q1_sha256": _hash_paths([registry_path, path("frozen_q1", "oof_predictions"), path("frozen_q1", "folds")]),
        "protocol": config["protocol"],
        "no_model_training": True,
        "no_top3_selection": True,
    }
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "protocol.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return audit

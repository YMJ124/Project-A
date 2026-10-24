"""Strict, read-only data loading for Question 1.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


EXPECTED_ROWS = 250
EXPECTED_FEATURES = 231
EXPECTED_TARGETS = 50
EXPECTED_MAIN_TARGETS = 30


@dataclass(frozen=True)
class Problem1Data:
    X: pd.DataFrame
    y: pd.DataFrame
    ids: pd.Series
    styles: pd.Series
    feature_names: List[str]
    target_names: List[str]
    main_target_names: List[str]
    appendix_target_names: List[str]
    audit: Dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_problem1_data(config: Dict[str, Any]) -> Problem1Data:
    """Load and validate attachment 1 and 4 without modifying either file."""
    data_cfg = config["data"]
    x_path = Path(data_cfg["chemicals_csv"])
    y_path = Path(data_cfg["sensory_csv"])
    id_col = data_cfg["id_column"]
    style_col = data_cfg["style_column"]
    forbidden = list(data_cfg["forbidden_input_columns"])

    chemicals = pd.read_csv(x_path)
    sensory = pd.read_csv(y_path)

    errors: List[str] = []
    if chemicals.shape[0] != EXPECTED_ROWS:
        errors.append(f"1.csv rows: expected {EXPECTED_ROWS}, got {chemicals.shape[0]}")
    if sensory.shape[0] != EXPECTED_ROWS:
        errors.append(f"4.csv rows: expected {EXPECTED_ROWS}, got {sensory.shape[0]}")
    for name, frame in (("1.csv", chemicals), ("4.csv", sensory)):
        for required in (id_col, "beer_id", style_col):
            if required not in frame.columns:
                errors.append(f"{name} missing required column {required}")
        if id_col in frame and not frame[id_col].is_unique:
            errors.append(f"{name} has duplicate {id_col} values")

    if not errors:
        x_ids = set(chemicals[id_col].tolist())
        y_ids = set(sensory[id_col].tolist())
        if x_ids != y_ids:
            errors.append("1.csv and 4.csv have different beer sets")

    if errors:
        raise ValueError("Data contract failed:\n- " + "\n- ".join(errors))

    # Integer beer is the only join key. beer_id is deliberately never used.
    sensory_labels = sensory[[id_col, style_col]].rename(
        columns={style_col: f"{style_col}__sensory"}
    )
    merged_labels = chemicals[[id_col, style_col]].merge(
        sensory_labels, on=id_col, how="inner", validate="one_to_one"
    )
    mismatch = merged_labels[style_col] != merged_labels[f"{style_col}__sensory"]
    if mismatch.any():
        bad = merged_labels.loc[mismatch, id_col].tolist()
        raise ValueError(f"Style mismatch between 1.csv and 4.csv for beer: {bad}")

    chemicals = chemicals.sort_values(id_col).reset_index(drop=True)
    sensory = sensory.sort_values(id_col).reset_index(drop=True)
    if not np.array_equal(chemicals[id_col].to_numpy(), sensory[id_col].to_numpy()):
        raise AssertionError("beer alignment failed after sorting")

    feature_names = [column for column in chemicals.columns if column not in forbidden]
    all_target_names = [column for column in sensory.columns if column not in forbidden]
    main_targets = [
        column for column in all_target_names if column.startswith(("A_", "F_"))
    ]
    appendix_targets = [column for column in all_target_names if column not in main_targets]

    if len(feature_names) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} chemical features, got {len(feature_names)}"
        )
    if len(all_target_names) != EXPECTED_TARGETS:
        raise ValueError(f"Expected {EXPECTED_TARGETS} targets, got {len(all_target_names)}")
    if len(main_targets) != EXPECTED_MAIN_TARGETS:
        raise ValueError(
            f"Expected {EXPECTED_MAIN_TARGETS} A_/F_ targets, got {len(main_targets)}"
        )
    if not all(pd.api.types.is_numeric_dtype(chemicals[c]) for c in feature_names):
        raise TypeError("All model features must be numeric")
    if not all(pd.api.types.is_numeric_dtype(sensory[c]) for c in all_target_names):
        raise TypeError("All targets must be numeric")
    if sensory[all_target_names].isna().any().any():
        raise ValueError("Sensory targets contain missing values")

    scope = data_cfg.get("target_scope", "main")
    if scope == "main":
        target_names = main_targets
    elif scope == "all":
        target_names = all_target_names
    else:
        raise ValueError("target_scope must be 'main' or 'all'")

    requested_targets = data_cfg.get("target_names")
    if requested_targets is not None:
        if not isinstance(requested_targets, list) or not requested_targets:
            raise ValueError("data.target_names must be a non-empty list")
        if len(requested_targets) != len(set(requested_targets)):
            raise ValueError("data.target_names must not contain duplicates")
        unknown_targets = sorted(set(requested_targets).difference(target_names))
        if unknown_targets:
            raise ValueError(
                f"data.target_names are outside target_scope: {unknown_targets}"
            )
        target_names = list(requested_targets)

    target_limit = config.get("runtime", {}).get("target_limit")
    if target_limit is not None:
        target_names = target_names[: int(target_limit)]

    X = chemicals[feature_names].copy()
    y = sensory[target_names].copy()
    finite_x = np.isfinite(X.to_numpy(dtype=float, na_value=np.nan)) | X.isna().to_numpy()
    if not finite_x.all():
        raise ValueError("Chemical features contain infinite values")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Targets contain non-finite values")

    missing_by_column = X.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
    audit: Dict[str, Any] = {
        "chemicals_path": str(x_path),
        "sensory_path": str(y_path),
        "chemicals_sha256": sha256_file(x_path),
        "sensory_sha256": sha256_file(y_path),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_all_targets": len(all_target_names),
        "n_selected_targets": len(target_names),
        "n_main_targets": len(main_targets),
        "n_appendix_targets": len(appendix_targets),
        "n_styles": int(chemicals[style_col].nunique()),
        "missing_values_total": int(X.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in missing_by_column.items()},
        "style_counts": {
            str(k): int(v)
            for k, v in chemicals[style_col].value_counts().sort_index().items()
        },
        "join_key": id_col,
        "forbidden_model_inputs": forbidden,
        "selected_targets": target_names,
    }
    return Problem1Data(
        X=X,
        y=y,
        ids=chemicals[id_col].copy(),
        styles=chemicals[style_col].copy(),
        feature_names=feature_names,
        target_names=target_names,
        main_target_names=main_targets,
        appendix_target_names=appendix_targets,
        audit=audit,
    )

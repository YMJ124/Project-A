"""Single-fold stage-0 smoke run for the Question 3 evaluator."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
import yaml

from beer_q1.data import load_problem1_data

from .baselines import empirical_resample
from .evaluation import (
    FoldGeometry,
    attach_reference_targets,
    build_real_real_reference,
    evaluate_generated_vs_real,
    evaluate_hard_gates,
    fit_style_classifier,
    summarize_reference,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["method", "target_style", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            median=("value", "median"),
            std=("value", "std"),
            q25=("value", lambda x: x.quantile(0.25)),
            q75=("value", lambda x: x.quantile(0.75)),
        )
    )


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_stage0_smoke(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Exercise one immutable outer fold without selecting a generator."""
    started = time.perf_counter()
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

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
    folds = pd.read_csv(fold_path)
    defects = pd.read_csv(defect_path)
    edges = pd.read_csv(edge_path)

    repeat = int(config["evaluation"]["repeat"])
    fold = int(config["evaluation"]["fold"])
    selected = folds.loc[folds["repeat"] == repeat, ["beer", "repeat", "fold", "seed"]]
    if len(selected) != len(data.X) or selected["beer"].duplicated().any():
        raise ValueError("The selected frozen repeat does not cover every beer exactly once")
    fold_by_beer = selected.set_index("beer")["fold"]
    assignments = data.ids.map(fold_by_beer)
    if assignments.isna().any():
        raise ValueError("Frozen folds and chemical data have different beer sets")
    train_mask = assignments.to_numpy() != fold
    valid_mask = ~train_mask
    X_train = data.X.loc[train_mask].reset_index(drop=True)
    X_valid = data.X.loc[valid_mask].reset_index(drop=True)
    style_train = data.styles.loc[train_mask].reset_index(drop=True)
    style_valid = data.styles.loc[valid_mask].reset_index(drop=True)
    id_train = data.ids.loc[train_mask].reset_index(drop=True)
    id_valid = data.ids.loc[valid_mask].reset_index(drop=True)
    seed = int(selected["seed"].iloc[0]) + fold * 10007

    geometry = FoldGeometry.fit(X_train)
    classifier = fit_style_classifier(X_train, style_train, seed=seed)
    target_styles = list(config["styles"]["generation_targets"])
    n_reference = int(config["evaluation"]["real_real_resamples"])

    generated_frames = []
    trace_frames = []
    gate_frames = []
    defect_frames = []
    reference_frames = []
    metric_frames = []
    for style_index, style in enumerate(target_styles):
        valid_style_mask = style_valid.astype(str).to_numpy() == str(style)
        validation_style = X_valid.loc[valid_style_mask].reset_index(drop=True)
        validation_ids = id_valid.loc[valid_style_mask].reset_index(drop=True)
        if validation_style.empty:
            raise ValueError(f"No validation rows for requested style {style}")
        target_training = X_train.loc[
            style_train.astype(str).to_numpy() == str(style)
        ].reset_index(drop=True)
        batch = empirical_resample(
            X_train,
            style_train,
            id_train,
            str(style),
            len(validation_style),
            seed=seed + style_index * 1009,
        )
        reference = build_real_real_reference(
            X_train,
            style_train,
            str(style),
            len(validation_style),
            geometry,
            edges,
            n_resamples=n_reference,
            seed=seed + style_index * 2027,
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
            str(style),
            geometry,
            classifier,
            edges,
        )

        metadata = pd.DataFrame(
            {
                "run_name": config["runtime"]["run_name"],
                "method": batch.method,
                "repeat": repeat,
                "fold": fold,
                "target_style": style,
                "sample_id": batch.chemicals.index.astype(str),
                "source_training_beer": batch.source_training_ids,
            }
        )
        generated_frames.append(
            pd.concat([metadata.reset_index(drop=True), batch.chemicals.reset_index(drop=True)], axis=1)
        )
        trace.insert(0, "sample_id", batch.chemicals.index.astype(str))
        trace.insert(0, "target_style", style)
        trace.insert(0, "fold", fold)
        trace.insert(0, "repeat", repeat)
        trace.insert(0, "method", batch.method)
        trace_frames.append(trace.reset_index(drop=True))
        gates.insert(0, "target_style", style)
        gates.insert(0, "fold", fold)
        gates.insert(0, "repeat", repeat)
        gates.insert(0, "method", batch.method)
        gate_frames.append(gates)
        defect_detail.insert(0, "target_style", style)
        defect_frames.append(defect_detail)
        reference.insert(0, "fold", fold)
        reference.insert(0, "repeat", repeat)
        reference_frames.append(reference)
        metrics.insert(0, "fold", fold)
        metrics.insert(0, "repeat", repeat)
        metrics.insert(0, "method", batch.method)
        metric_frames.append(metrics)

    generated = pd.concat(generated_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    gates = pd.concat(gate_frames, ignore_index=True)
    defect_details = pd.concat(defect_frames, ignore_index=True)
    references = pd.concat(reference_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    reference_summary = summarize_reference(references)
    distances = attach_reference_targets(metrics, reference_summary)
    metrics_summary = _summary(metrics)

    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    selected.assign(style=selected["beer"].map(dict(zip(data.ids, data.styles)))).to_csv(
        output / "folds.csv", index=False
    )
    _write_parquet_atomic(generated, output / "generated_samples.parquet")
    _write_parquet_atomic(traces, output / "sample_trace.parquet")
    gates.to_csv(output / "hard_gates.csv", index=False)
    defect_details.to_csv(output / "defect_gate_details.csv", index=False)
    references.to_csv(output / "real_real_reference.csv", index=False)
    metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    metrics_summary.to_csv(output / "metrics_summary.csv", index=False)
    distances.to_csv(output / "distance_to_target.csv", index=False)
    sensory = pd.DataFrame(
        columns=[
            "method",
            "repeat",
            "fold",
            "target_style",
            "sample_id",
            "target",
            "standardized_prediction",
            "relative_reconstructed_scale",
            "confidence_grade",
        ]
    )
    _write_parquet_atomic(sensory, output / "sensory_profiles.parquet")

    all_expected_copy_fail = bool(
        gates.loc[gates["gate"] == "exact_copy_rate", "pass"].eq(False).all()  # noqa: E712
    )
    manifest = {
        "status": "PASS_STAGE0_REFERENCE_REJECTED_AS_REQUIRED" if all_expected_copy_fail else "FAIL",
        "scope": "single_outer_fold_evaluator_smoke_not_generator_selection",
        "run_name": config["runtime"]["run_name"],
        "repeat": repeat,
        "fold": fold,
        "target_styles": target_styles,
        "method": "empirical_resample_reference_only",
        "expected_champion_eligible": False,
        "exact_copy_gate_failed_for_all_styles": all_expected_copy_fail,
        "sensory_status": "not_run_for_ineligible_copy_reference",
        "q1_status_inherited": "HOLD",
        "n_train": int(train_mask.sum()),
        "n_validation": int(valid_mask.sum()),
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit_at_run": _git_commit(project_root),
        "config_sha256": _sha256(config_path),
        "evaluator_source_sha256": _sha256(Path(__file__).with_name("evaluation.py")),
        "baseline_source_sha256": _sha256(Path(__file__).with_name("baselines.py")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "inputs": {
            "chemicals_csv": _sha256(data_paths["chemicals_csv"]),
            "sensory_csv": _sha256(data_paths["sensory_csv"]),
            "folds_csv": _sha256(fold_path),
            "defect_registry_csv": _sha256(defect_path),
            "consensus_edges_csv": _sha256(edge_path),
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    decision = f"""# Stage-0 evaluator smoke decision

- Status: `{manifest['status']}`
- Scope: one locked outer fold (repeat {repeat}, fold {fold}); no generator selection.
- Mechanism under test: empirical target-style resampling.
- Expected result: distributional reference may look strong, but exact-copy rate must be non-zero and the method must be rejected.
- Sensory inference was intentionally skipped because an ineligible copied reference cannot enter end-to-end champion comparison.
- Q1 remains `HOLD`; Q2 remains the frozen `READY` interface.
"""
    (output / "decision.md").write_text(decision, encoding="utf-8")
    return manifest

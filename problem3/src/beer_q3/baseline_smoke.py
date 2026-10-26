"""Single-fold comparison of the locked small-sample Q3 baselines."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml

from beer_q1.data import load_problem1_data

from .baselines import (
    empirical_resample,
    knn_jitter,
    pca_shrink_gaussian,
    rank_gaussian_copula,
)
from .derived import load_formula_config
from .evaluation import (
    FoldGeometry,
    attach_reference_targets,
    build_real_real_reference,
    evaluate_generated_vs_real,
    evaluate_hard_gates,
    fit_style_classifier,
    summarize_reference,
)
from .smoke import _git_commit, _resolve, _sha256, _summary, _write_parquet_atomic


def run_baseline_smoke(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Compare four fixed baselines on one locked outer fold.

    The run validates plumbing and hard gates only.  It must not be used for
    generator selection because it contains a single discovery fold.
    """
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
    formula_path = _resolve(project_root, config["interfaces"]["derived_formula_yaml"])
    folds = pd.read_csv(fold_path)
    defects = pd.read_csv(defect_path)
    edges = pd.read_csv(edge_path)
    formulas = load_formula_config(formula_path)

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
    seed = int(selected["seed"].iloc[0]) + fold * 10007

    geometry = FoldGeometry.fit(X_train)
    classifier = fit_style_classifier(X_train, style_train, seed=seed)
    target_styles = [str(value) for value in config["styles"]["generation_targets"]]
    n_reference = int(config["evaluation"]["real_real_resamples"])
    params = config["baselines"]

    generated_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    defect_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []

    for style_index, style in enumerate(target_styles):
        valid_style_mask = style_valid.astype(str).to_numpy() == style
        validation_style = X_valid.loc[valid_style_mask].reset_index(drop=True)
        if validation_style.empty:
            raise ValueError(f"No validation rows for requested style {style}")
        target_training = X_train.loc[
            style_train.astype(str).to_numpy() == style
        ].reset_index(drop=True)
        reference = build_real_real_reference(
            X_train,
            style_train,
            style,
            len(validation_style),
            geometry,
            edges,
            n_resamples=n_reference,
            seed=seed + style_index * 2027,
        )
        reference.insert(0, "fold", fold)
        reference.insert(0, "repeat", repeat)
        reference_frames.append(reference)

        common = {
            "training": X_train,
            "training_styles": style_train,
            "target_style": style,
            "n_samples": len(validation_style),
        }
        batches = [
            empirical_resample(
                X_train,
                style_train,
                id_train,
                style,
                len(validation_style),
                seed=seed + style_index * 1009,
            ),
            knn_jitter(
                **common,
                training_ids=id_train,
                full_geometry=geometry,
                defect_registry=defects,
                formula_config=formulas,
                seed=seed + style_index * 1009 + 101,
                **params["knn_jitter"],
            ),
            pca_shrink_gaussian(
                **common,
                full_geometry=geometry,
                defect_registry=defects,
                formula_config=formulas,
                seed=seed + style_index * 1009 + 211,
                **params["pca_shrink_gaussian"],
            ),
            rank_gaussian_copula(
                **common,
                full_geometry=geometry,
                defect_registry=defects,
                formula_config=formulas,
                seed=seed + style_index * 1009 + 307,
                **params["rank_gaussian_copula"],
            ),
        ]

        for batch in batches:
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
                    "method": batch.method,
                    "repeat": repeat,
                    "fold": fold,
                    "target_style": style,
                    "sample_id": batch.chemicals.index.astype(str),
                    "source_training_beer": batch.source_training_ids,
                }
            )
            generated_frames.append(
                pd.concat(
                    [metadata.reset_index(drop=True), batch.chemicals.reset_index(drop=True)],
                    axis=1,
                )
            )
            trace.insert(0, "sample_id", batch.chemicals.index.astype(str))
            trace.insert(0, "target_style", style)
            trace.insert(0, "fold", fold)
            trace.insert(0, "repeat", repeat)
            trace.insert(0, "method", batch.method)
            trace_frames.append(trace.reset_index(drop=True))
            for frame in (gates, defect_detail, metrics):
                frame.insert(0, "target_style", style) if "target_style" not in frame else None
                frame.insert(0, "fold", fold)
                frame.insert(0, "repeat", repeat)
                frame.insert(0, "method", batch.method)
            gate_frames.append(gates)
            defect_frames.append(defect_detail)
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

    eligible = gates.loc[gates["method"] != "empirical_resample_reference_only"]
    by_method = (
        eligible.groupby("method")["all_hard_gates_pass"].all().sort_index().to_dict()
    )
    passing = sorted(method for method, passed in by_method.items() if bool(passed))
    manifest = {
        "status": "PASS_BASELINE_PLUMBING" if passing else "HOLD_NO_NONCOPY_BASELINE_PASSED",
        "scope": "single_outer_fold_baseline_smoke_not_generator_selection",
        "run_name": config["runtime"]["run_name"],
        "repeat": repeat,
        "fold": fold,
        "target_styles": target_styles,
        "hard_gate_pass_by_noncopy_method": by_method,
        "noncopy_methods_passing_all_styles": passing,
        "sensory_status": "not_run_in_stage1_chemistry_plumbing_smoke",
        "q1_status_inherited": "HOLD",
        "n_train": int(train_mask.sum()),
        "n_validation": int(valid_mask.sum()),
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit_at_run": _git_commit(project_root),
        "config_sha256": _sha256(config_path),
        "formula_config_sha256": _sha256(formula_path),
        "source_hashes": {
            "baseline_smoke_py": _sha256(Path(__file__)),
            "baselines_py": _sha256(Path(__file__).with_name("baselines.py")),
            "evaluation_py": _sha256(Path(__file__).with_name("evaluation.py")),
            "derived_py": _sha256(Path(__file__).with_name("derived.py")),
        },
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
    failed_gate_rows = gates.loc[
        (gates["method"] != "empirical_resample_reference_only") & ~gates["pass"],
        ["method", "target_style", "gate", "failure_rate"],
    ]
    failed_lines = (
        "\n".join(
            f"- `{row.method}` / `{row.target_style}`: `{row.gate}` "
            f"(failure rate {row.failure_rate:.3f})"
            for row in failed_gate_rows.itertuples(index=False)
        )
        or "- None."
    )
    decision = f"""# Stage-1 single-fold baseline smoke decision

- Status: `{manifest['status']}`
- Scope: one locked discovery fold (repeat {repeat}, fold {fold}); this is plumbing evidence, not model selection.
- Fixed mechanisms: empirical resampling reference, KNN jitter, PCA shrink Gaussian, and rank Gaussian copula.
- Non-copy methods passing every hard gate for all three styles: {', '.join(passing) if passing else 'none'}.
- Q1 sensory inference is intentionally deferred until the chemistry evaluator and multi-fold generator configuration are locked.

## Hard-gate failures among non-copy methods

{failed_lines}

No method is declared a Question 3 champion from this run. The next formal step is discovery evaluation on repeats 0--2 with fold-local parameter selection where required.
"""
    (output / "decision.md").write_text(decision, encoding="utf-8")
    return manifest

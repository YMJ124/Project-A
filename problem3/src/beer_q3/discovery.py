"""Resumable discovery-fold orchestration for the Q3 statistical baselines."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import yaml

from .baseline_smoke import run_baseline_smoke
from .smoke import _git_commit, _resolve, _sha256, _write_parquet_atomic


DISTANCE_METRICS = {
    "wasserstein_median",
    "wasserstein_q75",
    "wasserstein_max",
    "spearman_structure_rmse",
    "q2_edge_strength_mae",
    "q2_edge_loss_rate",
}


def summarize_discovery_metrics(
    metrics: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize outer-fold values without treating synthetic rows as evidence."""
    rows: list[dict[str, Any]] = []
    group_columns = ["method", "target_style", "metric"]
    for group_index, (keys, group) in enumerate(metrics.groupby(group_columns, sort=True)):
        values = group["value"].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        rng = np.random.default_rng(seed + group_index * 104729)
        bootstrap = values[
            rng.integers(0, len(values), size=(int(bootstrap_resamples), len(values)))
        ].mean(axis=1)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_outer_fold_values": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "bootstrap_mean_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_mean_ci_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def build_method_scorecard(
    gates: pd.DataFrame,
    distances: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create a transparent method-level audit without a weighted total score."""
    gate_cells = (
        gates.groupby(["method", "repeat", "fold", "target_style"], as_index=False)
        .agg(all_hard_gates_pass=("pass", "all"))
    )
    distance = distances.loc[
        distances["metric"].isin(DISTANCE_METRICS) & distances["minimum_line"].notna()
    ].copy()
    distribution = distance.loc[
        distance["metric"].isin(["wasserstein_median", "wasserstein_q75"])
    ]
    metric_wide = metrics.pivot_table(
        index=["method", "repeat", "fold", "target_style"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    rows = []
    for method in sorted(gates["method"].unique()):
        method_gate = gate_cells.loc[gate_cells["method"] == method]
        method_distance = distance.loc[distance["method"] == method]
        method_distribution = distribution.loc[distribution["method"] == method]
        method_metrics = metric_wide.loc[metric_wide["method"] == method]
        rows.append(
            {
                "method": method,
                "n_fold_style_cells": int(len(method_gate)),
                "hard_gate_cell_pass_rate": float(method_gate["all_hard_gates_pass"].mean()),
                "all_hard_gates_all_cells": bool(method_gate["all_hard_gates_pass"].all()),
                "real_reference_distance_pass_rate": float(
                    method_distance["status"].eq("PASS").mean()
                ) if len(method_distance) else np.nan,
                "wasserstein_core_pass_rate": float(
                    method_distribution["status"].eq("PASS").mean()
                ) if len(method_distribution) else np.nan,
                "style_hit_rate_generated_mean": float(
                    method_metrics["style_hit_rate_generated"].mean()
                ),
                "style_hit_rate_real_mean": float(method_metrics["style_hit_rate_real"].mean()),
                "style_probability_gap_mean": float(
                    method_metrics["style_probability_gap"].mean()
                ),
                "generated_unique_rate_mean": float(
                    method_metrics["generated_unique_rate"].mean()
                ),
                "champion_eligible": False,
                "eligibility_reason": (
                    "reference_only_exact_copies"
                    if method == "empirical_resample_reference_only"
                    else "chemistry_discovery_only_sensory_not_run"
                ),
            }
        )
    return pd.DataFrame(rows)


def _concat_csv(paths: Iterable[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _concat_parquet(paths: Iterable[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _subrun_is_current(
    output: Path,
    subconfig: dict[str, Any],
    source_hashes: dict[str, str],
) -> bool:
    manifest_path = output / "run_manifest.json"
    resolved_path = output / "config.resolved.yaml"
    if not manifest_path.exists() or not resolved_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return resolved == subconfig and manifest.get("source_hashes") == source_hashes


def run_baseline_discovery(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Run fixed baselines over configured repeats and aggregate evidence."""
    started = time.perf_counter()
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)

    baseline_smoke_path = Path(__file__).with_name("baseline_smoke.py")
    source_hashes = {
        "baseline_smoke_py": _sha256(baseline_smoke_path),
        "baselines_py": _sha256(Path(__file__).with_name("baselines.py")),
        "evaluation_py": _sha256(Path(__file__).with_name("evaluation.py")),
        "derived_py": _sha256(Path(__file__).with_name("derived.py")),
    }
    repeats = [int(value) for value in config["evaluation"]["repeats"]]
    folds = [int(value) for value in config["evaluation"]["folds"]]
    expected = {(repeat, fold) for repeat in repeats for fold in folds}
    completed: list[tuple[int, int]] = []
    resumed: list[tuple[int, int]] = []
    subrun_outputs: list[Path] = []

    for repeat in repeats:
        for fold in folds:
            subrun_output = output / "fold_runs" / f"repeat_{repeat}_fold_{fold}"
            subrun_output.mkdir(parents=True, exist_ok=True)
            subconfig = {
                "data": config["data"],
                "interfaces": config["interfaces"],
                "styles": config["styles"],
                "baselines": config["baselines"],
                "evaluation": {
                    "repeat": repeat,
                    "fold": fold,
                    "real_real_resamples": int(config["evaluation"]["real_real_resamples"]),
                },
                "runtime": {
                    "run_name": f"{config['runtime']['run_name']}__r{repeat}_f{fold}",
                    "output_root": str(subrun_output.relative_to(project_root)),
                },
            }
            subconfig_path = subrun_output / "subrun.config.yaml"
            subconfig_path.write_text(
                yaml.safe_dump(subconfig, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            if _subrun_is_current(subrun_output, subconfig, source_hashes):
                resumed.append((repeat, fold))
            else:
                run_baseline_smoke(project_root, subconfig_path)
            completed.append((repeat, fold))
            subrun_outputs.append(subrun_output)

    if set(completed) != expected:
        raise RuntimeError("Discovery did not complete every requested repeat/fold")

    folds_frame = _concat_csv(path / "folds.csv" for path in subrun_outputs)
    folds_frame = folds_frame.drop_duplicates(["beer", "repeat", "fold"]).sort_values(
        ["repeat", "beer"]
    )
    generated = _concat_parquet(path / "generated_samples.parquet" for path in subrun_outputs)
    traces = _concat_parquet(path / "sample_trace.parquet" for path in subrun_outputs)
    gates = _concat_csv(path / "hard_gates.csv" for path in subrun_outputs)
    defects = _concat_csv(path / "defect_gate_details.csv" for path in subrun_outputs)
    references = _concat_csv(path / "real_real_reference.csv" for path in subrun_outputs)
    metrics = _concat_csv(path / "metrics_by_fold.csv" for path in subrun_outputs)
    distances = _concat_csv(path / "distance_to_target.csv" for path in subrun_outputs)
    sensory = _concat_parquet(path / "sensory_profiles.parquet" for path in subrun_outputs)

    metrics_summary = summarize_discovery_metrics(
        metrics,
        bootstrap_resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    scorecard = build_method_scorecard(gates, distances, metrics)
    distance_summary = (
        distances.groupby(["method", "target_style", "metric"], as_index=False)
        .agg(
            n_outer_fold_values=("value", "count"),
            current_mean=("value", "mean"),
            minimum_line_mean=("minimum_line", "mean"),
            ideal_line_mean=("ideal_line", "mean"),
            relative_gap_mean=("relative_gap", "mean"),
            pass_rate=("status", lambda values: values.eq("PASS").mean()),
            near_rate=("status", lambda values: values.eq("NEAR").mean()),
            fail_rate=("status", lambda values: values.eq("FAIL").mean()),
        )
    )

    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    folds_frame.to_csv(output / "folds.csv", index=False)
    _write_parquet_atomic(generated, output / "generated_samples.parquet")
    _write_parquet_atomic(traces, output / "sample_trace.parquet")
    gates.to_csv(output / "hard_gates.csv", index=False)
    defects.to_csv(output / "defect_gate_details.csv", index=False)
    references.to_csv(output / "real_real_reference.csv", index=False)
    metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    metrics_summary.to_csv(output / "metrics_summary.csv", index=False)
    distances.to_csv(output / "distance_to_target.csv", index=False)
    distance_summary.to_csv(output / "distance_to_target_summary.csv", index=False)
    scorecard.to_csv(output / "method_scorecard.csv", index=False)
    _write_parquet_atomic(sensory, output / "sensory_profiles.parquet")

    noncopy = scorecard.loc[scorecard["method"] != "empirical_resample_reference_only"]
    hard_gate_methods = noncopy.loc[
        noncopy["all_hard_gates_all_cells"], "method"
    ].tolist()
    phase = str(config["runtime"].get("phase", "discovery"))
    reference_status = (
        "REFERENCE_READY_CHEMISTRY_ONLY"
        if phase == "confirmation_reference"
        else "HOLD_DISCOVERY_CHEMISTRY_ONLY"
    )
    manifest = {
        "status": reference_status,
        "scope": f"fixed_baseline_{phase}_reference_no_champion_selection",
        "phase": phase,
        "run_name": config["runtime"]["run_name"],
        "repeats": repeats,
        "folds": folds,
        "completed_subruns": [list(value) for value in completed],
        "resumed_subruns": [list(value) for value in resumed],
        "noncopy_methods_passing_all_hard_gates_all_cells": hard_gate_methods,
        "sensory_status": "not_run_until_chemistry_candidate_is_locked",
        "q1_status_inherited": "HOLD",
        "external_test_data_status": "cohorts_locked_not_scored",
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit_at_run": _git_commit(project_root),
        "config_sha256": _sha256(config_path),
        "source_hashes": source_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = []
    for row in scorecard.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: hard-gate fold/style pass rate "
            f"{row.hard_gate_cell_pass_rate:.1%}; real-reference distance pass rate "
            f"{row.real_reference_distance_pass_rate:.1%}; Wasserstein-core pass rate "
            f"{row.wasserstein_core_pass_rate:.1%}; generated style hit rate "
            f"{row.style_hit_rate_generated_mean:.1%}."
        )
    decision = f"""# Baseline {phase} reference decision

- Status: `{manifest['status']}`.
- Evidence boundary: frozen repeats {repeats}, folds {folds}; parameters are unchanged from the single-fold smoke.
- This run evaluates chemistry, structure, style and memory risk. Q1 sensory inference remains pending, so no method is champion-eligible.
- Non-copy methods passing every hard gate in every fold/style cell: {', '.join(hard_gate_methods) if hard_gate_methods else 'none'}.

## Method audit

{chr(10).join(lines)}

## Decision discipline

The empirical resampling method remains reference-only because it copies training rows. This artifact is a fold-matched reference and cannot select a generator by itself. Any candidate must preserve numeric/derived/defect validity and satisfy its separately predeclared comparison rules. The external public-data audit remains untouched.
"""
    (output / "decision.md").write_text(decision, encoding="utf-8")
    return manifest

"""Orchestration for the first leakage-safe Question 2 pilot experiment.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn
import yaml

from beer_q1.data import load_problem1_data

from .explain import explain_fold_target, summarize_importance
from .network import build_fold_network, community_stability
from .preflight import sha256_file


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _q1_data_config(root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data": {
            "chemicals_csv": str(_resolve(root, config["data"]["chemicals_csv"])),
            "sensory_csv": str(_resolve(root, config["data"]["sensory_csv"])),
            "id_column": config["data"]["id_column"],
            "style_column": config["data"]["style_column"],
            "forbidden_input_columns": config["data"]["forbidden_input_columns"],
            "target_scope": config["data"]["target_scope"],
        },
        "runtime": {},
    }


def _validate_output_boundary(root: Path, output: Path) -> None:
    problem2_root = (root / "problem2").resolve()
    resolved = output.resolve()
    if resolved != problem2_root and problem2_root not in resolved.parents:
        raise ValueError(f"Question 2 output must stay under {problem2_root}, got {resolved}")


def _write_decision(
    output: Path,
    config: Dict[str, Any],
    sensitivity: pd.DataFrame,
    community: pd.DataFrame,
    reconstruction: pd.DataFrame,
    pilot_view: pd.DataFrame,
) -> None:
    primary = float(config["network"]["primary_abs_rho_threshold"])
    primary_edges = sensitivity[sensitivity["abs_rho_threshold"] == primary]
    formal_chunk = str(config["experiment"].get("scope", "")).startswith(
        "formal_explanation_chunk"
    )
    if formal_chunk:
        title = "# Question 2 formal explanation chunk decision"
        scope_text = (
            "This chunk covers all 30 main targets on the configured frozen repeat. "
            "It is one resumable component of the formal 7-repeat analysis and does "
            "not independently produce Top-3 conclusions."
        )
        decision_text = (
            "The all-target held-out permutation chunk is complete. Do not freeze "
            "chemical Top-3 until all seven repeat chunks, TreeSHAP, community joint "
            "permutation, source audit, and cross-repeat candidate aggregation are complete."
        )
    else:
        title = "# Question 2 pilot v1 decision"
        scope_text = (
            "This is a first runnable pilot on three predeclared representative targets. "
            "It is not the formal 30-target Top-3 result."
        )
        decision_text = (
            "The data path, frozen-fold network, true-pipeline reconstruction, and held-out "
            "permutation path are operational. Do not freeze chemical Top-3 yet: this pilot "
            "contains one repeat, lacks TreeSHAP and community joint permutation, and covers "
            "only three targets. The next version must add those mechanisms before the "
            "30-target run."
        )
    lines = [
        title,
        "",
        "## Scope",
        "",
        scope_text,
        "",
        "## Integrity checks",
        "",
        f"- Reconstructed target-fold pipelines passed: {int(reconstruction['reconstruction_passed'].sum())}/{len(reconstruction)}.",
        f"- Maximum absolute difference from frozen Q1 OOF: {reconstruction['reconstruction_max_abs_error'].max():.3g}.",
        f"- Primary |rho| threshold: {primary:.2f}; fold edge counts: "
        + ", ".join(map(str, primary_edges["n_edges"].astype(int).tolist()))
        + ".",
        f"- Pairwise fold-community ARI median: {community['adjusted_rand_index'].median():.3f}."
        if len(community)
        else "- Community stability requires at least two analyzed folds.",
        "",
        "## Pilot candidate view",
        "",
        "The rows below are diagnostic leads from held-out within-style permutation, not adopted Top-3 conclusions.",
        "",
    ]
    for target, group in pilot_view.groupby("target", sort=False):
        lines.append(f"### {target}")
        lines.append("")
        for row in group.head(5).itertuples(index=False):
            lines.append(
                f"- {row.pilot_rank}. `{row.feature}`: mean delta RMSE="
                f"{row.mean_delta_rmse:.5f}, Top-10 frequency={row.top10_frequency:.1%}, "
                f"positive permutation frequency={row.positive_permutation_frequency:.1%}."
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            decision_text,
            "",
            "C-grade target findings remain exploratory regardless of apparent permutation rank.",
        ]
    )
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(project_root: Path, config_path: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = _resolve(root, config["runtime"]["output_root"])
    _validate_output_boundary(root, output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    data = load_problem1_data(_q1_data_config(root, config))
    folds_path = _resolve(root, config["frozen_q1"]["folds"])
    tuning_path = _resolve(root, config["frozen_q1"]["tuning_trials"])
    oof_path = _resolve(root, config["frozen_q1"]["oof_predictions"])
    registry_path = _resolve(root, config["frozen_q1"]["source_registry"])
    folds = pd.read_csv(folds_path)
    tuning = pd.read_csv(tuning_path)
    frozen_oof = pd.read_parquet(oof_path)
    registry = pd.read_csv(registry_path).set_index("target")
    configured_targets = config["experiment"]["targets"]
    targets = (
        list(map(str, data.main_target_names))
        if configured_targets == "all_main"
        else list(map(str, configured_targets))
    )
    repeats = list(map(int, config["experiment"]["repeats"]))
    outer_folds = list(map(int, config["experiment"]["outer_folds"]))

    unknown = sorted(set(targets).difference(data.main_target_names))
    if unknown:
        raise ValueError(f"Unknown main targets: {unknown}")
    selected_folds = folds[folds["repeat"].isin(repeats)].copy()
    coverage = selected_folds.groupby(["repeat", "beer"]).size()
    if len(coverage) != len(repeats) * len(data.ids) or not coverage.eq(1).all():
        raise ValueError("Frozen fold coverage failed for the selected repeats")
    selected_folds.to_csv(output / "folds.csv", index=False)

    edge_tables: List[pd.DataFrame] = []
    node_tables: List[pd.DataFrame] = []
    sensitivity_tables: List[pd.DataFrame] = []
    beer_to_position = {int(beer): index for index, beer in enumerate(data.ids.to_numpy())}
    for repeat in repeats:
        repeat_rows = folds[folds["repeat"] == repeat]
        for fold in outer_folds:
            train_beers = repeat_rows.loc[repeat_rows["fold"] != fold, "beer"].to_numpy()
            train_index = np.asarray([beer_to_position[int(beer)] for beer in train_beers], dtype=int)
            network = build_fold_network(
                data.X.iloc[train_index],
                data.styles.iloc[train_index].astype(str).to_numpy(),
                repeat=repeat,
                fold=fold,
                thresholds=config["network"]["abs_rho_thresholds"],
                primary_threshold=float(config["network"]["primary_abs_rho_threshold"]),
                fdr_alpha=float(config["network"]["fdr_alpha"]),
                bootstrap_repeats=int(config["network"]["bootstrap_repeats"]),
                bootstrap_seed=int(config["network"]["bootstrap_seed"]),
                community_method=str(config["network"]["community_method"]),
            )
            edge_tables.append(network.edges)
            node_tables.append(network.nodes)
            sensitivity_tables.append(network.sensitivity)

    edges = pd.concat(edge_tables, ignore_index=True)
    nodes = pd.concat(node_tables, ignore_index=True)
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True)
    communities = community_stability(nodes)
    edges.to_parquet(output / "network_edges_by_fold.parquet", index=False)
    nodes.to_csv(output / "network_nodes.csv", index=False)
    sensitivity.to_csv(output / "network_threshold_sensitivity.csv", index=False)
    communities.to_csv(output / "community_stability.csv", index=False)

    importance_tables: List[pd.DataFrame] = []
    reconstruction_rows: List[Dict[str, object]] = []
    for repeat in repeats:
        for fold in outer_folds:
            fold_model_cache = {}
            for target in targets:
                raw, audit = explain_fold_target(
                    data,
                    folds,
                    tuning,
                    frozen_oof,
                    target=target,
                    grade=str(registry.loc[target, "grade"]),
                    repeat=repeat,
                    fold=fold,
                    modes=config["explanation"]["modes"],
                    permutations=int(config["explanation"]["permutations"]),
                    permutation_seed=int(config["explanation"]["permutation_seed"]),
                    style_minimum_group_n=int(
                        config["explanation"]["style_minimum_group_n"]
                    ),
                    tolerance=float(config["explanation"]["reconstruction_tolerance"]),
                    model_cache=fold_model_cache,
                )
                importance_tables.append(raw)
                reconstruction_rows.append(audit)

    importance = pd.concat(importance_tables, ignore_index=True)
    reconstruction = pd.DataFrame(reconstruction_rows)
    stability, pilot_view = summarize_importance(
        importance,
        primary_mode=str(config["explanation"]["primary_mode"]),
        top_k=config["explanation"]["top_k"],
        top10_frequency_minimum=float(config["stability"]["top10_frequency_minimum"]),
        positive_importance_frequency_minimum=float(
            config["stability"]["positive_importance_frequency_minimum"]
        ),
        direction_consistency_minimum=float(
            config["stability"]["direction_consistency_minimum"]
        ),
    )
    importance.to_parquet(output / "oof_permutation_importance.parquet", index=False)
    reconstruction.to_csv(output / "reconstruction_audit.csv", index=False)
    stability.to_csv(output / "importance_stability.csv", index=False)
    pilot_view.to_csv(output / "pilot_top10_view.csv", index=False)

    chemicals_path = _resolve(root, config["data"]["chemicals_csv"])
    sensory_path = _resolve(root, config["data"]["sensory_csv"])
    audit = {
        "status": "PASS",
        "run_name": config["experiment"]["run_name"],
        "scope": config["experiment"].get(
            "scope",
            (
                "all_main_engineering_smoke_not_formal_top3"
                if len(targets) == len(data.main_target_names)
                else "limited_target_pilot_not_formal_top3"
            ),
        ),
        "targets": targets,
        "repeats": repeats,
        "outer_folds": outer_folds,
        "n_samples": int(len(data.X)),
        "n_features": int(len(data.feature_names)),
        "n_network_edge_rows": int(len(edges)),
        "n_importance_rows": int(len(importance)),
        "all_reconstructions_passed": bool(reconstruction["reconstruction_passed"].all()),
        "max_reconstruction_abs_error": float(
            reconstruction["reconstruction_max_abs_error"].max()
        ),
        "input_sha256": {
            "chemicals_csv": sha256_file(chemicals_path),
            "sensory_csv": sha256_file(sensory_path),
            "folds": sha256_file(folds_path),
            "frozen_oof": sha256_file(oof_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "formal_top3_generated": False,
    }
    (output / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_decision(output, config, sensitivity, communities, reconstruction, pilot_view)
    return audit

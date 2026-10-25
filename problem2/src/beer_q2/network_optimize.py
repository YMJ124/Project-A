"""Independent confirmation of bootstrap-stable network communities.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from beer_q1.data import load_problem1_data

from .network import _community_membership, build_fold_network
from .reconstruct import _problem1_config, fold_positions


def _edge_set(frame: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(frame["feature_left"], frame["feature_right"]))


def _jaccard(left: Set[Any], right: Set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _membership_table(
    features: List[str],
    edges: pd.DataFrame,
    repeat: int,
    fold: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    index = {feature: position for position, feature in enumerate(features)}
    pairs = [(index[left], index[right]) for left, right in zip(edges.feature_left, edges.feature_right)]
    membership, method = _community_membership(
        len(features), pairs, edges["rho"].abs().tolist(), "leiden", seed
    )
    return pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "feature": features,
            "variant": label,
            "community": membership,
            "community_method": method,
        }
    )


def _median_ari(nodes: pd.DataFrame, variant: str) -> Tuple[float, pd.DataFrame]:
    selected = nodes[nodes["variant"] == variant]
    rows = []
    folds = sorted(selected["fold"].unique())
    for left, right in combinations(folds, 2):
        a = selected[selected["fold"] == left].sort_values("feature")
        b = selected[selected["fold"] == right].sort_values("feature")
        rows.append(
            {
                "variant": variant,
                "fold_a": int(left),
                "fold_b": int(right),
                "adjusted_rand_index": float(
                    adjusted_rand_score(a["community"], b["community"])
                ),
            }
        )
    table = pd.DataFrame(rows)
    return float(table["adjusted_rand_index"].median()), table


def run_network_confirmation(project_root: Path, config_path: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = (root / config["output_root"]).resolve()
    if (root / "problem2").resolve() not in output.parents:
        raise ValueError("Question 2 output must stay under problem2")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    data = load_problem1_data(_problem1_config(root))
    folds = pd.read_csv(root / "problem1/freezes/problem1_v1/base_run_metadata/folds.csv")
    repeat = int(config["repeat"])
    all_edges = []
    all_nodes = []
    fold_summary = []
    filtered_sets = {}
    baseline_sets = {}
    for fold in map(int, config["outer_folds"]):
        train_index, _, _ = fold_positions(data, folds, repeat, fold)
        result = build_fold_network(
            data.X.iloc[train_index],
            data.styles.iloc[train_index].astype(str).to_numpy(),
            repeat=repeat,
            fold=fold,
            thresholds=[float(config["abs_rho_threshold"])],
            primary_threshold=float(config["abs_rho_threshold"]),
            fdr_alpha=float(config["fdr_alpha"]),
            bootstrap_repeats=int(config["bootstrap_repeats"]),
            bootstrap_seed=int(config["bootstrap_seed"]),
            community_method=str(config["community_method"]),
        )
        baseline = result.edges.copy()
        stable = baseline[
            (baseline["bootstrap_frequency"] >= float(config["stable_edge_frequency_minimum"]))
            & (
                baseline["bootstrap_sign_consistency"]
                >= float(config["stable_edge_sign_consistency_minimum"])
            )
        ].copy()
        seed = int(config["bootstrap_seed"]) + repeat * 1009 + fold * 9176
        all_nodes.extend(
            [
                _membership_table(
                    list(data.feature_names), baseline, repeat, fold, seed, "unfiltered"
                ),
                _membership_table(
                    list(data.feature_names), stable, repeat, fold, seed, "stable_filtered"
                ),
            ]
        )
        baseline_sets[fold] = _edge_set(baseline)
        filtered_sets[fold] = _edge_set(stable)
        baseline["variant"] = "unfiltered"
        stable["variant"] = "stable_filtered"
        all_edges.extend([baseline, stable])
        fold_summary.append(
            {
                "repeat": repeat,
                "fold": fold,
                "baseline_edges": int(len(baseline)),
                "stable_edges": int(len(stable)),
                "edge_retention": float(len(stable) / len(baseline)),
            }
        )

    nodes = pd.concat(all_nodes, ignore_index=True)
    edges = pd.concat(all_edges, ignore_index=True)
    fold_summary = pd.DataFrame(fold_summary)
    baseline_ari, baseline_pairs = _median_ari(nodes, "unfiltered")
    stable_ari, stable_pairs = _median_ari(nodes, "stable_filtered")
    jaccards = [
        _jaccard(filtered_sets[left], filtered_sets[right])
        for left, right in combinations(sorted(filtered_sets), 2)
    ]
    stable_edge_jaccard = float(np.median(jaccards))
    edge_retention = float(fold_summary["edge_retention"].median())
    adoption = config["adoption"]
    criteria = {
        "median_ari_minimum": stable_ari >= float(adoption["median_ari_minimum"]),
        "median_ari_improvement_minimum": (
            stable_ari - baseline_ari
            >= float(adoption["median_ari_improvement_minimum"])
        ),
        "median_edge_jaccard_minimum": stable_edge_jaccard
        >= float(adoption["median_edge_jaccard_minimum"]),
        "edge_retention_minimum": edge_retention
        >= float(adoption["edge_retention_minimum"]),
    }
    adopted = bool(all(criteria.values()))
    summary = {
        "status": "ADOPT" if adopted else "REJECT",
        "repeat": repeat,
        "bootstrap_repeats": int(config["bootstrap_repeats"]),
        "baseline_median_ari": baseline_ari,
        "stable_filtered_median_ari": stable_ari,
        "median_ari_improvement": stable_ari - baseline_ari,
        "stable_filtered_edge_jaccard": stable_edge_jaccard,
        "median_edge_retention": edge_retention,
        "criteria": criteria,
    }
    edges.to_parquet(output / "network_edges_confirmation.parquet", index=False)
    nodes.to_csv(output / "network_nodes_confirmation.csv", index=False)
    fold_summary.to_csv(output / "fold_summary.csv", index=False)
    pd.concat([baseline_pairs, stable_pairs], ignore_index=True).to_csv(
        output / "community_stability_confirmation.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    decision = [
        "# Stable-edge community confirmation",
        "",
        f"- Decision: **{summary['status']}**",
        f"- Baseline median ARI: {baseline_ari:.3f}",
        f"- Stable-filtered median ARI: {stable_ari:.3f}",
        f"- ARI improvement: {stable_ari - baseline_ari:.3f}",
        f"- Filtered edge Jaccard: {stable_edge_jaccard:.3f}",
        f"- Median edge retention: {edge_retention:.1%}",
        "",
        "The mechanism changes only community construction; the full FDR-qualified edge table remains available for descriptive reporting.",
    ]
    (output / "decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    return summary

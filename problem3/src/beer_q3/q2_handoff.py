"""Build a frozen, leakage-safe Question 2 handoff for Question 3.

Community identifiers produced independently in each fold are arbitrary. This
module compares partitions through pairwise co-assignment, never by directly
matching their integer labels.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_pair(left: str, right: str) -> Tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _load_network_partitions(
    project_root: Path, config: Mapping[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[Path]]:
    network = config["network"]
    edge_frames = []
    node_frames = []
    source_paths = []
    for value in network["repeat_directories"]:
        directory = _resolve(project_root, value)
        edge_path = directory / network["edge_file"]
        node_path = directory / network["node_file"]
        if not edge_path.is_file() or not node_path.is_file():
            raise FileNotFoundError(f"Incomplete Q2 network directory: {directory}")
        edge_frames.append(pd.read_parquet(edge_path))
        node_frames.append(pd.read_csv(node_path))
        source_paths.extend([edge_path, node_path])

    edges = pd.concat(edge_frames, ignore_index=True)
    nodes = pd.concat(node_frames, ignore_index=True)
    variant = network["variant"]
    edges = edges.loc[edges["variant"] == variant].copy()
    nodes = nodes.loc[nodes["variant"] == variant].copy()
    return edges, nodes, source_paths


def _validate_partitions(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    expected_repeats: int,
    expected_folds: int,
    expected_features: int,
) -> Tuple[list[Tuple[int, int]], list[str]]:
    partitions = sorted(
        (int(row.repeat), int(row.fold))
        for row in nodes[["repeat", "fold"]].drop_duplicates().itertuples(index=False)
    )
    expected_count = expected_repeats * expected_folds
    if len(partitions) != expected_count:
        raise ValueError(f"Expected {expected_count} network partitions, got {len(partitions)}")
    repeats = {repeat for repeat, _ in partitions}
    if len(repeats) != expected_repeats:
        raise ValueError("Network repeats do not match the locked protocol")
    if any(
        len({fold for repeat_value, fold in partitions if repeat_value == repeat})
        != expected_folds
        for repeat in repeats
    ):
        raise ValueError("A network repeat does not contain all expected folds")

    feature_sets = []
    for repeat, fold in partitions:
        part = nodes[(nodes["repeat"] == repeat) & (nodes["fold"] == fold)]
        if part["feature"].duplicated().any():
            raise ValueError(f"Duplicate node in partition {(repeat, fold)}")
        feature_sets.append(set(part["feature"].astype(str)))
    if any(current != feature_sets[0] for current in feature_sets[1:]):
        raise ValueError("Network node sets differ across partitions")
    features = sorted(feature_sets[0])
    if len(features) != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {len(features)}")
    edge_partitions = set(
        map(tuple, edges[["repeat", "fold"]].drop_duplicates().to_numpy())
    )
    if edge_partitions != set(partitions):
        raise ValueError("Stable edge partitions differ from node partitions")
    return partitions, features


def build_consensus_edges(
    edges: pd.DataFrame,
    n_partitions: int,
    minimum_frequency: float,
    minimum_sign_consistency: float,
) -> pd.DataFrame:
    """Aggregate stable edges across independent outer-training partitions."""
    work = edges.copy()
    pairs = [
        _canonical_pair(str(left), str(right))
        for left, right in zip(work["feature_left"], work["feature_right"])
    ]
    work["feature_left"] = [pair[0] for pair in pairs]
    work["feature_right"] = [pair[1] for pair in pairs]
    if work.duplicated(["repeat", "fold", "feature_left", "feature_right"]).any():
        raise ValueError("Duplicate stable edge inside a network partition")

    rows = []
    for (left, right), group in work.groupby(["feature_left", "feature_right"], sort=True):
        positive = float((group["rho"] > 0).mean())
        cross_sign = max(positive, 1.0 - positive)
        presence_count = int(len(group))
        presence_frequency = presence_count / n_partitions
        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "partition_presence_count": presence_count,
                "partition_count": n_partitions,
                "partition_presence_frequency": presence_frequency,
                "rho_median": float(group["rho"].median()),
                "rho_q25": float(group["rho"].quantile(0.25)),
                "rho_q75": float(group["rho"].quantile(0.75)),
                "absolute_rho_median": float(group["rho"].abs().median()),
                "cross_partition_sign_consistency": cross_sign,
                "bootstrap_frequency_median": float(group["bootstrap_frequency"].median()),
                "bootstrap_sign_consistency_median": float(
                    group["bootstrap_sign_consistency"].median()
                ),
                "consensus_edge": bool(
                    presence_frequency + 1e-12 >= minimum_frequency
                    and cross_sign + 1e-12 >= minimum_sign_consistency
                ),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["consensus_edge", "partition_presence_frequency", "absolute_rho_median"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_consensus_communities(
    nodes: pd.DataFrame,
    partitions: Sequence[Tuple[int, int]],
    features: Sequence[str],
    minimum_frequency: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create consensus modules from pairwise community co-assignment rates."""
    pair_counts: Dict[Tuple[str, str], int] = {}
    for repeat, fold in partitions:
        partition = nodes[(nodes["repeat"] == repeat) & (nodes["fold"] == fold)]
        for _, group in partition.groupby("community"):
            members = sorted(group["feature"].astype(str))
            for pair in itertools.combinations(members, 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    n_partitions = len(partitions)
    coassignment = pd.DataFrame(
        [
            {
                "feature_left": left,
                "feature_right": right,
                "coassignment_count": count,
                "partition_count": n_partitions,
                "coassignment_frequency": count / n_partitions,
                "consensus_link": bool(count / n_partitions + 1e-12 >= minimum_frequency),
            }
            for (left, right), count in sorted(pair_counts.items())
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(features)
    stable_pairs = coassignment.loc[coassignment["consensus_link"]]
    graph.add_edges_from(
        stable_pairs[["feature_left", "feature_right"]].itertuples(index=False, name=None)
    )
    components = sorted(
        (sorted(component) for component in nx.connected_components(graph)),
        key=lambda values: (-len(values), values[0]),
    )
    module_by_feature = {
        feature: f"M{index:03d}"
        for index, component in enumerate(components, start=1)
        for feature in component
    }

    frequency_by_pair = {
        _canonical_pair(row.feature_left, row.feature_right): row.coassignment_frequency
        for row in coassignment.itertuples(index=False)
    }
    rows = []
    for component in components:
        size = len(component)
        for feature in component:
            peer_frequencies = [
                frequency_by_pair.get(_canonical_pair(feature, peer), 0.0)
                for peer in component
                if peer != feature
            ]
            rows.append(
                {
                    "feature": feature,
                    "consensus_module": module_by_feature[feature],
                    "module_size": size,
                    "mean_within_module_coassignment": (
                        float(np.mean(peer_frequencies)) if peer_frequencies else 1.0
                    ),
                    "minimum_within_module_coassignment": (
                        float(np.min(peer_frequencies)) if peer_frequencies else 1.0
                    ),
                }
            )
    communities = pd.DataFrame(rows).sort_values(
        ["module_size", "consensus_module", "feature"], ascending=[False, True, True]
    )
    return communities.reset_index(drop=True), coassignment


def _join_unique(values: Iterable[Any]) -> str:
    clean = sorted({str(value) for value in values if pd.notna(value) and str(value)})
    return ";".join(clean)


def build_candidate_handoff(
    candidates: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    community_columns = communities.rename(columns={"feature": "representative_feature"})
    result = candidates.merge(
        community_columns, on="representative_feature", how="left", validate="many_to_one"
    )
    if result["consensus_module"].isna().any():
        raise ValueError("A Q2 candidate is absent from the consensus communities")
    rho = pd.to_numeric(result["attachment6_rho"], errors="coerce")
    result["association_direction"] = np.select(
        [rho > 0, rho < 0], ["positive", "negative"], default="unknown"
    )
    result["association_basis"] = np.where(
        rho.notna(), "attachment6_descriptive_spearman", "unresolved"
    )
    result["causal_direction_claimed"] = False
    result["q3_use"] = np.where(
        result["candidate_status"].astype(str).str.endswith("CANDIDATE"),
        "candidate_guidance_with_validation",
        "audit_only",
    )
    return result


def build_actionability_registry(
    feature_schema: pd.DataFrame,
    field_dictionary: pd.DataFrame,
    candidates: pd.DataFrame,
    communities: pd.DataFrame,
    actionability: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    registry = feature_schema[
        ["feature", "minimum", "maximum", "q01", "q05", "q50", "q95", "q99"]
    ].copy()
    names = field_dictionary[["field_1_csv", "node_2_csv"]].rename(
        columns={"field_1_csv": "feature", "node_2_csv": "readable_name"}
    )
    registry = registry.merge(names, on="feature", how="left", validate="one_to_one")
    registry = registry.merge(communities, on="feature", how="left", validate="one_to_one")
    if registry[["readable_name", "consensus_module"]].isna().any().any():
        raise ValueError("Actionability registry has unresolved names or modules")

    candidate_summary = (
        candidates.groupby("representative_feature")
        .agg(
            q2_candidate_count=("target", "size"),
            q2_candidate_targets=("target", _join_unique),
            q2_best_rank=("candidate_rank", "min"),
            q2_evidence_tiers=("evidence_tier", _join_unique),
        )
        .reset_index()
        .rename(columns={"representative_feature": "feature"})
    )
    registry = registry.merge(candidate_summary, on="feature", how="left", validate="one_to_one")
    registry["q2_candidate_count"] = registry["q2_candidate_count"].fillna(0).astype(int)
    registry["q2_candidate_targets"] = registry["q2_candidate_targets"].fillna("")
    registry["q2_evidence_tiers"] = registry["q2_evidence_tiers"].fillna("")
    registry["q2_candidate_representative"] = registry["q2_candidate_count"] > 0

    derived = set(actionability["derived_read_only"])
    context = set(actionability["bounded_context"])
    defects = set(actionability["defect_chemical_features"])
    features = set(registry["feature"])
    unknown = (derived | context | defects) - features
    if unknown:
        raise ValueError(f"Configured actionability features are absent: {sorted(unknown)}")

    registry["generation_role"] = "profile_coordinate"
    registry.loc[registry["feature"].isin(context), "generation_role"] = "bounded_context"
    registry.loc[registry["feature"].isin(derived), "generation_role"] = "derived_read_only"
    registry["generator_decision_variable"] = registry["generation_role"].eq(
        "profile_coordinate"
    )
    registry["requires_process_validation"] = ~registry["feature"].isin(derived)
    registry["official_defect_chemical"] = registry["feature"].isin(defects)
    registry["causal_actionability_claimed"] = False
    registry["direct_process_control_claimed"] = False
    registry["recommended_bounds"] = "empirical_q01_q99"
    registry.loc[
        registry["feature"].isin(derived), "recommended_bounds"
    ] = "recompute_from_components"
    return registry.sort_values("feature").reset_index(drop=True)


def build_q2_handoff(project_root: Path, config_path: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    network = config["network"]
    edges, nodes, network_sources = _load_network_partitions(project_root, config)
    partitions, features = _validate_partitions(
        edges,
        nodes,
        int(network["expected_repeats"]),
        int(network["expected_folds_per_repeat"]),
        int(network["expected_features"]),
    )
    minimum_frequency = float(network["consensus_minimum_frequency"])
    consensus_edges = build_consensus_edges(
        edges,
        len(partitions),
        minimum_frequency,
        float(network["consensus_minimum_sign_consistency"]),
    )
    communities, coassignment = build_consensus_communities(
        nodes, partitions, features, minimum_frequency
    )

    inputs = {
        key: _resolve(project_root, value) for key, value in config["inputs"].items()
    }
    missing = [key for key, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Q2 handoff inputs: {missing}")
    candidates = pd.read_csv(inputs["candidate_catalog"])
    decisions = pd.read_csv(inputs["target_decisions"])
    field_dictionary = pd.read_csv(inputs["field_dictionary"])
    feature_schema = pd.read_csv(inputs["chemical_feature_schema"])
    if len(candidates) != 46 or candidates["target"].nunique() != 25:
        raise ValueError("Frozen Q2 candidate catalog does not match its locked contract")
    if len(decisions) != 30 or decisions["target"].nunique() != 30:
        raise ValueError("Frozen Q2 target decisions do not cover 30 targets")
    if set(features) != set(feature_schema["feature"]):
        raise ValueError("Q2 network features differ from the Q3 chemical schema")

    candidate_handoff = build_candidate_handoff(candidates, communities)
    registry = build_actionability_registry(
        feature_schema,
        field_dictionary,
        candidates,
        communities,
        config["actionability"],
    )
    output = _resolve(project_root, config["runtime"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    consensus_edges.to_csv(output / "consensus_edges.csv", index=False)
    communities.to_csv(output / "consensus_communities.csv", index=False)
    coassignment.to_csv(output / "community_coassignment.csv", index=False)
    candidate_handoff.to_csv(output / "candidate_handoff.csv", index=False)
    registry.to_csv(output / "actionability_registry.csv", index=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    consensus_edge_count = int(consensus_edges["consensus_edge"].sum())
    module_sizes = communities.groupby("consensus_module").size()
    summary = {
        "status": "PASS",
        "network_variant": network["variant"],
        "partition_count": len(partitions),
        "repeat_count": int(nodes["repeat"].nunique()),
        "folds_per_repeat": int(nodes["fold"].nunique()),
        "feature_count": len(features),
        "observed_stable_edge_pair_count": int(len(consensus_edges)),
        "consensus_edge_count": consensus_edge_count,
        "consensus_minimum_frequency": minimum_frequency,
        "consensus_minimum_partition_count": int(
            math.ceil(minimum_frequency * len(partitions) - 1e-12)
        ),
        "consensus_module_count": int(communities["consensus_module"].nunique()),
        "largest_consensus_module": int(module_sizes.max()),
        "singleton_consensus_modules": int((module_sizes == 1).sum()),
        "q2_candidate_count": int(len(candidate_handoff)),
        "q2_candidate_target_count": int(candidate_handoff["target"].nunique()),
        "q2_target_count": int(decisions["target"].nunique()),
        "q2_targets_without_candidates": sorted(
            set(decisions["target"]) - set(candidate_handoff["target"])
        ),
        "derived_read_only_count": int(
            (registry["generation_role"] == "derived_read_only").sum()
        ),
        "bounded_context_count": int(
            (registry["generation_role"] == "bounded_context").sum()
        ),
        "causal_claims": False,
        "community_label_alignment_used": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_paths = [*network_sources, *inputs.values()]
    manifest = {
        str(path.relative_to(project_root)): _sha256(path) for path in manifest_paths
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def write_readme(output: Path, summary: Mapping[str, Any]) -> None:
    readme = f"""# 问题二到问题三交接接口 v1

本目录把问题二的 7 重复×3 折网络稳定性结果压缩成问题三可直接读取的固定接口。

- 稳定边：在至少 {summary['consensus_minimum_partition_count']}/{summary['partition_count']} 个外层训练分区出现，且跨分区符号一致率至少 90%。
- 共识社区：先计算每对特征在 21 个分区中的同社区频率，再对达到 2/3 的共归属关系取连通分量；没有比较不同分区中任意的社区数字标签。
- 候选物：保留问题二冻结的 {summary['q2_candidate_count']} 条候选记录，覆盖 {summary['q2_candidate_target_count']}/{summary['q2_target_count']} 个主感官目标；其余 5 个目标保留“无合格候选”结论，不自动补齐。
- 可操作性：这里只定义生成变量角色和经验边界，不宣称化学变量可被配方直接、独立或因果地控制。

`derived_read_only` 字段必须由其组成项重算，不能作为独立优化维度。`bounded_context` 字段可以作为条件或约束，但在建立工艺到化学的模型前也不能称为直接配方控制量。

主要文件：

- `consensus_edges.csv`：跨 21 分区的稳定边统计。
- `consensus_communities.csv`：231 个字段的共识模块。
- `community_coassignment.csv`：所有曾经同社区的字段对及其频率。
- `candidate_handoff.csv`：问题二正式候选及其模块、证据与非因果方向标签。
- `actionability_registry.csv`：231 个生成字段的角色、边界和问题二候选覆盖。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )

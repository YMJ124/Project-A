"""Leakage-safe chemical abundance networks for Question 2.

Networks are recomputed from each frozen Q1 outer-training fold.  Attachment
2 is a descriptive cross-check only and is never used to define fold edges.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class NetworkResult:
    edges: pd.DataFrame
    nodes: pd.DataFrame
    sensitivity: pd.DataFrame


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values while preserving NaN positions."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    selected = values[finite]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def spearman_with_pvalues(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise-complete Spearman rho and its standard t approximation."""
    numeric = frame.astype(float)
    # pandas recomputes ranks on the pairwise-complete rows for every pair.
    # This exactly matches attachment 2, including the nine fields with NAs.
    rho = numeric.corr(method="spearman", min_periods=3).to_numpy(dtype=float)
    observed = numeric.notna().to_numpy(dtype=np.int16)
    n_pair = observed.T @ observed
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.maximum(1.0 - rho**2, np.finfo(float).eps)
        statistic = rho * np.sqrt(np.maximum(n_pair - 2, 0) / denominator)
        p_value = 2.0 * student_t.sf(np.abs(statistic), df=np.maximum(n_pair - 2, 1))
    p_value[n_pair < 3] = np.nan
    np.fill_diagonal(p_value, 0.0)
    return rho, p_value, n_pair


def stratified_bootstrap_indices(styles: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Resample within every style, preserving each style's training-fold size."""
    style_array = np.asarray(styles, dtype=object)
    sampled: List[np.ndarray] = []
    for style in pd.unique(style_array):
        positions = np.flatnonzero(style_array == style)
        sampled.append(rng.choice(positions, size=len(positions), replace=True))
    return np.concatenate(sampled)


def _community_membership(
    n_nodes: int,
    edge_indices: Sequence[Tuple[int, int]],
    weights: Sequence[float],
    method: str,
    seed: int,
) -> Tuple[np.ndarray, str]:
    if not edge_indices:
        return np.arange(n_nodes, dtype=int), "isolates"
    if method == "leiden":
        try:
            import igraph as ig
            import leidenalg

            graph = ig.Graph(n=n_nodes, edges=list(edge_indices), directed=False)
            partition = leidenalg.find_partition(
                graph,
                leidenalg.ModularityVertexPartition,
                weights=list(weights),
                seed=int(seed),
            )
            return np.asarray(partition.membership, dtype=int), "leiden"
        except (ImportError, RuntimeError, ValueError):
            pass

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(
        [(left, right, float(weight)) for (left, right), weight in zip(edge_indices, weights)]
    )
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    membership = np.full(n_nodes, -1, dtype=int)
    for community_id, members in enumerate(communities):
        membership[list(members)] = community_id
    for node in np.flatnonzero(membership < 0):
        membership[node] = int(membership.max() + 1)
    return membership, "greedy_modularity"


def build_fold_network(
    X_train: pd.DataFrame,
    styles_train: Sequence[str],
    repeat: int,
    fold: int,
    thresholds: Iterable[float],
    primary_threshold: float,
    fdr_alpha: float,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    community_method: str = "leiden",
) -> NetworkResult:
    """Build one fold-local network and bootstrap its edge stability."""
    thresholds = sorted(set(float(value) for value in thresholds))
    if primary_threshold not in thresholds:
        raise ValueError("primary_threshold must be included in thresholds")
    names = list(map(str, X_train.columns))
    n_features = len(names)
    upper_left, upper_right = np.triu_indices(n_features, k=1)

    rho, p_value, n_pair = spearman_with_pvalues(X_train)
    upper_p = p_value[upper_left, upper_right]
    upper_q = benjamini_hochberg(upper_p)
    q_value = np.full_like(rho, np.nan, dtype=float)
    q_value[upper_left, upper_right] = upper_q
    q_value[upper_right, upper_left] = upper_q
    np.fill_diagonal(q_value, 0.0)

    rng = np.random.default_rng(int(bootstrap_seed) + int(repeat) * 1009 + int(fold) * 9176)
    bootstrap_rho = np.empty((bootstrap_repeats, len(upper_left)), dtype=np.float32)
    for bootstrap in range(bootstrap_repeats):
        indices = stratified_bootstrap_indices(styles_train, rng)
        sampled = X_train.iloc[indices]
        sampled_rho, _, _ = spearman_with_pvalues(sampled)
        bootstrap_rho[bootstrap] = sampled_rho[upper_left, upper_right]

    base_rho = rho[upper_left, upper_right]
    base_sign = np.sign(base_rho)
    same_sign = np.sign(bootstrap_rho) == base_sign[None, :]
    q_low, q_high = np.nanquantile(bootstrap_rho, [0.025, 0.975], axis=0)

    edge_rows: List[Dict[str, object]] = []
    sensitivity_rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        selected = (np.abs(base_rho) >= threshold) & (upper_q <= fdr_alpha)
        sensitivity_rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "abs_rho_threshold": threshold,
                "fdr_alpha": float(fdr_alpha),
                "n_edges": int(selected.sum()),
                "positive_edges": int((selected & (base_rho > 0)).sum()),
                "negative_edges": int((selected & (base_rho < 0)).sum()),
            }
        )
        selected_positions = np.flatnonzero(selected)
        frequency = np.nanmean(np.abs(bootstrap_rho) >= threshold, axis=0)
        sign_consistency = np.nanmean(same_sign, axis=0)
        for position in selected_positions:
            left = int(upper_left[position])
            right = int(upper_right[position])
            edge_rows.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "abs_rho_threshold": threshold,
                    "feature_left": names[left],
                    "feature_right": names[right],
                    "rho": float(base_rho[position]),
                    "sign": "positive" if base_rho[position] > 0 else "negative",
                    "p_value": float(upper_p[position]),
                    "q_value": float(upper_q[position]),
                    "n_pair": int(n_pair[left, right]),
                    "bootstrap_frequency": float(frequency[position]),
                    "bootstrap_sign_consistency": float(sign_consistency[position]),
                    "bootstrap_rho_low": float(q_low[position]),
                    "bootstrap_rho_high": float(q_high[position]),
                }
            )

    primary_mask = (np.abs(base_rho) >= primary_threshold) & (upper_q <= fdr_alpha)
    primary_positions = np.flatnonzero(primary_mask)
    edge_indices = [
        (int(upper_left[position]), int(upper_right[position]))
        for position in primary_positions
    ]
    weights = [float(abs(base_rho[position])) for position in primary_positions]
    membership, method_used = _community_membership(
        n_features,
        edge_indices,
        weights,
        community_method,
        int(bootstrap_seed) + int(repeat) * 1009 + int(fold),
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(n_features))
    for position in primary_positions:
        left = int(upper_left[position])
        right = int(upper_right[position])
        graph.add_edge(left, right, weight=float(abs(base_rho[position])))
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {i: 0.0 for i in graph}
    positive_degree = np.zeros(n_features, dtype=float)
    negative_degree = np.zeros(n_features, dtype=float)
    for position in primary_positions:
        left = int(upper_left[position])
        right = int(upper_right[position])
        target = positive_degree if base_rho[position] > 0 else negative_degree
        target[left] += abs(base_rho[position])
        target[right] += abs(base_rho[position])

    connected_by_community: Dict[int, bool] = {}
    for community in np.unique(membership):
        members = np.flatnonzero(membership == community).tolist()
        subgraph = graph.subgraph(members)
        connected_by_community[int(community)] = len(members) == 1 or nx.is_connected(subgraph)

    node_rows = [
        {
            "repeat": int(repeat),
            "fold": int(fold),
            "feature": name,
            "community": int(membership[index]),
            "community_method": method_used,
            "community_connected": connected_by_community[int(membership[index])],
            "weighted_degree_positive": float(positive_degree[index]),
            "weighted_degree_negative": float(negative_degree[index]),
            "weighted_degree_absolute": float(positive_degree[index] + negative_degree[index]),
            "pagerank_absolute": float(pagerank[index]),
        }
        for index, name in enumerate(names)
    ]
    return NetworkResult(
        edges=pd.DataFrame(edge_rows),
        nodes=pd.DataFrame(node_rows),
        sensitivity=pd.DataFrame(sensitivity_rows),
    )


def community_stability(node_table: pd.DataFrame) -> pd.DataFrame:
    """Compare community assignments between every pair of analyzed folds."""
    rows: List[Dict[str, object]] = []
    keys = list(node_table[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None))
    for (repeat_a, fold_a), (repeat_b, fold_b) in combinations(keys, 2):
        left = node_table[(node_table["repeat"] == repeat_a) & (node_table["fold"] == fold_a)]
        right = node_table[(node_table["repeat"] == repeat_b) & (node_table["fold"] == fold_b)]
        paired = left[["feature", "community"]].merge(
            right[["feature", "community"]],
            on="feature",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        rows.append(
            {
                "repeat_a": int(repeat_a),
                "fold_a": int(fold_a),
                "repeat_b": int(repeat_b),
                "fold_b": int(fold_b),
                "adjusted_rand_index": float(
                    adjusted_rand_score(paired["community_a"], paired["community_b"])
                ),
            }
        )
    return pd.DataFrame(rows)

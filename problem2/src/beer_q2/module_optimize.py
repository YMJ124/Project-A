"""Confirm whether stable network modules improve attribution agreement.

The mechanism is predeclared before inspecting repeat 2: aggregate absolute
TreeSHAP within each stable community and compare it with held-out joint
community permutation.  No model, fold, or candidate threshold is changed.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


def _jaccard(left: Set[int], right: Set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def module_method_agreement(
    shap_values: pd.DataFrame,
    group_importance: pd.DataFrame,
    network_nodes: pd.DataFrame,
    feature_agreement: pd.DataFrame,
    top_k: int,
    network_variant: str = "stable_filtered",
) -> pd.DataFrame:
    """Return paired feature- and module-level agreement per target/fold."""
    nodes = network_nodes.copy()
    if "variant" in nodes.columns:
        nodes = nodes[nodes["variant"] == network_variant].copy()
    shap_fold = (
        shap_values.groupby(
            ["target", "repeat", "fold", "feature"], as_index=False
        )["abs_shap_value"]
        .mean()
        .merge(
            nodes[["repeat", "fold", "feature", "community"]],
            on=["repeat", "fold", "feature"],
            how="left",
            validate="many_to_one",
        )
    )
    if shap_fold["community"].isna().any():
        raise ValueError("SHAP features are missing stable-community assignments")
    module_shap = (
        shap_fold.groupby(
            ["target", "repeat", "fold", "community"], as_index=False
        )["abs_shap_value"]
        .sum()
        .rename(columns={"abs_shap_value": "module_abs_shap"})
    )
    module_permutation = (
        group_importance.groupby(
            ["target", "repeat", "fold", "community"], as_index=False
        )["delta_rmse"]
        .mean()
        .rename(columns={"delta_rmse": "module_delta_rmse"})
    )
    feature_column = f"permutation_shap_top{top_k}_jaccard"
    rows: List[Dict[str, Any]] = []
    keys = module_shap[["target", "repeat", "fold"]].drop_duplicates()
    for target, repeat, fold in keys.itertuples(index=False, name=None):
        shap_group = module_shap[
            (module_shap["target"] == target)
            & (module_shap["repeat"] == repeat)
            & (module_shap["fold"] == fold)
        ]
        permutation_group = module_permutation[
            (module_permutation["target"] == target)
            & (module_permutation["repeat"] == repeat)
            & (module_permutation["fold"] == fold)
        ]
        shap_top = set(
            shap_group.nlargest(top_k, "module_abs_shap")["community"].astype(int)
        )
        permutation_top = set(
            permutation_group.nlargest(top_k, "module_delta_rmse")["community"].astype(int)
        )
        feature_row = feature_agreement[
            (feature_agreement["target"] == target)
            & (feature_agreement["repeat"] == repeat)
            & (feature_agreement["fold"] == fold)
        ]
        if len(feature_row) != 1:
            raise ValueError("Expected one feature agreement row per target/fold")
        feature_value = float(feature_row.iloc[0][feature_column])
        module_value = _jaccard(shap_top, permutation_top)
        rows.append(
            {
                "target": target,
                "repeat": int(repeat),
                "fold": int(fold),
                "top_k": int(top_k),
                "feature_level_jaccard": feature_value,
                "module_level_jaccard": module_value,
                "module_minus_feature": module_value - feature_value,
                "shared_modules": "|".join(map(str, sorted(shap_top & permutation_top))),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "repeat", "fold"])


def evaluate_module_confirmation(
    advanced_run: Path,
    network_run: Path,
    output_dir: Path,
    top_k: int = 10,
    median_module_minimum: float = 0.50,
    median_improvement_minimum: float = 0.10,
    improved_rows_minimum: int = 6,
    ab_target_median_minimum: float = 0.30,
) -> Dict[str, Any]:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shap_values = pd.read_parquet(advanced_run / "oof_shap_values.parquet")
    group_importance = pd.read_parquet(
        advanced_run / "oof_group_permutation_importance.parquet"
    )
    feature_agreement = pd.read_csv(
        advanced_run / "permutation_shap_agreement.csv"
    )
    network_nodes = pd.read_csv(network_run / "network_nodes_confirmation.csv")
    paired = module_method_agreement(
        shap_values,
        group_importance,
        network_nodes,
        feature_agreement,
        top_k,
    )
    target_summary = (
        paired.groupby("target", as_index=False)
        .agg(
            median_feature_jaccard=("feature_level_jaccard", "median"),
            median_module_jaccard=("module_level_jaccard", "median"),
            median_improvement=("module_minus_feature", "median"),
            improved_folds=("module_minus_feature", lambda values: int((values > 0).sum())),
        )
    )
    median_feature = float(paired["feature_level_jaccard"].median())
    median_module = float(paired["module_level_jaccard"].median())
    median_improvement = float(paired["module_minus_feature"].median())
    improved_rows = int((paired["module_minus_feature"] > 0).sum())
    ab_targets = {"A_hops_all", "F_esters_isoaa"}
    ab_minimum = float(
        target_summary[target_summary["target"].isin(ab_targets)][
            "median_module_jaccard"
        ].min()
    )
    criteria = {
        "median_module_jaccard_minimum": median_module
        >= float(median_module_minimum),
        "median_improvement_minimum": median_improvement
        >= float(median_improvement_minimum),
        "improved_target_folds_minimum": improved_rows
        >= int(improved_rows_minimum),
        "each_ab_target_median_minimum": ab_minimum
        >= float(ab_target_median_minimum),
    }
    status = "ADOPT" if all(criteria.values()) else "REJECT"
    summary = {
        "status": status,
        "mechanism": "stable_community_aggregated_shap_vs_joint_permutation",
        "repeat": int(paired["repeat"].iloc[0]),
        "top_k": int(top_k),
        "median_feature_level_jaccard": median_feature,
        "median_module_level_jaccard": median_module,
        "median_improvement": median_improvement,
        "improved_target_folds": improved_rows,
        "total_target_folds": int(len(paired)),
        "minimum_ab_target_median_module_jaccard": ab_minimum,
        "adoption_thresholds": {
            "median_module_jaccard_minimum": float(median_module_minimum),
            "median_improvement_minimum": float(median_improvement_minimum),
            "improved_target_folds_minimum": int(improved_rows_minimum),
            "ab_target_median_minimum": float(ab_target_median_minimum),
        },
        "criteria": criteria,
    }
    paired.to_csv(output / "paired_method_agreement.csv", index=False)
    target_summary.to_csv(output / "target_summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Stable-module attribution confirmation",
        "",
        f"- Decision: **{status}**",
        f"- Feature-level median Top-{top_k} Jaccard: {median_feature:.3f}",
        f"- Module-level median Top-{top_k} Jaccard: {median_module:.3f}",
        f"- Median paired improvement: {median_improvement:.3f}",
        f"- Improved target-folds: {improved_rows}/{len(paired)}",
        f"- Minimum A/B target median module Jaccard: {ab_minimum:.3f}",
        "",
        "The mechanism changes the reporting unit from correlated single features to stable communities. It does not alter predictions or promote the C-grade target.",
    ]
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

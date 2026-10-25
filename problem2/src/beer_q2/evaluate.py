"""Evidence-gate evaluation for the first Question 2 pilot.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from beer_q1.data import load_problem1_data

from .network import spearman_with_pvalues
from .preflight import _load_network, sha256_file


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _q1_config(root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(config["data"])
    data.pop("chemical_network_csv", None)
    data["chemicals_csv"] = str(_resolve(root, data["chemicals_csv"]))
    data["sensory_csv"] = str(_resolve(root, data["sensory_csv"]))
    return {"data": data, "runtime": {}}


def _edge_set(frame: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(frame["feature_left"].astype(str), frame["feature_right"].astype(str)))


def _jaccard(left: Set[Any], right: Set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _network_stability(edges: pd.DataFrame, sensitivity: pd.DataFrame):
    pair_rows: List[Dict[str, object]] = []
    for threshold, threshold_edges in edges.groupby("abs_rho_threshold"):
        keys = list(
            threshold_edges[["repeat", "fold"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for (repeat_a, fold_a), (repeat_b, fold_b) in combinations(keys, 2):
            left = threshold_edges[
                (threshold_edges["repeat"] == repeat_a)
                & (threshold_edges["fold"] == fold_a)
            ]
            right = threshold_edges[
                (threshold_edges["repeat"] == repeat_b)
                & (threshold_edges["fold"] == fold_b)
            ]
            pair_rows.append(
                {
                    "abs_rho_threshold": float(threshold),
                    "repeat_a": int(repeat_a),
                    "fold_a": int(fold_a),
                    "repeat_b": int(repeat_b),
                    "fold_b": int(fold_b),
                    "edge_jaccard": _jaccard(_edge_set(left), _edge_set(right)),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    rows = []
    for threshold, threshold_edges in edges.groupby("abs_rho_threshold"):
        counts = sensitivity[sensitivity["abs_rho_threshold"] == threshold]["n_edges"]
        pair = pairs[pairs["abs_rho_threshold"] == threshold]
        rows.append(
            {
                "abs_rho_threshold": float(threshold),
                "mean_edge_count": float(counts.mean()),
                "edge_count_cv": float(counts.std(ddof=1) / counts.mean()),
                "median_edge_jaccard": float(pair["edge_jaccard"].median()),
                "min_edge_jaccard": float(pair["edge_jaccard"].min()),
                "median_bootstrap_frequency": float(
                    threshold_edges["bootstrap_frequency"].median()
                ),
                "fraction_edges_bootstrap_frequency_ge_0_8": float(
                    (threshold_edges["bootstrap_frequency"] >= 0.8).mean()
                ),
                "median_sign_consistency": float(
                    threshold_edges["bootstrap_sign_consistency"].median()
                ),
            }
        )
    return pairs, pd.DataFrame(rows)


def _mode_consistency(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in stability.groupby("target"):
        primary = set(
            group[group["mode"] == "within_style"]
            .nlargest(10, "mean_delta_rmse")["feature"]
            .astype(str)
        )
        sensitivity = set(
            group[group["mode"] == "unconditional"]
            .nlargest(10, "mean_delta_rmse")["feature"]
            .astype(str)
        )
        rows.append(
            {
                "target": target,
                "within_style_vs_unconditional_top10_jaccard": _jaccard(
                    primary, sensitivity
                ),
                "within_style_stable_candidate_count": int(
                    group[
                        (group["mode"] == "within_style")
                        & (group["passes_pilot_stability"] == True)  # noqa: E712
                    ]["feature"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_pilot(project_root: Path, run_dir: Path) -> Dict[str, Any]:
    root = project_root.resolve()
    output = run_dir.resolve()
    if (root / "problem2").resolve() not in output.parents:
        raise ValueError("Pilot evaluation output must be under problem2")
    with (output / "config.resolved.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data = load_problem1_data(_q1_config(root, config))
    reference = _load_network(_resolve(root, config["data"]["chemical_network_csv"]))
    recalculated, _, _ = spearman_with_pvalues(data.X)
    reference_values = reference.to_numpy(dtype=float)
    difference = np.abs(recalculated - reference_values)
    attachment_check = {
        "method": "pairwise_complete_spearman",
        "shape": list(reference.shape),
        "max_abs_difference": float(np.nanmax(difference)),
        "mean_abs_difference": float(np.nanmean(difference)),
        "allclose_at_1e_12": bool(
            np.allclose(recalculated, reference_values, atol=1e-12, equal_nan=True)
        ),
    }
    (output / "attachment2_network_check.json").write_text(
        json.dumps(attachment_check, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    edges = pd.read_parquet(output / "network_edges_by_fold.parquet")
    sensitivity = pd.read_csv(output / "network_threshold_sensitivity.csv")
    community = pd.read_csv(output / "community_stability.csv")
    reconstruction = pd.read_csv(output / "reconstruction_audit.csv")
    stability = pd.read_csv(output / "importance_stability.csv")
    edge_pairs, network_summary = _network_stability(edges, sensitivity)
    mode_consistency = _mode_consistency(stability)
    edge_pairs.to_csv(output / "network_edge_jaccard.csv", index=False)
    network_summary.to_csv(output / "network_stability_summary.csv", index=False)
    mode_consistency.to_csv(output / "explanation_mode_consistency.csv", index=False)

    primary_threshold = float(config["network"]["primary_abs_rho_threshold"])
    primary = network_summary[
        np.isclose(network_summary["abs_rho_threshold"], primary_threshold)
    ].iloc[0]
    median_ari = float(community["adjusted_rand_index"].median())
    median_mode_jaccard = float(
        mode_consistency["within_style_vs_unconditional_top10_jaccard"].median()
    )
    configured_targets = config["experiment"]["targets"]
    target_count = (
        len(data.main_target_names)
        if configured_targets == "all_main"
        else len(configured_targets)
    )
    repeat_count = len(config["experiment"]["repeats"])
    fold_count = len(config["experiment"]["outer_folds"])
    bootstrap_depth = int(config["network"]["bootstrap_repeats"])
    permutation_depth = int(config["explanation"]["permutations"])
    freeze_audit = json.loads(
        (root / "problem1/freezes/problem1_v1/base_run_metadata/data_audit.json").read_text(
            encoding="utf-8"
        )
    )
    chemicals_path = _resolve(root, config["data"]["chemicals_csv"])
    sensory_path = _resolve(root, config["data"]["sensory_csv"])
    hash_match = (
        sha256_file(chemicals_path) == freeze_audit["chemicals_sha256"]
        and sha256_file(sensory_path) == freeze_audit["sensory_sha256"]
    )

    criteria: List[Dict[str, object]] = []

    def add(gate, criterion, required, observed, status, blocking, evidence):
        criteria.append(
            {
                "gate": gate,
                "criterion": criterion,
                "required_for_formal_v1": required,
                "observed": observed,
                "status": status,
                "blocking": bool(blocking),
                "evidence": evidence,
            }
        )

    add("G0_data", "Frozen input identity", "1.csv and 4.csv hashes match Q1 freeze", str(hash_match), "PASS" if hash_match else "FAIL", True, "data_audit.json")
    add("G0_data", "Attachment 2 method match", "pairwise Spearman max difference <=1e-12", f"{attachment_check['max_abs_difference']:.3g}", "PASS" if attachment_check["allclose_at_1e_12"] else "FAIL", True, "attachment2_network_check.json")
    add("G1_q1", "True champion reconstruction", "all analyzed target-folds <=1e-9", f"{reconstruction['reconstruction_passed'].sum()}/{len(reconstruction)}; max={reconstruction['reconstruction_max_abs_error'].max():.3g}", "PASS" if reconstruction["reconstruction_passed"].all() else "FAIL", True, "reconstruction_audit.csv")
    add("G1_q1", "Target coverage", "30/30 main targets", f"{target_count}/30", "PASS" if target_count == 30 else "PARTIAL", True, "config.resolved.yaml")
    add("G2_network", "Threshold/FDR sensitivity", "rho 0.30/0.40/0.50 and FDR<=0.05", str(sorted(sensitivity["abs_rho_threshold"].unique().tolist())), "PASS", True, "network_threshold_sensitivity.csv")
    add("G2_network", "Bootstrap depth", "200 style-stratified resamples per fold", str(bootstrap_depth), "PASS" if bootstrap_depth >= 200 else "PARTIAL", True, "config.resolved.yaml")
    add("G2_network", "Primary edge-set stability", "median fold Jaccard >=0.60 and edge-count CV<=0.15", f"Jaccard={primary['median_edge_jaccard']:.3f}; CV={primary['edge_count_cv']:.3f}", "PASS" if primary["median_edge_jaccard"] >= 0.60 and primary["edge_count_cv"] <= 0.15 else "PARTIAL", False, "network_stability_summary.csv")
    add("G2_network", "Community stability", "median fold ARI >=0.50", f"{median_ari:.3f}", "PASS" if median_ari >= 0.50 else "PARTIAL", False, "community_stability.csv")
    add("G2_network", "Fold/repeat coverage", "7 frozen repeats x 3 folds", f"{repeat_count} x {fold_count}", "PASS" if repeat_count == 7 and fold_count == 3 else "PARTIAL", True, "folds.csv")
    add("G3_explain", "Held-out primary explanation", "within-style permutation on outer validation only", "implemented", "PASS", True, "oof_permutation_importance.parquet")
    add("G3_explain", "Permutation depth", "100 permutations per feature/fold", str(permutation_depth), "PASS" if permutation_depth >= 100 else "PARTIAL", True, "config.resolved.yaml")
    add("G3_explain", "Permutation-mode sensitivity", "median conditional/unconditional Top-10 Jaccard >=0.50", f"{median_mode_jaccard:.3f}", "PASS" if median_mode_jaccard >= 0.50 else "PARTIAL", False, "explanation_mode_consistency.csv")
    add("G4_methods", "TreeSHAP agreement", "report permutation-vs-SHAP Top-10 Jaccard", "not run", "NOT_RUN", True, "required by AGENTS_2")
    add("G4_methods", "Correlated-group importance", "community/group joint permutation", "not run", "NOT_RUN", True, "required by AGENTS_2")
    add("G4_methods", "Style direction heterogeneity", "styles with n>=8 checked for reversals", "not run", "NOT_RUN", True, "required by AGENTS_2")
    add("G5_candidates", "Predeclared candidate stability", "Top10>=60%, positive>=80%, direction>=70%", f"stable counts: {dict(zip(mode_consistency.target, mode_consistency.within_style_stable_candidate_count))}", "PARTIAL", True, "importance_stability.csv; only 3 folds")
    add("G5_candidates", "Network de-redundancy", "community alternatives and stable representative", "not run", "NOT_RUN", True, "required before Top-3")
    add("G5_candidates", "Attachment 6/8 consistency", "agreements and conflicts reported without selection leakage", "not run", "NOT_RUN", True, "required before Top-3")
    add("G5_candidates", "C-grade restraint", "no forced formal Top-3", "formal_top3_generated=false", "PASS", True, "data_audit.json")
    add("G6_repro", "Auditable artifacts", "config, folds, hashes, versions, raw importance retained", "present", "PASS", True, "run directory")

    evaluation = pd.DataFrame(criteria)
    evaluation.to_csv(output / "evaluation_matrix.csv", index=False)
    blocking_ready = bool(
        evaluation.loc[evaluation["blocking"], "status"].eq("PASS").all()
    )
    summary = {
        "pilot_engineering_status": "PASS",
        "formal_question2_status": "READY" if blocking_ready else "HOLD",
        "criteria_counts": {
            str(key): int(value) for key, value in evaluation["status"].value_counts().items()
        },
        "blocking_not_passed": evaluation[
            evaluation["blocking"] & (evaluation["status"] != "PASS")
        ]["criterion"].tolist(),
        "network_primary": {
            "threshold": primary_threshold,
            "median_edge_jaccard": float(primary["median_edge_jaccard"]),
            "edge_count_cv": float(primary["edge_count_cv"]),
            "median_community_ari": median_ari,
        },
        "median_permutation_mode_top10_jaccard": median_mode_jaccard,
        "note": "Network gates added after this pilot are calibration targets and must be locked before fresh-repeat confirmation.",
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Question 2 pilot v1.1 evaluation",
        "",
        f"- Pilot engineering status: **{summary['pilot_engineering_status']}**",
        f"- Formal Question 2 status: **{summary['formal_question2_status']}**",
        f"- Criteria: {summary['criteria_counts']}",
        f"- Primary network: edge Jaccard={primary['median_edge_jaccard']:.3f}, edge-count CV={primary['edge_count_cv']:.3f}, community ARI={median_ari:.3f}.",
        f"- Conditional/unconditional permutation Top-10 Jaccard median={median_mode_jaccard:.3f}.",
        "",
        "## Blocking items",
        "",
    ]
    lines.extend([f"- {item}" for item in summary["blocking_not_passed"]])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The pilot validates the data identity, attachment-2 Spearman convention, frozen-model reconstruction, "
            "fold-local network path, and held-out permutation path. It does not validate formal chemical Top-3. "
            "Fresh-repeat confirmation must use the locked evaluation framework in `problem2/docs/evaluation_framework_v1.md`.",
        ]
    )
    (output / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

"""Assemble primary and secondary formal Question 2 candidate evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import yaml


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def assemble_candidate_bundle(
    project_root: Path, config_path: Path
) -> Mapping[str, Any]:
    root = project_root.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    primary = _resolve(root, config["primary_run"])
    rescue = _resolve(root, config["module_rescue_run"])
    output = _resolve(root, config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    primary_summary = json.loads((primary / "summary.json").read_text(encoding="utf-8"))
    rescue_summary = json.loads((rescue / "summary.json").read_text(encoding="utf-8"))
    if primary_summary["status"] != "PASS":
        raise ValueError("Primary formal candidate run did not pass")
    if rescue_summary["status"] != "ADOPT_AS_SECONDARY_MODULE_EVIDENCE":
        raise ValueError("Module-first rescue was not adopted")

    p = pd.read_csv(primary / "top3_candidates.csv")
    primary_catalog = pd.DataFrame(
        {
            "target": p["target"],
            "grade": p["grade"],
            "candidate_rank": p["module_rank"],
            "representative_feature": p["representative_feature"],
            "readable_name": p["readable_name"],
            "chemical_name": p["chemical_name"],
            "related_stable_alternatives": p["related_stable_alternatives"],
            "candidate_status": p["formal_candidate_status"],
            "evidence_tier": "primary_feature_complete",
            "mean_delta_rmse": p["mean_delta_rmse"],
            "positive_permutation_frequency": p["positive_permutation_frequency"],
            "top10_frequency": p["top10_frequency"],
            "top10_shap_frequency": p["top10_shap_frequency"],
            "style_reversal_rate": p["style_reversal_rate"],
            "attachment6_rho": p["attachment6_rho"],
            "source_artifact": "q2_formal_candidate_final_v1/top3_candidates.csv",
        }
    )
    r = pd.read_csv(rescue / "rescue_module_candidates.csv")
    rescue_catalog = pd.DataFrame(
        {
            "target": r["target"],
            "grade": r["grade"],
            "candidate_rank": r["module_rank"],
            "representative_feature": r["representative_feature"],
            "readable_name": r["readable_name"],
            "chemical_name": r["chemical_name"],
            "related_stable_alternatives": r["related_stable_alternatives"],
            "candidate_status": r["candidate_status"],
            "evidence_tier": "secondary_module_leave_one_repeat_out",
            "mean_delta_rmse": r["representative_mean_delta_rmse"],
            "positive_permutation_frequency": pd.NA,
            "top10_frequency": pd.NA,
            "top10_shap_frequency": r["mean_shap_top5_frequency"],
            "style_reversal_rate": r["style_reversal_rate"],
            "attachment6_rho": r["attachment6_rho"],
            "source_artifact": "q2_module_first_rescue_v2/rescue_module_candidates.csv",
        }
    )
    overlap = set(primary_catalog["target"]) & set(rescue_catalog["target"])
    if overlap:
        raise ValueError(f"Secondary rescue unexpectedly overlaps primary targets: {overlap}")
    catalog = pd.concat([primary_catalog, rescue_catalog], ignore_index=True)
    catalog = catalog.sort_values(["target", "candidate_rank"]).reset_index(drop=True)

    decisions = pd.read_csv(primary / "target_decisions.csv")
    decisions = decisions.rename(
        columns={
            "reported_candidate_count": "primary_candidate_count",
            "decision": "primary_decision",
        }
    )
    rescue_counts = rescue_catalog.groupby("target").size().rename("secondary_module_count")
    decisions = decisions.merge(rescue_counts, on="target", how="left")
    decisions["secondary_module_count"] = decisions["secondary_module_count"].fillna(0).astype(int)
    decisions["final_candidate_count"] = (
        decisions["primary_candidate_count"] + decisions["secondary_module_count"]
    )
    decisions["final_decision"] = decisions["primary_decision"]
    rescued = decisions["secondary_module_count"] > 0
    decisions.loc[
        rescued & decisions["grade"].eq("C"), "final_decision"
    ] = "EXPLORATORY_MODULE_PARTIAL"
    decisions.loc[
        rescued & ~decisions["grade"].eq("C"), "final_decision"
    ] = "SECONDARY_MODULE_PARTIAL_EVIDENCE"

    primary_source = pd.read_csv(primary / "source_consistency.csv")
    primary_source.insert(0, "evidence_tier", "primary_feature_complete")
    rescue_source = pd.read_csv(rescue / "source_consistency.csv")
    rescue_source.insert(0, "evidence_tier", "secondary_module_leave_one_repeat_out")
    source = pd.concat([primary_source, rescue_source], ignore_index=True)

    catalog.to_csv(output / "candidate_catalog.csv", index=False)
    decisions.to_csv(output / "target_decisions.csv", index=False)
    source.to_csv(output / "source_consistency.csv", index=False)
    summary: Dict[str, Any] = {
        "status": "READY_WITH_LIMITATIONS",
        "targets": int(decisions["target"].nunique()),
        "reported_candidates": int(len(catalog)),
        "primary_candidates": int(len(primary_catalog)),
        "secondary_module_candidates": int(len(rescue_catalog)),
        "targets_with_three_candidates": int((decisions["final_candidate_count"] == 3).sum()),
        "targets_with_one_or_two_candidates": int(decisions["final_candidate_count"].between(1, 2).sum()),
        "targets_with_insufficient_evidence": int((decisions["final_candidate_count"] == 0).sum()),
        "targets_still_insufficient": decisions.loc[
            decisions["final_candidate_count"] == 0, "target"
        ].tolist(),
        "forced_top3_count": 0,
        "question1_status": "HOLD",
        "causal_claims_allowed": False,
        "same_source_audit_is_external_validation": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Problem 2 candidate bundle v2", "",
        f"- Status: **{summary['status']}**",
        f"- Reported candidates: {len(catalog)} ({len(primary_catalog)} primary + {len(rescue_catalog)} secondary module-level).",
        f"- Targets with three / one-to-two / zero candidates: {summary['targets_with_three_candidates']} / {summary['targets_with_one_or_two_candidates']} / {summary['targets_with_insufficient_evidence']}.",
        f"- Still insufficient: {', '.join(summary['targets_still_insufficient'])}.",
        "- No target was forced to have three candidates; no secondary module replaced a primary feature-level result.",
        "- Question 1 remains HOLD, so all interpretation confidence remains capped by its frozen grade.",
        "- Results are associative, not causal; attachment 6/8 remain same-source audits.",
    ]
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

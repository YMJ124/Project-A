"""Assemble current Q2 evaluation after the first confirmed optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))

from beer_q2.current_status import assemble_current_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="problem2/artifacts/q2_pilot_v1_1")
    parser.add_argument("--advanced", default="problem2/artifacts/q2_explain_pilot_v1")
    parser.add_argument(
        "--network-confirmation",
        default="problem2/artifacts/q2_network_stability_confirmation_v1",
    )
    parser.add_argument(
        "--source-audit",
        default="problem2/artifacts/q2_source_audit_pilot_v1",
    )
    parser.add_argument(
        "--module-confirmation",
        default="problem2/artifacts/q2_module_top5_confirmation_v1",
    )
    parser.add_argument(
        "--reconstruction-all30",
        default="problem2/artifacts/q1_reconstruction_all30_repeat4_v1",
    )
    parser.add_argument(
        "--source-shap-audit",
        default="problem2/artifacts/q1_source_shap_audit_v1",
    )
    parser.add_argument(
        "--network-audits",
        nargs="+",
        default=[
            "problem2/artifacts/q2_network_repeat0_full_v1",
            "problem2/artifacts/q2_network_stability_confirmation_v1",
            "problem2/artifacts/q2_network_repeat2_v1",
            "problem2/artifacts/q2_network_repeat3_v1",
            "problem2/artifacts/q2_network_repeat4_v1",
            "problem2/artifacts/q2_network_repeat5_v1",
            "problem2/artifacts/q2_network_repeat6_v1",
        ],
    )
    parser.add_argument("--all30-smoke", default="problem2/artifacts/q2_all30_smoke_v1")
    parser.add_argument(
        "--all30-advanced-smoke",
        default="problem2/artifacts/q2_all30_advanced_smoke_v1",
    )
    parser.add_argument(
        "--formal-run", default="problem2/artifacts/q2_formal_all7_fast_v1"
    )
    parser.add_argument(
        "--formal-advanced-run",
        default="problem2/artifacts/q2_formal_all7_advanced_fast_v1",
    )
    parser.add_argument(
        "--formal-candidate-run",
        default="problem2/artifacts/q2_formal_candidate_final_v1",
    )
    parser.add_argument(
        "--formal-bundle-run",
        default="problem2/artifacts/q2_formal_candidate_bundle_v2",
    )
    parser.add_argument("--output", default="problem2/artifacts/q2_current_evaluation_v8")
    args = parser.parse_args()
    summary = assemble_current_status(
        PROJECT_ROOT / args.base,
        PROJECT_ROOT / args.advanced,
        PROJECT_ROOT / args.network_confirmation,
        PROJECT_ROOT / args.source_audit,
        PROJECT_ROOT / args.module_confirmation,
        PROJECT_ROOT / args.output,
        reconstruction_all30=PROJECT_ROOT / args.reconstruction_all30,
        source_shap_audit=PROJECT_ROOT / args.source_shap_audit,
        network_audits=[PROJECT_ROOT / path for path in args.network_audits],
        all30_smoke=PROJECT_ROOT / args.all30_smoke,
        all30_advanced_smoke=PROJECT_ROOT / args.all30_advanced_smoke,
        formal_run=PROJECT_ROOT / args.formal_run,
        formal_advanced_run=PROJECT_ROOT / args.formal_advanced_run,
        formal_candidate_run=PROJECT_ROOT / args.formal_candidate_run,
        formal_bundle_run=PROJECT_ROOT / args.formal_bundle_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

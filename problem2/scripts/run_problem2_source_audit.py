"""Run the post-selection attachment and stable-network pilot audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))

from beer_q2.source_audit import run_source_and_network_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="problem2/artifacts/q2_explain_pilot_v1/candidate_evidence.csv",
    )
    parser.add_argument(
        "--network-confirmation",
        default="problem2/artifacts/q2_network_stability_confirmation_v1",
    )
    parser.add_argument(
        "--output", default="problem2/artifacts/q2_source_audit_pilot_v1"
    )
    args = parser.parse_args()
    summary = run_source_and_network_audit(
        PROJECT_ROOT,
        PROJECT_ROOT / args.candidates,
        PROJECT_ROOT / args.network_confirmation,
        PROJECT_ROOT / args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

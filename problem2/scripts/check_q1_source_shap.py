"""Audit TreeSHAP additivity for all frozen Q1 source architectures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))
sys.path.insert(0, str(PROJECT_ROOT / "problem1/src"))

from beer_q2.source_shap_audit import run_source_shap_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output", default="problem2/artifacts/q1_source_shap_audit_v1"
    )
    args = parser.parse_args()
    summary = run_source_shap_audit(
        PROJECT_ROOT,
        args.repeat,
        args.fold,
        PROJECT_ROOT / args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "PASS":
        raise SystemExit("Frozen source TreeSHAP audit FAILED")


if __name__ == "__main__":
    main()

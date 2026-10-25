"""Build gap and diagnosis tables for the advanced Q2 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))

from beer_q2.advanced_evaluate import evaluate_advanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", default="problem2/artifacts/q2_pilot_v1_1")
    parser.add_argument("--advanced-run", default="problem2/artifacts/q2_explain_pilot_v1")
    args = parser.parse_args()
    summary = evaluate_advanced(
        PROJECT_ROOT / args.base_run, PROJECT_ROOT / args.advanced_run
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

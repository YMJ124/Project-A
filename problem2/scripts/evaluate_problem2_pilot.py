"""Evaluate an existing Question 2 pilot against the evidence gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))
sys.path.insert(0, str(PROJECT_ROOT / "problem1/src"))

from beer_q2.evaluate import evaluate_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="problem2/artifacts/q2_pilot_v1_1")
    args = parser.parse_args()
    summary = evaluate_pilot(PROJECT_ROOT, PROJECT_ROOT / args.run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

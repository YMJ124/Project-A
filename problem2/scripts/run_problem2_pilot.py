"""Run the first three-target Question 2 pilot in the beer-q1 environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))
sys.path.insert(0, str(PROJECT_ROOT / "problem1/src"))

from beer_q2.experiment import run_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="problem2/configs/problem2_pilot_v1.yaml"
    )
    args = parser.parse_args()
    result = run_pilot(PROJECT_ROOT, PROJECT_ROOT / args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

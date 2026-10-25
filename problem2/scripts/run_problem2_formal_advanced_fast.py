"""Run resumable formal TreeSHAP and community explanations for Question 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))
sys.path.insert(0, str(PROJECT_ROOT / "problem1/src"))

from beer_q2.formal_advanced_fast import run_formal_advanced_fast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="problem2/configs/problem2_formal_all7_advanced_fast_v1.yaml",
    )
    args = parser.parse_args()
    result = run_formal_advanced_fast(PROJECT_ROOT, PROJECT_ROOT / args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

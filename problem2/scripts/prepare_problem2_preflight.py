"""Run the short, non-training Question 2 preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))
sys.path.insert(0, str(PROJECT_ROOT / "problem1/src"))

from beer_q2.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="problem2/configs/problem2_preflight.yaml")
    args = parser.parse_args()
    root = PROJECT_ROOT
    audit = run_preflight(root, root / args.config)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

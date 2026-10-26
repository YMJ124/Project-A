#!/usr/bin/env python3
"""Run the Q3 copula shrinkage smoke or discovery screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in ("problem1/src", "problem2/src", "problem3/src"):
    path = str(PROJECT_ROOT / source_root)
    if path not in sys.path:
        sys.path.insert(0, path)

from beer_q3.copula_screening import run_copula_screening


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="problem3/configs/problem3_copula_shrink_smoke_v1.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    result = run_copula_screening(PROJECT_ROOT, config_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

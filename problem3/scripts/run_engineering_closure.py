#!/usr/bin/env python3
"""Run the locked Question 3 engineering end-to-end closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for relative in ["problem1/src", "problem2/src", "problem3/src"]:
    value = str(PROJECT_ROOT / relative)
    if value not in sys.path:
        sys.path.insert(0, value)

from beer_q3.closure import run_engineering_closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="problem3/configs/problem3_engineering_closure_v1.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    result = run_engineering_closure(PROJECT_ROOT, config_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

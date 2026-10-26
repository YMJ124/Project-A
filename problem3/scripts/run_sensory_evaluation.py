#!/usr/bin/env python3
"""Run fold-isolated frozen-Q1 sensory evaluation for a Q3 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source in ("problem1/src", "problem2/src", "problem3/src"):
    value = str(PROJECT_ROOT / source)
    if value not in sys.path:
        sys.path.insert(0, value)

from beer_q3.sensory_evaluation import run_sensory_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = Path(args.config)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    print(json.dumps(run_sensory_evaluation(PROJECT_ROOT, config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


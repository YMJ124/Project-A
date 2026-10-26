#!/usr/bin/env python3
"""Audit and freeze the four deterministic aggregate chemistry formulas."""

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

from beer_q3.derived import audit_derived_formulas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="problem3/configs/derived_feature_formulas_v1.yaml"
    )
    parser.add_argument(
        "--chemicals",
        default="2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv",
    )
    args = parser.parse_args()
    config = Path(args.config)
    chemicals = Path(args.chemicals)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    if not chemicals.is_absolute():
        chemicals = PROJECT_ROOT / chemicals
    print(
        json.dumps(
            audit_derived_formulas(PROJECT_ROOT, config, chemicals),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run leakage-safe OOF evaluation for the five Q3 defect sensory targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
source = str(PROJECT_ROOT / "problem1/src")
if source not in sys.path:
    sys.path.insert(0, source)

from beer_q1.config import load_config, resolve_paths
from beer_q1.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="problem3/configs/defect_sensory_oof_v1.yaml",
    )
    parser.add_argument("--run-id", default="defect_sensory_oof_v1")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = resolve_paths(load_config(config_path), PROJECT_ROOT)
    output = run_experiment(config, run_id=args.run_id)
    print(f"Defect sensory OOF complete: {output}")


if __name__ == "__main__":
    main()

"""Run the cross-repeat module-first rescue audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))

from beer_q2.module_first_rescue import run_module_first_rescue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="problem2/configs/problem2_module_first_rescue_v1.yaml",
    )
    args = parser.parse_args()
    result = run_module_first_rescue(PROJECT_ROOT, PROJECT_ROOT / args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

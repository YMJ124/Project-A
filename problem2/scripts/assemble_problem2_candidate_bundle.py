"""Assemble the formal and module-rescue Q2 candidate catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))

from beer_q2.candidate_bundle import assemble_candidate_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="problem2/configs/problem2_candidate_bundle_v2.yaml"
    )
    args = parser.parse_args()
    result = assemble_candidate_bundle(PROJECT_ROOT, PROJECT_ROOT / args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

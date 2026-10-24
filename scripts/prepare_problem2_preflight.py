"""Run the short, non-training Question 2 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beer_q2.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/problem2_preflight.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = run_preflight(root, root / args.config)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

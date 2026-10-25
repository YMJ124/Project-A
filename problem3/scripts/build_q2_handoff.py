#!/usr/bin/env python3
"""Build the frozen Question 2 -> Question 3 handoff tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from beer_q3.q2_handoff import build_q2_handoff, write_readme


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="problem3/configs/q2_handoff_v1.yaml",
        help="Configuration path relative to the project root",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    summary = build_q2_handoff(PROJECT_ROOT, config_path)
    output = PROJECT_ROOT / "problem3/interfaces/q2_handoff_v1"
    write_readme(output, summary)
    print(summary)


if __name__ == "__main__":
    main()

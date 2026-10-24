"""Command-line interface for data audit and OOF experiments.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import apply_smoke_overrides, load_config, resolve_paths
from .data import load_problem1_data
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["audit", "run"], help="Audit data or run repeated nested CV"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/problem1_screening.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 1 repeat, 3 targets, and only baselines plus Ridge",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    config = resolve_paths(load_config(args.config), project_root)
    if args.smoke:
        config = apply_smoke_overrides(config)

    if args.command == "audit":
        data = load_problem1_data(config)
        print(json.dumps(data.audit, indent=2, ensure_ascii=False))
        return

    output_dir = run_experiment(config, run_id=args.run_id)
    print(f"Run complete: {output_dir}")


if __name__ == "__main__":
    main()


"""Evaluate the locked stable-module attribution mechanism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))

from beer_q2.module_optimize import evaluate_module_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--advanced-run",
        default="problem2/artifacts/q2_module_attribution_repeat2_v1",
    )
    parser.add_argument(
        "--network-run", default="problem2/artifacts/q2_network_repeat2_v1"
    )
    parser.add_argument(
        "--output",
        default="problem2/artifacts/q2_module_attribution_confirmation_v1",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()
    thresholds = {}
    if args.protocol:
        with (PROJECT_ROOT / args.protocol).open("r", encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)
        thresholds = {
            "median_module_minimum": protocol["adoption"][
                "median_module_jaccard_minimum"
            ],
            "median_improvement_minimum": protocol["adoption"][
                "median_improvement_minimum"
            ],
            "improved_rows_minimum": protocol["adoption"][
                "improved_target_folds_minimum"
            ],
            "ab_target_median_minimum": protocol["adoption"][
                "ab_target_median_minimum"
            ],
        }
    summary = evaluate_module_confirmation(
        PROJECT_ROOT / args.advanced_run,
        PROJECT_ROOT / args.network_run,
        PROJECT_ROOT / args.output,
        args.top_k,
        **thresholds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

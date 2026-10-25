"""Check that representative frozen Q1 models can be reconstructed exactly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "problem2/src"))
sys.path.insert(0, str(PROJECT_ROOT / "problem1/src"))

from beer_q2.reconstruct import DEFAULT_TARGETS, reconstruct_and_compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", default="0-6", help="inclusive repeat range, e.g. 0-0 or 0-6")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", default="problem2/artifacts/q1_reconstruction_v1")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="validate all 30 frozen main targets instead of --targets",
    )
    args = parser.parse_args()
    start, end = (int(value) for value in args.repeats.split("-", maxsplit=1))
    root = PROJECT_ROOT
    result = reconstruct_and_compare(
        root,
        targets=None if args.all_targets else args.targets,
        repeats=range(start, end + 1),
        tolerance=args.tolerance,
        output_dir=root / args.output,
    )
    print(result.to_string(index=False))
    if not bool(result["passed_tolerance"].all()):
        raise SystemExit("Q1 reconstruction consistency check FAILED")
    print("Q1 reconstruction consistency check PASSED")


if __name__ == "__main__":
    main()

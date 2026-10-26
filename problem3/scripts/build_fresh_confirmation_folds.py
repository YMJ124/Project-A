#!/usr/bin/env python3
"""Create pre-registered fresh style-balanced folds for Q3 confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
source_root = str(PROJECT_ROOT / "problem1/src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from beer_q1.folds import build_repeated_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemicals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--repeat-start", type=int, required=True)
    parser.add_argument("--n-repeats", type=int, default=4)
    args = parser.parse_args()

    chemicals = pd.read_csv(args.chemicals)
    folds = build_repeated_folds(
        chemicals["beer"],
        chemicals["tasting_category_fine"],
        n_splits=3,
        n_repeats=args.n_repeats,
        base_seed=args.base_seed,
    )
    folds["repeat"] = folds["repeat"] + args.repeat_start
    if folds.groupby(["repeat", "beer"]).size().ne(1).any():
        raise AssertionError("Each beer must occur exactly once per repeat")
    if folds.groupby("repeat")["fold"].nunique().ne(3).any():
        raise AssertionError("Every repeat must contain three folds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output, index=False)
    manifest = {
        "status": "FRESH_CONFIRMATION_FOLDS_LOCKED",
        "base_seed": args.base_seed,
        "repeat_start": args.repeat_start,
        "n_repeats": args.n_repeats,
        "repeats": sorted(folds["repeat"].unique().astype(int).tolist()),
        "n_beers": int(folds["beer"].nunique()),
        "rows": int(len(folds)),
        "source_chemicals": str(args.chemicals),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Deterministic style-balanced repeated folds.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def style_balanced_fold_ids(
    styles: Iterable[str], n_splits: int, seed: int
) -> np.ndarray:
    """Assign each style nearly equally across folds while balancing total size."""
    style_series = pd.Series(list(styles), dtype="object")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    counts = style_series.value_counts()
    too_small = counts[counts < n_splits]
    if not too_small.empty:
        raise ValueError(
            "Every style needs at least n_splits rows; too small: "
            + str(too_small.to_dict())
        )

    rng = np.random.default_rng(seed)
    assignments = np.full(len(style_series), -1, dtype=int)
    total_load = np.zeros(n_splits, dtype=int)

    # Large styles first reduces global fold-size drift.
    ordered_styles = sorted(counts.index, key=lambda value: (-counts[value], str(value)))
    for style in ordered_styles:
        positions = np.flatnonzero(style_series.to_numpy() == style)
        positions = rng.permutation(positions)
        base, extra = divmod(len(positions), n_splits)
        fold_counts = np.full(n_splits, base, dtype=int)
        tie_break = rng.random(n_splits)
        extra_folds = np.lexsort((tie_break, total_load))[:extra]
        fold_counts[extra_folds] += 1
        labels = np.concatenate(
            [np.full(count, fold, dtype=int) for fold, count in enumerate(fold_counts)]
        )
        labels = rng.permutation(labels)
        assignments[positions] = labels
        total_load += fold_counts

    if (assignments < 0).any():
        raise AssertionError("Some rows were not assigned to a fold")
    return assignments


def build_repeated_folds(
    ids: pd.Series,
    styles: pd.Series,
    n_splits: int,
    n_repeats: int,
    base_seed: int,
) -> pd.DataFrame:
    """Return one persistent long-form fold assignment table."""
    frames = []
    for repeat in range(n_repeats):
        seed = int(base_seed + repeat * 1009)
        fold = style_balanced_fold_ids(styles, n_splits=n_splits, seed=seed)
        frames.append(
            pd.DataFrame(
                {
                    "beer": ids.to_numpy(),
                    "style": styles.to_numpy(),
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    expected = len(ids) * n_repeats
    if len(result) != expected:
        raise AssertionError(f"Expected {expected} fold rows, got {len(result)}")
    if result.duplicated(["beer", "repeat"]).any():
        raise AssertionError("A beer was assigned more than once within a repeat")
    return result


"""Fold invariants and reproducibility tests.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

import numpy as np
import pandas as pd

from beer_q1.folds import build_repeated_folds, style_balanced_fold_ids


def test_style_balanced_folds_are_deterministic_and_balanced() -> None:
    styles = pd.Series(["a"] * 7 + ["b"] * 5 + ["c"] * 3)
    first = style_balanced_fold_ids(styles, n_splits=3, seed=42)
    second = style_balanced_fold_ids(styles, n_splits=3, seed=42)
    assert np.array_equal(first, second)
    table = pd.DataFrame({"style": styles, "fold": first})
    for _, counts in table.groupby("style")["fold"].value_counts().groupby(level=0):
        assert counts.max() - counts.min() <= 1
    assert table.groupby(["style", "fold"]).size().min() >= 1


def test_repeated_folds_assign_every_row_once_per_repeat() -> None:
    ids = pd.Series(range(15))
    styles = pd.Series(["a"] * 7 + ["b"] * 5 + ["c"] * 3)
    folds = build_repeated_folds(ids, styles, n_splits=3, n_repeats=4, base_seed=7)
    assert len(folds) == 60
    assert not folds.duplicated(["beer", "repeat"]).any()
    assert folds.groupby("repeat")["beer"].nunique().eq(15).all()


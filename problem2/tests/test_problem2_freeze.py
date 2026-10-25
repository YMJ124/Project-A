import hashlib
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_problem2_v1_freeze_is_complete_and_valid():
    root = Path(__file__).resolve().parents[1] / "freezes/problem2_v1"
    assert (root / "README.md").is_file()
    decisions = pd.read_csv(root / "candidates/target_decisions.csv")
    candidates = pd.read_csv(root / "candidates/candidate_catalog.csv")
    assert len(decisions) == 30
    assert decisions["target"].nunique() == 30
    assert len(candidates) == 46

    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        expected, relative = line.split("  ", maxsplit=1)
        assert _sha256(root / relative) == expected

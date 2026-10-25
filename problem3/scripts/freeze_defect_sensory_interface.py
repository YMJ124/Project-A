#!/usr/bin/env python3
"""Freeze the compact OOF evidence for the five defect sensory targets."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "problem3/artifacts/interfaces/defect_sensory_oof_v1"
DESTINATION = PROJECT_ROOT / "problem3/interfaces/defect_sensory_oof_v1"
FILES = [
    "config.resolved.yaml",
    "data_audit.json",
    "folds.csv",
    "metrics_summary.csv",
    "model_family_frequency.csv",
    "model_selection.csv",
    "run_manifest.json",
    "stage_gate.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in FILES if not (SOURCE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Defect sensory OOF output is incomplete: {missing}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(SOURCE / name, DESTINATION / name)

    selection = pd.read_csv(DESTINATION / "model_selection.csv")
    expected = {"dms", "diacetyl", "X4vg", "t2n", "stale_hops"}
    if set(selection["target"]) != expected:
        raise ValueError("Defect sensory selection does not contain the five locked targets")
    stale = selection.set_index("target").loc["stale_hops"]
    readme = f"""# 缺陷感官 OOF 接口 v1

本接口使用问题一相同的 7 重复×3 折风格平衡外层验证，并在每个外层训练折内比较多输出 ET、逐目标 ET 和强正则 GBR。

- `X4vg`：A 级，可作为补充感官审计。
- `dms`、`diacetyl`、`t2n`、`stale_hops`：C 级，不得作为可靠硬约束。
- `stale_hops`：R2={stale['r2_mean']:.4f}，Spearman={stale['spearman_mean']:.4f}，相对全局均值 RMSE 改善={stale['rmse_improvement_vs_global']:.2%}。

因此问题三中的 DMS/Diacetyl/4VG/trans-2-nonenal 以化学尺度约束为主；`stale_hops` 只能报告探索性预测和不确定性，不得宣称已可靠受控。
"""
    (DESTINATION / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(
        path for path in DESTINATION.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    (DESTINATION / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(f"Frozen defect sensory interface: {DESTINATION}")


if __name__ == "__main__":
    main()

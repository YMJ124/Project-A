#!/usr/bin/env python3
"""Create the compact, Git-trackable Question 2 v1 handoff freeze."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_ROOT = PROJECT_ROOT / "problem2/freezes/problem2_v1"

FILES = {
    "evaluation/evaluation_matrix.csv": "problem2/artifacts/q2_current_evaluation_v8/evaluation_matrix.csv",
    "evaluation/distance_to_optimal.csv": "problem2/artifacts/q2_current_evaluation_v8/distance_to_optimal_after_optimization.csv",
    "evaluation/network_repeat_robustness.csv": "problem2/artifacts/q2_current_evaluation_v8/network_repeat_robustness.csv",
    "evaluation/summary.json": "problem2/artifacts/q2_current_evaluation_v8/summary.json",
    "evaluation/report.md": "problem2/artifacts/q2_current_evaluation_v8/report.md",
    "candidates/candidate_catalog.csv": "problem2/artifacts/q2_formal_candidate_bundle_v2/candidate_catalog.csv",
    "candidates/target_decisions.csv": "problem2/artifacts/q2_formal_candidate_bundle_v2/target_decisions.csv",
    "candidates/source_consistency.csv": "problem2/artifacts/q2_formal_candidate_bundle_v2/source_consistency.csv",
    "candidates/summary.json": "problem2/artifacts/q2_formal_candidate_bundle_v2/summary.json",
    "configs/problem2_formal_all7_fast_v1.yaml": "problem2/configs/problem2_formal_all7_fast_v1.yaml",
    "configs/problem2_formal_all7_advanced_fast_v1.yaml": "problem2/configs/problem2_formal_all7_advanced_fast_v1.yaml",
    "configs/problem2_formal_finalize_v1.yaml": "problem2/configs/problem2_formal_finalize_v1.yaml",
    "configs/problem2_candidate_bundle_v2.yaml": "problem2/configs/problem2_candidate_bundle_v2.yaml",
    "docs/problem2_formal_advanced_analysis_v2.md": "problem2/docs/problem2_formal_advanced_analysis_v2.md",
    "upstream/problem1_source_registry.csv": "problem1/freezes/problem1_v1/source_registry.csv",
    "upstream/problem1_stage_gate.json": "problem1/freezes/problem1_v1/champion_artifacts/all_stage_gate.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [source for source in FILES.values() if not (PROJECT_ROOT / source).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze Question 2; missing inputs: {missing}")

    for destination, source in FILES.items():
        target = FREEZE_ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / source, target)

    summary = json.loads((FREEZE_ROOT / "evaluation/summary.json").read_text(encoding="utf-8"))
    decisions = pd.read_csv(FREEZE_ROOT / "candidates/target_decisions.csv")
    candidates = pd.read_csv(FREEZE_ROOT / "candidates/candidate_catalog.csv")
    if summary.get("formal_question2_status") != "READY":
        raise ValueError("Only a READY Question 2 result may be frozen")
    if len(decisions) != 30 or decisions["target"].nunique() != 30:
        raise ValueError("Frozen target decisions must cover 30 unique targets")
    if len(candidates) != 46:
        raise ValueError(f"Expected 46 frozen candidates, got {len(candidates)}")

    readme = """# 问题二阶段冻结：problem2_v1

本目录是问题二供问题三使用的轻量、可回滚交接包。

- 上游问题一：`problem1_v1`，状态 `HOLD`。
- 问题二状态：`READY`，20/20 评估项通过。
- 正式候选：46 个；7 个目标有完整 Top-3，18 个有1--2个，5 个仍证据不足。
- 所有候选只是样本外预测和稳定性支持的关联证据，不是因果干预结论。

`evaluation/` 和 `candidates/` 保留问题三直接需要的结果；`configs/` 保留重建配置；
`SHA256SUMS` 用于校验冻结内容。大型逐折 SHAP/置换 Parquet 不进入 Git，应按配置重建。
"""
    (FREEZE_ROOT / "README.md").write_text(readme, encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (FREEZE_ROOT / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    frozen_files = sorted(
        path for path in FREEZE_ROOT.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_text = "".join(
        f"{sha256(path)}  {path.relative_to(FREEZE_ROOT)}\n" for path in frozen_files
    )
    (FREEZE_ROOT / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "files": len(frozen_files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Configuration loading and validation.

AI-assisted implementation: OpenAI Codex (GPT-5), 2026-10-24.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML and enforce the small set of required sections."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    for section in ("data", "validation", "runtime", "models"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    return config


def apply_smoke_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a tiny but end-to-end configuration for a short smoke run."""
    result = deepcopy(config)
    result["validation"]["outer_repeats"] = 1
    result["runtime"]["target_limit"] = 3
    result["models"]["enabled"] = ["global_mean", "style_mean", "ridge"]
    result["models"]["grids"]["ridge"] = result["models"]["grids"]["ridge"][:2]
    return result


def resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Resolve data/output paths relative to the project root."""
    result = deepcopy(config)
    for key in ("chemicals_csv", "sensory_csv"):
        path = Path(result["data"][key])
        if not path.is_absolute():
            path = project_root / path
        result["data"][key] = str(path.resolve())
    output = Path(result["runtime"]["output_root"])
    if not output.is_absolute():
        output = project_root / output
    result["runtime"]["output_root"] = str(output.resolve())
    return result


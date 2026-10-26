"""Question 3 conditional generation and style-transfer package."""

from .evaluation import FoldGeometry, map_relative_0_6
from .derived import apply_derived_features, recompute_derived_features
from .external_audit import freeze_external_audit
from .preflight import run_preflight
from .q1_adapter import FullDataQ1Predictor, load_full_data_predictor
from .q2_handoff import build_q2_handoff

__all__ = [
    "FullDataQ1Predictor",
    "FoldGeometry",
    "apply_derived_features",
    "build_q2_handoff",
    "freeze_external_audit",
    "load_full_data_predictor",
    "map_relative_0_6",
    "recompute_derived_features",
    "run_preflight",
]

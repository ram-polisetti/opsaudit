"""opsaudit audits disparity signals in operational decision systems."""

__version__ = "0.1.1"

from .context import load_context
from .data import generate_dispatch, generate_staffing
from .gate import DEFAULT_THRESHOLDS, evaluate_gate, evaluate_gate_status
from .metrics import AuditResult, audit_disparities
from .report import save_report
from .rmf import RMF_MAPPING

__all__ = [
    "audit_disparities",
    "AuditResult",
    "generate_dispatch",
    "generate_staffing",
    "save_report",
    "evaluate_gate",
    "evaluate_gate_status",
    "DEFAULT_THRESHOLDS",
    "RMF_MAPPING",
    "load_context",
    "__version__",
]

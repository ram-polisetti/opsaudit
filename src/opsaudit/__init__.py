"""opsaudit audits disparity signals in operational decision systems."""

__version__ = "0.1.1"

from .context import load_context
from .data import generate_dispatch, generate_staffing
from .gate import DEFAULT_THRESHOLDS, evaluate_gate, evaluate_gate_status
from .gate_ci import (
    append_verdict_log,
    decide_exit,
    load_gate_config,
    render_verdict_markdown,
    run_gate,
)
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
    "load_gate_config",
    "run_gate",
    "decide_exit",
    "render_verdict_markdown",
    "append_verdict_log",
    "__version__",
]

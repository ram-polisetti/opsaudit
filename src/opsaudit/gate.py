"""Deployment-gate defaults for opsaudit results."""

from __future__ import annotations

from .metrics import AuditResult


DEFAULT_THRESHOLDS = {
    "disparate_impact_ratio": {"min": 0.8},
    "demographic_parity_diff": {"max": 0.2},
    "tpr_gap": {"max": 0.15},
    "fpr_gap": {"max": 0.15},
}


def evaluate_gate(
    result: AuditResult, thresholds: dict | None = None
) -> tuple[bool, list[dict]]:
    """Evaluate an audit against deployment thresholds.

    Example:
        >>> evaluate_gate(AuditResult([], 0, 0.0, 1.0, None, None, []))[0]
        True
    """
    active = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    findings: list[dict] = []
    for check, threshold in active.items():
        value = getattr(result, check)
        if value is None:
            status = "skipped"
        elif "min" in threshold:
            status = "pass" if value >= threshold["min"] else "fail"
        else:
            status = "pass" if value <= threshold["max"] else "fail"
        findings.append(
            {"check": check, "value": value, "threshold": threshold, "status": status}
        )
    return all(finding["status"] != "fail" for finding in findings), findings

"""Deployment-gate defaults for opsaudit results."""

from __future__ import annotations

import math
from typing import Any

from .metrics import AuditResult


DEFAULT_THRESHOLDS = {
    "disparate_impact_ratio": {"min": 0.8},
    "demographic_parity_diff": {"max": 0.2},
    "tpr_gap": {"max": 0.15},
    "fpr_gap": {"max": 0.15},
}


def evaluate_gate(
    result: AuditResult, thresholds: dict[str, dict[str, float]] | None = None
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate an audit against deployment thresholds.

    Example:
        >>> evaluate_gate(AuditResult([], 0, 0.0, 1.0, None, None, []))[0]
        True
    """
    active = _merge_thresholds(thresholds)
    findings: list[dict[str, Any]] = []
    for check, threshold in active.items():
        value = getattr(result, check)
        if value is None:
            status = "skipped"
        elif "min" in threshold:
            status = "pass" if value >= threshold["min"] else "fail"
        elif "max" in threshold:
            status = "pass" if value <= threshold["max"] else "fail"
        else:  # Defensive guard; _merge_thresholds validates the shape.
            raise ValueError(f"threshold for {check!r} must contain min or max")
        findings.append(
            {"check": check, "value": value, "threshold": threshold, "status": status}
        )
    return all(finding["status"] != "fail" for finding in findings), findings


def _merge_thresholds(
    overrides: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    """Validate threshold overrides and merge them with the defaults."""
    merged = {check: dict(rule) for check, rule in DEFAULT_THRESHOLDS.items()}
    if overrides is None:
        return merged
    if not isinstance(overrides, dict):
        raise ValueError("thresholds must be a mapping of checks to rules")
    for check, rule in overrides.items():
        if check not in DEFAULT_THRESHOLDS:
            raise ValueError(f"unknown threshold check: {check!r}")
        if not isinstance(rule, dict) or set(rule) != set(DEFAULT_THRESHOLDS[check]):
            expected_operator = next(iter(DEFAULT_THRESHOLDS[check]))
            raise ValueError(
                f"threshold for {check!r} must be {{{expected_operator!r}: number}}"
            )
        value = rule[next(iter(rule))]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"threshold for {check!r} must be a finite number")
        merged[check] = {next(iter(rule)): float(value)}
    return merged

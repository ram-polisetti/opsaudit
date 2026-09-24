"""Deployment-gate thresholds and PASS/FAIL/REVIEW evaluation."""

from __future__ import annotations

import math
from typing import Any, Literal

from .metrics import AuditResult

GateStatus = Literal["pass", "fail", "review"]

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

    This backwards-compatible helper returns ``True`` only for a ``PASS``.
    A ``REVIEW`` result is therefore non-passing, even when no metric fails.
    Use :func:`evaluate_gate_status` when the three-state status is needed.

    Example:
        >>> evaluate_gate(AuditResult([], 0, 0.0, 1.0, None, None, []))[0]
        True
    """
    status, findings = evaluate_gate_status(result, thresholds)
    return status == "pass", findings


def evaluate_gate_status(
    result: AuditResult, thresholds: dict[str, dict[str, float]] | None = None
) -> tuple[GateStatus, list[dict[str, Any]]]:
    """Return ``pass``, ``fail``, or ``review`` with machine-readable findings.

    Metric failures take precedence. A review is emitted when evidence-quality
    checks require a human decision or a passing point estimate has a bootstrap
    confidence interval crossing a configured threshold.

    Example:
        >>> evaluate_gate_status(AuditResult([], 0, 0.0, 1.0, None, None, []))[0]
        'pass'
    """
    active = _merge_thresholds(thresholds)
    findings: list[dict[str, Any]] = []
    metric_failed = False
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
        metric_failed = metric_failed or status == "fail"
        findings.append(
            {"check": check, "value": value, "threshold": threshold, "status": status}
        )

    review_findings = _review_findings(result, active)
    findings.extend(review_findings)
    if metric_failed:
        return "fail", findings
    if review_findings:
        return "review", findings
    return "pass", findings


def _review_findings(
    result: AuditResult, thresholds: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Translate evidence-quality concerns into gate findings."""
    findings: list[dict[str, Any]] = []
    for reason in result.review_reasons:
        findings.append(
            {
                "check": f"evidence:{reason['code']}",
                "value": None,
                "threshold": None,
                "status": "review",
                "reason": reason["message"],
                "details": reason,
            }
        )
    for check, threshold in thresholds.items():
        interval = result.confidence_intervals.get(check)
        value = getattr(result, check)
        if interval is None or value is None:
            continue
        crosses = (
            interval["lower"] < threshold["min"]
            if "min" in threshold
            else interval["upper"] > threshold["max"]
        )
        if crosses and _value_passes(value, threshold):
            findings.append(
                {
                    "check": f"evidence:confidence_interval:{check}",
                    "value": value,
                    "threshold": threshold,
                    "status": "review",
                    "reason": "The 95% bootstrap confidence interval crosses the configured threshold.",
                    "confidence_interval": interval,
                }
            )
    return findings


def _value_passes(value: float, threshold: dict[str, float]) -> bool:
    """Return whether a single numeric value passes its threshold rule."""
    return value >= threshold["min"] if "min" in threshold else value <= threshold["max"]


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

"""Disparity metrics for binary operational decision systems.

The module uses only NumPy and standard-library data structures for its metric
calculations. Undefined rates are represented by ``None``, never ``NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class GroupMetrics:
    """Confusion-matrix-derived measures for one named group.

    Example:
        >>> GroupMetrics("A", 4, 0.5, 1.0, 0.0, 1.0, 1.0).group
        'A'
    """

    group: str
    n: int
    selection_rate: float
    tpr: float | None
    fpr: float | None
    precision: float | None
    accuracy: float


@dataclass
class AuditResult:
    """The complete result of an operational disparity audit.

    Example:
        >>> AuditResult([], 0, 0.0, 1.0, None, None, []).n_total
        0
    """

    groups: list[GroupMetrics]
    n_total: int
    demographic_parity_diff: float
    disparate_impact_ratio: float
    tpr_gap: float | None
    fpr_gap: float | None
    flags: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the audit result.

        Undefined rates are rendered as ``None`` so ``json.dumps`` emits
        ``null`` rather than a non-standard floating-point value.

        Example:
            >>> AuditResult([], 0, 0.0, 1.0, None, None, []).to_dict()["n_total"]
            0
        """
        return {
            "groups": [
                {
                    "group": group.group,
                    "n": int(group.n),
                    "selection_rate": float(group.selection_rate),
                    "tpr": _json_number(group.tpr),
                    "fpr": _json_number(group.fpr),
                    "precision": _json_number(group.precision),
                    "accuracy": float(group.accuracy),
                }
                for group in self.groups
            ],
            "n_total": int(self.n_total),
            "demographic_parity_diff": float(self.demographic_parity_diff),
            "disparate_impact_ratio": float(self.disparate_impact_ratio),
            "tpr_gap": _json_number(self.tpr_gap),
            "fpr_gap": _json_number(self.fpr_gap),
            "flags": _json_safe_flags(self.flags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuditResult":
        """Recreate an audit result saved by :meth:`to_dict`.

        Example:
            >>> AuditResult.from_dict(AuditResult([], 0, 0.0, 1.0, None, None, []).to_dict()).n_total
            0
        """
        return cls(
            groups=[
                GroupMetrics(
                    group=str(group["group"]),
                    n=int(group["n"]),
                    selection_rate=float(group["selection_rate"]),
                    tpr=_optional_float(group.get("tpr")),
                    fpr=_optional_float(group.get("fpr")),
                    precision=_optional_float(group.get("precision")),
                    accuracy=float(group["accuracy"]),
                )
                for group in d["groups"]
            ],
            n_total=int(d["n_total"]),
            demographic_parity_diff=float(d["demographic_parity_diff"]),
            disparate_impact_ratio=float(d["disparate_impact_ratio"]),
            tpr_gap=_optional_float(d.get("tpr_gap")),
            fpr_gap=_optional_float(d.get("fpr_gap")),
            flags=_json_safe_flags(list(d.get("flags", []))),
        )


def audit_disparities(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> AuditResult:
    """Audit binary predictions for group-level disparities.

    ``tpr`` is ``None`` when a group has no positive true labels; ``fpr`` is
    ``None`` when it has no negative true labels; ``precision`` is ``None``
    when it has no positive predictions. ``tpr_gap`` and ``fpr_gap`` are
    ``None`` when fewer than two groups have a defined rate. If no observation
    receives a positive prediction, disparate impact is defined as ``1.0``:
    there is no between-group allocation difference to compare.

    Example:
        >>> result = audit_disparities([1, 0], [1, 0], ["A", "B"])
        >>> result.disparate_impact_ratio
        0.0
    """
    true_values = _as_binary_array(y_true, "y_true")
    pred_values = _as_binary_array(y_pred, "y_pred")
    group_values = np.asarray(groups, dtype=object)

    if true_values.ndim != 1 or pred_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("y_true, y_pred, and groups must be one-dimensional")
    if not (len(true_values) == len(pred_values) == len(group_values)):
        raise ValueError("y_true, y_pred, and groups must have equal length")
    if len(true_values) == 0:
        raise ValueError("y_true, y_pred, and groups must not be empty")

    group_labels = group_values.astype(str)
    group_metrics: list[GroupMetrics] = []
    for label in np.unique(group_labels):
        mask = group_labels == label
        group_true = true_values[mask]
        group_pred = pred_values[mask]
        tp = int(np.sum((group_true == 1) & (group_pred == 1)))
        tn = int(np.sum((group_true == 0) & (group_pred == 0)))
        fp = int(np.sum((group_true == 0) & (group_pred == 1)))
        fn = int(np.sum((group_true == 1) & (group_pred == 0)))
        n = int(mask.sum())
        group_metrics.append(
            GroupMetrics(
                group=str(label),
                n=n,
                selection_rate=float(np.mean(group_pred)),
                tpr=_safe_div(tp, tp + fn),
                fpr=_safe_div(fp, fp + tn),
                precision=_safe_div(tp, tp + fp),
                accuracy=float((tp + tn) / n),
            )
        )

    selection_rates = [metric.selection_rate for metric in group_metrics]
    max_selection_rate = max(selection_rates)
    demographic_parity_diff = float(max_selection_rate - min(selection_rates))
    disparate_impact_ratio = (
        1.0
        if max_selection_rate == 0
        else float(min(selection_rates) / max_selection_rate)
    )
    tpr_gap = _max_gap([metric.tpr for metric in group_metrics])
    fpr_gap = _max_gap([metric.fpr for metric in group_metrics])

    result = AuditResult(
        groups=group_metrics,
        n_total=int(len(true_values)),
        demographic_parity_diff=demographic_parity_diff,
        disparate_impact_ratio=disparate_impact_ratio,
        tpr_gap=tpr_gap,
        fpr_gap=fpr_gap,
        flags=[],
    )
    # Import at call time: gate imports AuditResult, so module-level import would
    # create a circular dependency.
    from .gate import DEFAULT_THRESHOLDS

    active_thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    result.flags = _build_flags(result, active_thresholds)
    return result


def _safe_div(num: int, den: int) -> float | None:
    """Divide integers, returning ``None`` when ``den`` is zero."""
    if den == 0:
        return None
    return float(num / den)


def _as_binary_array(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Convert a one-dimensional binary array-like to an integer array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(int)


def _max_gap(values: Sequence[float | None]) -> float | None:
    """Return the range of defined rates, or ``None`` if fewer than two exist."""
    defined = [value for value in values if value is not None]
    if len(defined) < 2:
        return None
    return float(max(defined) - min(defined))


def _build_flags(
    result: AuditResult, thresholds: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Build JSON-safe default-check flags for an audit result."""
    flags: list[dict[str, Any]] = []
    for check, threshold in thresholds.items():
        value = getattr(result, check)
        if value is None:
            passed = True
        elif "min" in threshold:
            passed = bool(value >= threshold["min"])
        elif "max" in threshold:
            passed = bool(value <= threshold["max"])
        else:
            raise ValueError(f"threshold for {check!r} must contain min or max")
        flags.append(
            {
                "check": check,
                "value": _json_number(value),
                "threshold": dict(threshold),
                "passed": passed,
            }
        )
    return flags


def _optional_float(value: object) -> float | None:
    """Convert an optional numeric value to a built-in float."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("audit results cannot contain NaN or infinite values")
    return number


def _json_number(value: float | None) -> float | None:
    """Return a finite built-in float or ``None`` for JSON serialization."""
    return _optional_float(value)


def _json_safe_flags(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy flags while ensuring all nested numerical values are JSON-safe."""
    copied: list[dict[str, Any]] = []
    for flag in flags:
        copied_flag = dict(flag)
        copied_flag["value"] = _optional_float(copied_flag.get("value"))
        threshold = copied_flag.get("threshold")
        if isinstance(threshold, dict):
            copied_flag["threshold"] = {
                str(key): _optional_float(value)
                for key, value in threshold.items()
            }
        copied.append(copied_flag)
    return copied

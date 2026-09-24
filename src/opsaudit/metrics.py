"""Disparity and evidence-quality metrics for operational decision systems.

The module uses only NumPy and standard-library data structures for its metric
calculations. Undefined rates are represented by ``None``, never ``NaN``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .context import normalize_context


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

    ``warnings`` preserve non-blocking evidence-quality observations, while
    ``review_reasons`` identify conditions that require a human decision before
    the deployment gate can pass.

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
    context: dict[str, str] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    review_reasons: list[dict[str, Any]] = field(default_factory=list)
    confidence_intervals: dict[str, dict[str, float]] = field(default_factory=dict)
    min_group_n: int | None = None
    bootstrap_samples: int = 0

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
            "flags": _json_safe_value(self.flags),
            "context": _json_safe_value(self.context),
            "warnings": _json_safe_value(self.warnings),
            "review_reasons": _json_safe_value(self.review_reasons),
            "confidence_intervals": _json_safe_value(self.confidence_intervals),
            "min_group_n": self.min_group_n,
            "bootstrap_samples": int(self.bootstrap_samples),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditResult:
        """Recreate an audit result saved by :meth:`to_dict`.

        Reports generated before the evidence-quality fields were introduced
        remain compatible and load with empty context and warning values.

        Example:
            >>> AuditResult.from_dict(AuditResult([], 0, 0.0, 1.0, None, None, []).to_dict()).n_total
            0
        """
        min_group_n = d.get("min_group_n")
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
            flags=list(_json_safe_value(d.get("flags", []))),
            context=normalize_context(d.get("context", {})),
            warnings=list(_json_safe_value(d.get("warnings", []))),
            review_reasons=list(_json_safe_value(d.get("review_reasons", []))),
            confidence_intervals=_confidence_intervals_from_dict(
                d.get("confidence_intervals", {})
            ),
            min_group_n=None if min_group_n is None else int(min_group_n),
            bootstrap_samples=int(d.get("bootstrap_samples", 0)),
        )


def audit_disparities(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    thresholds: dict[str, dict[str, float]] | None = None,
    min_group_n: int | None = None,
    bootstrap: int = 0,
    bootstrap_seed: int = 42,
    context: Mapping[str, object] | None = None,
) -> AuditResult:
    """Audit binary predictions for group-level disparities and evidence quality.

    ``tpr`` is ``None`` when a group has no positive true labels; ``fpr`` is
    ``None`` when it has no negative true labels; ``precision`` is ``None``
    when it has no positive predictions. ``tpr_gap`` and ``fpr_gap`` are
    ``None`` when fewer than two groups have a defined rate. If no observation
    receives a positive prediction, disparate impact is defined as ``1.0`` but
    the result records a human-review warning. ``min_group_n`` records a review
    reason for every group below the configured size. ``bootstrap`` is optional,
    capped at 1,000, and produces deterministic 95% confidence intervals.

    Example:
        >>> result = audit_disparities([1, 0, 1, 0], [1, 0, 1, 0], ["A", "A", "B", "B"], min_group_n=3)
        >>> result.review_reasons[0]["code"]
        'minimum_group_size'
    """
    true_values = _as_binary_array(y_true, "y_true")
    pred_values = _as_binary_array(y_pred, "y_pred")
    group_values = np.asarray(groups, dtype=object)
    _validate_audit_options(min_group_n, bootstrap)

    if true_values.ndim != 1 or pred_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("y_true, y_pred, and groups must be one-dimensional")
    if not (len(true_values) == len(pred_values) == len(group_values)):
        raise ValueError("y_true, y_pred, and groups must have equal length")
    if len(true_values) == 0:
        raise ValueError("y_true, y_pred, and groups must not be empty")

    group_labels = group_values.astype(str)
    group_metrics, demographic_parity_diff, disparate_impact_ratio, tpr_gap, fpr_gap = (
        _summarize(true_values, pred_values, group_labels)
    )
    warnings, review_reasons = _evidence_observations(
        group_metrics, pred_values, min_group_n
    )
    confidence_intervals = _bootstrap_confidence_intervals(
        true_values, pred_values, group_labels, bootstrap, bootstrap_seed
    )
    result = AuditResult(
        groups=group_metrics,
        n_total=len(true_values),
        demographic_parity_diff=demographic_parity_diff,
        disparate_impact_ratio=disparate_impact_ratio,
        tpr_gap=tpr_gap,
        fpr_gap=fpr_gap,
        flags=[],
        context=normalize_context(context or {}),
        warnings=warnings,
        review_reasons=review_reasons,
        confidence_intervals=confidence_intervals,
        min_group_n=min_group_n,
        bootstrap_samples=bootstrap,
    )
    # Import at call time: gate imports AuditResult, so module-level import would
    # create a circular dependency.
    from .gate import _merge_thresholds

    result.flags = _build_flags(result, _merge_thresholds(thresholds))
    return result


def _summarize(
    true_values: np.ndarray, pred_values: np.ndarray, group_labels: np.ndarray
) -> tuple[list[GroupMetrics], float, float, float | None, float | None]:
    """Calculate group metrics and aggregate disparity values without flags."""
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
    return (
        group_metrics,
        demographic_parity_diff,
        disparate_impact_ratio,
        _max_gap([metric.tpr for metric in group_metrics]),
        _max_gap([metric.fpr for metric in group_metrics]),
    )


def _evidence_observations(
    group_metrics: list[GroupMetrics], pred_values: np.ndarray, min_group_n: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create non-numeric evidence warnings and review reasons."""
    warnings: list[dict[str, Any]] = []
    review_reasons: list[dict[str, Any]] = []
    if min_group_n is not None:
        undersized = [
            {"group": group.group, "n": group.n}
            for group in group_metrics
            if group.n < min_group_n
        ]
        if undersized:
            observation = {
                "code": "minimum_group_size",
                "message": f"One or more groups have fewer than {min_group_n} records.",
                "min_group_n": min_group_n,
                "groups": undersized,
            }
            warnings.append(observation)
            review_reasons.append(observation)
    if not np.any(pred_values == 1):
        observation = {
            "code": "no_positive_predictions",
            "message": "No positive decisions were observed; a disparate-impact ratio of 1.0 does not establish healthy allocation.",
        }
        warnings.append(observation)
        review_reasons.append(observation)
    return warnings, review_reasons


def _bootstrap_confidence_intervals(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    group_labels: np.ndarray,
    bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, float]]:
    """Create deterministic stratified 95% confidence intervals when requested."""
    if bootstrap == 0:
        return {}
    rng = np.random.default_rng(bootstrap_seed)
    labels = np.unique(group_labels)
    indices_by_group = [np.flatnonzero(group_labels == label) for label in labels]
    values: dict[str, list[float]] = {
        "demographic_parity_diff": [],
        "disparate_impact_ratio": [],
        "tpr_gap": [],
        "fpr_gap": [],
    }
    values.update({f"selection_rate:{label}": [] for label in labels})

    for _ in range(bootstrap):
        sampled_indices = np.concatenate(
            [
                indexes[rng.integers(0, len(indexes), size=len(indexes))]
                for indexes in indices_by_group
            ]
        )
        group_metrics, parity_diff, impact_ratio, tpr_gap, fpr_gap = _summarize(
            true_values[sampled_indices],
            pred_values[sampled_indices],
            group_labels[sampled_indices],
        )
        bootstrap_values: dict[str, float | None] = {
            "demographic_parity_diff": parity_diff,
            "disparate_impact_ratio": impact_ratio,
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
        }
        bootstrap_values.update(
            {
                f"selection_rate:{group.group}": group.selection_rate
                for group in group_metrics
            }
        )
        for check, value in bootstrap_values.items():
            if value is not None:
                values[check].append(float(value))

    return {
        check: {
            "lower": float(np.percentile(sample, 2.5)),
            "upper": float(np.percentile(sample, 97.5)),
        }
        for check, sample in values.items()
        if len(sample) == bootstrap
    }


def _validate_audit_options(min_group_n: int | None, bootstrap: int) -> None:
    """Validate evidence-quality options before calculating an audit."""
    if min_group_n is not None and (
        isinstance(min_group_n, bool)
        or not isinstance(min_group_n, int)
        or min_group_n < 1
    ):
        raise ValueError("min_group_n must be a positive integer or None")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or not 0 <= bootstrap <= 1000:
        raise ValueError("bootstrap must be an integer between 0 and 1000")


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
    """Convert an optional numeric value to a built-in finite float."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("audit results cannot contain NaN or infinite values")
    return number


def _json_number(value: float | None) -> float | None:
    """Return a finite built-in float or ``None`` for JSON serialization."""
    return _optional_float(value)


def _json_safe_value(value: Any) -> Any:
    """Recursively convert NumPy values to finite JSON-safe built-in values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _optional_float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    raise ValueError(f"audit result contains a non-JSON-safe value: {type(value).__name__}")


def _confidence_intervals_from_dict(value: object) -> dict[str, dict[str, float]]:
    """Load finite confidence intervals from a saved JSON audit report."""
    if not isinstance(value, Mapping):
        raise ValueError("confidence_intervals must be a mapping")
    intervals: dict[str, dict[str, float]] = {}
    for check, interval in value.items():
        if not isinstance(interval, Mapping) or set(interval) != {"lower", "upper"}:
            raise ValueError("each confidence interval must contain lower and upper")
        lower = _optional_float(interval["lower"])
        upper = _optional_float(interval["upper"])
        if lower is None or upper is None or lower > upper:
            raise ValueError("confidence interval bounds must be finite and ordered")
        intervals[str(check)] = {"lower": lower, "upper": upper}
    return intervals

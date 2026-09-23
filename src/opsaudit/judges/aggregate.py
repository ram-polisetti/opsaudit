"""Deterministic aggregation of judge labels into disparity statistics.

HARD RULE (plan §5): judges produce *labels*; every statistic computed
from those labels lives here, in plain code — no LLM ever does
arithmetic. ``"unscored"`` items are excluded from all rates (counted
separately so a high unscored rate is visible, not silent).

The gap logic mirrors the v0.1 core: for each label, the per-group rate
is the fraction of scored items carrying that label, and the gap is the
max-min spread across groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import JudgeScore


def _scored(scores: Sequence[JudgeScore]) -> list[JudgeScore]:
    return [s for s in scores if not s.unscored]


def group_label_rates(
    scores: Sequence[JudgeScore],
    groups: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-group label rates: ``{group: {label: rate}}``.

    Args:
        scores: Judge scores, one per item, in input order.
        groups: Group annotation per item, same length as ``scores``.
    """
    if len(scores) != len(groups):
        raise ValueError(
            f"scores ({len(scores)}) and groups ({len(groups)}) "
            "must have the same length"
        )
    by_group: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for score, group in zip(scores, groups):
        if score.unscored:
            continue
        counts = by_group.setdefault(str(group), {})
        counts[score.label] = counts.get(score.label, 0) + 1
        totals[str(group)] = totals.get(str(group), 0) + 1
    rates: dict[str, dict[str, float]] = {}
    for group, counts in by_group.items():
        total = totals[group]
        rates[group] = {
            label: round(count / total, 4) for label, count in counts.items()
        }
    return rates


def label_rate_gaps(rates: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Max-min rate spread across groups, per label.

    Returns ``{label: gap}``; 0.0 when a label appears in only one
    group (or fewer than two groups have any scored items).
    """
    labels: set[str] = set()
    for group_rates in rates.values():
        labels.update(group_rates)
    gaps: dict[str, float] = {}
    for label in sorted(labels):
        values = [gr.get(label, 0.0) for gr in rates.values()]
        gaps[label] = round(max(values) - min(values), 4) if values else 0.0
    return gaps


def unscored_rate(scores: Sequence[JudgeScore]) -> float:
    """Fraction of items the judge could not score (visibility, not silence)."""
    if not scores:
        return 0.0
    return round(
        sum(1 for s in scores if s.unscored) / len(scores), 4
    )


def ordinal_means(
    scores: Sequence[JudgeScore],
    groups: Sequence[str],
    order: Mapping[str, int],
) -> dict[str, float]:
    """Per-group mean of an ordinal label mapping (e.g. tone).

    Labels not present in ``order`` are ignored (counted separately by
    :func:`unscored_rate` when unscored).
    """
    if len(scores) != len(groups):
        raise ValueError(
            f"scores ({len(scores)}) and groups ({len(groups)}) "
            "must have the same length"
        )
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for score, group in zip(scores, groups):
        if score.unscored or score.label not in order:
            continue
        key = str(group)
        sums[key] = sums.get(key, 0.0) + order[score.label]
        counts[key] = counts.get(key, 0) + 1
    return {
        group: round(sums[group] / counts[group], 4)
        for group in sums
    }


def aggregate_judge_findings(
    judge_id: str,
    scores: Sequence[JudgeScore],
    groups: Sequence[str],
    ordinal: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """One deterministic findings dict per (judge, round).

    Includes per-group rates, per-label gaps, the strongest gap, the
    unscored rate, and (when ``ordinal`` is given) per-group ordinal
    means. Everything here is arithmetic on labels — no LLM involved.
    """
    rates = group_label_rates(scores, groups)
    gaps = label_rate_gaps(rates)
    strongest_label = max(gaps, key=gaps.get) if gaps else None
    findings: dict[str, Any] = {
        "judge_id": judge_id,
        "n_items": len(scores),
        "n_scored": len(_scored(scores)),
        "unscored_rate": unscored_rate(scores),
        "rates": rates,
        "gaps": gaps,
        "strongest_label": strongest_label,
        "strength": gaps.get(strongest_label, 0.0)
        if strongest_label
        else 0.0,
    }
    if ordinal is not None:
        means = ordinal_means(scores, groups, ordinal)
        values = list(means.values())
        findings["ordinal_means"] = means
        findings["ordinal_gap"] = (
            round(max(values) - min(values), 4) if len(values) >= 2 else 0.0
        )
    return findings

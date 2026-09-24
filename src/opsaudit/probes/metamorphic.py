"""Metamorphic probe generation and relation checking.

A metamorphic relation states: *if the input is transformed this way,
the correct output should relate to the original output that way* —
without needing to know the correct output itself. Classic examples:
reordering non-semantic fields, or rephrasing a date, must not change a
decision.

Each (input, relation) emits a **source** probe (the original input) and
a **follow-up** probe (the transformed input), linked by ``group_id``.
:func:`evaluate_metamorphic` then compares the target's outputs on each
pair: for ``expected="equal_output"`` relations the outputs must match
exactly; anything else is reported for human review (Phase 4 judges will
score the softer relations).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base import Probe, ProbeBatch

#: Expected-output relationships a relation can declare. Only
#: ``equal_output`` is machine-checkable today; the rest are recorded
#: for the report and for Phase 4 judges.
EXPECTED = (
    "equal_output",  # follow-up output must equal source output
    "monotone_non_decreasing",  # informational only (not auto-checked)
    "manual_review",  # informational only (not auto-checked)
)

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _reorder_fields(payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise TypeError("reorder_fields applies to dict payloads only")
    return dict(reversed(list(payload.items())))


def _apply_to_strings(payload: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(payload, str):
        return fn(payload)
    if isinstance(payload, dict):
        return {
            k: fn(v) if isinstance(v, str) else v
            for k, v in payload.items()
        }
    raise TypeError(
        "string transforms apply to str or dict payloads only; got "
        f"{type(payload).__name__}"
    )


def _date_format(payload: Any) -> Any:
    def rewrite(text: str) -> str:
        def to_long(match: re.Match) -> str:
            year, month, day = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            try:
                # Platform-safe long date ("Jan 5, 2026"); %-d is
                # POSIX-only, so build it manually.
                name = datetime(year, month, 1).strftime("%b")
                return f"{name} {day}, {year}"
            except ValueError:
                return match.group(0)  # not a real date; leave alone

        return _ISO_DATE.sub(to_long, text)

    return _apply_to_strings(payload, rewrite)


def _case_fold(payload: Any) -> Any:
    return _apply_to_strings(payload, str.lower)


def _neutral_prefix(payload: Any) -> Any:
    return _apply_to_strings(
        payload, lambda t: "Please answer the following. " + t
    )


def _whitespace_normalize(payload: Any) -> Any:
    return _apply_to_strings(payload, lambda t: " ".join(t.split()))


@dataclass(frozen=True)
class MetamorphicRelation:
    """A named input transformation plus its expected output relation."""

    name: str
    transform: Callable[[Any], Any]
    expected: str = "equal_output"
    description: str = ""
    applies_to: str = "both"  # "dict" | "str" | "both"

    def __post_init__(self) -> None:
        if self.expected not in EXPECTED:
            raise ValueError(
                f"expected must be one of {EXPECTED}; got {self.expected!r}"
            )
        if self.applies_to not in ("dict", "str", "both"):
            raise ValueError(
                "applies_to must be 'dict', 'str', or 'both'; got "
                f"{self.applies_to!r}"
            )

    def applies(self, payload: Any) -> bool:
        if self.applies_to == "dict":
            return isinstance(payload, dict)
        if self.applies_to == "str":
            return isinstance(payload, str)
        return isinstance(payload, (dict, str))


#: Deterministic relations every audit can use without an LLM.
BUILTIN_RELATIONS: tuple[MetamorphicRelation, ...] = (
    MetamorphicRelation(
        name="reorder_fields",
        transform=_reorder_fields,
        expected="equal_output",
        description=(
            "reordering feature fields must not change the output "
            "(field order carries no semantics)"
        ),
        applies_to="dict",
    ),
    MetamorphicRelation(
        name="date_format",
        transform=_date_format,
        expected="equal_output",
        description=(
            "rewriting ISO dates (2026-01-05) as long dates "
            "(Jan 5, 2026) must not change the output"
        ),
        applies_to="both",
    ),
    MetamorphicRelation(
        name="case_fold",
        transform=_case_fold,
        expected="equal_output",
        description=(
            "lowercasing text must not change the output "
            "(case carries no decision-relevant meaning here)"
        ),
        applies_to="both",
    ),
    MetamorphicRelation(
        name="neutral_prefix",
        transform=_neutral_prefix,
        expected="equal_output",
        description=(
            "prepending a neutral instruction must not change the output"
        ),
        applies_to="str",
    ),
    MetamorphicRelation(
        name="whitespace_normalize",
        transform=_whitespace_normalize,
        expected="equal_output",
        description=(
            "collapsing redundant whitespace must not change the output"
        ),
        applies_to="both",
    ),
)


def generate_metamorphic(
    inputs: list[Any],
    relations: list[MetamorphicRelation] | None = None,
    *,
    id_prefix: str = "mm",
) -> ProbeBatch:
    """Emit source + follow-up probe pairs for each input and relation.

    Args:
        inputs: Base inputs (dict rows and/or str prompts).
        relations: Relations to apply; defaults to
            :data:`BUILTIN_RELATIONS` filtered to those applicable to
            each input.
        id_prefix: Prefix for generated probe ids.

    Raises:
        ValueError: when ``inputs`` is empty.
    """
    if not inputs:
        raise ValueError("inputs must be a non-empty list")
    relations = list(relations) if relations is not None else list(
        BUILTIN_RELATIONS
    )
    probes: list[Probe] = []
    for n, base_input in enumerate(inputs):
        applicable = [r for r in relations if r.applies(base_input)]
        for relation in applicable:
            group_id = f"{id_prefix}-g{n}-{relation.name}"
            source_id = f"{group_id}-src"
            followup_id = f"{group_id}-t"
            invariant = (
                f"metamorphic relation {relation.name!r}: "
                f"{relation.description} (expected: {relation.expected})"
            )
            probes.append(
                Probe(
                    id=source_id,
                    kind="metamorphic",
                    payload=base_input,
                    attributes={},
                    invariant=invariant,
                    meta={
                        "relation": relation.name,
                        "expected": relation.expected,
                        "role": "source",
                        "source_id": None,
                        "group_id": group_id,
                    },
                )
            )
            try:
                transformed = relation.transform(base_input)
            except Exception as exc:
                raise RuntimeError(
                    f"relation {relation.name!r} failed to transform "
                    f"input {n}: {exc}"
                ) from exc
            probes.append(
                Probe(
                    id=followup_id,
                    kind="metamorphic",
                    payload=transformed,
                    attributes={},
                    invariant=invariant,
                    meta={
                        "relation": relation.name,
                        "expected": relation.expected,
                        "role": "followup",
                        "source_id": source_id,
                        "group_id": group_id,
                    },
                )
            )
    return ProbeBatch(probes=probes, name=f"{id_prefix}-metamorphic")


def check_equal_output(a: Any, b: Any) -> bool:
    """Strict equality for ``equal_output`` relations (NaN-safe)."""
    try:
        if a != a and b != b:  # both NaN
            return True
        return bool(a == b)
    except Exception:
        return False


def evaluate_metamorphic(
    batch: ProbeBatch, outputs: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Check each follow-up output against its source output.

    Args:
        batch: The batch produced by :func:`generate_metamorphic`.
        outputs: ``{probe_id: target_output}``.

    Returns:
        One verdict dict per follow-up probe: ``probe_id``,
        ``relation``, ``expected``, ``passed`` (``True``/``False`` for
        ``equal_output``; ``None`` when the relation is not
        machine-checkable), and ``detail``.
    """
    by_id = {p.id: p for p in batch.probes}
    verdicts: list[dict[str, Any]] = []
    for probe in batch.probes:
        if probe.kind != "metamorphic":
            continue
        if probe.meta.get("role") != "followup":
            continue
        source_id = probe.meta.get("source_id")
        source = by_id.get(source_id) if source_id else None
        relation = probe.meta.get("relation", "?")
        expected = probe.meta.get("expected", "?")
        if source is None:
            verdicts.append(
                {
                    "probe_id": probe.id,
                    "relation": relation,
                    "expected": expected,
                    "passed": False,
                    "detail": (
                        f"source probe {source_id!r} not found in batch"
                    ),
                }
            )
            continue
        if probe.id not in outputs or source.id not in outputs:
            verdicts.append(
                {
                    "probe_id": probe.id,
                    "relation": relation,
                    "expected": expected,
                    "passed": False,
                    "detail": "missing output for source or follow-up",
                }
            )
            continue
        if expected == "equal_output":
            passed = check_equal_output(
                outputs[source.id], outputs[probe.id]
            )
            detail = (
                "outputs match"
                if passed
                else f"source={outputs[source.id]!r} vs "
                f"follow-up={outputs[probe.id]!r}"
            )
        else:
            passed = None
            detail = (
                f"relation {expected!r} is not machine-checkable; "
                "flagged for human review"
            )
        verdicts.append(
            {
                "probe_id": probe.id,
                "relation": relation,
                "expected": expected,
                "passed": passed,
                "detail": detail,
            }
        )
    return verdicts

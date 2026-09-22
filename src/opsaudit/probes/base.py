"""Probe: the atomic unit of an agentic audit campaign.

A :class:`Probe` is one input crafted to test a specific fairness or
robustness property of a target. Payloads are generic on purpose: a
``str`` payload is a prompt for text targets, a ``dict`` payload is a
feature row for tabular targets. :class:`ProbeBatch` groups probes and
converts them into whatever shape a target adapter needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

#: Probe kinds produced by the deterministic generators (Phase 2).
#: ``llm_assisted`` marks probes proposed by an LLM and then validated
#: by the deterministic invariants (see ``opsaudit.probes.llm_assisted``).
KINDS = (
    "counterfactual",
    "adversarial",
    "metamorphic",
    "llm_assisted",
)


@dataclass(frozen=True)
class Probe:
    """One audit input with its fairness/robustness contract attached.

    Attributes:
        id: Unique within its batch (e.g. ``"cf-group-0-1-a"``).
        kind: One of :data:`KINDS`.
        payload: The input itself — ``str`` for text targets,
            ``dict`` for tabular (feature-name -> value) targets.
        attributes: Protected-attribute annotations, e.g.
            ``{"group": "PT"}``. Empty when the probe does not vary a
            protected attribute.
        invariant: Human-readable statement of what *should* hold, e.g.
            ``"model output should not depend on group"``.
        meta: Machine-readable extras (``pair_id`` for counterfactual
            mirrors, ``relation`` for metamorphic probes, ...).
    """

    id: str
    kind: str
    payload: Any
    attributes: dict[str, Any] = field(default_factory=dict)
    invariant: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-safe one-line description for logs and reports."""
        return {
            "id": self.id,
            "kind": self.kind,
            "attributes": dict(self.attributes),
            "invariant": self.invariant,
        }


@dataclass
class ProbeBatch:
    """An ordered set of probes plus target-facing converters."""

    probes: list[Probe] = field(default_factory=list)
    name: str = ""

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.probes)

    def __iter__(self) -> Iterator[Probe]:
        return iter(self.probes)

    def __add__(self, other: "ProbeBatch") -> "ProbeBatch":
        if not isinstance(other, ProbeBatch):
            return NotImplemented
        name = self.name or other.name
        if self.name and other.name and self.name != other.name:
            name = f"{self.name}+{other.name}"
        return ProbeBatch(
            probes=[*self.probes, *other.probes], name=name
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def filter(
        self,
        kind: str | None = None,
        attribute: str | None = None,
        value: Any = None,
    ) -> "ProbeBatch":
        """Return the subset matching all given criteria."""
        selected = self.probes
        if kind is not None:
            selected = [p for p in selected if p.kind == kind]
        if attribute is not None:
            selected = [p for p in selected if attribute in p.attributes]
        if value is not None:
            selected = [
                p
                for p in selected
                if value in p.attributes.values()
            ]
        return ProbeBatch(probes=selected, name=self.name)

    def pair_groups(self) -> dict[str, list[Probe]]:
        """Group probes by ``meta["pair_id"]`` (counterfactual mirrors)."""
        groups: dict[str, list[Probe]] = {}
        for probe in self.probes:
            pair_id = probe.meta.get("pair_id")
            if pair_id is not None:
                groups.setdefault(str(pair_id), []).append(probe)
        return groups

    # ------------------------------------------------------------------
    # Target-facing converters
    # ------------------------------------------------------------------
    def text_prompts(self) -> list[str]:
        """Payloads as an ordered prompt list for text targets."""
        prompts = []
        for probe in self.probes:
            if not isinstance(probe.payload, str):
                raise TypeError(
                    f"probe {probe.id!r} has a non-text payload "
                    f"({type(probe.payload).__name__}); text_prompts() "
                    "requires every payload to be a string"
                )
            prompts.append(probe.payload)
        return prompts

    def rows(self) -> list[dict[str, Any]]:
        """Payloads as an ordered row list for tabular targets."""
        rows = []
        for probe in self.probes:
            if not isinstance(probe.payload, dict):
                raise TypeError(
                    f"probe {probe.id!r} has a non-row payload "
                    f"({type(probe.payload).__name__}); rows() requires "
                    "every payload to be a dict"
                )
            rows.append(dict(probe.payload))
        return rows

    def to_dataframe(self, columns: list[str] | None = None):
        """Payloads as a ``pandas.DataFrame`` in ``columns`` order.

        ``pandas`` is a core dependency of opsaudit, so this is always
        available.
        """
        import pandas as pd

        frame = pd.DataFrame(self.rows())
        if columns is not None:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise ValueError(
                    f"columns not present in probe payloads: {missing}"
                )
            frame = frame[columns]
        return frame

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    def to_event(self) -> dict[str, Any]:
        """Render as an evidence-log event (``kind="probe_batch"``)."""
        return {
            "kind": "probe_batch",
            "batch_name": self.name,
            "n": len(self.probes),
            "probes": [p.summary() for p in self.probes],
        }

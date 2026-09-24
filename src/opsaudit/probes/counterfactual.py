"""Symmetric counterfactual probe generation.

A counterfactual audit asks: *would the model have decided differently
if only the protected attribute had been different?* That question is
only valid when each pair of probes differs in **exactly** the protected
attribute(s) under test — everything else held constant (*ceteris
paribus*). This module generates such pairs and can verify the symmetry
after the fact (:func:`check_symmetry`), which is also what the
LLM-assisted generator uses to validate model-proposed candidates.

Two modes:

- **Tabular** (``text_field=None``): ``base`` is a feature dict; each
  probe payload is a copy of ``base`` with one attribute swapped.
- **Text** (``text_field`` set): ``base[text_field]`` is a template with
  ``{attribute}`` placeholders; each probe payload is the rendered
  string. The template and the non-varied defaults are stored in
  ``meta`` so symmetry stays checkable.
"""

from __future__ import annotations

import itertools
from typing import Any

from .base import Probe, ProbeBatch


def _distinct(values: list[Any]) -> list[Any]:
    """Deduplicate preserving order (values may be unhashable)."""
    out: list[Any] = []
    for value in values:
        if not any(_safe_equal(value, seen) for seen in out):
            out.append(value)
    return out


def _safe_equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        return False


def generate_counterfactuals(
    base: dict[str, Any],
    attributes: dict[str, list[Any]],
    *,
    text_field: str | None = None,
    id_prefix: str = "cf",
) -> ProbeBatch:
    """Generate symmetric counterfactual pairs, one attribute at a time.

    Args:
        base: Reference input. Tabular mode: feature dict. Text mode:
            dict containing the template under ``text_field`` plus
            default values for every attribute placeholder.
        attributes: ``{attribute_name: [value_a, value_b, ...]}``. For
            each attribute and each unordered pair of distinct values,
            two mirror probes are emitted. Attributes are varied **one
            at a time** — all others stay at their base values.
        text_field: When set, ``base[text_field]`` is treated as a
            ``str.format`` template with ``{attribute}`` placeholders and
            probe payloads are rendered strings.
        id_prefix: Prefix for generated probe ids.

    Raises:
        ValueError: on empty ``attributes``, an attribute with fewer
            than two distinct values, a missing attribute key
            (tabular), or a missing ``{attribute}`` placeholder (text).
    """
    if not attributes:
        raise ValueError("attributes must be a non-empty dict")
    if not isinstance(base, dict):
        raise ValueError(
            f"base must be a dict; got {type(base).__name__}"
        )

    mode = "text" if text_field is not None else "tabular"
    if text_field is not None:
        template = base.get(text_field)
        if not isinstance(template, str):
            raise ValueError(
                f"base[{text_field!r}] must be a string template; got "
                f"{type(template).__name__}"
            )

    probes: list[Probe] = []
    for attr, raw_values in attributes.items():
        values = _distinct(list(raw_values))
        if len(values) < 2:
            raise ValueError(
                f"attribute {attr!r} needs at least two distinct values "
                f"to form counterfactual pairs; got {values!r}"
            )
        if mode == "tabular" and attr not in base:
            raise ValueError(
                f"attribute {attr!r} is not a key of base "
                f"(keys: {sorted(base)})"
            )
        if mode == "text" and f"{{{attr}}}" not in template:
            raise ValueError(
                f"template has no {{{attr}}} placeholder: {template!r}"
            )
        for i, j in itertools.combinations(range(len(values)), 2):
            v1, v2 = values[i], values[j]
            pair_id = f"{id_prefix}-{attr}-{i}-{j}"
            invariant = (
                f"counterfactual pair: identical except {attr}="
                f"{v1!r} vs {v2!r}; model output should not depend "
                f"on {attr}"
            )
            for tag, value in (("a", v1), ("b", v2)):
                if mode == "tabular":
                    payload = dict(base)
                    payload[attr] = value
                    meta = {
                        "pair_id": pair_id,
                        "attribute": attr,
                        "values": [v1, v2],
                        "mode": mode,
                    }
                else:
                    ctx = {
                        k: v for k, v in base.items() if k != text_field
                    }
                    ctx[attr] = value
                    payload = template.format(**ctx)
                    meta = {
                        "pair_id": pair_id,
                        "attribute": attr,
                        "values": [v1, v2],
                        "mode": mode,
                        "template": template,
                        "defaults": {
                            k: v
                            for k, v in base.items()
                            if k != text_field
                        },
                    }
                probes.append(
                    Probe(
                        id=f"{pair_id}-{tag}",
                        kind="counterfactual",
                        payload=payload,
                        attributes={attr: value},
                        invariant=invariant,
                        meta=meta,
                    )
                )
    return ProbeBatch(probes=probes, name=f"{id_prefix}-counterfactuals")


def check_symmetry(batch: ProbeBatch) -> tuple[bool, list[str]]:
    """Verify every counterfactual pair in ``batch`` is truly symmetric.

    Returns ``(ok, problems)``. A pair is symmetric when it has exactly
    two probes, both ``kind="counterfactual"``, whose attributes differ
    in exactly the annotated attribute, and whose payloads differ in
    nothing else.
    """
    problems: list[str] = []
    # Both deterministic counterfactuals and validated LLM-proposed
    # counterfactuals (kind="llm_assisted", proposed_kind="counterfactual")
    # carry pair_id/attribute metadata and are checked identically.
    cf_probes = [
        p
        for p in batch.probes
        if p.kind == "counterfactual"
        or (
            p.kind == "llm_assisted"
            and p.meta.get("proposed_kind") == "counterfactual"
        )
    ]
    if not cf_probes:
        problems.append("batch contains no counterfactual probes")
        return False, problems

    for probe in cf_probes:
        if "pair_id" not in probe.meta:
            problems.append(f"probe {probe.id!r} has no meta['pair_id']")
    groups = batch.pair_groups()
    for pair_id, members in groups.items():
        members = [
            m
            for m in members
            if m.kind == "counterfactual"
            or (
                m.kind == "llm_assisted"
                and m.meta.get("proposed_kind") == "counterfactual"
            )
        ]
        if len(members) != 2:
            problems.append(
                f"pair {pair_id!r}: expected 2 mirrors, found "
                f"{len(members)}"
            )
            continue
        first, second = members
        attr = first.meta.get("attribute")
        if second.meta.get("attribute") != attr:
            problems.append(
                f"pair {pair_id!r}: mirrors annotate different "
                "attributes"
            )
            continue
        # Attributes must differ in exactly the varied attribute.
        keys1, keys2 = set(first.attributes), set(second.attributes)
        if keys1 != keys2:
            problems.append(
                f"pair {pair_id!r}: mirrors annotate different "
                f"attribute sets ({sorted(keys1)} vs {sorted(keys2)})"
            )
            continue
        differing = [
            k
            for k in keys1
            if not _safe_equal(first.attributes[k], second.attributes[k])
        ]
        if differing != [attr]:
            problems.append(
                f"pair {pair_id!r}: attributes differ in {differing}, "
                f"expected only [{attr!r}]"
            )
            continue
        # Payloads must differ in nothing else.
        mode = first.meta.get("mode", "tabular")
        if mode == "tabular":
            if not (
                isinstance(first.payload, dict)
                and isinstance(second.payload, dict)
            ):
                problems.append(
                    f"pair {pair_id!r}: tabular mode requires dict "
                    "payloads"
                )
                continue
            pkeys1, pkeys2 = set(first.payload), set(second.payload)
            if pkeys1 != pkeys2:
                problems.append(
                    f"pair {pair_id!r}: payload keys differ "
                    f"({sorted(pkeys1)} vs {sorted(pkeys2)})"
                )
                continue
            drifted = [
                k
                for k in pkeys1
                if k != attr
                and not _safe_equal(first.payload[k], second.payload[k])
            ]
            if drifted:
                problems.append(
                    f"pair {pair_id!r}: non-attribute fields drifted "
                    f"between mirrors: {drifted}"
                )
            for probe in (first, second):
                if not _safe_equal(
                    probe.payload.get(attr), probe.attributes[attr]
                ):
                    problems.append(
                        f"probe {probe.id!r}: payload[{attr!r}] does not "
                        "match its attribute annotation"
                    )
        else:  # text mode: re-render from the stored template
            template = first.meta.get("template")
            defaults = first.meta.get("defaults", {})
            if not isinstance(template, str) or not isinstance(
                defaults, dict
            ):
                problems.append(
                    f"pair {pair_id!r}: text mode requires "
                    "meta['template'] and meta['defaults']"
                )
                continue
            for probe in (first, second):
                ctx = dict(defaults)
                ctx[attr] = probe.attributes[attr]
                try:
                    expected = template.format(**ctx)
                except (KeyError, IndexError) as exc:
                    problems.append(
                        f"probe {probe.id!r}: template render failed: "
                        f"{exc}"
                    )
                    continue
                if expected != probe.payload:
                    problems.append(
                        f"probe {probe.id!r}: payload is not the "
                        "template rendered with its own attributes "
                        "(something else drifted)"
                    )
        # Mirror values must match the declared pair values.
        declared = first.meta.get("values")
        if isinstance(declared, list) and len(declared) == 2:
            got = [first.attributes[attr], second.attributes[attr]]
            if not (
                _safe_equal(got[0], declared[0])
                and _safe_equal(got[1], declared[1])
            ) and not (
                _safe_equal(got[0], declared[1])
                and _safe_equal(got[1], declared[0])
            ):
                problems.append(
                    f"pair {pair_id!r}: mirror values {got!r} do not "
                    f"match declared values {declared!r}"
                )
    # Every counterfactual probe must belong to a pair.
    paired_ids = {p.id for ps in groups.values() for p in ps}
    for probe in cf_probes:
        if probe.id not in paired_ids:
            problems.append(
                f"probe {probe.id!r} is not assigned to any pair"
            )
    return (len(problems) == 0), problems

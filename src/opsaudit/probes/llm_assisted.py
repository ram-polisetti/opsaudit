"""LLM-assisted probe generation with deterministic validation.

An LLM is good at *inventing* audit probes a template author would not
think of (domain-specific phrasings, realistic edge cases). It is bad at
being *trustworthy*: it drifts, hallucinates schemas, and breaks the
symmetry counterfactual auditing depends on. So this module treats the
LLM as an untrusted proposer:

1. Ask a text :class:`~opsaudit.targets.Target` for probe candidates
   (short prompt, JSON-only response).
2. Validate every candidate against deterministic invariants:
   - schema check for all kinds (known ``kind``, present payload,
     dict ``attributes``, non-empty ``invariant``);
   - counterfactual candidates must come in **symmetric pairs** —
     unpaired or asymmetric candidates are discarded, and surviving
     pairs get ``pair_id``\\ s so :func:`check_symmetry` passes;
   - text-payload counterfactuals are discarded (without the template
     metadata only :func:`generate_counterfactuals` can guarantee
     symmetry — use it instead).
3. Invalid candidates are **discarded with reasons recorded**, never
   silently kept and never silently "fixed".

With no target configured the module raises a clear error instead of
returning a deceptively empty battery.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import Probe, ProbeBatch
from .counterfactual import _distinct, _safe_equal

#: Kinds the LLM is allowed to propose. ``llm_assisted`` itself is not
#: proposable — kept probes are re-tagged by this module.
PROPOSABLE_KINDS = ("counterfactual", "adversarial", "metamorphic")

_PROMPT_TEMPLATE = """You are helping audit an AI system for bias and robustness.
{brief}

Propose exactly {n} audit probe candidates as a JSON array and NOTHING else.
Each candidate must be an object with:
- "kind": one of "counterfactual", "adversarial", "metamorphic"
- "payload": the probe input. For counterfactual and adversarial probes on tabular data, a JSON object of feature names to values. For text probes, a string.
- "attributes": a JSON object of protected-attribute annotations, e.g. {{"group": "PT"}}. Use {{}} when none apply.
- "invariant": one sentence stating what should hold, e.g. "model output should not depend on group".
- "relation": (metamorphic only) the relation name being tested, e.g. "paraphrase".

Rules:
- Counterfactual candidates MUST be proposed in mirror pairs: for every candidate there must be another identical except for one protected attribute value.
- Keep payloads small and realistic. Respond with ONLY the JSON array.
"""


def _extract_json_array(text: str) -> Any:
    """Pull the first JSON array out of ``text`` (tolerates fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in model response")
    return json.loads(cleaned[start : end + 1])


def _schema_problems(candidate: Any, index: int) -> list[str]:
    problems: list[str] = []
    tag = f"candidate[{index}]"
    if not isinstance(candidate, dict):
        return [f"{tag}: not a JSON object"]
    kind = candidate.get("kind")
    if kind not in PROPOSABLE_KINDS:
        problems.append(
            f"{tag}: kind must be one of {PROPOSABLE_KINDS}; got {kind!r}"
        )
    if "payload" not in candidate:
        problems.append(f"{tag}: missing 'payload'")
    attributes = candidate.get("attributes", {})
    if not isinstance(attributes, dict):
        problems.append(f"{tag}: 'attributes' must be an object")
    invariant = candidate.get("invariant", "")
    if not isinstance(invariant, str) or not invariant.strip():
        problems.append(f"{tag}: 'invariant' must be a non-empty string")
    if kind == "metamorphic" and not candidate.get("relation"):
        problems.append(f"{tag}: metamorphic probes need 'relation'")
    return problems


def _pair_key(payload: Any, attributes: dict) -> Any:
    """Identity of a counterfactual probe ignoring attribute values."""
    if isinstance(payload, dict):
        rest = {
            k: v for k, v in payload.items() if k not in attributes
        }
        return ("dict", json.dumps(rest, sort_keys=True, default=str))
    return None  # text payloads cannot be paired safely


def _pair_counterfactuals(
    candidates: list[dict],
) -> tuple[list[tuple[dict, dict, str]], list[tuple[dict, str]]]:
    """Group counterfactual candidates into symmetric mirror pairs.

    Returns ``(pairs, unpaired)`` where each pair is
    ``(cand_a, cand_b, differing_attribute)`` and each unpaired entry is
    ``(candidate, reason)``.
    """
    indexed = list(enumerate(candidates))
    used: set[int] = set()
    pairs: list[tuple[dict, dict, str]] = []
    unpaired: list[tuple[dict, str]] = []

    def attrs(i: int) -> dict:
        return candidates[i].get("attributes", {})

    def payload(i: int) -> Any:
        return candidates[i].get("payload")

    for i, _ in indexed:
        if i in used:
            continue
        if not isinstance(payload(i), dict):
            unpaired.append(
                (
                    candidates[i],
                    "text-payload counterfactual: symmetry needs "
                    "template metadata — use generate_counterfactuals "
                    "instead",
                )
            )
            used.add(i)
            continue
        key_i = _pair_key(payload(i), attrs(i))
        if key_i is None:
            unpaired.append((candidates[i], "unpairable payload"))
            used.add(i)
            continue
        mirror = None
        for j, _ in indexed:
            if j in used or j == i:
                continue
            if not isinstance(payload(j), dict):
                continue
            if _pair_key(payload(j), attrs(j)) != key_i:
                continue
            ai, aj = attrs(i), attrs(j)
            if set(ai) != set(aj):
                continue
            differing = [
                k for k in ai if not _safe_equal(ai[k], aj[k])
            ]
            if len(differing) != 1:
                continue
            # Payloads must agree on everything but the attribute.
            pi, pj = payload(i), payload(j)
            attr = differing[0]
            if set(pi) != set(pj):
                continue
            if any(
                not _safe_equal(pi[k], pj[k])
                for k in pi
                if k != attr
            ):
                continue
            if not _safe_equal(pi.get(attr), ai[attr]):
                continue
            mirror = (j, attr)
            break
        if mirror is None:
            unpaired.append(
                (
                    candidates[i],
                    "no symmetric mirror found differing in exactly "
                    "one protected attribute",
                )
            )
        else:
            j, attr = mirror
            pairs.append((candidates[i], candidates[j], attr))
            used.add(i)
            used.add(j)
    return pairs, unpaired


def generate_llm_assisted(
    target: Any,
    brief: str,
    *,
    n: int = 10,
    seed: int | None = None,
) -> tuple[ProbeBatch, dict[str, Any]]:
    """Ask an LLM target for probe candidates; keep only valid ones.

    Args:
        target: A text :class:`~opsaudit.targets.Target`
            (``supports_generate`` must be true). ``None`` or a
            non-text target raises a clear error.
        brief: Short description of the system under audit and the risk
            areas to probe (kept short: long prompts hit the Ollama
            Cloud empty-response quirk).
        n: How many candidates to request.
        seed: Recorded in the report for reproducibility (prompt text is
            deterministic given ``brief`` and ``n``).

    Returns:
        ``(batch, report)`` where kept probes are tagged
        ``kind="llm_assisted"`` (original kind in
        ``meta["proposed_kind"]``) and ``report`` records
        ``requested``/``kept``/``discarded`` plus per-reason counts.

    Raises:
        RuntimeError: when ``target`` is ``None`` or does not support
            text generation.
    """
    if target is None:
        raise RuntimeError(
            "generate_llm_assisted requires a text Target, got None. "
            "Configure one (e.g. OllamaTarget) or use the deterministic "
            "generators instead — an empty battery would be a silent lie."
        )
    if not getattr(target, "supports_generate", False):
        raise RuntimeError(
            f"generate_llm_assisted requires a text Target; "
            f"{type(target).__name__} does not support generate()."
        )
    if not brief or not brief.strip():
        raise ValueError("brief must be a non-empty string")
    if n <= 0:
        raise ValueError("n must be positive")

    prompt = _PROMPT_TEMPLATE.format(brief=brief.strip(), n=n)
    responses = target.generate([prompt])
    raw = responses[0] if responses else ""
    # The Ollama Cloud API occasionally returns empty responses on
    # longer prompts: one retry, then give up with the reason recorded.
    if not raw.strip():
        responses = target.generate([prompt])
        raw = responses[0] if responses else ""

    report: dict[str, Any] = {
        "requested": n,
        "kept": 0,
        "discarded": 0,
        "reasons": {},
        "parse_ok": True,
        "seed": seed,
    }

    def discard(reason: str) -> None:
        report["discarded"] += 1
        report["reasons"][reason] = report["reasons"].get(reason, 0) + 1

    try:
        candidates = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError):
        report["parse_ok"] = False
        discard("response was not a parseable JSON array")
        return ProbeBatch(probes=[], name="llm-assisted"), report
    if not isinstance(candidates, list):
        report["parse_ok"] = False
        discard("response JSON was not an array")
        return ProbeBatch(probes=[], name="llm-assisted"), report

    valid: list[dict] = []
    for index, candidate in enumerate(candidates):
        problems = _schema_problems(candidate, index)
        if problems:
            for problem in problems:
                discard(f"schema: {problem.split(': ', 1)[-1]}")
            continue
        valid.append(candidate)

    kept: list[Probe] = []
    seq = 0

    def mint(
        candidate: dict,
        payload: Any,
        attributes: dict,
        probe_id: str,
        meta_extra: dict | None = None,
    ) -> Probe:
        meta = {
            "proposed_kind": candidate["kind"],
            "llm_generated": True,
            **(meta_extra or {}),
        }
        if candidate["kind"] == "metamorphic":
            meta["relation"] = candidate.get("relation")
        return Probe(
            id=probe_id,
            kind="llm_assisted",
            payload=payload,
            attributes=dict(attributes),
            invariant=str(candidate["invariant"]).strip(),
            meta=meta,
        )

    cf_candidates = [c for c in valid if c["kind"] == "counterfactual"]
    other = [c for c in valid if c["kind"] != "counterfactual"]

    pairs, unpaired = _pair_counterfactuals(cf_candidates)
    for pair_n, (cand_a, cand_b, attr) in enumerate(pairs):
        pair_id = f"llm-pair-{pair_n}"
        for tag, cand in (("a", cand_a), ("b", cand_b)):
            kept.append(
                mint(
                    cand,
                    payload=dict(cand["payload"]),
                    attributes=dict(cand["attributes"]),
                    probe_id=f"{pair_id}-{tag}",
                    meta_extra={
                        "pair_id": pair_id,
                        "attribute": attr,
                        "values": _distinct(
                            [
                                cand_a["attributes"][attr],
                                cand_b["attributes"][attr],
                            ]
                        ),
                        "mode": "tabular",
                    },
                )
            )
    for cand, reason in unpaired:
        discard(f"counterfactual: {reason}")

    for cand in other:
        kept.append(
            mint(
                cand,
                payload=cand["payload"],
                attributes=dict(cand.get("attributes", {})),
                probe_id=f"llm-{seq}",
            )
        )
        seq += 1

    report["kept"] = len(kept)
    return ProbeBatch(probes=kept, name="llm-assisted"), report

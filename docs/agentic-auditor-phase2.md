# Agentic auditor — Phase 2: probe generators

Phase 2 adds the deterministic probe-generation layer. Generators propose
audit inputs; the agentic planner (Phase 3) will decide which to run, and
the judges (Phase 4) will score the unstructured outputs. Nothing here
changes v0.1 behavior or the Phase 1 interfaces.

## The core guarantee: symmetry

A counterfactual pair that drifts in anything *but* the protected
attribute is a broken audit instrument — it manufactures the disparity
it claims to find. Two mechanisms enforce this:

- `generate_counterfactuals` builds pairs by construction (deep copies of
  the base input, one attribute swapped) and never mutates the caller's
  base dict.
- `check_symmetry(batch)` verifies pairs after the fact and reports
  exactly what drifted. It is also the validator for LLM-proposed
  candidates (see below).

## New modules

- `src/opsaudit/probes/base.py` — `Probe` (frozen dataclass: id, kind,
  payload, protected-attribute annotations, invariant, meta) and
  `ProbeBatch` (filtering, `text_prompts()`/`rows()`/`to_dataframe()`
  converters, `pair_groups()`, `to_event()` for the evidence log).
  Payloads are generic: `str` for text targets, `dict` rows for tabular.
- `src/opsaudit/probes/counterfactual.py` — symmetric pairs, one
  attribute varied at a time (*ceteris paribus*). Tabular mode swaps dict
  keys; text mode renders a `{attr}` template (template + defaults kept
  in `meta` so symmetry stays checkable). `check_symmetry()` returns
  `(ok, problems)`.
- `src/opsaudit/probes/adversarial.py` — boundary probes: extreme
  numerics, unseen categories, null rows (tabular); empty/whitespace/
  very-long/unicode/ambiguous/prompt-injection phrasings (text). The
  invariant is graceful degradation, not correctness.
- `src/opsaudit/probes/metamorphic.py` — source + follow-up pairs per
  (input, relation); `BUILTIN_RELATIONS` (reorder_fields, date_format,
  case_fold, neutral_prefix, whitespace_normalize); `evaluate_metamorphic`
  machine-checks `equal_output` relations and flags the rest for human
  review.
- `src/opsaudit/probes/llm_assisted.py` — the LLM as **untrusted
  proposer**: candidates are requested as JSON, then validated —
  schema checks for all kinds, mirror-pairing for counterfactuals
  (unpaired/asymmetric candidates discarded with reasons, never
  silently fixed), text counterfactuals rejected outright. Returns
  `(batch, report)` with kept/discarded counts. `None`/non-text target
  raises a clear `RuntimeError` instead of returning a deceptively empty
  battery. Empty model responses are retried once (Ollama Cloud quirk).

## Dependency policy (unchanged)

No new dependencies. Everything is stdlib + `pandas`/`numpy` (already
core). Unit tests never touch the network — the LLM-assisted tests use a
canned `Target` subclass.

## Conventions for Phase 3+ authors

- Generators never call targets; they only produce `ProbeBatch`es. The
  planner (Phase 3) executes via the Phase 1 adapters.
- `Probe.kind` is the *semantic* kind; `meta["proposed_kind"]` records
  what the LLM originally proposed for `llm_assisted` probes.
- `check_symmetry` covers both `counterfactual` and validated
  `llm_assisted` counterfactuals (selected via `proposed_kind`).
- Evidence events: reuse `batch.to_event()` (`kind="probe_batch"`).

## Demo

`examples/agentic-audit-demo/phase2_demo.py` — counterfactual battery
against a `TabularTarget` with a planted disparity (caught by the real
`audit_disparities`), adversarial no-crash check, metamorphic
reorder-stability check; writes `phase2_evidence.jsonl`.

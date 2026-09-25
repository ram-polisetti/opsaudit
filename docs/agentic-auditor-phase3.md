# Agentic auditor — Phase 3: the planner loop

Phase 3 adds the adaptive layer: an LLM planner decides *what to probe
next*, deterministic code builds the probes, runs them, measures, and
stops. Nothing here changes v0.1 behavior or the Phase 1/2 interfaces.

## The division of labor (the design bet)

An LLM is good at *strategy* ("the temp gap looks real — drill into
FT-vs-temp") and bad at *trustworthy measurement*. So the planner's job
is deliberately tiny:

- **The LLM decides:** which generator to use, which attributes to vary,
  how many probes — emitted as a short JSON-only `ProbeSpec`.
- **Deterministic code does everything else:** Phase 2 generators build
  every probe (symmetry guarantees live there, not in the prompt),
  Phase 1 adapters execute them, and all statistics are computed in
  code. The planner never sees raw model outputs — only compact numeric
  summaries (per-attribute gaps, error rates). Judging unstructured
  text outputs is Phase 4's job; here text outputs are recorded
  unscored.

This is what keeps an *agentic* audit defensible: the clever part
proposes, the boring part disposes, and the evidence log records both.

## New modules

- `src/opsaudit/agents/planner.py` — `AuditPlanner` + `ProbeSpec`.
  Prompts are kept short on purpose (the Ollama Cloud API returns empty
  responses on long prompts): one compact JSON-only request per round,
  one retry on empty, then a `stop` spec. Malformed/empty planner output
  never crashes a campaign — it ends it with a logged reason
  (`planner_invalid_spec` / `planner_empty_response`). Specs are
  validated before use (known generator, counterfactuals need
  `{attr: [2+ values]}`, metamorphic needs `inputs`; counterfactual
  attribute names must be keys of the brief's `base_input`, and every
  proposed value must come from the brief's `protected_attributes` (or
  the optional `attribute_values` allowlist for non-protected
  attributes) — invented values stop the campaign before any probe
  executes).
- `src/opsaudit/agents/budgets.py` — `Budget` (max probes/rounds/cost,
  per-probe and per-planner-call cost model, flatness rule) and
  `BudgetTracker`. Every stopping rule is a pure function of counters:
  `max_probes`, `max_rounds`, `max_cost` exhaustion; `is_flat()` fires
  when the best finding strength hasn't improved for `flat_rounds`
  consecutive rounds. Same history → same stop decision, always.
- `src/opsaudit/agents/cache.py` — `ResponseCache`: SHA-256 keys over a
  canonical JSON encoding, in-memory with optional JSON file backing.
  Identical prompts/rows are never re-queried — for the planner *and*
  the audited target. Predict-row keys preserve dict key order, because
  metamorphic reorder probes differ *only* in key order.
- `src/opsaudit/agents/campaign.py` — `AuditCampaign.run()`:
  plan → probe → observe → replan. Probe batches are capped to the
  remaining probe budget without splitting counterfactual pairs or
  metamorphic groups (`_cap_batch`). Target exceptions are recorded as
  error observations, never allowed to kill the campaign. Round
  summaries are deterministic: counterfactual → per-attribute outcome
  rates and gaps; adversarial → error rate; metamorphic → violation rate
  on `equal_output` relations. Returns a `CampaignReport` (rounds,
  probes used, stop reason, per-round findings, evidence path).

## Reproducibility

Same brief + seed + deterministic targets → same prompts → same cached
responses → same campaign. The seed is recorded in the evidence log
(`campaign_start`) and seeds stdlib `random`; the planner gets its own
cache namespace so planner prompts and target inputs can never collide.

## Known limitation (documented, not hidden)

`reorder_fields` metamorphic probes cannot observe order-sensitivity
through DataFrame-backed tabular adapters: pandas normalizes dict key
order when building the frame, so the difference never reaches the
estimator. The relation remains meaningful for text targets and
raw-dict targets. (Found while testing Phase 3; the violation counter
itself is unit-tested with synthetic outputs.)

## Dependency policy (unchanged)

No new dependencies. Unit tests never touch the network — the planner
brain is a scripted `Target` subclass.

## Demo

`examples/agentic-audit-demo/phase3_demo.py` — a scripted planner brain
(no credentials) runs a 3-round campaign against a `TabularTarget` with
a planted disparity: broad group sweep → drill-down into FT-vs-temp →
adversarial robustness check → planner-initiated stop. Prints the
adaptive drill-down; writes `phase3_evidence.jsonl`.

## Conventions for Phase 4+ authors

- The planner only ever emits `ProbeSpec`s; it never constructs probes
  or reads raw outputs. Keep it that way.
- New finding-strength metrics go in `campaign._summarize_round`;
  judges (Phase 4) produce *labels* that summaries may consume, but
  summaries never call an LLM.
- Evidence event kinds used: `campaign_start`, `plan`, `probe_batch`,
  `observations`, `round_summary`, `campaign_end`.

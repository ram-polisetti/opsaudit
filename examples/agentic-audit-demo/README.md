# Phase 1 demo: target adapters + evidence log

Runs one fixed counterfactual probe battery against three target types:

| target | what it is |
|---|---|
| `dispatch-threshold-model` | `TabularTarget` wrapping a toy classifier on synthetic dispatch data (`opsaudit.data.generate_dispatch`). A zone-C disparity is planted; the audit uses opsaudit's real deterministic metrics. |
| `mock-llm` | deterministic canned-response text target with a planted tone gap. |
| `ollama-live` | a **real** `OllamaTarget` HTTP call — only when `OLLAMA_MODEL` is set (plus `OLLAMA_API_KEY` for Ollama Cloud, or a local server via `OLLAMA_BASE_URL`). Skipped cleanly otherwise. |

Every probe, response, and computed metric lands in `phase1_evidence.jsonl`
(an `EvidenceLog` transcript).

## Run it

```bash
# from the repo root (package installed, e.g. pip install -e ".[dev,llm]")
python examples/agentic-audit-demo/phase1_demo.py --n 2000 --seed 7

# with a live Ollama target (Cloud example)
OLLAMA_MODEL=kimi-k2.7-code \
OLLAMA_BASE_URL=https://ollama.com \
OLLAMA_API_KEY="$OLLAMA_API_KEY" \
python examples/agentic-audit-demo/phase1_demo.py
```

No network is required: without `OLLAMA_MODEL` the live target skips with
a one-line message and the demo still exercises the tabular and mock
targets plus the evidence log.

## What "gap" means

- tabular: `1 - disparate_impact_ratio` from `opsaudit.audit_disparities`.
- text targets: max pairwise approval-rate gap across the counterfactual
  pairs (a toy deterministic labeler — real judges arrive in Phase 4).

Disparities in the mock targets are planted for the demo. The tabular
audit is real.

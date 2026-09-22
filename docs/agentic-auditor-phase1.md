# Agentic auditor — Phase 1: target adapters + evidence log

Phase 1 adds the model-agnostic probing layer. Later phases build on it;
nothing here changes v0.1 behavior.

## New modules

- `src/opsaudit/targets/base.py` — `Target` ABC. `generate(prompts) ->
  list[str]` for text targets, `predict(X) -> list` for scoring targets,
  `describe() -> dict` for JSON-safe metadata (must never need network).
  `supports_generate` / `supports_predict` introspection flags.
- `src/opsaudit/targets/hf.py` — `HuggingFaceTarget` (lazy `transformers`).
- `src/opsaudit/targets/openai_compat.py` — `OpenAICompatTarget` for any
  `/chat/completions` endpoint (lazy `httpx`).
- `src/opsaudit/targets/ollama.py` — `OllamaTarget` for Ollama Cloud/local
  `/api/chat`. Config from `OLLAMA_MODEL` / `OLLAMA_BASE_URL` /
  `OLLAMA_API_KEY` env vars (or explicit args); the key is deliberately
  excluded from `describe()` so it can never leak into the evidence log.
- `src/opsaudit/targets/tabular.py` — `TabularTarget` wrapping any
  sklearn-style estimator (duck-typed; no sklearn dependency).
- `src/opsaudit/targets/rag.py` — `RagTarget` wrapping a
  `rag_fn(query) -> str` callable.
- `src/opsaudit/evidence.py` — `EvidenceLog`: append-only JSONL transcript.
  `record(event)` stamps `seq`, `ts` (UTC), and the target descriptor;
  `read()` returns records in order. No edit/delete API by design.

## Dependency policy

Core stays dependency-light (`pandas`, `numpy`, `click`, `jinja2`,
`pyyaml`). HTTP/ML backends are **lazy optional**: every adapter imports
cleanly without its dependency and raises an actionable `ImportError`
only when the missing backend is actually used. `httpx` is available via
the `llm` extra: `pip install -e ".[llm]"`.

## Conventions for Phase 2+ authors

- Adapters never do I/O at import or in `describe()`.
- `generate([])` / `predict` on empty input returns `[]` without network.
- Secrets never appear in `describe()` output or logs.
- Event kinds are documented in `evidence.EVENT_KINDS`; `record()` stays
  schema-generic.
- Unit tests must not hit real networks — fake the HTTP layer (see
  `tests/test_targets.py::_install_fake_httpx`).

## Demo

`examples/agentic-audit-demo/phase1_demo.py` (+ README) — one fixed
counterfactual battery against a tabular target, a mock LLM target, and
(optionally) a live Ollama target; writes `phase1_evidence.jsonl`.

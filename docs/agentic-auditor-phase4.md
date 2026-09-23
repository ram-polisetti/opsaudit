# Agentic auditor — Phase 4: judges + calibration

Phase 4 adds LLM judges for open-ended outputs and the calibration
harness that measures them. Nothing here changes v0.1 behavior or the
Phase 1–3 interfaces (the only Phase 3 touch is two new *optional*
`AuditCampaign` parameters).

## The hard rule (plan §5)

Judges score **unstructured text only** — they never touch numbers. A
judge turns text into a *label*; every statistic computed from those
labels (group rates, gaps, ordinal means) lives in deterministic code
(`opsaudit.judges.aggregate`). No LLM ever does arithmetic.

## New modules

- `src/opsaudit/judges/prompts.py` — versioned prompt templates. Every
  template carries an explicit version; the `judge_id` recorded in
  every score is `"{name}-v{version}"` (e.g. `stereotype-judge-v1`).
  Rubrics change by version bump, never by editing a shipped template.
- `src/opsaudit/judges/base.py` — `JudgeScore` (frozen dataclass:
  label, confidence, rationale, judge_id, model_id, unscored flag) and
  the `Judge` ABC. `LLMJudge` holds the shared execution contract: one
  short JSON-only prompt per text, **one retry on empty**, then the
  item is marked `unscored` — a label is never invented, and malformed
  JSON becomes `unscored`, never a guess.
- `src/opsaudit/judges/stereotype.py` — stereotype detection
  (`stereotype`/`no_stereotype` + severity 0–2); rubric shipped in the
  module docstring. The prompt names the group of interest.
- `src/opsaudit/judges/refusal.py` — refusal/hedge detection
  (`refusal`/`compliance`); a *wrong* answer is still compliance —
  this judge measures willingness, not correctness.
- `src/opsaudit/judges/tone.py` — tone toward the recipient
  (`cold`/`neutral`/`warm`) with `TONE_ORDER` for ordinal aggregation.
- `src/opsaudit/judges/aggregate.py` — deterministic aggregation:
  per-group label rates, per-label max–min gaps, unscored rate,
  ordinal means. Mirrors the v0.1 gap logic; `unscored` items are
  excluded from rates but counted visibly.
- `src/opsaudit/calibration/harness.py` — `CalibrationHarness` runs a
  judge over human-labeled `(text, label)` pairs and reports accuracy,
  **Cohen's kappa**, per-label precision/recall, and a confusion
  summary — all deterministic. The `CalibrationReport` carries a
  PASS/FAIL recommendation against a configurable kappa threshold
  (default 0.6, the conventional "substantial agreement" boundary). A
  FAIL prints a loud stderr warning; the report is stored on
  `judge.calibration`.
- `src/opsaudit/calibration/datasets.py` — **synthetic, hand-written**
  starter datasets (~40 items across the three judge tasks) so the
  harness runs on day one. Marked synthetic everywhere; production
  audits must supply real human labels.

## Campaign wiring (optional, off by default)

`AuditCampaign(..., judges=[...])`: when a round's outputs are
non-numeric text and judges are configured, the round summary includes
deterministic per-attribute judge-label findings
(`_summarize_with_judges`); the round strength is the strongest
judge-label gap. Calibration is enforced at construction: a judge
without a *passing* calibration report is rejected unless the operator
passes `allow_uncalibrated=True` — and that override is logged as an
`uncalibrated_judge_override` evidence event, right up front.

## What "calibrated" means here

The shipped judges have no real-model calibration scores yet: the
starter datasets are synthetic and the test/demo judges are scripted.
The success-criteria gate for Phase 5 ("every judge ships with a
measured calibration score in the docs") is satisfied by the *harness*,
not by pretending scripted scores are real. Before any production use,
an operator must run `CalibrationHarness` with a real judging model
against real human-labeled data and record the report.

## Dependency policy (unchanged)

No new dependencies. Unit tests never touch the network — judge
"models" are scripted `Target` subclasses.

## Demo

`examples/agentic-audit-demo/phase4_demo.py` — runs the harness over
the starter datasets (scripted perfect judges → kappa 1.0 PASS, plus a
deliberately constant judge → kappa 0.0 FAIL with the warning), then a
mini campaign with a calibrated `ToneJudge` against a scripted target
with a planted tone gap (warm for full-time, cold for temp): caught via
judge labels, gap 1.0, ordinal gap 2.0.

## Conventions for Phase 5 authors

- Judges emit labels; summaries consume labels. Never let an LLM touch
  a number — enforce in review.
- New judges: new module + versioned prompt + rubric in the docstring +
  starter dataset entries + calibration before use.
- Evidence event kinds added: `uncalibrated_judge_override`
  (Phase 3 kinds unchanged).

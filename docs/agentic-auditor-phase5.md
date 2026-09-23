# Agentic auditor — Phase 5: reports, RMF integration, v0.2.0

Phase 5 closes the loop: a campaign's evidence becomes a defensible
report, mapped onto NIST AI RMF 1.0, shippable as v0.2.0. Nothing here
changes v0.1 behavior or the Phase 1–4 interfaces — `opsaudit.reports`
is purely additive.

## The design bet

An agentic audit is only as defensible as its paper trail. The report
is therefore **deterministic template fill, never LLM prose**: the
executive summary, findings table, methodology, limitations, and
reproducibility block are all rendered from numbers in the evidence
log. If two runs have the same evidence, they have the same report —
word for word.

## New modules

- `src/opsaudit/reports/report.py` — `AuditReport` (frozen dataclass)
  and `build_report(campaign_report, target=..., judges=...,
  budget=...)`. Sections: executive summary (plain-language sentences
  built from numbers), findings table (per-round generator, probe
  count, strength, key signal), methodology (brief, budget, generators
  used, judge cards with rubric + prompt version + calibration scores),
  limitations (honest, always printed), NIST AI RMF mapping, and a
  reproducibility block (seed, package version, git SHA when
  available, `target.describe()`, evidence path).
- `src/opsaudit/reports/markdown.py` / `html.py` — Markdown and
  standalone-HTML renders. The HTML is one file: inline CSS, no
  external assets, no scripts, no sticky/fixed positioning.
- `src/opsaudit/reports/rmf.py` — `AGENTIC_RMF_MAPPING`: eight
  evidence-contribution mappings onto real NIST AI RMF 1.0
  subcategories (verified against the framework core, e.g. MEASURE
  2.11 for fairness-and-bias evaluation, MEASURE 2.1 for documented
  test sets, MAP 5.1 for impact characterization). Every mapping
  carries a `limits` field; `mappings_for_report` includes the
  judge-calibration mapping only when judges were used.
- `docs/RMF_MAPPINGS.md` — what each mapping claim means and where it
  stops. The mappings are evidence contributions, not compliance
  claims.

## The verdict, honestly labeled

The report flags `FINDINGS WARRANT REVIEW` when the best finding
strength meets a heuristic flagging threshold (default 0.2), otherwise
`NO MATERIAL FINDINGS`. The threshold is printed next to the verdict
in every render, labeled a heuristic — it is not a significance test.

## Demo

`examples/agentic-audit-demo/phase5_demo.py` — the v0.2 acceptance
proof, fully scripted (no network, no credentials):

1. A `TabularTarget` with a planted disparity on `employment_type`
   (an attribute the fixed v0.1 probe battery never groups by, so the
   fixed battery misses it); the agentic campaign's brief names it,
   the planner drills in, gap 1.000 found.
2. A scripted text target with a planted tone gap (warm for FT, cold
   for temp) caught via a calibrated `ToneJudge`.
3. A scripted RAG pipeline (`RagTarget`) that refuses benefits questions
   for temp workers but answers them for FT — caught via a calibrated
   `RefusalJudge`. Text-output disparities are definitionally invisible
   to the fixed v0.1 tabular battery.

Each campaign produces Markdown + HTML reports. The demo exits 0 and
runs from any working directory.

## Conventions for future authors

- Reports never contain LLM-generated prose. Template fill only.
- New report sections: add a builder in `report.py`, renders in
  `markdown.py`/`html.py`, and a test in `tests/test_reports.py`.
- RMF mappings: verify subcategory numbers against the RMF 1.0 core
  before adding; when in doubt, claim the function level and say so.

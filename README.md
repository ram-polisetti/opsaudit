# opsaudit

![CI](https://github.com/ram-polisetti/opsaudit/actions/workflows/ci.yml/badge.svg?branch=main) ![Python 3.10 or newer](assets/python.svg) ![Apache-2.0 license](assets/license.svg)

> **A practical pre-deployment fairness and governance check for operational ML decisions.**

opsaudit is a practical, open-source Python toolkit for checking whether operational ML decisions—such as dispatch, routing, demand forecasts, and warehouse staffing—produce materially different outcomes across groups. It is built by an operator, for operators: generate a reproducible scenario, measure the disparities, save a reviewable report, and use the gate in a release workflow. It helps teams identify disparity signals, document them, and decide what must be investigated before deployment; it does not claim to certify a system as fair.

## Installation

Install the current development version directly from GitHub:

```bash
pip install "git+https://github.com/ram-polisetti/opsaudit.git"
```

The validated `v0.1.0` release remains available at the matching Git tag:

```bash
pip install "git+https://github.com/ram-polisetti/opsaudit.git@v0.1.0"
```

PyPI publication is prepared but intentionally pending PyPI account setup. Until then, use a GitHub install or clone the repository for local development:

```bash
git clone https://github.com/ram-polisetti/opsaudit.git
cd opsaudit
pip install -e .[dev]
```

## Quickstart

```bash
pip install -e .
opsaudit generate --scenario dispatch --n 5000 --bias 0.0 --seed 42 --out dispatch.csv
opsaudit audit --data dispatch.csv --truth on_time --pred priority_route --group group --out report
opsaudit gate --report report.json        # expect: pass
opsaudit generate --scenario dispatch --n 5000 --bias 0.5 --seed 42 --out biased.csv
opsaudit audit --data biased.csv --truth on_time --pred priority_route --group group --out biased_report
opsaudit gate --report biased_report.json # expect: FAIL, exit code 1
```

The `generate` command creates only local, synthetic data. `audit` writes matching `.md`, `.html`, and `.json` reports. The gate returns exit code 0 for a passing result and 1 for a failing result, so it can be used directly in CI.

## Agentic auditor (v0.2)

Point opsaudit at any model — a tabular classifier, a chat model, a RAG pipeline — and get an *adaptive* audit: an LLM planner decides what to probe next, deterministic code builds the probes and computes every statistic, and LLM judges score open-ended text outputs only (with measured calibration). Every step lands in an append-only evidence log, and the campaign renders into a Markdown/HTML report mapped onto NIST AI RMF 1.0.

```
brief ──▶ AuditPlanner ──ProbeSpec──▶ Probe generators ──▶ Target adapter ──▶ target
   ▲            │  (LLM picks WHAT            │ (deterministic:        │ (tabular / HF /
   │            │   to probe; short            │  counterfactual,       │  OpenAI-compat /
   │            │   JSON, never builds          │  adversarial,          │  Ollama / RAG)
   │            │   probes)                     │  metamorphic)          │
   │            ▼                               ▼                      ▼
   │     ResponseCache ◀── observations ── deterministic summaries ──┘
   │            │        (gaps, error/violation rates; judge LABELS only)
   │            ▼
   └──── replan (or stop: budget exhausted / findings flat / planner stops)
                                  │
                                  ▼
                    AuditReport → Markdown / HTML (+ RMF mapping)
```

```python
from opsaudit import TabularTarget, AuditPlanner, AuditCampaign, Budget, build_report
from opsaudit.reports import save_agentic_report

brief = {
    "target_type": "tabular",
    "protected_attributes": {"employment_type": ["FT", "temp"]},
    "risk_areas": ["hiring decisions"],
    "base_input": {"tenure_months": 12, "employment_type": "FT"},
}
planner = AuditPlanner(brief, planner_llm_target, budget=Budget(max_probes=200))
campaign = AuditCampaign(audited_target=TabularTarget(model), planner=planner,
                         evidence_path="evidence.jsonl", seed=42)
result = campaign.run()
report = build_report(result, target=campaign.audited_target,
                      budget={"max_probes": 200})
save_agentic_report(report, "agentic_report")   # .md + .html + .json
```

### Bring your own model: the 15-line adapter

The auditor never touches your model — it only touches the `Target`
contract: `predict(X)` for scoring models, `generate(prompts)` for text
models, plus `describe()` for the evidence log. Ready-made adapters cover
the common cases:

| Your model | Adapter | Notes |
|---|---|---|
| sklearn, XGBoost, LightGBM | `TabularTarget` | duck-typed on `.predict(X)` — no sklearn import needed |
| OpenAI-compatible API | `OpenAICompatTarget` | chat-completions endpoint |
| Ollama (local or Cloud) | `OllamaTarget` / `SecureOllamaTarget` | |
| HuggingFace pipeline | `HuggingFaceTarget` | text-generation pipelines |
| RAG pipeline | `RagTarget` | wraps any `query -> str` function |

Anything else — PyTorch, TensorFlow, Spark ML, a REST endpoint — gets a
small adapter *you* write, translating the contract into your model's
dialect. This is deliberate: the framework stays small by defining one
interface instead of learning every library's API. PyTorch example:

```python
from opsaudit.targets import Target

class TorchTarget(Target):
    """Audit a PyTorch classifier: Target contract on one side,
    whatever dialect your model speaks on the other."""

    def __init__(self, module, encode_row, *, name="torch-model"):
        self.module = module            # your nn.Module
        self.encode_row = encode_row    # applicant dict -> tensor
        self._name = name

    def predict(self, X):
        import pandas as pd, torch
        rows = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.module.eval()
        with torch.no_grad():
            batch = torch.stack([self.encode_row(r) for _, r in rows.iterrows()])
            return [int(v) for v in self.module(batch).argmax(dim=1).tolist()]

    def describe(self):
        return {"target_type": "tabular", "name": self._name,
                "framework": "pytorch"}
```

Then `AuditCampaign(audited_target=TorchTarget(...), ...)` — and everything
downstream (probes, metrics, planner, evidence log, report) is unchanged.
Rule of thumb: if you can turn rows into outputs, you can audit it.

**The hard rules** (enforced in code review, not just docs):

- The LLM proposes; the statistics dispose. Judges turn text into *labels*; every rate, gap, and verdict is computed in deterministic code.
- No judge ships without a measured calibration score: run `CalibrationHarness` against real human labels (accuracy, Cohen's kappa) before trusting one. The campaign *rejects* uncalibrated judges unless you pass an explicit, evidence-logged `allow_uncalibrated=True`.
- The report verdict (`FINDINGS WARRANT REVIEW` / `NO MATERIAL FINDINGS`) uses a heuristic flagging threshold (default 0.2), printed next to the verdict — it is not a significance test.
- RMF mappings name the *evidence the audit contributes* toward real NIST AI RMF 1.0 subcategories (e.g. MEASURE 2.11 for bias evaluation, MEASURE 2.1 for documented test sets); each carries its limits — the tool assists a subcategory, it never satisfies one. See [docs/RMF_MAPPINGS.md](docs/RMF_MAPPINGS.md).

**Limitations, honestly:** the campaign only probes what the planner chose within budget; LLM planners/judges have their own biases (calibration measures agreement, it doesn't remove bias); scripted demos are pipeline proofs, not real audits. Try the scripted end-to-end proof: `python examples/agentic-audit-demo/phase5_demo.py` (no credentials, no network).

## Provenance, verification, and human sign-off

Every `audit` run embeds a tamper-evident `provenance` block in the report JSON: SHA-256 of the input data, row count, opsaudit version and git SHA, the full resolved CLI arguments, the random seed, a UTC timestamp, the Python version, and the gate verdict computed at audit time. `generate` writes a matching provenance sidecar (`<name>.provenance.json`) next to its CSV.

```bash
opsaudit verify --report report.json --data dispatch.csv
# {"verified": true, "checks": [...]} — exit 0

opsaudit signoff --report report.json --reviewer "A. Operator" \
  --decision approve --note "Reviewed FPR gap; acceptable for this lane."
# recorded approve by A. Operator (sign-off #1 on this report)

opsaudit verify --report report.json --data dispatch.csv  # still exit 0
```

`verify` checks six things: the provenance block exists, the data hash matches, the row count matches, the report body hash matches, the sign-off hash chain is intact, and a deterministic re-run of the audit reproduces the recorded metrics and gate status. Any edit — to the data, the metrics, or a sign-off note — fails verification with a named reason (exit 1). Sign-offs are hash-chained to the report body and to each other, so a reviewer's `reject` on a failing gate cannot be quietly rewritten to `approve`. See `docs/PROVENANCE.md` for the full design, threat model, and a worked validation on the real UCI Adult dataset.

## Evidence-quality options

Use an audit-context manifest to preserve decision metadata with the report. The manifest is for compact review context—not raw records, credentials, or personal data.

```bash
opsaudit audit \
  --data dispatch.csv \
  --truth on_time \
  --pred priority_route \
  --group group \
  --context examples/audit-context.yaml \
  --min-group-n 30 \
  --bootstrap 500 \
  --out evidence_report
```

- Repeat `--group` to form an intersectional label, such as `--group delivery_zone --group contract_type`.
- `--min-group-n` records a review condition when a group is too small for the configured policy.
- `--bootstrap 0` is the default; use a bounded value from `1` to `1000` for deterministic 95% stratified bootstrap intervals.
- The Markdown, HTML, and JSON reports preserve context, evidence-quality conditions, and intervals.

The deployment gate now has an explicit three-state contract:

| status | exit code | meaning |
| --- | ---: | --- |
| `PASS` | 0 | Configured checks passed and no review condition was recorded. |
| `FAIL` | 1 | One or more configured disparity checks failed. |
| `REVIEW` | 2 | Evidence is insufficient or uncertain; a human decision is required. |

Most CI systems treat exit code `2` as blocking. If a team deliberately wants a warning-only exception, it should handle exit `2` explicitly in its pipeline rather than weaken the default gate.

## Metrics glossary

| metric | definition | default gate |
| --- | --- | --- |
| selection rate | Share of records receiving a positive model decision within a group. | n/a |
| demographic parity difference | Highest group selection rate minus lowest group selection rate. | `<= 0.20` |
| disparate impact ratio | Lowest group selection rate divided by highest group selection rate; `1.0` when no positive decision exists for any group. | `>= 0.80` |
| TPR gap | Largest pairwise difference in true-positive rate across groups with positive labels. | `<= 0.15` |
| FPR gap | Largest pairwise difference in false-positive rate across groups with negative labels. | `<= 0.15` |
| precision | True positives divided by positive model decisions for a group. | n/a |
| accuracy | Correct decisions divided by all records for a group. | n/a |

## NIST AI RMF mapping

opsaudit connects practical evidence to the NIST AI Risk Management Framework at the function level: synthetic scenarios help **Map** contexts, disparity checks **Measure** risk, the deployment gate helps **Manage** release decisions, and saved reports support **Govern** oversight. See [rmf.py](src/opsaudit/rmf.py) for the complete mapping; it intentionally does not claim RMF subcategory numbers.

## What this audit can and cannot tell you

opsaudit is a screening and documentation tool, not a fairness certification. A passing gate means the configured checks passed for the supplied data and thresholds; it does not prove that a system is fair.

- Treat the disparate-impact ratio and other thresholds as investigation signals, not universal decision rules.
- Check group definitions and sample sizes before drawing conclusions; small groups can produce noisy rates.
- Review data and outcome-label quality. For example, on-time delivery can be affected by route difficulty, staffing, infrastructure, and package mix—not just a decision model.
- Use group fields only for a legitimate audit purpose, with appropriate access controls and accountable review.
- Record the reviewer, decision, rationale, and remediation evidence alongside the report.

## Interpreting a failed gate

The biased dispatch quickstart intentionally produces a failed report, including a disparate-impact ratio below the default `0.80` threshold. A failed gate is a prompt to investigate, not an automatic diagnosis of cause.

1. Confirm the decision, outcome, group columns, and row counts are correct.
2. Inspect per-group sample sizes and rates; check for data-quality or labeling issues.
3. Review the allocation rule and surrounding operational process for plausible causes.
4. Decide whether to remediate, change an approved policy threshold, or hold deployment; document the owner and rationale.
5. Rerun the audit after remediation and retain both reports as decision evidence.

The gate uses exit `0` for `PASS`, `1` for `FAIL`, and `2` for `REVIEW`. A review condition means the tool cannot make a reliable release recommendation without a human decision.

## Disparity-gate GitHub Action

The gate ships as a reusable composite action that turns the thesis — the
bias check belongs in the deployment pipeline, as a gate that can say no —
into a merge-blocking check on every PR.

### Adopt it in your repo

1. Add `.opsaudit-gate.yml` to your repo root (see
   [examples/gate-consumer](examples/gate-consumer/) for a complete,
   deliberately-failing example):
   ```yaml
   version: 1
   audit:
     data: data/decisions.csv   # CSV with one row per decision
     truth: hired               # ground-truth outcome column
     pred: model_decision       # model decision column (omit for outcome-rate-only audits)
     groups: [gender]           # one or more group columns
     min_group_n: 30
     bootstrap: 200
   thresholds:                  # overrides; defaults are the four-fifths rule + 0.15 error gaps
     disparate_impact_ratio: {min: 0.8}
   gate:
     fail_mode: block           # block | advisory
     review: block              # block | pass — how a REVIEW verdict treats the check
   ```
2. Add `.github/workflows/disparity-gate.yml` (copy
   `examples/gate-consumer/workflow.yml`), set the `paths:` filter to your
   model code, data, and features, and grant the three permissions
   (`contents: write`, `pull-requests: write`, `actions: write`).
3. Open a PR. The action installs opsaudit, runs
   `opsaudit gate-run`, uploads the verdict + full report bundle as an
   artifact, posts the verdict as a PR comment, and appends the verdict to
   the append-only audit log.

### What "the gate says no" looks like

A failing check posts a comment with the decision banner
(`⛔ Disparity gate: THE GATE SAYS NO`), a table of every check with its
value vs. threshold, the affected slices sorted by selection rate, any
evidence-quality notes, and a link to the full report artifact. The check
stays red until the disparity is fixed or the threshold is deliberately
relaxed in `.opsaudit-gate.yml` with justification.

When the evidence is too thin to judge (tiny groups, missing predictions),
the gate returns `REVIEW` instead of guessing: the check blocks until a
human records `opsaudit signoff --report <report.json> --reviewer <name>
--decision approve|reject`, hash-chained to the report.

### The audit log: design and tradeoff

Every verdict (pass/fail/review, commit SHA, metrics, timestamp) is appended
as one JSONL line to `verdicts.jsonl` on a dedicated orphan branch,
`opsaudit-audit-log`, plus a 90-day workflow artifact as backup. The branch
design was chosen because PR workflows never touch that ref, so the log
survives force-pushes to feature branches and history rewrites on `main` —
a log committed to `main` itself would not. Tradeoffs, stated plainly:
appending needs `contents: write`; concurrent runs rebase-and-retry on push
conflicts (5 attempts); and a repository admin can still delete the branch,
so for evidentiary-grade immutability you would mirror `verdicts.jsonl` to
external immutable storage — that mirroring is out of scope here.

### Limitations

- The gate audits the data you point it at; it cannot detect disparities in
  data it never sees, and a passing gate is not a fairness certification.
- Thresholds are policy, not physics: the defaults encode the four-fifths
  rule and should be set deliberately per deployment.
- `review: block` is the safe default, but it means thin data blocks merges
  until a human signs off — plan reviewer capacity accordingly.
- Fork-PR setups need the `pull_request_target` pattern (the bundled
  workflow uses `pull_request`); see the example for the permission notes.

## Synthetic data auditor

Before synthetic data is allowed to train anything, `audit-synthetic` evaluates it
against the source data on three axes and issues a go/no-go verdict (exit 0/1/2
for pass/fail/review):

- **Fidelity** — per-column distribution similarity (KS for numerics, total
  variation distance for categoricals), correlation drift, marginal coverage.
  Losing a well-represented source category escalates to at least `review`.
- **Privacy** — membership-inference attack surface: nearest-neighbor distance
  ratios, memorization rate, and exact-duplicate excess over chance collisions.
- **Bias amplification** — runs the real `audit_disparities` on source and
  synthetic and compares disparate impact, parity gaps, and TPR/FPR gaps,
  with bootstrap confidence-interval separation so sampling noise is not
  mistaken for new bias.

```bash
opsaudit audit-synthetic --source source.csv --synthetic synthetic.csv \
  --config synthetic-audit.yml --out synthetic_report.json
```

See [docs/SYNTHETICS.md](docs/SYNTHETICS.md) for the full methodology and
limitations, and
[examples/synthetic-audit-demo/demo.py](examples/synthetic-audit-demo/demo.py)
for a worked UCI Adult example (a privacy-safe synthetic set passes; a
memorizing, biased one fails on privacy and bias).

## Project status and roadmap

### v0.2.0 — the agentic LLM auditor

Five phases, each merged with a green CI gate and an independent demo:

1. **Target adapters + evidence log** — one `Target` interface (`.predict()` / `.generate()`) with adapters for HuggingFace, OpenAI-compatible APIs, Ollama Cloud, tabular/sklearn-style models, and RAG pipelines; append-only JSONL evidence transcript.
2. **Probe generators** — deterministic counterfactual, adversarial, and metamorphic generators (plus an LLM-assisted proposer treated as untrusted: every candidate validated, invalid ones discarded with reasons).
3. **Agentic planner loop** — plan → probe → observe → replan with budgets, response caching, deterministic stopping rules, and seeded reproducibility.
4. **Judges + calibration** — stereotype, refusal, and tone judges that emit labels only; a calibration harness measuring judge-vs-human agreement (Cohen's kappa, PASS/FAIL at 0.6); uncalibrated judges are rejected unless explicitly overridden.
5. **Reports + RMF + release** — deterministic Markdown/HTML reports with methodology, limitations, and reproducibility blocks; evidence-contribution mappings onto real NIST AI RMF 1.0 subcategories ([docs/RMF_MAPPINGS.md](docs/RMF_MAPPINGS.md)).

Acceptance proof: `examples/agentic-audit-demo/phase5_demo.py` audits three target types and adaptively finds a disparity the fixed v0.1 battery missed. All v0.1 tests remain green.

### v0.1.0 — available from GitHub

- Deterministic dispatch and staffing scenarios, binary disparity metrics, Markdown/HTML/JSON reports, function-level NIST AI RMF mapping, and a CI-friendly deployment gate.
- GitHub Actions tests package builds and PyPI metadata; the `v0.1.0` tag and draft release are prepared.
- Cite the project using [CITATION.cff](CITATION.cff), and see [CHANGELOG.md](CHANGELOG.md) for release contents.

### v0.1.1 — development on `main`

This development release prioritizes decision context and evidence quality over more surface area:

- Audit-context manifests for the system, model version, intended use, decision/audit owners, decision period, threshold policy, reviewer decision, and remediation notes.
- Configurable minimum-group-size review conditions and an explicit warning when every decision is negative.
- Optional, capped bootstrap confidence intervals for selection rates and disparity metrics.
- A `REVIEW` result for insufficient evidence, with exit `0` = pass, `1` = fail, and `2` = review.
- Tamper-evident provenance: `provenance` blocks in report JSON, `opsaudit verify` for data/body/sign-off/deterministic-re-run checks, and `opsaudit signoff` for hash-chained human review records (see `docs/PROVENANCE.md`).

### Later: intersectional analysis

Support combined group definitions—such as region plus contract type—only after the minimum-sample and uncertainty work is in place. Intersectional analysis should report small-group limitations rather than overstate weak evidence.

## Intentional non-goals

opsaudit deliberately does not provide a dashboard, web application, model training, model selection, live drift monitoring, telemetry, legal compliance certification, real company data, or client-branded examples. The project stays focused on explainable, reproducible operational AI-audit evidence.

## Learn and present the project

- [Operational AI Audit Playbook](docs/OPERATIONAL_AI_AUDIT_PLAYBOOK.md) — the practical review workflow behind an audit.
- [Three-minute demo script](docs/DEMO_SCRIPT.md) — a clean-pass versus biased-fail walkthrough.
- [Evidence-quality design](docs/EVIDENCE_QUALITY_DESIGN.md) — the `PASS`/`FAIL`/`REVIEW` contract and technical decisions.

## Development

```bash
pip install -e .[dev]
pytest
pytest --cov=src/opsaudit --cov-fail-under=80
```

To build and validate release artifacts locally:

```bash
pip install -e .[dev,release]
python -m build
twine check dist/*
```

Contributions are welcome—see [CONTRIBUTING.md](CONTRIBUTING.md). Release maintainers should also read [PUBLISHING.md](PUBLISHING.md).

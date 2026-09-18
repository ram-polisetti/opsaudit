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

## Project status and roadmap

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

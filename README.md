# opsaudit

![CI](assets/ci.svg) ![Python 3.10 or newer](assets/python.svg) ![Apache-2.0 license](assets/license.svg)

opsaudit is a practical, open-source Python toolkit for checking whether operational ML decisions—such as dispatch, routing, demand forecasts, and warehouse staffing—produce materially different outcomes across groups. It is built by an operator, for operators: generate a reproducible scenario, measure the disparities, save a reviewable report, and use the gate in a release workflow.

## Installation

Install the published package:

```bash
pip install opsaudit
```

For local development, clone the repository and use `pip install -e .[dev]` instead.

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

## Interpret results responsibly

opsaudit is a screening and documentation tool, not a fairness certification. A passing gate means the configured checks passed for the supplied data and thresholds; it does not prove that a system is fair. Before acting, verify group definitions and sample sizes, review data and outcome-label quality, and investigate operational conditions that may affect outcomes. Record the reviewer, decision, and rationale alongside the report.

## Interpreting a failed gate

The biased dispatch quickstart intentionally produces a failed report, including a disparate-impact ratio below the default `0.80` threshold. A failed gate is a prompt to investigate, not an automatic diagnosis of cause.

1. Confirm the decision, outcome, group columns, and row counts are correct.
2. Inspect per-group sample sizes and rates; check for data-quality or labeling issues.
3. Review the allocation rule and surrounding operational process for plausible causes.
4. Decide whether to remediate, change an approved policy threshold, or hold deployment; document the owner and rationale.
5. Rerun the audit after remediation and retain both reports as decision evidence.

The current CLI exit contract is `0` for `PASS` and `1` for `FAIL`. A future `REVIEW` state will use exit code `2`, indicating that the gate cannot make a reliable decision and should block CI unless a pipeline explicitly handles that condition.

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

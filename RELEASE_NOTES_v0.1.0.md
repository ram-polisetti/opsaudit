# opsaudit v0.1.0 release notes

opsaudit v0.1.0 is the first public release of a practical Python toolkit for auditing disparity signals in operational machine-learning decisions.

## Highlights

- Audit binary decisions across groups with selection rate, demographic parity difference, disparate impact ratio, TPR/FPR gaps, precision, and accuracy.
- Generate deterministic synthetic dispatch and warehouse-staffing scenarios for demonstrations and tests.
- Produce Markdown, HTML, and JSON evidence artifacts, including a function-level NIST AI RMF mapping.
- Use `opsaudit gate` as a CI-friendly go/no-go check: exit `0` passes and exit `1` fails.
- Configure threshold overrides in YAML without changing application code.

## Install

```bash
pip install opsaudit
```

## Scope

This release does not train models, provide legal or fairness certification, monitor drift, send telemetry, use real company data, or make runtime network calls. It is designed to surface disparity signals and support a documented operational review.

See [CHANGELOG.md](CHANGELOG.md) for the complete v0.1.0 inventory and [README.md](README.md) for the quickstart and interpretation guidance.

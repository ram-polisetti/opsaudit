# Changelog

All notable changes to this project are documented in this file.

## 0.1.1 — Unreleased

### Added

- Flat YAML audit-context manifests, preserved in Markdown, HTML, and JSON audit evidence.
- Configurable minimum-group-size review conditions and an all-negative-decision review warning.
- Optional, capped, deterministic stratified bootstrap confidence intervals.
- Three-state deployment-gate status: `PASS` (exit 0), `FAIL` (exit 1), and `REVIEW` (exit 2).
- Repeated `--group` options for intersectional group labels.
- Operational audit playbook, demo script, and v0.1.1 evidence-quality design documentation.
- Tamper-evident provenance trail: every `audit` run embeds a `provenance` block in the report JSON (input SHA-256, row count, code version and git SHA, full resolved arguments, seed, UTC timestamp, Python version, gate verdict); `generate` writes a provenance sidecar next to its CSV. New `opsaudit verify` command recomputes the data hash, checks the report body hash and sign-off chain, and deterministically re-runs the audit to confirm recorded metrics and gate status (exit 0 = verified, 1 = tampered). New `opsaudit signoff` command appends hash-chained human review records (approve/reject) — the escalation point for REVIEW gates — and refuses to sign tampered reports. Full design, threat model, and real-data validation in `docs/PROVENANCE.md`.

## 0.1.0 — 2026-09-18

### Added

- Binary group-disparity metrics for selection rate, demographic parity difference, disparate impact ratio, TPR gap, FPR gap, precision, and accuracy.
- Seeded synthetic dispatch and warehouse-staffing scenarios.
- Markdown, HTML, and JSON audit reports with function-level NIST AI RMF mapping.
- A CI-friendly deployment gate with YAML threshold overrides.
- `opsaudit generate`, `opsaudit audit`, and `opsaudit gate` commands.
- Test coverage for metrics, data generation, reports, gate behavior, and CLI workflows.

### Scope boundaries

- No model training, dashboard, web application, drift monitoring, telemetry, real company data, or runtime network calls.

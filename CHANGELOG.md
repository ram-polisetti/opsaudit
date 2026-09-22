# Changelog

All notable changes to this project are documented in this file.

## 0.1.1 — Unreleased

### Added

- Synthetic data auditor (`opsaudit.synthetics` + `opsaudit audit-synthetic` CLI): three-axis go/no-go evaluation of synthetic datasets against their source data — statistical fidelity (per-column KS/TVD, correlation drift, marginal coverage, support-aware `vanished_categories` escalation), privacy attack-surface probe (distance-ratio risk, memorization rate, exact-duplicate excess over chance), and bias amplification via the real `audit_disparities` on both frames with bootstrap-CI separation guards. YAML config with per-axis threshold overrides; JSON report with provenance block; exits 0/1/2 for pass/fail/review. Methodology and limitations in `docs/SYNTHETICS.md`; worked UCI Adult demo in `examples/synthetic-audit-demo/` (devcontainer pre-fetches the dataset); 33 tests in `tests/test_synthetics.py`.
- Flat YAML audit-context manifests, preserved in Markdown, HTML, and JSON audit evidence.
- Configurable minimum-group-size review conditions and an all-negative-decision review warning.
- Optional, capped, deterministic stratified bootstrap confidence intervals.
- Three-state deployment-gate status: `PASS` (exit 0), `FAIL` (exit 1), and `REVIEW` (exit 2).
- Repeated `--group` options for intersectional group labels.
- Operational audit playbook, demo script, and v0.1.1 evidence-quality design documentation.
- Tamper-evident provenance trail: every `audit` run embeds a `provenance` block in the report JSON (input SHA-256, row count, code version and git SHA, full resolved arguments, seed, UTC timestamp, Python version, gate verdict); `generate` writes a provenance sidecar next to its CSV. New `opsaudit verify` command recomputes the data hash, checks the report body hash and sign-off chain, and deterministically re-runs the audit to confirm recorded metrics and gate status (exit 0 = verified, 1 = tampered). New `opsaudit signoff` command appends hash-chained human review records (approve/reject) — the escalation point for REVIEW gates — and refuses to sign tampered reports. Full design, threat model, and real-data validation in `docs/PROVENANCE.md`.
- Disparity-gate GitHub Action (`.github/actions/disparity-gate`): reusable composite action that runs the gate on PRs touching model code/data, blocks the merge on threshold breach, posts the verdict with evidence as a PR comment, uploads the verdict + report artifact, and appends every verdict as JSONL to the append-only `opsaudit-audit-log` branch (survives force-pushes to PR branches; tradeoffs documented in README). Backed by a new `opsaudit.gate_ci` module and `opsaudit gate-run` CLI (`.opsaudit-gate.yml` config: audit spec, threshold overrides, `fail_mode: block|advisory`, `review: block|pass`; exits 0/1/2). The repo dogfoods the gate on its own PRs (`.github/workflows/disparity-gate.yml`); `examples/gate-consumer/` is a complete, deliberately-failing adoption example.

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

# Changelog

All notable changes to this project are documented in this file.

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

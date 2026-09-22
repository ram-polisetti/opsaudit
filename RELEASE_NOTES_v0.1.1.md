# opsaudit v0.1.1 release notes (draft — unreleased)

opsaudit v0.1.1 adds evidence-quality controls and a tamper-evident provenance trail on top of the v0.1.0 disparity-audit toolkit.

## Highlights

- Evidence-quality audit controls: YAML audit-context manifests, minimum-group-size review conditions, all-negative-decision review warnings, and optional deterministic stratified bootstrap confidence intervals.
- Three-state deployment gate: `PASS` (exit 0), `FAIL` (exit 1), `REVIEW` (exit 2) — uncertain or thin evidence now escalates to a human instead of silently passing.
- Repeated `--group` options for intersectional group labels.
- **Tamper-evident provenance trail:** every `audit` run embeds a `provenance` block in the report JSON (input SHA-256, row count, code version and git SHA, full resolved arguments, seed, UTC timestamp, Python version, gate verdict). `opsaudit verify` recomputes the data hash, checks the report body hash and sign-off chain, and deterministically re-runs the audit to confirm the recorded metrics and gate status. `opsaudit signoff` appends hash-chained human review records (approve/reject) and refuses to sign tampered reports.
- Validated end to end on the real UCI Adult dataset (30,162 rows): the gate correctly FAILs a biased rule-based predictor (disparate impact ratio 0.264), and `verify` detects single-row data tampering, metric edits, and sign-off rewrites. See `docs/PROVENANCE.md`.

## Scope

This release does not train models, provide legal or fairness certification, monitor drift, send telemetry, use real company data, or make runtime network calls. Provenance attests to the file as presented to `audit`; reviewer identities are self-asserted (no PKI). See `docs/PROVENANCE.md` for the full threat model and limitations.

See [CHANGELOG.md](CHANGELOG.md) for the complete v0.1.1 inventory and [README.md](README.md) for the quickstart and interpretation guidance.

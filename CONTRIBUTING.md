# Contributing to opsaudit

Keep changes practical, explainable, and covered by tests. To add a metric:

1. Add a function in `src/opsaudit/metrics.py` that returns JSON-safe values (`None`, never `NaN`).
2. Wire the result into `AuditResult` and the README glossary.
3. Add a unit test with hand-computed expected values.
4. Optionally add a default threshold entry when the metric should affect the deployment gate.

Pull requests must keep `pytest` green. Please avoid adding runtime network calls, telemetry, model training, dashboards, or real operational data to v0.1.

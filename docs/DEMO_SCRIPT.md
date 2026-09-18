# Three-Minute Demo Script

This is a short, repeatable walkthrough for a class, interview, or project discussion. Run it from a clean directory after installing `opsaudit` from GitHub or from a local clone.

## Setup — 15 seconds

> “opsaudit is a command-line pre-deployment fairness and governance check for operational ML decisions. It does not certify fairness; it creates evidence for a release decision.”

## 1. Generate and audit a clean dispatch scenario — 45 seconds

```bash
opsaudit generate --scenario dispatch --n 5000 --bias 0.0 --seed 42 --out dispatch.csv
opsaudit audit --data dispatch.csv --truth on_time --pred priority_route --group group --out clean_report
opsaudit gate --report clean_report.json
```

> “The gate exits 0. In this synthetic example, priority-route selection and error-rate gaps stay within the configured policy thresholds. The audit also writes Markdown, HTML, and JSON evidence files.”

## 2. Generate and audit a biased scenario — 60 seconds

```bash
opsaudit generate --scenario dispatch --n 5000 --bias 0.5 --seed 42 --out biased.csv
opsaudit audit --data biased.csv --truth on_time --pred priority_route --group group --out biased_report
opsaudit gate --report biased_report.json
```

> “This gate exits 1. The disparate-impact ratio and other disparity signals breach the default thresholds. A failure is not an automatic explanation of cause; it is a reason to hold or escalate the decision and investigate.”

## 3. Show evidence quality and context — 45 seconds

```bash
opsaudit audit \
  --data biased.csv \
  --truth on_time \
  --pred priority_route \
  --group group \
  --context examples/audit-context.yaml \
  --min-group-n 30 \
  --bootstrap 500 \
  --out evidence_report
```

> “The report now records who owns the decision, the intended use, the review disposition, group-size conditions, and optional uncertainty intervals. If evidence is insufficient, the gate uses exit 2 for REVIEW rather than issuing a misleading pass.”

## Close — 15 seconds

> “The point is not to add a dashboard to a metric. It is to give operations and governance stakeholders a reproducible way to decide whether a decision system needs investigation before release.”

## Discussion prompts

- What group definitions are legitimate and useful in this operational setting?
- Which owner approves the threshold policy and any exception?
- What could make `on_time` an incomplete or biased outcome label?
- What evidence should exist before a failed model decision is released anyway?

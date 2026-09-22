# Example consumer: the disparity gate saying no

This directory is a complete, copy-paste example of adopting the opsaudit
disparity gate in your own repo.

## What's here

- `.opsaudit-gate.yml` — the gate config: which CSV to audit, which columns
  are truth/prediction/groups, the fairness thresholds, and the blocking
  policy.
- `data/decisions.csv` — a small, deliberately **biased** dispatch dataset
  (generated with `opsaudit generate --scenario dispatch --n 2000 --bias 0.5
  --seed 42`). The gate fails on it on purpose.
- `data/decisions.provenance.json` — the provenance sidecar written by
  `opsaudit generate` (input hash, row count, seed).
- `workflow.yml` — copy to `.github/workflows/disparity-gate.yml`.

## Try it

1. Copy `.opsaudit-gate.yml` to your repo root (fix the `data:` path) and
   `workflow.yml` to `.github/workflows/disparity-gate.yml`.
2. Open a PR that touches a path in the workflow's `paths:` filter.
3. The gate audits the data, finds the disparate impact ratio at ~0.58
   against the 0.80 four-fifths threshold, **fails the check**, posts the
   verdict as a PR comment with the evidence table, and appends the verdict
   to the `opsaudit-audit-log` branch.
4. Regenerate the data with `--bias 0.0` and push again: the gate passes and
   the PR unblocks.

## The review path

If the data is too thin to judge (tiny groups, no prediction column), the
gate returns `review` instead of guessing: the check blocks until a human
records a decision with
`opsaudit signoff --report <report.json> --reviewer <name> --decision approve|reject`,
which is hash-chained to the report and verifiable with `opsaudit verify`.

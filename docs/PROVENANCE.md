# Provenance trail — design and validation

Tamper-evident provenance for opsaudit reports, introduced in v0.1.1.
An audit report is only as trustworthy as the link between the data that was
audited, the code that audited it, and the verdict that was recorded. This
document describes how opsaudit makes that link checkable.

## What is captured

Every `opsaudit audit` run embeds a `provenance` block in the report JSON:

| field | content |
| --- | --- |
| `tool` / `opsaudit_version` | `"opsaudit"` and the package version |
| `code.git_sha` / `code.git_dirty` | git HEAD of the checkout the code ran from (editable installs), plus whether the tree was dirty; `null` for installed wheels outside a checkout |
| `input.path` / `input.sha256` / `input.row_count` | the audited CSV: path as given, SHA-256 of its raw bytes, row count |
| `run.command` / `run.args` | `"audit"` and the full resolved arguments (truth, pred, groups, min-group-n, bootstrap, bootstrap-seed, context file) |
| `run.seed` / `run.timestamp_utc` / `run.python_version` | the random seed used, UTC timestamp, interpreter version |
| `gate.status` / `gate.findings` | the deployment-gate verdict computed at audit time |
| `body_sha256` | SHA-256 over the canonical JSON of the report body (metrics + provenance, excluding sign-offs) |

`opsaudit generate` writes a provenance sidecar (`<name>.provenance.json`) next to
its CSV, recording the scenario parameters, seed, and the output file's hash
and row count.

## What is hashed, and how

- Files: SHA-256 over raw bytes, streamed in 64 KiB chunks.
- In-memory structures: SHA-256 over canonical JSON (`sort_keys=True`, compact
  separators, UTF-8) — key order and whitespace cannot change the digest.
- The body hash covers the audit metrics **and** the provenance block, but not
  the `signoffs` list, so appending a human review never invalidates it. The
  hash field itself is excluded from its own computation.
- Each sign-off record carries `prev_hash` (the body hash for the first
  record, the canonical hash of the previous record afterwards) **and**
  `record_hash` (a hash of the record's own content), so edits to any record —
  including the last one — are detectable.

## Verification procedure

`opsaudit verify --report report.json --data data.csv` runs these checks and
prints a JSON summary (`{"verified": bool, "checks": [...]}`), exiting 0 only
when every check passes:

1. **provenance_present** — the report carries a provenance block (reports
   predating v0.1.1 fail here with guidance).
2. **data_hash** — the data file's SHA-256 matches the recorded input hash.
3. **row_count** — the data file's row count matches.
4. **body_hash** — the report body hash recomputes to the recorded value.
5. **signoff_chain** — every sign-off record's content hash and chain link verify.
6. **deterministic_rerun** — the audit is re-run from the recorded arguments
   against the supplied data; the recomputed metrics must equal the recorded
   ones byte-for-byte (bootstrap uses the recorded seed, so this is exact).
7. **gate_status** — the re-run's gate verdict matches the verdict recorded at
   audit time (compared only when no custom `--thresholds` overrides are given).

`opsaudit signoff --report … --reviewer … --decision approve|reject --note …`
appends a review record, but first refuses to sign a report whose body hash or
existing sign-off chain does not verify.

## Threat model

Covered — detected by `verify`:

- Silent edits to the audited CSV after the report was produced (even a single
  flipped prediction in 30,162 rows).
- Editing recorded metrics to make a failing gate look passing.
- Rewriting, deleting, or reordering human sign-off records (e.g. changing a
  `reject` to an `approve`).
- Appending a sign-off to a report whose metrics were already tampered with
  (`signoff` refuses).

Not covered — out of scope by design:

- **Key management / identity.** Reviewer names are self-asserted strings;
  there is no cryptographic signature binding a record to a person. A reviewer
  who can write the report file can forge a sign-off attributed to anyone.
  This is an append-only *evidence* log, not a PKI.
- **Data seen before hashing.** Provenance attests to the file as presented
  to `audit`; it cannot prove the file itself is the true production extract.
  Pair with access-controlled data pipelines for that guarantee.
- **Code substitution.** `git_sha` records which checkout ran, but a
  compromised interpreter or dependency could lie about it. Reproducible
  builds and pinned environments are the complementary control.
- **History rewriting.** The JSON report is a file, not a ledger; `verify`
  detects *modification*, not *deletion* of the report itself. Store reports
  in version control or immutable storage for retention guarantees.

## Limitations

- Verification re-runs the audit, so it needs the original data file and the
  same opsaudit version semantics; cross-version metric changes will (correctly)
  fail the deterministic re-run check.
- `generate` sidecars are informational only — `verify` covers audit reports.
- Timestamps and reviewer names are recorded as given; only hashes are verified.

## Real-world validation: UCI Adult dataset

Methodology (reproducible end to end; data stays local, never committed):

- **Source:** UCI Machine Learning Repository, Adult / Census Income dataset —
  <https://archive.ics.uci.edu/dataset/2/adult> (raw file:
  `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`,
  32,561 rows, no header). Donor: Barry Becker / Census Bureau 1994.
- **Preparation** (deterministic, scripted): parse with the documented column
  names, drop 2,399 rows with missing values → 30,162 rows. Truth: income
  `>50K`. Prediction: a fixed rule — predict 1 when
  `(education_num >= 13 and hours_per_week >= 40) or capital_gain > 0`
  (no model training, per this repo's scope). Groups: `sex`, `race`
  (intersectional via repeated `--group`).
- **Run:**
  ```bash
  opsaudit audit --data adult_audit.csv --truth truth --pred pred \
    --group sex --group race --min-group-n 100 --bootstrap 200 \
    --out adult_report
  opsaudit gate --report adult_report.json      # FAIL, exit 1
  opsaudit signoff --report adult_report.json --reviewer "Ram Charan Polisetti" \
    --decision reject --note "Gate FAIL: disparate impact 0.264 < 0.800; TPR gap 0.338."
  opsaudit verify --report adult_report.json --data adult_audit.csv  # verified: true
  ```
- **Observed results (2026-09-21, opsaudit 0.1.1):** disparate impact ratio
  **0.264** (threshold ≥ 0.800), demographic parity difference **0.321**
  (≤ 0.200), TPR gap **0.338** (≤ 0.150), FPR gap **0.208** (≤ 0.150) —
  gate **FAIL**, plus a `minimum_group_size` review flag for
  `sex=Female | race=Other` (n=87 < 100). The rule-based predictor reproduces
  the dataset's well-known sex/race disparities, which is exactly what the
  gate is for.
- **Tamper drills on the real report:** flipping a single prediction in
  30,162 rows → `verify` fails on `data_hash` (exit 1); editing a recorded
  metric → fails on `body_hash` and `deterministic_rerun`; rewriting the
  sign-off note → fails on `signoff_chain`. Untampered report with sign-off →
  all seven checks pass.

## Reproducing the claims

1. Install: `pip install -e ".[dev]"` (Python ≥ 3.10).
2. `pytest tests -q` — the suite includes 12 provenance tests covering every
   check above (modified data, modified metrics, modified sign-off, legacy
   reports, no-prediction audits, cross-run determinism).
3. Re-run the Adult walkthrough with the commands in this document; `verify`
   must print `"verified": true` and the tamper drills must fail as described.

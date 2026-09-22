# Synthetic data auditor — methodology

`opsaudit.synthetics` evaluates a synthetic dataset against the source data it
was derived from on three axes, then issues a go/no-go verdict (`pass`,
`review`, `fail`). The overall verdict is the worst of the three axis
verdicts. Everything is deterministic: every randomized step takes an
explicit seed.

CLI: `opsaudit audit-synthetic --source src.csv --synthetic syn.csv --config config.yml --out report.json`
(exits 0/1/2 for pass/fail/review). The JSON report carries a provenance
block with SHA-256 hashes and row counts of both inputs.

## Config

```yaml
truth: on_time            # binary outcome column (bias axis)
pred: priority_route      # binary decision column (bias axis)
groups: [group]           # protected-attribute column(s) (bias axis)
numeric: [packages_assigned]   # optional; inferred from dtypes when omitted
categorical: [group]           # optional; inferred from dtypes when omitted
exclude: [driver_id]           # id-like columns skipped by fidelity/privacy
seed: 42
thresholds:
  fidelity: {pass_score: 0.90, review_score: 0.75}
  privacy: {pass_risk: 0.20, review_risk: 0.45}
  bias: {max_amplification: 1.25, review_amplification: 1.10,
         max_gap_growth: 0.10, review_gap_growth: 0.05,
         introduced_disparity_di: 0.80, review_introduced_di: 0.88,
         source_fair_di: 0.90}
```

## Axis 1 — statistical fidelity

- **Numeric columns:** two-sample Kolmogorov–Smirnov statistic (max difference
  of empirical CDFs), implemented in NumPy (no SciPy dependency).
- **Categorical columns:** total variation distance between the category
  distributions over the union of both supports; categories present in the
  source but absent from the synthetic data are flagged as
  `vanished_categories` (with their source support recorded). Losing a
  well-represented category (≥1% of source rows) escalates fidelity to at
  least `review` — a downstream model never sees it, which is a utility
  failure even when the aggregate score looks fine. Losing a singleton is
  expected sampling noise under any resampling-based synthesizer, so it is
  recorded but does not escalate.
- **Correlation drift:** mean absolute difference of off-diagonal Pearson
  correlations over numeric columns (reported raw and normalized to [0, 1]).
- **Marginal coverage:** fraction of synthetic values inside the source's
  observed range (numeric) or seen categories (categorical).
- `fidelity_score` is the mean per-column similarity (`1 − KS` / `1 − TVD`).

## Axis 2 — privacy (membership-inference attack surface)

A distance-based membership-inference attack scores candidate records by their
distance to the nearest synthetic row: memorized source rows sit suspiciously
close. The probe measures that signal directly:

- Features are encoded once: numerics z-scored on **source** statistics,
  categoricals ordinal-encoded on combined categories. Pairwise distances are
  computed in chunks with the `||a−b||²` expansion (a 2000×2000 comparison
  needs ~32 MB). Both frames are deterministically subsampled to 2000 rows.
- `distance_ratio_risk = 1 − median(d_src→src) / median(d_src→synth)`, clipped
  to [0, 1]: 0 when synthetic rows are no closer than source rows are to each
  other, →1 as synthetic rows collapse onto source rows.
- `memorization_rate`: fraction of source rows closer to synthetic than the
  5th percentile of source→source nearest distances.
- `duplicate_rate` / `excess_duplicate_rate`: exact-match fraction of
  synthetic rows, minus the source's own self-collision rate — chance
  collisions in low-cardinality spaces are not evidence of copying.
- `privacy_risk` is the max of the three signals. Every fail/review verdict
  is explained by a finding: `close_synthetic_neighbors` (the distance-ratio
  signal itself), `high_memorization`, `exact_duplicates`, or
  `low_feature_cardinality`.

## Axis 3 — bias amplification

Runs the real `opsaudit.metrics.audit_disparities` on source and synthetic
with identical truth/pred/group roles and compares:

- `amplification_factor = (1 − DI_synth) / (1 − DI_src)`: >1 amplified, <1
  dampened. When the source is already fair (DI ≥ 0.90) the ratio is
  meaningless, so the synthetic side is judged in absolute terms instead
  (`disparity_introduced` / `disparity_drift`).
- Gap growth on demographic-parity difference, TPR gap, and FPR gap.
- Both audits use deterministic bootstrap 95% confidence intervals (default
  200 resamples). **A verdict only fires when the point-estimate threshold is
  breached *and* the intervals are separated in the worse direction** — gap
  statistics are noisy in small samples, and sampling noise must not read as
  new bias. Pass `bootstrap=0` to fall back to point-estimate rules.

## Limitations

- **Fidelity is marginal + pairwise.** Per-column and correlation checks miss
  higher-order joint-structure failures (e.g. a synthetic set that preserves
  every marginal and every pairwise correlation can still break a three-way
  interaction a downstream model relies on).
- **Privacy is an attack-surface probe, not a labeled attack.** It quantifies
  the signal a nearest-neighbor membership-inference attack would exploit; it
  does not train an attack model or report a true attack AUC, which would need
  ground-truth member/non-member labels the auditor does not have.
- **Low-cardinality feature spaces blind the privacy probe.** When most source
  rows already collide with another source row, closeness carries no signal;
  the report records a `low_feature_cardinality` finding — read a pass there
  as weak evidence, not a clean bill of health.
- **The probe sees the generator's output, not its training.** It cannot tell
  whether closeness came from memorization or from a small, clumpy source
  distribution.
- **Bias comparison needs adequate samples.** Gap estimates are noisy when the
  smallest group has only hundreds of rows; the bootstrap-CI separation rule
  guards against false alarms but also reduces power — small but real
  amplification can hide inside overlapping intervals.
- **Thresholds are policy, not math.** The defaults encode a risk appetite;
  set them from your own governance policy.
- **Tabular only.** Text, image, and sequential synthetic data are out of
  scope.

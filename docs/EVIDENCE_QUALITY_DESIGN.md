# Evidence-Quality Design

This document defines the v0.1.1 evidence-quality behavior so users and CI systems can rely on a stable contract.

## Decision contract

| status | CLI exit | interpretation | default CI posture |
| --- | ---: | --- | --- |
| `PASS` | 0 | All configured metric checks passed and no review condition was recorded. | Continue |
| `FAIL` | 1 | At least one configured metric check failed. | Block |
| `REVIEW` | 2 | The evidence does not support a reliable automatic recommendation. | Block pending human decision |

Metric failure takes precedence over review. A report can therefore contain both a failed metric and review findings while the final gate status remains `FAIL`.

## Review conditions

v0.1.1 records review conditions when:

- A configured `--min-group-n` is not met by one or more groups.
- No positive model decisions occur. The disparate-impact ratio is mathematically defined as `1.0` in that case, but it does not establish healthy allocation.
- The prediction column is omitted, so error-rate metrics cannot be assessed.
- A passing point estimate has an optional 95% bootstrap confidence interval that crosses the active policy threshold.

## Bootstrap design

- Disabled by default with `--bootstrap 0`.
- Explicitly enabled from `1` through `1000`; the cap protects normal CLI use from unexpectedly expensive workloads.
- Stratified by group, preserving each observed group size in every resample.
- Deterministic for a fixed input and `--bootstrap-seed`.
- Reports 2.5th and 97.5th percentile intervals for group selection rates and aggregate disparity metrics when the metric is defined in every resample.

Bootstrap intervals are evidence-quality signals. They do not replace a decision owner’s judgment about data-generating conditions, group definitions, or operational risk.

## Intersectional labels

Repeated `--group` options combine named columns into labels such as:

```bash
opsaudit audit --data data.csv --truth outcome --pred decision \
  --group delivery_zone --group contract_type --out report
```

This exposes intersections without silently hiding small groups. Use `--min-group-n` whenever intersectional analysis is enabled.

## Test matrix

| scenario | expected status |
| --- | --- |
| Clean dispatch, default options | `PASS` / 0 |
| Biased dispatch, default options | `FAIL` / 1 |
| Passing metrics with undersized group | `REVIEW` / 2 |
| Passing point estimate with a threshold-crossing confidence interval | `REVIEW` / 2 |
| Failed metric plus undersized group | `FAIL` / 1 |

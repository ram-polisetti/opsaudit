"""Synthetic data auditor: three-axis go/no-go evaluation of synthetic datasets.

Before synthetic data is allowed to train anything, it is audited against the
source data it was derived from on three axes:

1. **Fidelity** — does the synthetic data preserve the source distributions?
   Two-sample Kolmogorov-Smirnov per numeric column, total variation distance
   per categorical column, correlation-structure drift, and marginal coverage.
2. **Privacy** — membership-inference attack surface. Nearest-neighbor
   distances measure whether synthetic rows hug source rows (memorization),
   which is exactly the signal a distance-based membership-inference attack
   exploits.
3. **Bias amplification** — runs opsaudit's real ``audit_disparities`` on the
   source and on the synthetic data and compares: does the synthetic data
   amplify, preserve, or dampen group disparities?

Only NumPy and pandas are used (no SciPy); every randomized step takes an
explicit seed and is deterministic.

Example:
    >>> import pandas as pd
    >>> src = pd.DataFrame({"y": [1, 0, 1, 0], "g": ["a", "a", "b", "b"]})
    >>> syn = src.copy()  # a verbatim copy is the worst-case privacy outcome
    >>> result = audit_synthetic(src, syn, {"truth": "y", "pred": "y",
    ...     "groups": ["g"], "numeric": ["y"], "categorical": ["g"]},
    ...     bootstrap=0)
    >>> result["verdict"]
    'fail'
    >>> result["axes"]["privacy"]["verdict"]
    'fail'
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .metrics import audit_disparities

#: Axes evaluated by the auditor, in report order.
AXES = ("fidelity", "privacy", "bias")

#: Verdicts, ordered from best to worst for aggregation.
_VERDICT_ORDER = {"pass": 0, "review": 1, "fail": 2}

_DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "fidelity": {"pass_score": 0.90, "review_score": 0.75},
    "privacy": {"pass_risk": 0.20, "review_risk": 0.45},
    "bias": {"max_amplification": 1.25, "review_amplification": 1.10,
             "max_gap_growth": 0.10, "review_gap_growth": 0.05,
             "introduced_disparity_di": 0.80, "review_introduced_di": 0.88,
             "source_fair_di": 0.90},
}

#: Row cap for the pairwise-distance privacy probe (keeps memory bounded).
_PRIVACY_SAMPLE_N = 2000

#: A vanished category only escalates the fidelity verdict when it held at
#: least this share of source rows. Losing a singleton (or near-singleton)
#: to sampling noise is expected under any resampling-based synthesizer, so
#: the finding is recorded but does not force human review below this line.
_VANISHED_MIN_SUPPORT = 0.01


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticAuditConfig:
    """Validated configuration for a synthetic-data audit.

    ``truth``/``pred``/``groups`` name the columns used by the bias axis
    (the same roles as ``opsaudit audit``). ``numeric`` and ``categorical``
    name the columns used by the fidelity and privacy axes; when omitted
    they are inferred from dtypes. ``exclude`` names id-like columns that
    are skipped by the fidelity and privacy axes.

    Example:
        >>> cfg = SyntheticAuditConfig.from_dict({"truth": "y", "pred": "p",
        ...     "groups": ["g"]})
        >>> cfg.truth
        'y'
    """

    truth: str
    pred: str
    groups: list[str]
    numeric: list[str] | None = None
    categorical: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SyntheticAuditConfig":
        """Validate a config mapping, raising ``ValueError`` on problems."""
        if not isinstance(raw, Mapping):
            raise ValueError("synthetic-audit config must be a mapping")
        for key in ("truth", "pred"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"config must name a {key!r} column")
        groups = raw.get("groups")
        if isinstance(groups, str):
            groups = [groups]
        if not isinstance(groups, (list, tuple)) or not groups or not all(
            isinstance(g, str) and g for g in groups
        ):
            raise ValueError("config 'groups' must be a non-empty list of column names")
        thresholds = _merge_synthetic_thresholds(raw.get("thresholds"))
        seed = raw.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("config 'seed' must be an integer")

        def _column_list(name: str) -> list[str] | None:
            value = raw.get(name)
            if value is None:
                return None
            items = [value] if isinstance(value, str) else list(value)
            if not all(isinstance(item, str) and item for item in items):
                raise ValueError(f"config {name!r} must list column names")
            return items

        return cls(
            truth=raw["truth"],
            pred=raw["pred"],
            groups=list(groups),
            numeric=_column_list("numeric"),
            categorical=_column_list("categorical"),
            exclude=_column_list("exclude") or [],
            thresholds=thresholds,
            seed=seed,
        )


def _merge_synthetic_thresholds(
    overrides: Mapping[str, Any] | None,
) -> dict[str, dict[str, float]]:
    """Merge threshold overrides onto defaults, validating every value."""
    merged = {axis: dict(values) for axis, values in _DEFAULT_THRESHOLDS.items()}
    if overrides is None:
        return merged
    if not isinstance(overrides, Mapping):
        raise ValueError("config 'thresholds' must be a mapping of axis to rules")
    for axis, rules in overrides.items():
        if axis not in merged:
            raise ValueError(f"unknown threshold axis: {axis!r}")
        if not isinstance(rules, Mapping):
            raise ValueError(f"thresholds for {axis!r} must be a mapping")
        for name, value in rules.items():
            if name not in merged[axis]:
                raise ValueError(f"unknown threshold {name!r} for axis {axis!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"threshold {axis}.{name} must be a number, got {value!r}"
                )
            merged[axis][name] = float(value)
    return merged


def _resolve_columns(
    source: pd.DataFrame, synthetic: pd.DataFrame, config: SyntheticAuditConfig
) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) columns shared by both frames.

    Raises ``ValueError`` when the frames share no usable columns or when a
    configured column is missing.
    """
    shared = [c for c in source.columns if c in synthetic.columns]
    excluded = set(config.exclude)
    for name in ("numeric", "categorical"):
        for column in getattr(config, name) or []:
            if column not in shared:
                raise ValueError(
                    f"configured {name} column {column!r} is not present in both datasets"
                )
    numeric = (
        list(config.numeric)
        if config.numeric is not None
        else [c for c in shared if _is_numeric(source[c]) and c not in excluded]
    )
    categorical = (
        list(config.categorical)
        if config.categorical is not None
        else [
            c
            for c in shared
            if c not in numeric and c not in excluded and c in shared
        ]
    )
    numeric = [c for c in numeric if c not in excluded]
    categorical = [c for c in categorical if c not in excluded]
    if not numeric and not categorical:
        raise ValueError("no usable columns shared by source and synthetic datasets")
    return numeric, categorical


def _is_numeric(series: pd.Series) -> bool:
    """Return True for real numeric dtypes (bools count as categorical)."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


# ---------------------------------------------------------------------------
# Axis 1 — statistical fidelity
# ---------------------------------------------------------------------------

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF difference).

    Implemented with NumPy only (no SciPy dependency).

    Example:
        >>> ks_statistic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        0.0
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("ks_statistic requires non-empty inputs")
    grid = np.concatenate([a, b])
    grid.sort(kind="mergesort")
    cdf_a = np.searchsorted(a, grid, side="right", sorter=np.argsort(a)) / len(a)
    cdf_b = np.searchsorted(b, grid, side="right", sorter=np.argsort(b)) / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def total_variation_distance(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two categorical distributions.

    The category support is the union of both series' values; unseen
    categories count as zero probability on that side.

    Example:
        >>> total_variation_distance(pd.Series(["x", "x"]), pd.Series(["x", "x"]))
        0.0
    """
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    support = pa.index.union(pb.index)
    return float(0.5 * np.abs(pa.reindex(support, fill_value=0.0)
                             - pb.reindex(support, fill_value=0.0)).sum())


def audit_fidelity(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Measure statistical fidelity of ``synthetic`` against ``source``.

    Returns a JSON-safe dict with per-column similarities, an aggregate
    ``fidelity_score`` in [0, 1], correlation drift, marginal coverage, and
    a pass/review/fail verdict.

    Example:
        >>> import pandas as pd
        >>> src = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        >>> out = audit_fidelity(src, src.copy(), ["x"], [])
        >>> out["verdict"]
        'pass'
    """
    active = dict(_DEFAULT_THRESHOLDS["fidelity"])
    if thresholds:
        active.update(thresholds)

    column_scores: dict[str, dict[str, Any]] = {}
    similarities: list[float] = []
    for column in numeric:
        statistic = ks_statistic(
            source[column].to_numpy(dtype=float), synthetic[column].to_numpy(dtype=float)
        )
        similarity = 1.0 - statistic
        column_scores[column] = {
            "type": "numeric",
            "ks_statistic": statistic,
            "similarity": similarity,
        }
        similarities.append(similarity)
    for column in categorical:
        distance = total_variation_distance(
            source[column].astype(str), synthetic[column].astype(str)
        )
        similarity = 1.0 - distance
        entry: dict[str, Any] = {
            "type": "categorical",
            "total_variation_distance": distance,
            "similarity": similarity,
        }
        vanished = sorted(
            set(source[column].astype(str).unique())
            - set(synthetic[column].astype(str).unique())
        )
        if vanished:
            entry["vanished_categories"] = vanished
        column_scores[column] = entry
        similarities.append(similarity)

    fidelity_score = float(np.mean(similarities)) if similarities else 0.0
    correlation_drift = _correlation_drift(source, synthetic, numeric)
    coverage = _marginal_coverage(source, synthetic, numeric, categorical)

    findings: list[dict[str, Any]] = []
    for column, entry in column_scores.items():
        if entry["similarity"] <= 0.5:
            findings.append(
                {
                    "code": "low_column_fidelity",
                    "column": column,
                    "similarity": entry["similarity"],
                    "message": f"column {column!r} poorly preserved "
                    f"(similarity {entry['similarity']:.3f})",
                }
            )
        if entry.get("vanished_categories"):
            source_values = source[column].astype(str)
            support = {
                category: float((source_values == category).mean())
                for category in entry["vanished_categories"]
            }
            findings.append(
                {
                    "code": "vanished_categories",
                    "column": column,
                    "categories": entry["vanished_categories"],
                    "max_source_support": max(support.values()),
                    "message": f"column {column!r} lost categories present in source: "
                    f"{', '.join(entry['vanished_categories'])}",
                }
            )

    if fidelity_score >= active["pass_score"]:
        verdict = "pass"
    elif fidelity_score >= active["review_score"]:
        verdict = "review"
    else:
        verdict = "fail"
    if any(
        f["code"] == "vanished_categories"
        and f["max_source_support"] >= _VANISHED_MIN_SUPPORT
        for f in findings
    ):
        # Losing a whole well-represented category is a utility failure
        # even when the aggregate score looks fine: a downstream model
        # never sees it. (Singleton losses are sampling noise and do not
        # escalate.)
        if _VERDICT_ORDER[verdict] < _VERDICT_ORDER["review"]:
            verdict = "review"

    return {
        "verdict": verdict,
        "fidelity_score": fidelity_score,
        "n_columns": len(similarities),
        "columns": column_scores,
        "correlation_drift": correlation_drift,
        "marginal_coverage": coverage,
        "findings": findings,
    }


def _correlation_drift(
    source: pd.DataFrame, synthetic: pd.DataFrame, numeric: list[str]
) -> dict[str, Any] | None:
    """Mean absolute off-diagonal correlation difference, or None if <2 columns."""
    if len(numeric) < 2:
        return None
    src_corr = source[numeric].corr(numeric_only=True).to_numpy(dtype=float)
    syn_corr = synthetic[numeric].corr(numeric_only=True).to_numpy(dtype=float)
    mask = ~np.eye(len(numeric), dtype=bool)
    src_vals = np.nan_to_num(src_corr[mask], nan=0.0)
    syn_vals = np.nan_to_num(syn_corr[mask], nan=0.0)
    drift = float(np.mean(np.abs(src_vals - syn_vals)))
    return {"mean_abs_diff": drift, "normalized": drift / 2.0}


def _marginal_coverage(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, Any]:
    """Fraction of synthetic values inside the source's observed support."""
    coverages: list[float] = []
    for column in numeric:
        values = synthetic[column].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        lo = float(np.nanmin(source[column].to_numpy(dtype=float)))
        hi = float(np.nanmax(source[column].to_numpy(dtype=float)))
        coverages.append(float(np.mean((values >= lo) & (values <= hi))))
    for column in categorical:
        seen = set(source[column].astype(str).unique())
        values = synthetic[column].astype(str)
        if len(values) == 0:
            continue
        coverages.append(float(values.isin(seen).mean()))
    return {
        "mean_coverage": float(np.mean(coverages)) if coverages else 1.0,
        "n_columns": len(coverages),
    }


# ---------------------------------------------------------------------------
# Axis 2 — privacy (membership-inference attack surface)
# ---------------------------------------------------------------------------

def _encode_features(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode frames to a numeric matrix: z-scored numerics + ordinal codes.

    Scalers and category codes are fit on the source only, so a synthetic
    row landing exactly on a source row means a real match, not an
    encoding artifact.
    """
    parts_src: list[np.ndarray] = []
    parts_syn: list[np.ndarray] = []
    for column in numeric:
        values = source[column].to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        parts_src.append(((values - mean) / std).reshape(-1, 1))
        syn_values = synthetic[column].to_numpy(dtype=float)
        parts_syn.append(((np.nan_to_num(syn_values, nan=mean) - mean) / std).reshape(-1, 1))
    for column in categorical:
        codes, _ = pd.factorize(
            pd.concat([source[column].astype(str), synthetic[column].astype(str)],
                      ignore_index=True)
        )
        n_src = len(source)
        parts_src.append(codes[:n_src].astype(float).reshape(-1, 1))
        parts_syn.append(codes[n_src:].astype(float).reshape(-1, 1))
    if not parts_src:
        raise ValueError("privacy probe needs at least one feature column")
    return np.hstack(parts_src), np.hstack(parts_syn)


def _nearest_distances(
    query: np.ndarray, reference: np.ndarray, exclude_self: bool
) -> np.ndarray:
    """Row-wise Euclidean distance to the nearest reference row.

    Computed in chunks with the ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b``
    expansion so a 2000x2000 comparison needs ~32 MB, not ~320 MB.
    ``exclude_self`` masks the diagonal (query is reference).
    """
    chunk = 500
    out = np.empty(len(query))
    ref_sq = np.einsum("ij,ij->i", reference, reference)
    for start in range(0, len(query), chunk):
        block = query[start : start + chunk]
        dist_sq = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + ref_sq[None, :]
            - 2.0 * block @ reference.T
        )
        np.maximum(dist_sq, 0.0, out=dist_sq)
        if exclude_self:
            idx = np.arange(start, min(start + chunk, len(query)))
            dist_sq[np.arange(len(idx)), idx] = np.inf
        out[start : start + chunk] = np.sqrt(dist_sq.min(axis=1))
    return out


def audit_privacy(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    thresholds: dict[str, float] | None = None,
    seed: int = 42,
    sample_n: int = _PRIVACY_SAMPLE_N,
) -> dict[str, Any]:
    """Probe the membership-inference attack surface of ``synthetic``.

    A distance-based membership-inference attack scores candidate records by
    their distance to the nearest synthetic row: memorized source rows sit
    suspiciously close. This probe measures that signal directly — the
    ``privacy_risk`` in [0, 1] is 0 when synthetic rows are no closer to
    source rows than source rows are to each other, and approaches 1 as
    synthetic rows collapse onto source rows.

    Both frames are deterministically subsampled to ``sample_n`` rows for the
    pairwise computation.

    Example:
        >>> import pandas as pd
        >>> src = pd.DataFrame({"x": [float(i) for i in range(50)]})
        >>> out = audit_privacy(src, src.copy(), ["x"], [], seed=1)
        >>> out["verdict"]
        'fail'
    """
    active = dict(_DEFAULT_THRESHOLDS["privacy"])
    if thresholds:
        active.update(thresholds)

    rng = np.random.default_rng(seed)
    src_idx = rng.choice(len(source), size=min(sample_n, len(source)), replace=False)
    syn_idx = rng.choice(len(synthetic), size=min(sample_n, len(synthetic)), replace=False)
    src_sample = source.iloc[src_idx].reset_index(drop=True)
    syn_sample = synthetic.iloc[syn_idx].reset_index(drop=True)

    encoded_src, encoded_syn = _encode_features(src_sample, syn_sample, numeric, categorical)

    # Distance from each source row to its nearest synthetic neighbor...
    dist_src_syn = _nearest_distances(encoded_src, encoded_syn, exclude_self=False)
    # ...versus to its nearest *other* source row (the no-memorization baseline).
    dist_src_src = _nearest_distances(encoded_src, encoded_src, exclude_self=True)

    finite_src_src = dist_src_src[np.isfinite(dist_src_src)]
    if len(finite_src_src) == 0:
        raise ValueError("privacy probe needs at least two distinct source rows")
    median_src_src = float(np.median(finite_src_src))
    median_src_syn = float(np.median(dist_src_syn))

    if median_src_src == 0:
        # Degenerate feature space (e.g. very low cardinality): source rows
        # already sit on top of each other, so closeness carries no signal.
        ratio_risk = 0.0
    elif median_src_syn == 0:
        ratio_risk = 1.0  # synthetic rows sit exactly on source rows
    else:
        ratio = median_src_src / median_src_syn
        ratio_risk = float(np.clip(1.0 - 1.0 / ratio, 0.0, 1.0))

    # Memorization: source rows closer to synthetic than 95% of source-source
    # nearest distances are suspiciously close.
    close_cutoff = float(np.percentile(finite_src_src, 5))
    memorization_rate = float(np.mean(dist_src_syn < close_cutoff))

    # Exact duplicates: synthetic rows identical to a source row on all features.
    # Chance collisions are normal in low-cardinality feature spaces, so only
    # the *excess* over the source's own self-collision rate is suspicious.
    src_tuples = [tuple(row) for row in encoded_src.tolist()]
    syn_tuples = [tuple(row) for row in encoded_syn.tolist()]
    src_tuple_set = set(src_tuples)
    duplicate_rate = float(np.mean([t in src_tuple_set for t in syn_tuples]))
    src_counts = Counter(src_tuples)
    source_self_duplicate_rate = float(
        np.mean([src_counts[t] > 1 for t in src_tuples])
    )
    excess_duplicate_rate = float(
        max(0.0, duplicate_rate - source_self_duplicate_rate)
    )

    privacy_risk = float(max(ratio_risk, memorization_rate, excess_duplicate_rate))

    findings: list[dict[str, Any]] = []
    if ratio_risk > active["review_risk"]:
        # The distance-ratio signal is the core of a distance-based
        # membership-inference attack: it deserves its own finding so a
        # fail/review verdict is never unexplained.
        if median_src_syn > 0:
            detail = (
                f"synthetic rows sit {median_src_src / median_src_syn:.1f}x "
                "closer to source rows than source rows sit to each other"
            )
        else:
            detail = "synthetic rows sit exactly on source rows"
        findings.append(
            {
                "code": "close_synthetic_neighbors",
                "distance_ratio_risk": ratio_risk,
                "median_source_to_synthetic": median_src_syn,
                "median_source_to_source": median_src_src,
                "message": detail
                + " — the distance signal a membership-inference attack exploits",
            }
        )
    if excess_duplicate_rate > 0:
        findings.append(
            {
                "code": "exact_duplicates",
                "duplicate_rate": duplicate_rate,
                "source_self_duplicate_rate": source_self_duplicate_rate,
                "excess_duplicate_rate": excess_duplicate_rate,
                "message": f"{excess_duplicate_rate:.1%} of sampled synthetic rows are "
                "exact copies of source rows beyond chance collisions",
            }
        )
    if memorization_rate > 0.10:
        findings.append(
            {
                "code": "high_memorization",
                "memorization_rate": memorization_rate,
                "message": f"{memorization_rate:.1%} of source rows have a "
                "suspiciously close synthetic neighbor",
            }
        )
    if source_self_duplicate_rate > 0.5:
        # The feature space is too small for distance/duplicate signals to
        # discriminate memorization from chance: most source rows already
        # collide with another source row. Record the caveat so a privacy
        # pass is read as weak evidence, not a clean bill of health.
        findings.append(
            {
                "code": "low_feature_cardinality",
                "source_self_duplicate_rate": source_self_duplicate_rate,
                "message": f"{source_self_duplicate_rate:.0%} of source rows collide "
                "with another source row — the privacy probe has limited power "
                "in this feature space; treat a pass as weak evidence",
            }
        )

    if privacy_risk <= active["pass_risk"]:
        verdict = "pass"
    elif privacy_risk <= active["review_risk"]:
        verdict = "review"
    else:
        verdict = "fail"

    return {
        "verdict": verdict,
        "privacy_risk": privacy_risk,
        "distance_ratio_risk": ratio_risk,
        "memorization_rate": memorization_rate,
        "duplicate_rate": duplicate_rate,
        "source_self_duplicate_rate": source_self_duplicate_rate,
        "excess_duplicate_rate": excess_duplicate_rate,
        "median_source_to_synthetic": median_src_syn,
        "median_source_to_source": median_src_src,
        "sample_n": int(min(sample_n, len(source))),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Axis 3 — bias amplification
# ---------------------------------------------------------------------------

#: Disparity metrics compared between source and synthetic audits.
_BIAS_METRICS = (
    "disparate_impact_ratio",
    "demographic_parity_diff",
    "tpr_gap",
    "fpr_gap",
)

#: Metrics where a larger value means a larger disparity.
_HIGHER_WORSE = {"demographic_parity_diff", "tpr_gap", "fpr_gap"}


def _group_labels(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build one group-label series, combining repeated group columns."""
    if len(columns) == 1:
        return frame[columns[0]].astype(str)
    return frame.loc[:, columns].astype(str).apply(
        lambda row: " | ".join(f"{column}={row[column]}" for column in columns), axis=1
    )


def audit_bias_amplification(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    truth: str,
    pred: str,
    group_columns: list[str],
    thresholds: dict[str, float] | None = None,
    bootstrap: int = 200,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Compare opsaudit disparity metrics on source vs. synthetic data.

    Runs the real :func:`opsaudit.metrics.audit_disparities` on both frames
    with identical roles. ``amplification_factor`` measures how the
    disparate-impact disparity ``(1 - DI)`` changed: >1 means the synthetic
    data amplified the source's disparity, <1 means it dampened it.

    Both audits use deterministic bootstrap confidence intervals
    (``bootstrap`` resamples, ``bootstrap_seed`` seed). A verdict only fires
    when the point-estimate threshold is breached *and* the 95% intervals are
    separated in the worse direction — gap statistics are noisy in small
    samples, and the auditor must not mistake sampling noise for new bias.
    Pass ``bootstrap=0`` to fall back to point-estimate-only rules.

    Example:
        >>> import pandas as pd
        >>> from opsaudit.data import generate_dispatch
        >>> src = generate_dispatch(n=600, bias_strength=0.0, seed=11)
        >>> syn = generate_dispatch(n=600, bias_strength=0.0, seed=12)
        >>> out = audit_bias_amplification(src, syn, "on_time",
        ...     "priority_route", ["group"])
        >>> out["verdict"] in ("pass", "review")
        True
    """
    active = dict(_DEFAULT_THRESHOLDS["bias"])
    if thresholds:
        active.update(thresholds)

    for frame, name in ((source, "source"), (synthetic, "synthetic")):
        missing = [c for c in [truth, pred, *group_columns] if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} dataset is missing column(s): {', '.join(missing)}")

    result_src = audit_disparities(
        source[truth], source[pred], _group_labels(source, group_columns),
        bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
    )
    result_syn = audit_disparities(
        synthetic[truth], synthetic[pred], _group_labels(synthetic, group_columns),
        bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
    )

    comparisons: dict[str, dict[str, Any]] = {}
    for metric in _BIAS_METRICS:
        value_src = getattr(result_src, metric)
        value_syn = getattr(result_syn, metric)
        entry: dict[str, Any] = {"source": value_src, "synthetic": value_syn}
        if value_src is None or value_syn is None:
            entry["delta"] = None
            entry["note"] = "undefined on at least one side"
        else:
            entry["delta"] = float(value_syn - value_src)
            entry["gap_growth"] = (
                float(value_syn - value_src) if metric in _HIGHER_WORSE else None
            )
        comparisons[metric] = entry

    di_src = result_src.disparate_impact_ratio
    di_syn = result_syn.disparate_impact_ratio
    disparity_src = 1.0 - di_src
    disparity_syn = 1.0 - di_syn
    if di_src >= active["source_fair_di"]:
        # Source is (essentially) fair: ratios against a ~zero disparity are
        # meaningless, so judge the synthetic side in absolute terms.
        amplification_factor = None
        introduced = disparity_syn
    else:
        amplification_factor = float(disparity_syn / disparity_src)
        introduced = 0.0

    findings: list[dict[str, Any]] = []
    verdict = "pass"

    def separated(metric: str, worse_higher: bool) -> bool:
        """True when the 95% bootstrap intervals are separated in the worse direction."""
        if bootstrap == 0:
            return True  # no intervals available; judge on point estimates
        ci_src = result_src.confidence_intervals.get(metric)
        ci_syn = result_syn.confidence_intervals.get(metric)
        if not ci_src or not ci_syn:
            return True  # metric undefined on some side; judge on point estimates
        if worse_higher:
            return bool(ci_syn["lower"] > ci_src["upper"])
        return bool(ci_syn["upper"] < ci_src["lower"])

    if (
        amplification_factor is not None
        and amplification_factor > active["max_amplification"]
        and separated("disparate_impact_ratio", worse_higher=False)
    ):
        verdict = "fail"
        findings.append(
            {
                "code": "bias_amplified",
                "amplification_factor": amplification_factor,
                "message": f"synthetic data amplified disparate-impact disparity "
                f"x{amplification_factor:.2f} vs. source",
            }
        )
    if amplification_factor is None:
        # Source was fair: a real disparity in the synthetic data is
        # introduced, not amplified.
        if di_syn < active["introduced_disparity_di"] and separated(
            "disparate_impact_ratio", worse_higher=False
        ):
            verdict = "fail"
            findings.append(
                {
                    "code": "disparity_introduced",
                    "source_di": di_src,
                    "synthetic_di": di_syn,
                    "message": f"source was fair (DI {di_src:.3f}) but synthetic data "
                    f"introduces disparity (DI {di_syn:.3f})",
                }
            )
        elif di_syn < active["review_introduced_di"] and separated(
            "disparate_impact_ratio", worse_higher=False
        ):
            verdict = "review"
            findings.append(
                {
                    "code": "disparity_drift",
                    "source_di": di_src,
                    "synthetic_di": di_syn,
                    "message": f"source was fair (DI {di_src:.3f}); synthetic DI "
                    f"{di_syn:.3f} drifted down — human review advised",
                }
            )
    max_gap_growth = max(
        (
            comparisons[m]["gap_growth"]
            for m in _HIGHER_WORSE
            if comparisons[m]["gap_growth"] is not None
        ),
        default=0.0,
    )
    worst_gap_metric = max(
        _HIGHER_WORSE,
        key=lambda m: comparisons[m]["gap_growth"]
        if comparisons[m]["gap_growth"] is not None
        else float("-inf"),
    )
    if max_gap_growth > active["max_gap_growth"] and separated(
        worst_gap_metric, worse_higher=True
    ):
        verdict = "fail"
        findings.append(
            {
                "code": "gap_grew",
                "max_gap_growth": max_gap_growth,
                "message": f"a disparity gap grew by {max_gap_growth:.3f} "
                "in the synthetic data",
            }
        )
    if verdict == "pass":
        if (
            amplification_factor is not None
            and amplification_factor > active["review_amplification"]
            and separated("disparate_impact_ratio", worse_higher=False)
        ) or (
            max_gap_growth > active["review_gap_growth"]
            and separated(worst_gap_metric, worse_higher=True)
        ):
            verdict = "review"
            findings.append(
                {
                    "code": "bias_drift",
                    "amplification_factor": amplification_factor,
                    "max_gap_growth": max_gap_growth,
                    "message": "disparity drifted upward in the synthetic data; "
                    "human review advised",
                }
            )
    if verdict == "pass" and amplification_factor is not None and amplification_factor < 0.9:
        findings.append(
            {
                "code": "bias_dampened",
                "amplification_factor": amplification_factor,
                "message": "synthetic data dampened the source disparity "
                "(check it did not erase real signal)",
            }
        )

    return {
        "verdict": verdict,
        "amplification_factor": amplification_factor,
        "introduced_disparity": introduced if amplification_factor is None else 0.0,
        "metrics": comparisons,
        "source_audit": result_src.to_dict(),
        "synthetic_audit": result_syn.to_dict(),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------

def audit_synthetic(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: SyntheticAuditConfig | Mapping[str, Any],
    bootstrap: int = 200,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Audit ``synthetic`` against ``source`` on all three axes.

    Returns a JSON-safe report dict with per-axis results and an overall
    verdict: ``fail`` if any axis fails, ``review`` if any axis needs
    review, otherwise ``pass``. ``bootstrap``/``bootstrap_seed`` control the
    deterministic confidence intervals behind the bias axis (0 disables
    them; the privacy axis always uses ``config``'s seed).

    Example:
        >>> import pandas as pd
        >>> src = pd.DataFrame({"y": [1, 0, 1, 0], "g": ["a", "a", "b", "b"]})
        >>> audit_synthetic(src, src.copy(),
        ...     {"truth": "y", "pred": "y", "groups": ["g"]},
        ...     bootstrap=0)["axes"]["fidelity"]["verdict"]
        'pass'
    """
    cfg = config if isinstance(config, SyntheticAuditConfig) else SyntheticAuditConfig.from_dict(config)
    if len(source) == 0 or len(synthetic) == 0:
        raise ValueError("source and synthetic datasets must not be empty")
    numeric, categorical = _resolve_columns(source, synthetic, cfg)

    fidelity = audit_fidelity(source, synthetic, numeric, categorical,
                              cfg.thresholds["fidelity"])
    privacy = audit_privacy(source, synthetic, numeric, categorical,
                            cfg.thresholds["privacy"], seed=cfg.seed)
    bias = audit_bias_amplification(source, synthetic, cfg.truth, cfg.pred,
                                    cfg.groups, cfg.thresholds["bias"],
                                    bootstrap=bootstrap,
                                    bootstrap_seed=bootstrap_seed)

    axes = {"fidelity": fidelity, "privacy": privacy, "bias": bias}
    overall = "pass"
    for axis in AXES:
        if _VERDICT_ORDER[axes[axis]["verdict"]] > _VERDICT_ORDER[overall]:
            overall = axes[axis]["verdict"]

    return {
        "verdict": overall,
        "opsaudit_version": __version__,
        "axes": axes,
        "columns": {"numeric": numeric, "categorical": categorical},
        "n_source": int(len(source)),
        "n_synthetic": int(len(synthetic)),
    }

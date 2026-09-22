"""Phase 2 demo: probe generators end to end.

Exercises the three deterministic generators against a tabular target
wrapping a toy classifier with a *planted* disparity (temp workers with
< 24 months tenure are denied shifts):

1. ``counterfactual`` — symmetric group-swapped pairs; the deterministic
   ``audit_disparities`` must catch the planted gap.
2. ``adversarial`` — extreme numerics / unseen categories / nulls; the
   target must not crash.
3. ``metamorphic`` — field reordering must not change the decision.

Every probe, response, and computed metric is recorded in an
``EvidenceLog`` (``phase2_evidence.jsonl`` next to this script), and a
summary table is printed at the end.

Usage:
    python examples/agentic-audit-demo/phase2_demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a core dependency
    sys.exit("phase2 demo needs pandas: pip install -e .")

from opsaudit import EvidenceLog, audit_disparities
from opsaudit.probes import (
    check_symmetry,
    evaluate_metamorphic,
    generate_adversarial,
    generate_counterfactuals,
    generate_metamorphic,
    BUILTIN_RELATIONS,
)
from opsaudit.targets import TabularTarget

HERE = Path(__file__).resolve().parent
COLUMNS = ["tenure_months", "shifts_requested", "group"]


class BiasedGrants:
    """Toy classifier with a planted disparity: temp workers with short
    tenure are denied shifts everyone else gets."""

    def predict(self, X):
        out = []
        for _, row in X.iterrows():
            if row["group"] == "temp" and row["tenure_months"] < 24:
                out.append(0)
            else:
                out.append(1)
        return out


def _frame(rows):
    return pd.DataFrame(rows)[COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    target = TabularTarget(BiasedGrants(), name="biased-grants-demo")
    log = EvidenceLog(HERE / "phase2_evidence.jsonl", target=target)
    summary: list[tuple[str, str]] = []

    # 1. Counterfactual battery ------------------------------------------------
    base = {"tenure_months": 6, "shifts_requested": 4, "group": "FT"}
    cf = generate_counterfactuals(base, {"group": ["FT", "PT", "temp"]})
    ok, problems = check_symmetry(cf)
    assert ok, f"symmetry broken: {problems}"
    preds = target.predict(_frame(cf.rows()))
    log.record(cf.to_event())
    log.record({"kind": "response_batch", "n": len(preds)})

    y_true = [1] * len(preds)  # everyone "deserved" the shifts
    groups = [p.attributes["group"] for p in cf]
    result = audit_disparities(y_true, preds, groups, min_group_n=1)
    log.record(
        {
            "kind": "metric_computed",
            "tpr_gap": result.tpr_gap,
            "disparate_impact_ratio": result.disparate_impact_ratio,
            "n_flags": len(result.flags),
        }
    )
    by_group = {g.group: g.tpr for g in result.groups}
    summary.append(
        (
            "counterfactual",
            f"{len(cf)} probes, symmetry OK; grant rate by group {by_group}; "
            f"TPR gap {result.tpr_gap:.2f} -> "
            f"{'DISPARITY FOUND' if result.flags else 'no flag'}",
        )
    )

    # 2. Adversarial battery ---------------------------------------------------
    adv = generate_adversarial(target_type="tabular", base_row=base)
    adv_preds = target.predict(_frame(adv.rows()))
    log.record(adv.to_event())
    log.record({"kind": "response_batch", "n": len(adv_preds)})
    crashed = len(adv_preds) != len(adv)
    summary.append(
        (
            "adversarial",
            f"{len(adv)} probes, target returned {len(adv_preds)} outputs "
            f"-> {'CRASH' if crashed else 'no crash, graceful degradation'}",
        )
    )

    # 3. Metamorphic battery ---------------------------------------------------
    mm = generate_metamorphic(
        [base],
        relations=[r for r in BUILTIN_RELATIONS if r.name == "reorder_fields"],
    )
    outputs = {}
    for probe in mm:
        outputs[probe.id] = target.predict(_frame([probe.payload]))[0]
    verdicts = evaluate_metamorphic(mm, outputs)
    log.record(mm.to_event())
    log.record({"kind": "response_batch", "n": len(outputs)})
    passed = sum(1 for v in verdicts if v["passed"])
    summary.append(
        (
            "metamorphic",
            f"{len(verdicts)} relation checks, {passed} passed "
            f"-> {'STABLE' if passed == len(verdicts) else 'VIOLATION'}",
        )
    )

    print(f"\nphase2 demo — seed {args.seed}")
    print(f"{'generator':<15} {'result'}")
    print("-" * 78)
    for name, line in summary:
        print(f"{name:<15} {line}")
    print(f"\nevidence: {HERE / 'phase2_evidence.jsonl'} ({len(log)} events)")

    if not result.flags:
        print("UNEXPECTED: planted disparity was not flagged")
        return 1
    print("\nOK: planted disparity caught by deterministic metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

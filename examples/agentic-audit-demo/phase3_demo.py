"""Phase 3 demo: the agentic planner loop end to end.

A scripted (deterministic, no credentials) planner brain runs a 3-round
adaptive campaign against a tabular target wrapping a toy classifier
with a *planted* disparity (temp workers with < 24 months tenure are
denied shifts):

1. Round 1 — broad counterfactual sweep over ``group`` (FT/PT/temp).
   The deterministic summary finds the gap on ``temp``.
2. Round 2 — the planner *drills down*: counterfactuals on just the
   FT-vs-temp pair driving the gap.
3. Round 3 — adversarial robustness sweep (extreme numerics, unseen
   categories, nulls); the target must not crash.

Then the planner calls stop. Every plan, probe batch, observation, and
round summary is recorded in an ``EvidenceLog``
(``phase3_evidence.jsonl`` next to this script), and the adaptive
drill-down is printed.

Usage:
    python examples/agentic-audit-demo/phase3_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from opsaudit.agents import AuditCampaign, AuditPlanner, Budget
from opsaudit.targets import TabularTarget, Target

HERE = Path(__file__).resolve().parent
COLUMNS = ["tenure_months", "shifts_requested", "group"]


class ScriptedPlannerBrain(Target):
    """Deterministic stand-in for the planner LLM.

    Returns one scripted JSON plan per call so the demo is reproducible
    without any model credentials. A real deployment would point
    ``AuditPlanner`` at an Ollama Cloud chat model instead.
    """

    name = "scripted-planner-brain"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompts):
        self.calls += 1
        plans = [
            {
                "action": "probe",
                "generator": "counterfactual",
                "params": {"attributes": {"group": ["FT", "PT", "temp"]}},
                "reason": "broad sweep: which groups diverge?",
            },
            {
                "action": "probe",
                "generator": "counterfactual",
                "params": {"attributes": {"group": ["FT", "temp"]}},
                "reason": "gap concentrated on temp; drill into FT-vs-temp",
            },
            {
                "action": "probe",
                "generator": "adversarial",
                "params": {},
                "reason": "robustness check on the audited target",
            },
            {"action": "stop", "reason": "disparity confirmed and target robust; done"},
        ]
        plan = plans[min(self.calls - 1, len(plans) - 1)]
        return [json.dumps(plan) for _ in prompts]

    def describe(self):
        return {"target_type": "planner", "name": self.name}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    brief = {
        "target_type": "tabular",
        "protected_attributes": {"group": ["FT", "PT", "temp"]},
        "risk_areas": ["shift allocation"],
        "base_input": {
            "tenure_months": 6,
            "shifts_requested": 4,
            "group": "FT",
        },
    }
    budget = Budget(max_probes=200, max_rounds=5, flat_rounds=0)
    planner = AuditPlanner(brief, ScriptedPlannerBrain(), budget=budget)
    campaign = AuditCampaign(
        audited_target=TabularTarget(BiasedGrants(), name="biased-grants-demo"),
        planner=planner,
        evidence_path=HERE / "phase3_evidence.jsonl",
        seed=args.seed,
    )

    print("=== Phase 3 demo: adaptive audit campaign ===\n")
    report = campaign.run()

    for finding in report.findings:
        details = finding["details"]
        print(
            f"round {finding['round']}: {finding['generator']} "
            f"({finding['n_probes']} probes) "
            f"strength={finding['strength']:.3f}"
        )
        for attr, gap in sorted(details.get("gaps", {}).items()):
            rates = details["outcome_rates"][attr]
            print(f"    {attr}: gap={gap:.3f} rates={rates}")
        if "error_rate" in details:
            print(
                f"    adversarial errors: {details['n_errors']} "
                f"(rate={details['error_rate']:.2f})"
            )
    print()
    print(f"rounds:        {report.rounds}")
    print(f"probes used:   {report.probes_used}")
    print(f"best strength: {report.best_strength:.3f}")
    print(f"stop reason:   {report.stop_reason}")
    print(f"evidence:      {report.evidence_path} ({report.n_events} events)")

    strongest = max(report.findings, key=lambda f: f["strength"])
    assert strongest["strength"] > 0, "demo should find the planted disparity"
    assert report.stop_reason.startswith("planner_stop")
    print("\nDEMO OK: planner adaptively drilled into the temp-vs-FT gap "
          "and stopped on its own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

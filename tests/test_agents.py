"""Tests for the Phase 3 agentic planner loop.

No network anywhere: the planner "LLM" is a scripted fake Target and the
audited targets are deterministic toys. These tests prove the adaptive
behavior (drill-down), the stopping rules, reproducibility, caching,
and graceful handling of malformed planner output.
"""

from __future__ import annotations

import json

import pytest

from opsaudit import EvidenceLog
from opsaudit.agents import (
    AuditCampaign,
    AuditPlanner,
    Budget,
    BudgetTracker,
    ProbeSpec,
    ResponseCache,
)
from opsaudit.agents.campaign import _cap_batch
from opsaudit.probes import generate_counterfactuals
from opsaudit.targets import TabularTarget, Target


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class ScriptedPlannerTarget(Target):
    """A planner brain that returns scripted JSON responses in order."""

    name = "scripted-planner"

    def __init__(self, script):
        # script: list of raw response strings (JSON or garbage)
        self.script = list(script)
        self.calls = 0

    def generate(self, prompts):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return [self.script[idx] for _ in prompts]

    def describe(self):
        return {"target_type": "planner", "name": self.name}


def _probe_spec(generator, params, reason="test"):
    return json.dumps(
        {
            "action": "probe",
            "generator": generator,
            "params": params,
            "reason": reason,
        }
    )


_STOP = json.dumps({"action": "stop", "reason": "done for test"})


class BiasedGrants:
    """Planted disparity: temp workers with < 24 months tenure denied."""

    def predict(self, X):
        out = []
        for _, row in X.iterrows():
            if row["group"] == "temp" and row["tenure_months"] < 24:
                out.append(0)
            else:
                out.append(1)
        return out


class ConstantGrants:
    """No disparity at all."""

    def predict(self, X):
        return [1] * len(X)


class CountingTarget(Target):
    """Wraps a predict callable and counts underlying calls."""

    name = "counting"

    def __init__(self, fn):
        self.fn = fn
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return self.fn(X)

    def describe(self):
        return {"target_type": "tabular", "name": self.name}


class ExplodingTarget(Target):
    """Raises on extreme numeric inputs."""

    name = "exploding"

    def predict(self, X):
        for _, row in X.iterrows():
            if abs(float(row["tenure_months"])) > 1e6:
                raise ValueError("tenure out of range")
        return [1] * len(X)

    def describe(self):
        return {"target_type": "tabular", "name": self.name}


BRIEF = {
    "target_type": "tabular",
    "protected_attributes": {"group": ["FT", "PT", "temp"]},
    "risk_areas": ["shift allocation"],
    "base_input": {"tenure_months": 6, "shifts_requested": 4, "group": "FT"},
}


def _campaign(planner_script, audited, tmp_path, budget=None, seed=7):
    planner_target = ScriptedPlannerTarget(planner_script)
    planner = AuditPlanner(
        dict(BRIEF), planner_target, budget=budget or Budget()
    )
    campaign = AuditCampaign(
        audited_target=audited,
        planner=planner,
        evidence_path=tmp_path / "campaign.jsonl",
        seed=seed,
    )
    return campaign.run(), planner_target


# ----------------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------------
def test_budget_validation():
    with pytest.raises(ValueError):
        Budget(max_probes=0)
    with pytest.raises(ValueError):
        Budget(max_rounds=0)
    with pytest.raises(ValueError):
        Budget(max_cost=-1.0)


def test_budget_exhaustion_reasons():
    t = BudgetTracker(Budget(max_probes=4, max_rounds=3, max_cost=1.0,
                             probe_cost=0.5))
    assert t.exhausted() == (False, "ok")
    t.record_probes(4)
    assert t.exhausted() == (True, "max_probes")

    t = BudgetTracker(Budget(max_rounds=2))
    t.record_round(0.1)
    t.record_round(0.2)
    assert t.exhausted() == (True, "max_rounds")

    t = BudgetTracker(Budget(max_cost=1.0, probe_cost=0.6))
    t.record_probes(2)
    assert t.exhausted() == (True, "max_cost")

    assert t.remaining_probes() == 198
    assert t.remaining_rounds() == 10


def test_budget_flatness():
    t = BudgetTracker(Budget(flat_rounds=2))
    t.record_round(0.5)
    assert not t.is_flat()  # need at least flat_rounds rounds
    t.record_round(0.5)
    # Window [0.5, 0.5] vs empty baseline: counts as the initial gain.
    assert not t.is_flat()
    t.record_round(0.5)
    assert t.is_flat()
    t.record_round(0.9)  # improvement resets the rule
    assert not t.is_flat()
    t.record_round(0.9)
    assert not t.is_flat()  # only one flat round so far
    t.record_round(0.9)
    assert t.is_flat()

    t = BudgetTracker(Budget(flat_rounds=0))
    t.record_round(0.1)
    t.record_round(0.1)
    assert not t.is_flat()


# ----------------------------------------------------------------------
# Planner spec validation
# ----------------------------------------------------------------------
def test_planner_accepts_valid_spec():
    target = ScriptedPlannerTarget(
        [_probe_spec("counterfactual", {"attributes": {"group": ["FT", "temp"]}})]
    )
    planner = AuditPlanner(dict(BRIEF), target)
    spec, meta = planner.propose_spec()
    assert spec.action == "probe"
    assert spec.generator == "counterfactual"
    assert spec.params["attributes"] == {"group": ["FT", "temp"]}
    assert not meta["cache_hit"]


def test_planner_rejects_unknown_generator_gracefully():
    target = ScriptedPlannerTarget(
        [_probe_spec("telepathy", {"attributes": {"group": ["FT", "temp"]}})]
    )
    planner = AuditPlanner(dict(BRIEF), target)
    spec, _ = planner.propose_spec()
    assert spec.action == "stop"
    assert "planner_invalid_spec" in spec.reason


def test_planner_malformed_json_stops_gracefully():
    target = ScriptedPlannerTarget(["this is not json {"])
    planner = AuditPlanner(dict(BRIEF), target)
    spec, _ = planner.propose_spec()
    assert spec.action == "stop"
    assert "planner_invalid_spec" in spec.reason


def test_planner_prompt_uses_real_attribute_names():
    # Regression: the prompt once showed a literal {"attr": ["A","B"]}
    # placeholder, which real LLMs copied verbatim into their specs.
    planner = AuditPlanner(dict(BRIEF), ScriptedPlannerTarget(["{}"]))
    prompt = planner._plan_prompt([], 10, 3)
    assert '"group"' in prompt  # real attribute name from the brief
    assert '"attr"' not in prompt  # no literal placeholder left


def test_planner_rejects_unknown_counterfactual_attribute_gracefully():
    # A syntactically valid spec naming an attribute that is not a key of
    # base_input must stop the campaign with a logged reason, not crash
    # the generator mid-run.
    target = ScriptedPlannerTarget(
        [_probe_spec("counterfactual", {"attributes": {"attr": ["A", "B"]}})]
    )
    planner = AuditPlanner(dict(BRIEF), target)
    spec, _ = planner.propose_spec()
    assert spec.action == "stop"
    assert "planner_invalid_spec" in spec.reason
    assert "base_input" in spec.reason


def test_planner_rejects_invented_counterfactual_values_gracefully():
    # Regression (2026-09-24): a planner LLM invented race/age-bin values
    # the attribute-name check could not catch; the sklearn target silently
    # routed the resulting NaNs and polluted pooled metrics. Invented
    # values must stop the campaign with a logged reason before any probe
    # executes.
    target = ScriptedPlannerTarget(
        [
            _probe_spec(
                "counterfactual",
                {"attributes": {"group": ["FT", "Martian"]}},
            )
        ]
    )
    planner = AuditPlanner(dict(BRIEF), target)
    spec, _ = planner.propose_spec()
    assert spec.action == "stop"
    assert "planner_invalid_spec" in spec.reason
    assert "Martian" in spec.reason
    assert "allowed values" in spec.reason


def test_planner_accepts_subset_of_allowed_values():
    target = ScriptedPlannerTarget(
        [_probe_spec("counterfactual", {"attributes": {"group": ["PT", "temp"]}})]
    )
    planner = AuditPlanner(dict(BRIEF), target)
    spec, _ = planner.propose_spec()
    assert spec.action == "probe"
    assert spec.params["attributes"] == {"group": ["PT", "temp"]}


def test_planner_skips_value_check_for_unlisted_attribute():
    # Attributes with no allowlist in the brief keep the legacy behavior:
    # names are validated, values are not. This documents the boundary of
    # the value check rather than silently extending it.
    target = ScriptedPlannerTarget(
        [_probe_spec("counterfactual", {"attributes": {"tenure_months": [6, 600]}})]
    )
    planner = AuditPlanner(dict(BRIEF), target)
    spec, _ = planner.propose_spec()
    assert spec.action == "probe"


def test_planner_honors_attribute_values_allowlist():
    brief = dict(BRIEF)
    brief["attribute_values"] = {"tenure_months": [6, 12, 24]}
    target = ScriptedPlannerTarget(
        [_probe_spec("counterfactual", {"attributes": {"tenure_months": [6, 600]}})]
    )
    planner = AuditPlanner(brief, target)
    spec, _ = planner.propose_spec()
    assert spec.action == "stop"
    assert "planner_invalid_spec" in spec.reason
    assert "600" in spec.reason


def test_campaign_never_executes_probes_with_invented_values(tmp_path):
    # End-to-end: invented values stop the campaign before the audited
    # target sees a single probe.
    audited = CountingTarget(lambda X: [1] * len(X))
    report, _ = _campaign(
        [
            _probe_spec(
                "counterfactual",
                {"attributes": {"group": ["FT", "Martian"]}},
            )
        ],
        audited,
        tmp_path,
    )
    assert audited.predict_calls == 0
    assert report.stop_reason.startswith("planner_stop:planner_invalid_spec")


def test_planner_empty_response_retries_then_stops():
    target = ScriptedPlannerTarget(["", "   "])
    planner = AuditPlanner(dict(BRIEF), target)
    spec, meta = planner.propose_spec()
    assert spec.action == "stop"
    assert "planner_empty_response" in spec.reason
    assert meta["retried"]
    assert target.calls == 2  # exactly one retry


def test_planner_requires_text_target():
    with pytest.raises(ValueError):
        AuditPlanner(dict(BRIEF), TabularTarget(BiasedGrants()))
    with pytest.raises(ValueError):
        AuditPlanner({"target_type": "quantum"}, ScriptedPlannerTarget([]))


# ----------------------------------------------------------------------
# Campaign behavior
# ----------------------------------------------------------------------
def test_campaign_drills_down_and_finds_disparity(tmp_path):
    script = [
        _probe_spec(
            "counterfactual",
            {"attributes": {"group": ["FT", "PT", "temp"]}},
            reason="broad sweep over groups",
        ),
        _probe_spec(
            "counterfactual",
            {"attributes": {"group": ["FT", "temp"]}},
            reason="gap on temp; drill into temp vs FT",
        ),
        _STOP,
    ]
    report, planner_target = _campaign(
        script, TabularTarget(BiasedGrants(), name="biased"), tmp_path
    )
    assert report.rounds == 2
    assert report.stop_reason.startswith("planner_stop")
    # Round 1 found the planted gap: temp=0 vs FT/PT=1.
    gaps = report.findings[0]["details"]["gaps"]
    assert gaps["group"] == pytest.approx(1.0)
    assert report.best_strength == pytest.approx(1.0)
    assert report.probes_used > 0
    assert planner_target.calls == 3


def test_campaign_stops_on_probe_budget(tmp_path):
    script = [
        _probe_spec(
            "counterfactual",
            {"attributes": {"group": ["FT", "PT", "temp"]}},
            reason="keep probing",
        )
    ] * 5
    report, _ = _campaign(
        script,
        TabularTarget(BiasedGrants()),
        tmp_path,
        budget=Budget(max_probes=4, max_rounds=10),
    )
    assert report.stop_reason == "budget:max_probes"
    assert report.probes_used <= 4


def test_campaign_stops_on_round_budget(tmp_path):
    script = [_probe_spec("adversarial", {}, reason="keep probing")] * 5
    report, _ = _campaign(
        script,
        TabularTarget(ConstantGrants()),
        tmp_path,
        budget=Budget(max_probes=1000, max_rounds=2, flat_rounds=0),
    )
    assert report.stop_reason == "budget:max_rounds"
    assert report.rounds == 2


def test_campaign_flat_findings_stop(tmp_path):
    script = [_probe_spec("adversarial", {}, reason="keep probing")] * 5
    report, _ = _campaign(
        script,
        TabularTarget(ConstantGrants()),
        tmp_path,
        budget=Budget(max_probes=1000, max_rounds=10, flat_rounds=2),
    )
    assert report.stop_reason.startswith("flat_findings")
    assert report.rounds == 2
    assert report.best_strength == 0.0


def test_campaign_planner_stop_first_round(tmp_path):
    report, _ = _campaign(
        [_STOP], TabularTarget(ConstantGrants()), tmp_path
    )
    assert report.rounds == 0
    assert report.probes_used == 0
    assert report.stop_reason.startswith("planner_stop")


def test_campaign_malformed_planner_json_never_crashes(tmp_path):
    report, _ = _campaign(
        ["definitely not json"], TabularTarget(ConstantGrants()), tmp_path
    )
    assert report.rounds == 0
    assert "planner_invalid_spec" in report.stop_reason


def test_campaign_reproducible_with_seed(tmp_path):
    script = [
        _probe_spec(
            "counterfactual",
            {"attributes": {"group": ["FT", "PT", "temp"]}},
        ),
        _STOP,
    ]
    report1, _ = _campaign(
        script, TabularTarget(BiasedGrants()), tmp_path / "a", seed=42
    )
    report2, _ = _campaign(
        script, TabularTarget(BiasedGrants()), tmp_path / "b", seed=42
    )
    assert report1.findings == report2.findings
    assert report1.stop_reason == report2.stop_reason
    assert report1.probes_used == report2.probes_used


def test_cache_prevents_duplicate_target_calls(tmp_path):
    # Same spec twice: the second round's identical rows must come from
    # the cache, not from the target.
    spec = _probe_spec(
        "counterfactual", {"attributes": {"group": ["FT", "temp"]}}
    )
    counting = CountingTarget(BiasedGrants().predict)
    report, _ = _campaign(
        [spec, spec, _STOP],
        counting,
        tmp_path,
        budget=Budget(max_probes=100, max_rounds=5, flat_rounds=0),
    )
    assert report.rounds == 2
    # Round 1 executes one predict call for the 2-row batch; round 2's
    # identical rows are served entirely from the cache.
    assert counting.predict_calls == 1


def test_adversarial_errors_summarized_not_crashing(tmp_path):
    report, _ = _campaign(
        [_probe_spec("adversarial", {}, reason="robustness"), _STOP],
        TabularTarget(ExplodingTarget()),
        tmp_path,
    )
    details = report.findings[0]["details"]
    assert details["n_errors"] > 0
    assert details["error_rate"] > 0
    assert report.best_strength == pytest.approx(details["error_rate"])


def test_metamorphic_violations_detected(tmp_path):
    # Unit-test the summarizer directly: DataFrame-backed tabular
    # adapters normalize dict key order by construction, so an
    # order-sensitive *execution* cannot be observed through them —
    # reorder_fields is meaningful for text / raw-dict targets. What
    # matters here is that violations in the outputs are counted.
    from opsaudit.probes import BUILTIN_RELATIONS, generate_metamorphic

    relations = [r for r in BUILTIN_RELATIONS if r.name == "reorder_fields"]
    batch = generate_metamorphic(
        [{"tenure_months": 6, "group": "FT"}],
        relations=relations,
        id_prefix="tmm",
    )
    assert len(batch) == 2
    campaign = AuditCampaign(
        audited_target=TabularTarget(ConstantGrants()),
        planner=AuditPlanner(dict(BRIEF), ScriptedPlannerTarget([_STOP])),
        evidence_path=tmp_path / "c.jsonl",
        seed=7,
    )
    # Simulate an order-sensitive target: source and follow-up differ.
    summary = campaign._summarize_round(
        batch, ["tenure_months=6,group=FT", "group=FT,tenure_months=6"], 1
    )
    details = summary["details"]
    assert details["n_pairs_checked"] == 1
    assert details["n_violations"] == 1
    assert details["violation_rate"] == pytest.approx(1.0)
    # A stable target shows no violation.
    stable = campaign._summarize_round(batch, ["same", "same"], 1)
    assert stable["details"]["n_violations"] == 0
    assert stable["strength"] == 0.0


def test_evidence_log_covers_full_loop(tmp_path):
    report, _ = _campaign(
        [
            _probe_spec(
                "counterfactual",
                {"attributes": {"group": ["FT", "temp"]}},
            ),
            _STOP,
        ],
        TabularTarget(BiasedGrants()),
        tmp_path,
    )
    log = EvidenceLog(tmp_path / "campaign.jsonl")
    kinds = [e["kind"] for e in log.read()]
    for expected in (
        "campaign_start",
        "plan",
        "probe_batch",
        "observations",
        "round_summary",
        "campaign_end",
    ):
        assert expected in kinds, f"missing event kind {expected}"
    assert report.n_events == len(kinds)


# ----------------------------------------------------------------------
# Batch capping keeps pairs intact
# ----------------------------------------------------------------------
def test_cap_batch_keeps_counterfactual_pairs():
    batch = generate_counterfactuals(
        {"tenure_months": 6, "group": "FT"},
        {"group": ["FT", "PT", "temp"]},
    )
    assert len(batch) == 6  # 3 pairs
    capped = _cap_batch(batch, 4)
    assert len(capped) == 4
    groups = capped.pair_groups()
    assert all(len(members) == 2 for members in groups.values())
    capped = _cap_batch(batch, 5)
    assert len(capped) == 4  # cannot split a pair


# ----------------------------------------------------------------------
# ResponseCache
# ----------------------------------------------------------------------
def test_response_cache_memory_and_file(tmp_path):
    cache = ResponseCache()
    hit, _ = cache.lookup({"q": 1})
    assert not hit
    cache.store({"q": 1}, {"a": "b"})
    hit, value = cache.lookup({"q": 1})
    assert hit and value == {"a": "b"}

    path = tmp_path / "cache.json"
    cache2 = ResponseCache(path)
    cache2.store("prompt", "response")
    cache2.save()
    cache3 = ResponseCache(path)
    hit, value = cache3.lookup("prompt")
    assert hit and value == "response"


def test_probe_spec_to_dict_roundtrip():
    spec = ProbeSpec(
        action="probe", generator="adversarial", params={}, reason="r"
    )
    d = spec.to_dict()
    assert d["action"] == "probe"
    assert json.loads(json.dumps(d)) == d

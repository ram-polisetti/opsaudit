"""Tests for the Phase 4 calibration harness.

No network anywhere: judges are deterministic stubs. These tests prove
the agreement metrics match hand-computed values, the PASS/FAIL
threshold behaves, unscored items are handled honestly, and the
campaign refuses uncalibrated judges without an explicit override.
"""

from __future__ import annotations

import json

import pytest

from opsaudit.agents import AuditCampaign, AuditPlanner, Budget
from opsaudit.calibration import (
    CalibrationHarness,
    CalibrationReport,
    cohen_kappa,
)
from opsaudit.judges import Judge, JudgeScore
from opsaudit.targets import Target


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FixedJudge(Judge):
    """A judge returning predetermined labels (no LLM)."""

    name = "fixed-judge"
    prompt_version = 1

    def __init__(self, labels, model_id="fixed-model"):
        super().__init__()
        self._labels = list(labels)
        self.model_id = model_id

    def score(self, texts):
        assert len(texts) == len(self._labels)
        return [
            JudgeScore(
                label=label,
                confidence=1.0,
                rationale="",
                judge_id=self.judge_id,
                model_id=self.model_id,
                unscored=(label == "unscored"),
            )
            for label in self._labels
        ]

    def labels(self):
        return tuple(sorted(set(self._labels)))


class ScriptedPlannerTarget(Target):
    name = "scripted-planner"

    def __init__(self, script):
        self.script = list(script)

    def generate(self, prompts):
        return [self.script[0] for _ in prompts]

    def describe(self):
        return {"target_type": "planner", "name": self.name}


class DummyTextTarget(Target):
    name = "dummy-text"

    def generate(self, prompts):
        return ["dummy output" for _ in prompts]

    def describe(self):
        return {"target_type": "text", "name": self.name}


_STOP = json.dumps({"action": "stop", "reason": "test done"})


def _campaign(judges, allow_uncalibrated=False, tmp_path=None):
    brief = {
        "target_type": "text",
        "protected_attributes": {"group": ["A", "B"]},
        "risk_areas": ["tone"],
    }
    planner = AuditPlanner(
        brief,
        ScriptedPlannerTarget([_STOP]),
        budget=Budget(max_rounds=2, max_probes=10),
    )
    return AuditCampaign(
        audited_target=DummyTextTarget(),
        planner=planner,
        evidence_path=str(tmp_path / "ev.jsonl"),
        judges=judges,
        allow_uncalibrated=allow_uncalibrated,
    )


# ----------------------------------------------------------------------
# cohen_kappa: hand-computed values
# ----------------------------------------------------------------------
def test_kappa_partial_agreement():
    # human: a a b b ; judge: a b b b
    # p_o = 3/4 = 0.75 ; p_e = 0.5*0.25 + 0.5*0.75 = 0.5
    # kappa = (0.75 - 0.5) / 0.5 = 0.5
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == 0.5


def test_kappa_perfect_agreement():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_kappa_single_label_is_perfect():
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_kappa_empty_is_zero():
    assert cohen_kappa([], []) == 0.0


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


# ----------------------------------------------------------------------
# Harness behavior
# ----------------------------------------------------------------------
def test_harness_reports_hand_computed_metrics():
    dataset = [("t1", "a"), ("t2", "a"), ("t3", "b"), ("t4", "b")]
    judge = FixedJudge(["a", "b", "b", "b"])
    report = CalibrationHarness(judge).run(dataset)
    assert isinstance(report, CalibrationReport)
    assert report.n_items == 4
    assert report.n_scored == 4
    assert report.n_unscored == 0
    assert report.accuracy == 0.75
    assert report.kappa == 0.5
    # per-label: "a": tp=1, fp=0, fn=1 → P=1.0, R=0.5, support=2
    assert report.per_label["a"] == {
        "precision": 1.0,
        "recall": 0.5,
        "support": 2,
    }
    # "b": tp=2, fp=1, fn=0 → P=2/3, R=1.0, support=2
    assert report.per_label["b"]["precision"] == pytest.approx(0.6667)
    assert report.per_label["b"]["recall"] == 1.0
    assert report.per_label["b"]["support"] == 2
    assert report.confusion == {"a": {"a": 1, "b": 1}, "b": {"b": 2}}
    json.dumps(report.to_dict())  # JSON-safe


def test_harness_fail_below_threshold_warns_and_records(capsys):
    dataset = [("t1", "a"), ("t2", "a"), ("t3", "b"), ("t4", "b")]
    judge = FixedJudge(["a", "b", "b", "b"])  # kappa 0.5 < 0.6
    report = CalibrationHarness(judge, kappa_threshold=0.6).run(dataset)
    assert not report.passed
    assert judge.calibration is report  # stored on the judge
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "FAILED calibration" in captured.err


def test_harness_pass_at_or_above_threshold(capsys):
    dataset = [("t1", "a"), ("t2", "b")]
    judge = FixedJudge(["a", "b"])  # kappa 1.0
    report = CalibrationHarness(judge).run(dataset)
    assert report.passed
    assert report.kappa == 1.0
    assert "WARNING" not in capsys.readouterr().err


def test_harness_excludes_unscored_from_accuracy():
    dataset = [("t1", "a"), ("t2", "a"), ("t3", "b")]
    judge = FixedJudge(["a", "unscored", "b"])
    report = CalibrationHarness(judge).run(dataset)
    assert report.n_scored == 2
    assert report.n_unscored == 1
    assert report.accuracy == 1.0  # 2/2 scored, not 2/3
    assert any("unscored" in note for note in report.notes)


def test_harness_all_unscored_is_fail():
    dataset = [("t1", "a"), ("t2", "b")]
    judge = FixedJudge(["unscored", "unscored"])
    report = CalibrationHarness(judge).run(dataset)
    assert not report.passed
    assert report.accuracy == 0.0


class ShortJudge(FixedJudge):
    """Returns fewer scores than texts (contract violation)."""

    def score(self, texts):
        return super().score(texts)[:1]


class LongJudge(FixedJudge):
    """Returns more scores than texts (contract violation)."""

    def score(self, texts):
        return super().score(texts) + super().score(texts)[:1]


def test_harness_rejects_short_score_list():
    dataset = [("t1", "a"), ("t2", "b")]
    with pytest.raises(ValueError, match="2 texts"):
        CalibrationHarness(ShortJudge(["a", "b"])).run(dataset)


def test_harness_rejects_long_score_list():
    dataset = [("t1", "a"), ("t2", "b")]
    with pytest.raises(ValueError, match="2 texts"):
        CalibrationHarness(LongJudge(["a", "b"])).run(dataset)


def test_harness_empty_dataset_is_fail():
    report = CalibrationHarness(FixedJudge([])).run([])
    assert not report.passed
    assert report.n_items == 0
    assert report.n_scored == 0
    assert "scored nothing" in report.notes[0]


def test_harness_rejects_bad_threshold():
    with pytest.raises(ValueError):
        CalibrationHarness(FixedJudge(["a"]), kappa_threshold=1.5)


def test_harness_rejects_none_judge():
    with pytest.raises(ValueError):
        CalibrationHarness(None)


# ----------------------------------------------------------------------
# Campaign wiring: calibrated judges required, override is explicit
# ----------------------------------------------------------------------
def test_campaign_rejects_uncalibrated_judge(tmp_path):
    judge = FixedJudge(["a", "b"])  # calibration never run
    assert judge.calibration is None
    with pytest.raises(ValueError, match="no passing calibration"):
        _campaign([judge], tmp_path=tmp_path)


def test_campaign_rejects_failed_calibration(tmp_path):
    judge = FixedJudge(["a", "b"])
    judge.calibration = CalibrationReport(
        judge_id=judge.judge_id,
        model_id="m",
        n_items=2,
        n_scored=2,
        n_unscored=0,
        accuracy=0.5,
        kappa=0.0,
        per_label={},
        confusion={},
        kappa_threshold=0.6,
        passed=False,
    )
    with pytest.raises(ValueError, match="no passing calibration"):
        _campaign([judge], tmp_path=tmp_path)


def test_campaign_accepts_calibrated_judge(tmp_path):
    judge = FixedJudge(["a", "b"])
    dataset = [("t1", "a"), ("t2", "b")]
    CalibrationHarness(judge).run(dataset)  # kappa 1.0 → PASS
    campaign = _campaign([judge], tmp_path=tmp_path)
    report = campaign.run()
    assert report.rounds == 0  # planner stopped immediately; no crash
    assert report.stop_reason.startswith("planner_stop")


def test_campaign_override_logs_evidence(tmp_path):
    judge = FixedJudge(["a", "b"])  # uncalibrated, explicit override
    campaign = _campaign(
        [judge], allow_uncalibrated=True, tmp_path=tmp_path
    )
    campaign.run()
    events = [
        json.loads(line)
        for line in (tmp_path / "ev.jsonl").read_text().splitlines()
    ]
    override = [e for e in events if e["kind"] == "uncalibrated_judge_override"]
    assert len(override) == 1
    assert override[0]["judge_ids"] == [judge.judge_id]
    assert override[0]["allow_uncalibrated"] is True

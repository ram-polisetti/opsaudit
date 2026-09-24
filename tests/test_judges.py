"""Tests for the Phase 4 LLM judges.

No network anywhere: the judging "LLM" is a scripted fake Target.
These tests prove label parsing, the empty-response retry → unscored
path, malformed-JSON → unscored (never an invented label), prompt
versioning in judge_id, and the deterministic aggregation of labels
into group rates/gaps.
"""

from __future__ import annotations

import json

import pytest

from opsaudit.judges import (
    JudgeScore,
    RefusalJudge,
    StereotypeJudge,
    ToneJudge,
    aggregate_judge_findings,
    group_label_rates,
    label_rate_gaps,
    ordinal_means,
    unscored_rate,
)
from opsaudit.judges.base import LLMJudge
from opsaudit.judges.tone import TONE_ORDER
from opsaudit.targets import Target


def test_judges_are_llm_judges_with_versioned_ids():
    for cls, expected_id in [
        (StereotypeJudge, "stereotype-judge-v1"),
        (RefusalJudge, "refusal-judge-v1"),
        (ToneJudge, "tone-judge-v1"),
    ]:
        assert issubclass(cls, LLMJudge)
        assert cls.prompt_version >= 1
        assert expected_id.startswith(cls.name)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class ScriptedJudgeTarget(Target):
    """A judging model that returns scripted raw responses in order."""

    name = "scripted-judge"

    def __init__(self, script):
        # script: list of raw response strings, consumed one per prompt.
        self.script = list(script)
        self.calls = []  # list of prompt-lists, one entry per generate()

    def generate(self, prompts):
        self.calls.append(list(prompts))
        out = []
        for _ in prompts:
            out.append(self.script.pop(0) if self.script else "")
        return out

    def describe(self):
        return {
            "target_type": "text",
            "name": self.name,
            "model": "scripted-test-model",
        }


class TextOnly:
    """Not a text target: generate() is the base-class stub."""


def _judge_json(**fields):
    return json.dumps(fields)


# ----------------------------------------------------------------------
# Stereotype judge
# ----------------------------------------------------------------------
def test_stereotype_parses_positive_label():
    target = ScriptedJudgeTarget([])
    judge = StereotypeJudge(target, group="women")
    score = judge._parse(
        _judge_json(
            label=1, severity=2, confidence=0.8, rationale="says women can't lead"
        ),
        "text",
    )
    assert score.label == "stereotype"
    assert score.meta["severity"] == 2
    assert score.confidence == pytest.approx(0.8)
    assert score.judge_id == "stereotype-judge-v1"  # name + prompt version
    assert score.model_id == "scripted-test-model"
    assert not score.unscored


def test_stereotype_parses_negative_label():
    target = ScriptedJudgeTarget([])
    judge = StereotypeJudge(target, group="women")
    score = judge._parse(
        _judge_json(label=0, severity=0, confidence=0.95, rationale="none"), "t"
    )
    assert score.label == "no_stereotype"
    assert score.meta["severity"] == 0


def test_stereotype_rejects_severity_mismatch():
    target = ScriptedJudgeTarget([])
    judge = StereotypeJudge(target, group="women")
    with pytest.raises(ValueError):
        judge._parse(
            _judge_json(label=0, severity=1, confidence=0.5, rationale="x"), "t"
        )


def test_stereotype_rejects_unknown_label():
    target = ScriptedJudgeTarget([])
    judge = StereotypeJudge(target, group="women")
    with pytest.raises(ValueError):
        judge._parse(
            _judge_json(label="maybe", severity=0, confidence=0.5, rationale="x"),
            "t",
        )


def test_stereotype_requires_group():
    target = ScriptedJudgeTarget([])
    with pytest.raises(ValueError):
        StereotypeJudge(target, group="  ")


def test_stereotype_prompt_names_the_group():
    target = ScriptedJudgeTarget(
        [_judge_json(label=0, severity=0, confidence=1.0, rationale="none")]
    )
    judge = StereotypeJudge(target, group="nurses")
    judge.score(["some text"])
    assert "nurses" in target.calls[0][0]


# ----------------------------------------------------------------------
# Refusal judge
# ----------------------------------------------------------------------
def test_refusal_parses_refusal():
    target = ScriptedJudgeTarget([])
    judge = RefusalJudge(target)
    score = judge._parse(
        _judge_json(label="refusal", confidence=0.7, rationale="declines"), "t"
    )
    assert score.label == "refusal"
    assert score.judge_id == "refusal-judge-v1"


def test_refusal_parses_compliance():
    target = ScriptedJudgeTarget([])
    judge = RefusalJudge(target)
    score = judge._parse(
        _judge_json(label="compliance", confidence=0.9, rationale="answers"), "t"
    )
    assert score.label == "compliance"


# ----------------------------------------------------------------------
# Tone judge
# ----------------------------------------------------------------------
def test_tone_parses_warm_with_ordinal():
    target = ScriptedJudgeTarget([])
    judge = ToneJudge(target)
    score = judge._parse(
        _judge_json(label="warm", confidence=0.85, rationale="friendly"), "t"
    )
    assert score.label == "warm"
    assert score.meta["ordinal"] == TONE_ORDER["warm"] == 2
    assert score.judge_id == "tone-judge-v1"


def test_tone_rejects_unknown_label():
    target = ScriptedJudgeTarget([])
    judge = ToneJudge(target)
    with pytest.raises(ValueError):
        judge._parse(
            _judge_json(label="spicy", confidence=0.5, rationale="x"), "t"
        )


# ----------------------------------------------------------------------
# Shared LLMJudge execution behavior
# ----------------------------------------------------------------------
def test_score_retries_empty_once_then_unscored():
    # One text, two empty responses: one retry, then unscored — never a
    # guessed label.
    target = ScriptedJudgeTarget(["", ""])
    judge = ToneJudge(target)
    scores = judge.score(["hello"])
    assert len(scores) == 1
    assert scores[0].unscored
    assert scores[0].label == "unscored"
    assert scores[0].confidence == 0.0
    assert len(target.calls) == 2  # initial + one retry


def test_score_recovers_on_retry():
    target = ScriptedJudgeTarget(
        ["", _judge_json(label="cold", confidence=0.6, rationale="curt")]
    )
    judge = ToneJudge(target)
    scores = judge.score(["hello"])
    assert scores[0].label == "cold"
    assert not scores[0].unscored


def test_score_malformed_json_becomes_unscored():
    target = ScriptedJudgeTarget(["definitely not json", "still not json"])
    judge = RefusalJudge(target)
    scores = judge.score(["hello"])
    assert scores[0].unscored
    assert "unparseable" in scores[0].rationale


def test_score_preserves_input_order():
    target = ScriptedJudgeTarget(
        [
            _judge_json(label="warm", confidence=0.9, rationale="a"),
            _judge_json(label="cold", confidence=0.9, rationale="b"),
        ]
    )
    judge = ToneJudge(target)
    scores = judge.score(["first", "second"])
    assert [s.label for s in scores] == ["warm", "cold"]


def test_judge_rejects_non_text_target():
    with pytest.raises(ValueError):
        ToneJudge(None)
    with pytest.raises(ValueError):
        ToneJudge(TextOnly())


def test_confidence_clamped_to_unit_interval():
    target = ScriptedJudgeTarget([])
    judge = ToneJudge(target)
    score = judge._parse(
        _judge_json(label="warm", confidence=99, rationale="x"), "t"
    )
    assert score.confidence == 1.0
    score = judge._parse(
        _judge_json(label="warm", confidence="nonsense", rationale="x"), "t"
    )
    assert score.confidence == 0.0


def test_judge_score_serializes_for_evidence_log():
    target = ScriptedJudgeTarget([])
    judge = ToneJudge(target)
    score = judge._parse(
        _judge_json(label="neutral", confidence=0.5, rationale="ok"), "t"
    )
    d = score.to_dict()
    json.dumps(d)  # must be JSON-safe
    assert d["judge_id"] == "tone-judge-v1"


# ----------------------------------------------------------------------
# Deterministic aggregation (no LLM arithmetic — pure code)
# ----------------------------------------------------------------------
def _mk(label, unscored=False):
    return JudgeScore(
        label=label,
        confidence=0.9,
        rationale="",
        judge_id="tone-judge-v1",
        model_id="m",
        unscored=unscored,
    )


def test_group_label_rates_hand_computed():
    scores = [_mk("warm"), _mk("warm"), _mk("cold"), _mk("cold")]
    groups = ["A", "A", "B", "B"]
    rates = group_label_rates(scores, groups)
    assert rates == {"A": {"warm": 1.0}, "B": {"cold": 1.0}}


def test_label_rate_gaps_hand_computed():
    rates = {"A": {"warm": 1.0}, "B": {"warm": 0.25, "cold": 0.75}}
    gaps = label_rate_gaps(rates)
    assert gaps["warm"] == pytest.approx(0.75)
    assert gaps["cold"] == pytest.approx(0.75)


def test_aggregation_skips_unscored_but_counts_them():
    scores = [_mk("warm"), _mk("warm", unscored=True), _mk("cold")]
    groups = ["A", "A", "B"]
    rates = group_label_rates(scores, groups)
    assert rates["A"] == {"warm": 1.0}  # unscored item excluded
    assert unscored_rate(scores) == 0.3333  # rounded, honest


def test_ordinal_means_for_tone():
    scores = [_mk("warm"), _mk("neutral"), _mk("cold"), _mk("cold")]
    groups = ["A", "A", "B", "B"]
    means = ordinal_means(scores, groups, TONE_ORDER)
    assert means == {"A": 1.5, "B": 0.0}


def test_aggregate_judge_findings_shape():
    scores = [_mk("warm"), _mk("warm"), _mk("cold"), _mk("neutral")]
    groups = ["A", "A", "B", "B"]
    findings = aggregate_judge_findings(
        "tone-judge-v1", scores, groups, ordinal=TONE_ORDER
    )
    assert findings["judge_id"] == "tone-judge-v1"
    assert findings["n_items"] == 4
    assert findings["n_scored"] == 4
    assert findings["strongest_label"] == "warm"
    assert findings["strength"] == pytest.approx(1.0)
    assert findings["ordinal_gap"] == pytest.approx(1.5)  # A:2.0 vs B:0.5
    json.dumps(findings)  # JSON-safe for the evidence log


def test_group_length_mismatch_rejected():
    with pytest.raises(ValueError):
        group_label_rates([_mk("warm")], ["A", "B"])


class ExplodingJudgeTarget(Target):
    """A judging model whose transport always fails."""

    name = "exploding-judge"

    def generate(self, prompts):
        raise ConnectionError("simulated network failure")

    def describe(self):
        return {"target_type": "exploding-judge", "name": self.name}


class FlakyJudgeTarget(Target):
    """Succeeds on the first call, explodes on the retry."""

    name = "flaky-judge"

    def __init__(self):
        self.calls = 0

    def generate(self, prompts):
        self.calls += 1
        if self.calls == 1:
            return [""] * len(prompts)  # empties -> triggers retry
        raise TimeoutError("simulated retry failure")

    def describe(self):
        return {"target_type": "flaky-judge", "name": self.name}


def test_judge_transport_failure_scores_all_unscored():
    judge = RefusalJudge(ExplodingJudgeTarget())
    scores = judge.score(["text one", "text two"])
    assert len(scores) == 2
    assert all(s.unscored for s in scores)
    assert all("judge model call failed" in s.rationale for s in scores)
    json.dumps([s.__dict__ for s in scores])  # JSON-safe


def test_judge_retry_failure_stays_unscored():
    judge = RefusalJudge(FlakyJudgeTarget())
    scores = judge.score(["text one"])
    assert len(scores) == 1
    assert scores[0].unscored

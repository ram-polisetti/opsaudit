"""Tests for opsaudit.reports: deterministic report assembly and rendering.

Everything here is canned fixtures and arithmetic — no LLM, no network.
"""

from __future__ import annotations

import json
import re

import pytest

from opsaudit import (
    AGENTIC_RMF_MAPPING,
    FLAG_THRESHOLD_DEFAULT,
    build_report,
    mappings_for_report,
    save_agentic_report,
)
from opsaudit.agents import CampaignReport
from opsaudit.reports import to_html, to_markdown

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _campaign(best_strength=1.0, seed=7):
    return CampaignReport(
        brief={
            "target_type": "tabular",
            "protected_attributes": {"employment_type": ["FT", "temp"]},
            "risk_areas": ["hiring"],
            "base_input": {"tenure_months": 12, "employment_type": "FT"},
        },
        seed=seed,
        rounds=2,
        probes_used=6,
        stop_reason="planner_stop:disparity confirmed",
        findings=[
            {
                "round": 1,
                "generator": "counterfactual",
                "n_probes": 4,
                "strength": 1.0,
                "details": {
                    "outcome_rates": {
                        "employment_type": {"FT": 1.0, "temp": 0.0}
                    },
                    "gaps": {"employment_type": 1.0},
                },
            },
            {
                "round": 2,
                "generator": "counterfactual",
                "n_probes": 2,
                "strength": 1.0,
                "details": {"gaps": {"employment_type": 1.0}},
            },
        ],
        best_strength=best_strength,
        evidence_path="/tmp/evidence.jsonl",
        n_events=8,
    )


class _FakeTarget:
    name = "fake-target"

    def describe(self):
        return {"target_type": "tabular", "name": self.name,
                "estimator": "FakeClf"}


class _FakeCalibration:
    passed = True

    def to_dict(self):
        return {"judge_id": "fake-judge-v1", "kappa": 0.85,
                "accuracy": 0.9, "n_items": 20, "n_scored": 20,
                "passed": True}


class _FakeJudge:
    name = "fake-judge"
    judge_id = "fake-judge-v1"
    prompt_version = 1
    model_id = "scripted"
    calibration = _FakeCalibration()

    def labels(self):
        return ("a", "b")


BUDGET = {"max_probes": 200, "max_rounds": 5, "flat_rounds": 0}


def _report(**kwargs):
    kwargs.setdefault("target", _FakeTarget())
    kwargs.setdefault("budget", BUDGET)
    return build_report(_campaign(), **kwargs)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def test_verdict_review_when_strength_meets_threshold():
    report = _report()
    assert report.verdict == "FINDINGS WARRANT REVIEW"
    assert report.best_strength == 1.0
    assert report.flag_threshold == FLAG_THRESHOLD_DEFAULT


def test_verdict_clear_when_strength_below_threshold():
    report = build_report(_campaign(best_strength=0.05), target=_FakeTarget(),
                          budget=BUDGET)
    assert report.verdict == "NO MATERIAL FINDINGS"


def test_verdict_boundary_is_inclusive():
    report = build_report(
        _campaign(best_strength=FLAG_THRESHOLD_DEFAULT),
        target=_FakeTarget(), budget=BUDGET,
        flag_threshold=FLAG_THRESHOLD_DEFAULT,
    )
    assert report.verdict == "FINDINGS WARRANT REVIEW"


def test_executive_summary_names_target_numbers_and_threshold():
    report = _report()
    summary = report.executive_summary
    assert "fake-target" in summary
    assert "2 round(s)" in summary
    assert "6 probe(s)" in summary
    assert "planner_stop:disparity confirmed" in summary
    assert "1.000" in summary
    assert "FINDINGS WARRANT REVIEW" in summary
    assert "not a significance test" in summary


def test_executive_summary_empty_findings():
    campaign = _campaign(best_strength=0.0)
    campaign = CampaignReport(
        brief=campaign.brief, seed=campaign.seed, rounds=0,
        probes_used=0, stop_reason="budget:max_rounds", findings=[],
        best_strength=0.0, evidence_path=campaign.evidence_path,
        n_events=2,
    )
    report = build_report(campaign, target=_FakeTarget(), budget=BUDGET)
    assert "No rounds produced findings" in report.executive_summary
    assert report.verdict == "NO MATERIAL FINDINGS"


def test_report_identifies_target_from_brief_without_target():
    report = build_report(_campaign(), budget=BUDGET)
    assert report.target_name == "unknown-target"
    assert report.target_type == "tabular"


def test_methodology_records_brief_budget_generators():
    report = _report()
    method = report.methodology
    assert method["brief"]["risk_areas"] == ["hiring"]
    assert method["budget"] == BUDGET
    assert method["generators_used"] == ["counterfactual"]
    assert method["probes_by_round"] == [
        {"round": 1, "generator": "counterfactual", "n_probes": 4},
        {"round": 2, "generator": "counterfactual", "n_probes": 2},
    ]


def test_judge_card_records_calibration():
    report = _report(judges=(_FakeJudge(),))
    cards = report.methodology["judges"]
    assert len(cards) == 1
    card = cards[0]
    assert card["judge_id"] == "fake-judge-v1"
    assert card["prompt_version"] == 1
    assert card["calibration"]["kappa"] == 0.85
    assert card["calibration"]["passed"] is True


def test_judge_without_calibration_recorded_as_none():
    judge = _FakeJudge()
    judge.calibration = None
    report = _report(judges=(judge,))
    assert report.methodology["judges"][0]["calibration"] is None


def test_limitations_always_present_plus_extras():
    report = _report(extra_limitations=("demo-only run",))
    assert len(report.limitations) >= 4
    assert "demo-only run" in report.limitations
    assert any("budget" in lim for lim in report.limitations)


def test_reproducibility_block_has_seed_and_version():
    report = _report()
    repro = report.reproducibility
    assert repro["seed"] == 7
    assert repro["package_version"]
    assert repro["evidence_path"] == "/tmp/evidence.jsonl"
    assert report.to_dict()["reproducibility"]["seed"] == 7


def test_to_dict_is_json_safe():
    report = _report(judges=(_FakeJudge(),))
    payload = json.dumps(report.to_dict())
    assert "fake-target" in payload


# ----------------------------------------------------------------------
# RMF mappings
# ----------------------------------------------------------------------


def test_all_mappings_reference_valid_functions():
    for mapping in AGENTIC_RMF_MAPPING:
        assert mapping["function"] in {"GOVERN", "MAP", "MEASURE", "MANAGE"}
        assert mapping["rationale"]
        assert mapping["limits"]  # honesty is mandatory


def test_subcategory_numbers_match_verified_rmf_core():
    subs = {
        m["subcategory"]
        for m in AGENTIC_RMF_MAPPING
        if m["subcategory"] is not None
    }
    assert subs <= {
        "MEASURE 2.11", "MEASURE 2.1", "MEASURE 2.3", "MEASURE 2.13",
        "MAP 5.1", "MAP 1.1",
    }
    for sub in subs:
        assert re.fullmatch(r"(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+", sub)


def test_judge_calibration_mapping_only_with_judges():
    without = mappings_for_report((), judges_used=False)
    with_judges = mappings_for_report((), judges_used=True)
    aspects_without = {m["aspect"] for m in without}
    aspects_with = {m["aspect"] for m in with_judges}
    assert "judge_calibration" not in aspects_without
    assert "judge_calibration" in aspects_with
    assert len(with_judges) == len(without) + 1


def test_report_carries_mappings_with_judges_flag():
    report = _report(judges=(_FakeJudge(),))
    aspects = {m["aspect"] for m in report.rmf_mappings}
    assert "judge_calibration" in aspects
    plain = _report()
    assert "judge_calibration" not in {m["aspect"] for m in plain.rmf_mappings}


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def test_markdown_has_all_sections_and_numbers():
    md = to_markdown(_report())
    for section in ("# Agentic audit report", "## Executive summary",
                    "## Findings", "## Methodology", "## Limitations",
                    "## NIST AI RMF mapping", "## Reproducibility"):
        assert section in md
    assert "1.000" in md
    assert "gap employment_type=1.000" in md
    assert "FINDINGS WARRANT REVIEW" in md
    assert "MEASURE 2.11" in md


def test_markdown_judge_section_lists_calibration():
    md = to_markdown(_report(judges=(_FakeJudge(),)))
    assert "### Judges" in md
    assert "fake-judge-v1" in md
    assert "kappa=0.850" in md


def test_html_is_standalone_and_contains_numbers():
    html = to_html(_report())
    assert html.startswith("<!DOCTYPE html>")
    assert "1.000" in html
    assert "FINDINGS WARRANT REVIEW" in html
    assert "MEASURE 2.11" in html
    # No external assets, no scripts, no sticky positioning.
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html
    assert "position: sticky" not in html
    assert "position:sticky" not in html
    assert "position: fixed" not in html


def test_html_escapes_target_name():
    class _Evil:
        name = "<script>alert(1)</script>"

        def describe(self):
            return {"target_type": "tabular", "name": self.name}

    html = to_html(build_report(_campaign(), target=_Evil(), budget=BUDGET))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_save_agentic_report_writes_three_files(tmp_path):
    files = save_agentic_report(_report(), tmp_path / "audit")
    assert [f.suffix for f in files] == [".md", ".html", ".json"]
    payload = json.loads((tmp_path / "audit.json").read_text())
    assert payload["verdict"] == "FINDINGS WARRANT REVIEW"
    assert (tmp_path / "audit.md").read_text().startswith(
        "# Agentic audit report")


def test_report_is_frozen():
    report = _report()
    with pytest.raises(Exception):
        report.verdict = "x"  # type: ignore[misc]


def _judge_campaign():
    return CampaignReport(
        brief={"target_type": "text",
               "protected_attributes": {"employment_type": ["FT", "temp"]}},
        seed=3,
        rounds=1,
        probes_used=2,
        stop_reason="planner_stop:tone gap confirmed",
        findings=[
            {
                "round": 1,
                "generator": "counterfactual",
                "n_probes": 2,
                "strength": 1.0,
                "details": {
                    "n_judged_outputs": 2,
                    "judge_findings": [
                        {
                            "judge_id": "tone-judge-v1",
                            "model_id": "scripted",
                            "calibrated": True,
                            "by_attribute": {
                                "employment_type": {
                                    "strongest_label": "warm",
                                    "strength": 1.0,
                                    "gaps": {"warm": 1.0, "cold": 1.0},
                                }
                            },
                            "best_strength": 1.0,
                        }
                    ],
                },
            }
        ],
        best_strength=1.0,
        evidence_path="/tmp/evidence.jsonl",
        n_events=5,
    )


def test_judge_findings_render_in_markdown_and_html():
    campaign = _judge_campaign()
    report = build_report(campaign, budget=BUDGET, judges=(_FakeJudge(),))
    md = to_markdown(report)
    assert "tone-judge-v1:employment_type warm gap=1.000" in md
    html = to_html(report)
    assert "tone-judge-v1:employment_type warm gap=1.000" in html
    assert "judge_calibration" in {m["aspect"] for m in report.rmf_mappings}
    assert "The strongest finding (strength 1.000)" in report.executive_summary
    assert "tone-judge-v1" in report.executive_summary


def test_strongest_signal_names_judge_gap():
    report = build_report(_judge_campaign(), budget=BUDGET)
    assert "'warm' label-rate gap of 1.000" in report.executive_summary

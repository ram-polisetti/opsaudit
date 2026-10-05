import pytest

from opsaudit.eu_ai_act import (
    USE_CASES,
    classify_system,
    cross_check_gate,
    eu_ai_act_section,
)


def test_employment_recruitment_is_high_risk():
    c = classify_system({"use_cases": ["employment_recruitment"]})

    assert c.tier == "high_risk"
    assert any("Annex III" in ref for _, ref, _ in c.matched_use_cases)
    assert any("Conformity assessment" in ob for ob in c.obligations)


def test_social_scoring_is_prohibited():
    c = classify_system({"use_cases": ["social_scoring"]})

    assert c.tier == "prohibited"
    assert any(ref == "Art. 5" for _, ref, _ in c.matched_use_cases)


def test_prohibited_takes_precedence_over_high_risk():
    c = classify_system({"use_cases": ["employment_recruitment", "social_scoring"]})

    assert c.tier == "prohibited"


def test_spam_filter_is_minimal_risk():
    c = classify_system({"use_cases": []})

    assert c.tier == "minimal"
    assert c.matched_use_cases == []


def test_chatbot_triggers_transparency_duties():
    c = classify_system({"use_cases": ["conversational_ai"]})

    assert c.tier == "transparency"
    assert any("Art. 50" in ob for ob in c.obligations)


def test_gpai_flags_add_duties_without_changing_tier():
    c = classify_system({"use_cases": [], "is_gpai": True, "gpai_systemic_risk": True})

    assert c.tier == "minimal"
    assert any("Art. 53" in d for d in c.gpai_duties)
    assert any("Art. 55" in d for d in c.gpai_duties)


def test_unknown_use_case_key_raises():
    with pytest.raises(ValueError, match="Unknown use-case keys"):
        classify_system({"use_cases": ["mind_reading"]})


def test_gate_pass_with_prohibited_tier_is_hard_stop():
    c = classify_system({"use_cases": ["social_scoring"]})
    check = cross_check_gate("pass", c)

    assert check.verdict == "hard_stop"
    assert "Art. 5" in check.rationale


def test_gate_pass_with_high_risk_tier_is_conditional():
    c = classify_system({"use_cases": ["credit_scoring"]})
    check = cross_check_gate("pass", c)

    assert check.verdict == "conditional_pass"
    assert "conformity" in check.rationale.lower()


def test_gate_fail_blocks_even_minimal_tier():
    c = classify_system({"use_cases": []})
    check = cross_check_gate("fail", c)

    assert check.verdict == "blocked"


def test_gate_review_blocks_high_risk():
    c = classify_system({"use_cases": ["employment_recruitment"]})
    check = cross_check_gate("review", c)

    assert check.verdict == "blocked"


def test_invalid_gate_status_raises():
    c = classify_system({"use_cases": []})
    with pytest.raises(ValueError, match="Unknown gate status"):
        cross_check_gate("maybe", c)


def test_report_section_renders():
    c = classify_system({"use_cases": ["employment_recruitment"]})
    section = eu_ai_act_section(c, cross_check_gate("pass", c))

    assert "## EU AI Act" in section
    assert "high_risk" in section
    assert "conditional_pass" in section
    assert "not legal advice" in section


def test_controlled_vocabulary_covers_all_tiers():
    tiers = {v["tier"] for v in USE_CASES.values()}

    assert tiers == {"prohibited", "high_risk", "transparency"}

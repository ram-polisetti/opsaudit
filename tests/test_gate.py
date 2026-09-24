import pytest
import yaml

from opsaudit.data import generate_dispatch
from opsaudit.gate import evaluate_gate, evaluate_gate_status
from opsaudit.metrics import AuditResult, audit_disparities


def _dispatch_result(bias_strength: float):
    frame = generate_dispatch(n=5000, bias_strength=bias_strength, seed=42)
    return audit_disparities(frame["on_time"], frame["priority_route"], frame["group"])


def test_clean_dispatch_passes_gate():
    passed, findings = evaluate_gate(_dispatch_result(0.0))

    assert passed is True
    assert all(finding["status"] != "fail" for finding in findings)


def test_biased_dispatch_fails_disparate_impact_gate():
    passed, findings = evaluate_gate(_dispatch_result(0.5))

    assert passed is False
    assert any(
        finding["status"] == "fail" and finding["check"] == "disparate_impact_ratio"
        for finding in findings
    )


def test_custom_yaml_thresholds_can_make_biased_run_pass(tmp_path):
    threshold_file = tmp_path / "thresholds.yaml"
    threshold_file.write_text(
        """disparate_impact_ratio:\n  min: 0.5\ndemographic_parity_diff:\n  max: 0.3\ntpr_gap:\n  max: 0.3\nfpr_gap:\n  max: 0.3\n"""
    )
    overrides = yaml.safe_load(threshold_file.read_text())

    passed, _ = evaluate_gate(_dispatch_result(0.5), overrides)

    assert passed is True


@pytest.mark.parametrize(
    "thresholds",
    [
        {"unknown": {"max": 0.2}},
        {"tpr_gap": {"min": 0.2}},
        {"fpr_gap": {"max": "low"}},
        {"disparate_impact_ratio": {"min": 0.8, "max": 1.0}},
    ],
)
def test_malformed_thresholds_raise_value_error(thresholds):
    with pytest.raises(ValueError):
        evaluate_gate(_dispatch_result(0.0), thresholds)


def test_minimum_group_review_is_not_a_passing_gate():
    result = audit_disparities([1, 1, 0], [1, 1, 1], ["A", "B", "B"], min_group_n=2)

    status, findings = evaluate_gate_status(result)
    passed, _ = evaluate_gate(result)

    assert status == "review"
    assert passed is False
    assert any(finding["status"] == "review" for finding in findings)


def test_passing_point_estimate_with_crossing_interval_requires_review():
    result = AuditResult([], 0, 0.1, 0.85, 0.1, 0.1, [])
    result.confidence_intervals = {
        "disparate_impact_ratio": {"lower": 0.7, "upper": 0.95}
    }

    status, findings = evaluate_gate_status(result)

    assert status == "review"
    assert any("confidence_interval" in finding["check"] for finding in findings)

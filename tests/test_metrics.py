import json

import pytest

from opsaudit.metrics import AuditResult, audit_disparities


def _group(result: AuditResult, name: str):
    return next(group for group in result.groups if group.group == name)


def test_worked_example_has_exact_metrics():
    result = audit_disparities(
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0],
        ["A", "A", "A", "A", "B", "B", "B", "B"],
    )

    a, b = _group(result, "A"), _group(result, "B")
    assert (a.n, a.selection_rate, a.tpr, a.fpr, a.precision, a.accuracy) == (
        4,
        0.25,
        0.5,
        0.0,
        1.0,
        0.75,
    )
    assert b.n == 4
    assert b.selection_rate == 0.75
    assert b.tpr == 1.0
    assert b.fpr == 0.5
    assert b.precision == pytest.approx(2 / 3)
    assert b.accuracy == 0.75
    assert result.demographic_parity_diff == 0.5
    assert result.disparate_impact_ratio == pytest.approx(1 / 3)
    assert result.tpr_gap == 0.5
    assert result.fpr_gap == 0.5


def test_perfect_parity_has_zero_gaps_and_ratio_one():
    result = audit_disparities([1, 0, 1, 0], [1, 0, 1, 0], ["A", "A", "B", "B"])

    assert result.demographic_parity_diff == 0.0
    assert result.disparate_impact_ratio == 1.0
    assert result.tpr_gap == 0.0
    assert result.fpr_gap == 0.0


def test_single_group_has_no_error_rate_gaps():
    result = audit_disparities([1, 0], [1, 0], ["A", "A"])

    assert result.tpr_gap is None
    assert result.fpr_gap is None


def test_all_zero_predictions_define_disparate_impact_as_one():
    result = audit_disparities([1, 0, 1, 0], [0, 0, 0, 0], ["A", "A", "B", "B"])

    assert result.disparate_impact_ratio == 1.0


def test_group_without_positive_labels_has_undefined_tpr():
    result = audit_disparities([0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 1, 0], ["A", "A", "B", "B", "C", "C"])

    assert _group(result, "A").tpr is None
    assert result.tpr_gap == 0.5


@pytest.mark.parametrize(
    ("y_true", "y_pred", "groups"),
    [
        ([1], [1, 0], ["A"]),
        ([1, 2], [1, 0], ["A", "B"]),
        ([1, 0], [1, 2], ["A", "B"]),
    ],
)
def test_invalid_inputs_raise_value_error(y_true, y_pred, groups):
    with pytest.raises(ValueError):
        audit_disparities(y_true, y_pred, groups)


def test_json_and_dict_round_trip():
    result = audit_disparities([1, 0, 1, 0], [1, 0, 1, 0], ["A", "A", "B", "B"])

    payload = result.to_dict()
    assert json.dumps(payload)
    assert AuditResult.from_dict(payload) == result

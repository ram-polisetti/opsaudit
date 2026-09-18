import pandas as pd
import pytest

from opsaudit.data import generate_dispatch, generate_staffing
from opsaudit.metrics import audit_disparities


@pytest.mark.parametrize(
    ("generator", "columns", "id_column", "decision", "outcome"),
    [
        (
            generate_dispatch,
            ["driver_id", "group", "packages_assigned", "priority_route", "on_time"],
            "driver_id",
            "priority_route",
            "on_time",
        ),
        (
            generate_staffing,
            [
                "worker_id",
                "group",
                "tenure_months",
                "shifts_requested",
                "shifts_granted",
                "completed_satisfactorily",
            ],
            "worker_id",
            "shifts_granted",
            "completed_satisfactorily",
        ),
    ],
)
def test_generator_columns_dtypes_and_binary_ranges(
    generator, columns, id_column, decision, outcome
):
    frame = generator(n=50, seed=7)

    assert frame.columns.tolist() == columns
    assert pd.api.types.is_string_dtype(frame[id_column])
    assert pd.api.types.is_string_dtype(frame["group"])
    assert pd.api.types.is_integer_dtype(frame[decision])
    assert pd.api.types.is_integer_dtype(frame[outcome])
    assert frame[decision].isin([0, 1]).all()
    assert frame[outcome].isin([0, 1]).all()
    if decision == "priority_route":
        assert frame["group"].isin(["A", "B", "C"]).all()
        assert frame["packages_assigned"].between(20, 119).all()
    else:
        assert frame["group"].isin(["FT", "PT", "temp"]).all()
        assert frame["tenure_months"].between(1, 59).all()
        assert frame["shifts_requested"].between(1, 7).all()


@pytest.mark.parametrize("generator", [generate_dispatch, generate_staffing])
def test_generators_are_deterministic(generator):
    pd.testing.assert_frame_equal(generator(n=50, seed=7), generator(n=50, seed=7))


def test_same_seed_produces_byte_identical_csvs(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    generate_dispatch(n=100, seed=7).to_csv(first, index=False)
    generate_dispatch(n=100, seed=7).to_csv(second, index=False)

    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("generator", [generate_dispatch, generate_staffing])
def test_different_seeds_produce_different_frames(generator):
    assert not generator(n=50, seed=7).equals(generator(n=50, seed=8))


@pytest.mark.parametrize(
    ("generator", "decision", "outcome"),
    [
        (generate_dispatch, "priority_route", "on_time"),
        (generate_staffing, "shifts_granted", "completed_satisfactorily"),
    ],
)
def test_bias_reduces_disparate_impact(generator, decision, outcome):
    clean = generator(n=5000, bias_strength=0.0, seed=42)
    biased = generator(n=5000, bias_strength=0.5, seed=42)

    clean_result = audit_disparities(clean[outcome], clean[decision], clean["group"])
    biased_result = audit_disparities(biased[outcome], biased[decision], biased["group"])
    assert biased_result.disparate_impact_ratio < clean_result.disparate_impact_ratio


@pytest.mark.parametrize("generator", [generate_dispatch, generate_staffing])
@pytest.mark.parametrize("bias_strength", [-0.01, 1.01])
def test_bias_outside_valid_range_raises_value_error(generator, bias_strength):
    with pytest.raises(ValueError):
        generator(bias_strength=bias_strength)

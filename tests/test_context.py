import pytest

from opsaudit.context import load_context, normalize_context


def test_load_context_normalizes_scalar_values(tmp_path):
    path = tmp_path / "context.yaml"
    path.write_text("system_name: dispatch\nmodel_version: 2\n")

    assert load_context(path) == {"system_name": "dispatch", "model_version": "2"}


@pytest.mark.parametrize(
    "context",
    [[], {"": "dispatch"}, {"system_name": ["dispatch"]}, {"system_name": None}],
)
def test_context_rejects_nonflat_or_missing_values(context):
    with pytest.raises(ValueError):
        normalize_context(context)

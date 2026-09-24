import json

from opsaudit.data import generate_dispatch
from opsaudit.metrics import AuditResult, audit_disparities
from opsaudit.report import save_report, to_html, to_markdown


def _result():
    frame = generate_dispatch(n=100, seed=7)
    return audit_disparities(frame["on_time"], frame["priority_route"], frame["group"])


def test_markdown_contains_groups_and_disparate_impact_ratio():
    markdown = to_markdown(_result())

    assert "A" in markdown
    assert "B" in markdown
    assert "C" in markdown
    assert "disparate_impact_ratio" in markdown


def test_html_has_document_start_and_group_labels():
    html = to_html(_result())

    assert html.startswith(("<!DOCTYPE html>", "<html"))
    assert "A" in html
    assert "B" in html
    assert "C" in html


def test_save_report_writes_all_formats_and_json_round_trips(tmp_path):
    paths = save_report(_result(), tmp_path / "nested" / "report")

    assert [path.suffix for path in paths] == [".md", ".html", ".json"]
    assert all(path.exists() for path in paths)
    loaded = json.loads(paths[-1].read_text())
    assert isinstance(AuditResult.from_dict(loaded), AuditResult)


def test_reports_include_context_review_and_bootstrap_evidence():
    result = audit_disparities(
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        ["A", "A", "B", "B"],
        min_group_n=3,
        bootstrap=10,
        context={"system_name": "dispatch"},
    )

    markdown = to_markdown(result)
    html = to_html(result)

    assert "## Audit context" in markdown
    assert "minimum_group_size" in markdown
    assert "Bootstrap confidence intervals" in markdown
    assert "Audit context" in html
    assert "minimum_group_size" in html

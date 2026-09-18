import json

import pandas as pd
from click.testing import CliRunner

from opsaudit.cli import main


def test_generate_writes_csv_with_expected_columns(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "dispatch.csv"

    result = runner.invoke(
        main,
        ["generate", "--scenario", "dispatch", "--n", "25", "--seed", "7", "--out", str(csv_path)],
    )

    assert result.exit_code == 0, result.output
    assert csv_path.exists()
    assert "priority_route" in csv_path.read_text()


def test_audit_writes_all_report_formats(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "dispatch.csv"
    report_prefix = tmp_path / "report"
    assert runner.invoke(
        main,
        ["generate", "--scenario", "dispatch", "--n", "5000", "--out", str(csv_path)],
    ).exit_code == 0

    result = runner.invoke(
        main,
        [
            "audit",
            "--data",
            str(csv_path),
            "--truth",
            "on_time",
            "--pred",
            "priority_route",
            "--group",
            "group",
            "--out",
            str(report_prefix),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.json").exists()


def test_audit_without_prediction_skips_error_rate_metrics(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "dispatch.csv"
    report_prefix = tmp_path / "outcome-rates"
    assert runner.invoke(
        main,
        ["generate", "--scenario", "dispatch", "--n", "50", "--out", str(csv_path)],
    ).exit_code == 0

    result = runner.invoke(
        main,
        [
            "audit",
            "--data",
            str(csv_path),
            "--truth",
            "on_time",
            "--group",
            "group",
            "--out",
            str(report_prefix),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "prediction column not provided; skipping error-rate metrics" in result.output
    assert "n/a" in (tmp_path / "outcome-rates.md").read_text()


def test_gate_exits_zero_for_clean_report_and_one_for_biased_report(tmp_path):
    runner = CliRunner()
    for bias, expected_exit_code in [(0.0, 0), (0.5, 1)]:
        csv_path = tmp_path / f"dispatch-{bias}.csv"
        report_prefix = tmp_path / f"report-{bias}"
        assert runner.invoke(
            main,
            [
                "generate",
                "--scenario",
                "dispatch",
                "--n",
                "5000",
                "--bias",
                str(bias),
                "--out",
                str(csv_path),
            ],
        ).exit_code == 0
        assert runner.invoke(
            main,
            [
                "audit",
                "--data",
                str(csv_path),
                "--truth",
                "on_time",
                "--pred",
                "priority_route",
                "--group",
                "group",
                "--out",
                str(report_prefix),
            ],
        ).exit_code == 0

        result = runner.invoke(main, ["gate", "--report", f"{report_prefix}.json"])
        assert result.exit_code == expected_exit_code, result.output


def test_audit_accepts_context_bootstrap_and_intersectional_groups(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "dispatch.csv"
    context_path = tmp_path / "context.yaml"
    report_prefix = tmp_path / "report"
    assert runner.invoke(
        main,
        ["generate", "--scenario", "dispatch", "--n", "100", "--out", str(csv_path)],
    ).exit_code == 0
    frame = pd.read_csv(csv_path)
    frame["service_level"] = ["standard" if index % 2 else "premium" for index in range(len(frame))]
    frame.to_csv(csv_path, index=False)
    context_path.write_text("system_name: route priority\nmodel_version: 1\n")

    result = runner.invoke(
        main,
        [
            "audit",
            "--data",
            str(csv_path),
            "--truth",
            "on_time",
            "--pred",
            "priority_route",
            "--group",
            "group",
            "--group",
            "service_level",
            "--context",
            str(context_path),
            "--min-group-n",
            "10",
            "--bootstrap",
            "10",
            "--out",
            str(report_prefix),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["context"]["system_name"] == "route priority"
    assert payload["context"]["audit_group_columns"] == "group, service_level"
    assert payload["bootstrap_samples"] == 10
    assert all("group=" in group["group"] for group in payload["groups"])


def test_gate_exits_two_when_evidence_requires_review(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "small.csv"
    report_prefix = tmp_path / "small-report"
    pd.DataFrame(
        {
            "on_time": [1, 1, 0],
            "priority_route": [1, 1, 1],
            "group": ["A", "B", "B"],
        }
    ).to_csv(csv_path, index=False)
    audit_result = runner.invoke(
        main,
        [
            "audit",
            "--data",
            str(csv_path),
            "--truth",
            "on_time",
            "--pred",
            "priority_route",
            "--group",
            "group",
            "--min-group-n",
            "2",
            "--out",
            str(report_prefix),
        ],
    )

    assert audit_result.exit_code == 0, audit_result.output
    result = runner.invoke(main, ["gate", "--report", f"{report_prefix}.json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["status"] == "review"

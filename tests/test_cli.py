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

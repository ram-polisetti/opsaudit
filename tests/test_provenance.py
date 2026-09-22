"""Tests for the tamper-evident provenance trail (audit/generate/verify/signoff)."""

import json

import pandas as pd
from click.testing import CliRunner

from opsaudit.cli import main
from opsaudit.provenance import (
    attach_body_hash,
    canonical_hash,
    check_body_hash,
    check_signoff_chain,
    sha256_file,
)


def _run_audit(tmp_path, **kwargs):
    """Generate data and audit it; return (csv_path, report_json_path)."""
    runner = CliRunner()
    csv_path = tmp_path / "data.csv"
    assert runner.invoke(
        main, ["generate", "--scenario", "dispatch", "--n", "500", "--out", str(csv_path)]
    ).exit_code == 0
    args = [
        "audit",
        "--data", str(csv_path),
        "--truth", "on_time",
        "--pred", "priority_route",
        "--group", "group",
        "--out", str(tmp_path / "report"),
    ]
    for key, value in kwargs.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    return csv_path, tmp_path / "report.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_audit_report_embeds_provenance_block(tmp_path):
    csv_path, report_path = _run_audit(tmp_path)
    document = _load(report_path)
    provenance = document["provenance"]
    assert provenance["tool"] == "opsaudit"
    assert provenance["opsaudit_version"]
    assert provenance["input"]["sha256"] == sha256_file(csv_path)
    assert provenance["input"]["row_count"] == 500
    assert provenance["run"]["command"] == "audit"
    assert provenance["run"]["args"]["truth"] == "on_time"
    assert provenance["run"]["args"]["groups"] == ["group"]
    assert provenance["run"]["seed"] == 42
    assert provenance["run"]["timestamp_utc"].endswith("Z")
    assert provenance["run"]["python_version"]
    assert provenance["gate"]["status"] in ("pass", "fail", "review")
    assert document["signoffs"] == []
    ok, reason = check_body_hash(document)
    assert ok, reason


def test_generate_writes_provenance_sidecar(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "data.csv"
    assert runner.invoke(
        main, ["generate", "--scenario", "staffing", "--n", "100", "--seed", "3", "--out", str(csv_path)]
    ).exit_code == 0
    sidecar = tmp_path / "data.provenance.json"
    assert sidecar.exists()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["run"]["command"] == "generate"
    assert provenance["run"]["args"] == {"scenario": "staffing", "n": 100, "bias": 0.0, "seed": 3}
    assert provenance["output"]["sha256"] == sha256_file(csv_path)
    assert provenance["output"]["row_count"] == 100


def test_verify_passes_on_untampered_report(tmp_path):
    csv_path, report_path = _run_audit(tmp_path)
    result = CliRunner().invoke(
        main, ["verify", "--report", str(report_path), "--data", str(csv_path)]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["verified"] is True
    assert all(check["passed"] for check in summary["checks"])


def test_verify_detects_modified_data(tmp_path):
    csv_path, report_path = _run_audit(tmp_path)
    frame = pd.read_csv(csv_path)
    frame.loc[0, "priority_route"] = 1 - frame.loc[0, "priority_route"]
    frame.to_csv(csv_path, index=False)
    result = CliRunner().invoke(
        main, ["verify", "--report", str(report_path), "--data", str(csv_path)]
    )
    assert result.exit_code == 1
    summary = json.loads(result.output)
    assert summary["verified"] is False
    data_check = next(c for c in summary["checks"] if c["check"] == "data_hash")
    assert data_check["passed"] is False
    assert "modified" in data_check["detail"]


def test_verify_detects_modified_report_metrics(tmp_path):
    csv_path, report_path = _run_audit(tmp_path)
    document = _load(report_path)
    document["disparate_impact_ratio"] = 0.999
    report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    result = CliRunner().invoke(
        main, ["verify", "--report", str(report_path), "--data", str(csv_path)]
    )
    assert result.exit_code == 1
    summary = json.loads(result.output)
    assert summary["verified"] is False
    assert any(
        c["check"] == "body_hash" and not c["passed"] for c in summary["checks"]
    )
    assert any(
        c["check"] == "deterministic_rerun" and not c["passed"]
        for c in summary["checks"]
    )


def test_verify_detects_modified_signoff(tmp_path):
    csv_path, report_path = _run_audit(tmp_path)
    runner = CliRunner()
    assert runner.invoke(
        main,
        ["signoff", "--report", str(report_path), "--reviewer", "R. E. Viewer",
         "--decision", "approve", "--note", "original note"],
    ).exit_code == 0
    document = _load(report_path)
    document["signoffs"][0]["note"] = "rewritten after the fact"
    report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    result = runner.invoke(
        main, ["verify", "--report", str(report_path), "--data", str(csv_path)]
    )
    assert result.exit_code == 1
    summary = json.loads(result.output)
    chain = next(c for c in summary["checks"] if c["check"] == "signoff_chain")
    assert chain["passed"] is False


def test_signoff_round_trip_and_chain(tmp_path):
    csv_path, report_path = _run_audit(tmp_path)
    runner = CliRunner()
    for reviewer, decision in (("Alice", "reject"), ("Bob", "approve")):
        result = runner.invoke(
            main,
            ["signoff", "--report", str(report_path), "--reviewer", reviewer,
             "--decision", decision, "--note", f"note by {reviewer}"],
        )
        assert result.exit_code == 0, result.output
    document = _load(report_path)
    assert len(document["signoffs"]) == 2
    assert document["signoffs"][0]["prev_hash"] == document["provenance"]["body_sha256"]
    assert (
        document["signoffs"][1]["prev_hash"]
        == canonical_hash(document["signoffs"][0])
    )
    ok, reason = check_signoff_chain(document)
    assert ok, reason
    result = runner.invoke(
        main, ["verify", "--report", str(report_path), "--data", str(csv_path)]
    )
    assert result.exit_code == 0, result.output


def test_signoff_refuses_tampered_report(tmp_path):
    _, report_path = _run_audit(tmp_path)
    document = _load(report_path)
    document["fpr_gap"] = 0.0
    report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        ["signoff", "--report", str(report_path), "--reviewer", "Mallory",
         "--decision", "approve"],
    )
    assert result.exit_code != 0
    assert "refusing to sign" in result.output


def test_verify_rejects_report_without_provenance(tmp_path):
    _, report_path = _run_audit(tmp_path)
    document = _load(report_path)
    del document["provenance"]
    del document["signoffs"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    result = CliRunner().invoke(
        main, ["verify", "--report", str(legacy), "--data", str(tmp_path / "data.csv")]
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["verified"] is False


def test_verify_no_prediction_path(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "data.csv"
    assert runner.invoke(
        main, ["generate", "--scenario", "dispatch", "--n", "300", "--out", str(csv_path)]
    ).exit_code == 0
    assert runner.invoke(
        main,
        ["audit", "--data", str(csv_path), "--truth", "on_time",
         "--group", "group", "--out", str(tmp_path / "nopred")],
    ).exit_code == 0
    result = runner.invoke(
        main, ["verify", "--report", str(tmp_path / "nopred.json"), "--data", str(csv_path)]
    )
    assert result.exit_code == 0, result.output


def test_deterministic_rerun_matches_across_runs(tmp_path):
    csv_path, first = _run_audit(tmp_path)
    runner = CliRunner()
    assert runner.invoke(
        main,
        ["audit", "--data", str(csv_path), "--truth", "on_time",
         "--pred", "priority_route", "--group", "group",
         "--out", str(tmp_path / "second")],
    ).exit_code == 0
    first_doc, second_doc = _load(first), _load(tmp_path / "second.json")
    strip = lambda doc: {k: v for k, v in doc.items() if k not in ("provenance", "signoffs")}
    assert strip(first_doc) == strip(second_doc)


def test_attach_body_hash_excludes_signoffs(tmp_path):
    _, report_path = _run_audit(tmp_path)
    document = _load(report_path)
    document["signoffs"].append({"reviewer": "X"})
    ok, reason = check_body_hash(document)
    assert ok, reason
    attach_body_hash(document)
    ok, reason = check_body_hash(document)
    assert ok, reason

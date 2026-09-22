"""Tests for the CI disparity-gate orchestration (opsaudit.gate_ci)."""

import json

import pytest

from opsaudit.data import generate_dispatch
from opsaudit.gate_ci import (
    append_verdict_log,
    decide_exit,
    load_gate_config,
    render_verdict_markdown,
    run_gate,
    write_gate_outputs,
)

FIXED_TIME = "2026-01-15T12:00:00+00:00"
FIXED_SHA = "abc123def456"

BASE_CONFIG = """\
version: 1
audit:
  data: {data}
  truth: on_time
  pred: priority_route
  groups: [group]
  min_group_n: 30
  bootstrap: 0
  bootstrap_seed: 42
thresholds:
  disparate_impact_ratio: {{min: 0.8}}
  demographic_parity_diff: {{max: 0.2}}
  tpr_gap: {{max: 0.15}}
  fpr_gap: {{max: 0.15}}
gate:
  fail_mode: block
  review: block
"""


def _setup(tmp_path, *, bias_strength, seed=42, n=4000):
    frame = generate_dispatch(n=n, bias_strength=bias_strength, seed=seed)
    frame.to_csv(tmp_path / "decisions.csv", index=False)
    config_path = tmp_path / ".opsaudit-gate.yml"
    config_path.write_text(BASE_CONFIG.format(data="decisions.csv"), encoding="utf-8")
    return load_gate_config(config_path), tmp_path


def test_clean_data_passes_gate(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.0)
    verdict = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)

    assert verdict["status"] == "pass"
    assert verdict["exit_code"] == 0
    assert verdict["commit_sha"] == FIXED_SHA
    assert verdict["timestamp_utc"] == FIXED_TIME
    assert verdict["schema"] == "opsaudit-gate-verdict/1"
    assert all(f["status"] != "fail" for f in verdict["findings"])
    assert verdict["metrics"]["n_total"] == 4000
    assert len(verdict["data_sha256"]) == 64


def test_biased_data_fails_gate(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.9)
    verdict = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)

    assert verdict["status"] == "fail"
    assert verdict["exit_code"] == 1
    failed = [f for f in verdict["findings"] if f["status"] == "fail"]
    assert failed, "expected at least one failing finding"
    assert any(f["check"] == "disparate_impact_ratio" for f in failed)


def test_review_status_when_evidence_too_thin(tmp_path):
    # No prediction column and groups below min_group_n: the audit cannot
    # compute error metrics and must escalate to a human instead of passing.
    frame = generate_dispatch(n=60, bias_strength=0.0, seed=7)
    frame.to_csv(tmp_path / "decisions.csv", index=False)
    config_path = tmp_path / ".opsaudit-gate.yml"
    config_path.write_text(
        "version: 1\n"
        "audit:\n"
        "  data: decisions.csv\n"
        "  truth: on_time\n"
        "  groups: [group]\n"
        "  min_group_n: 500\n"
        "gate:\n"
        "  fail_mode: block\n"
        "  review: block\n",
        encoding="utf-8",
    )
    verdict = run_gate(load_gate_config(config_path), tmp_path, timestamp=FIXED_TIME)

    assert verdict["status"] == "review"
    assert verdict["exit_code"] == 2
    assert verdict["review_reasons"], "expected evidence-quality review reasons"


def test_decide_exit_matrix(tmp_path):
    config, _ = _setup(tmp_path, bias_strength=0.0)

    assert decide_exit("pass", config) == 0
    assert decide_exit("fail", config) == 1
    assert decide_exit("review", config) == 2

    config.review_policy = "pass"
    assert decide_exit("review", config) == 0

    config.fail_mode = "advisory"
    assert decide_exit("pass", config) == 0
    assert decide_exit("fail", config) == 0
    assert decide_exit("review", config) == 0


def test_verdict_is_deterministic(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.5)
    first = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)
    second = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)

    assert first == second


def test_render_verdict_markdown_pass(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.0)
    verdict = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)
    body = render_verdict_markdown(verdict, run_url="https://example.invalid/run/1")

    assert "PASS" in body
    assert "disparate_impact_ratio" in body
    assert "https://example.invalid/run/1" in body
    assert "THE GATE SAYS NO" not in body


def test_render_verdict_markdown_fail_names_slices(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.9)
    verdict = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)
    body = render_verdict_markdown(verdict)

    assert "THE GATE SAYS NO" in body
    assert "Affected slices" in body
    assert "| C |" in body  # worst-off group appears in the slice table


def test_append_verdict_log_is_jsonl(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.0)
    log = tmp_path / "audit-log" / "verdicts.jsonl"
    append_verdict_log(run_gate(config, root, sha="aaa", timestamp=FIXED_TIME), log)
    append_verdict_log(run_gate(config, root, sha="bbb", timestamp=FIXED_TIME), log)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["commit_sha"] == "aaa"
    assert json.loads(lines[1])["commit_sha"] == "bbb"


def test_write_gate_outputs(tmp_path):
    config, root = _setup(tmp_path, bias_strength=0.0)
    verdict = run_gate(config, root, sha=FIXED_SHA, timestamp=FIXED_TIME)
    paths = write_gate_outputs(verdict, tmp_path / "gate-out")

    assert paths["verdict_json"].is_file()
    assert paths["verdict_markdown"].is_file()
    assert (
        json.loads(paths["verdict_json"].read_text(encoding="utf-8"))["status"]
        == "pass"
    )


def test_minimal_config_is_valid(tmp_path):
    config_path = tmp_path / ".opsaudit-gate.yml"
    config_path.write_text(
        "audit: {data: decisions.csv, truth: on_time, groups: [group]}",
        encoding="utf-8",
    )
    config = load_gate_config(config_path)  # version defaults to 1

    assert config.fail_mode == "block"
    assert config.review_policy == "block"
    assert config.bootstrap == 0


@pytest.mark.parametrize(
    "body",
    [
        "version: 2\naudit: {data: x.csv, truth: y, groups: [g]}",
        "version: 1\naudit: {truth: on_time, groups: [group]}",
        "version: 1\naudit: {data: decisions.csv, truth: on_time}",
        "version: 1\naudit: {data: decisions.csv, truth: on_time, groups: [group], bootstrap: -1}",
        "version: 1\naudit: {data: decisions.csv, truth: on_time, groups: [group], min_group_n: 0}",
        "version: 1\naudit: {data: decisions.csv, truth: on_time, groups: [group]}\n"
        "thresholds: {bogus_check: {max: 0.2}}",
        "version: 1\naudit: {data: decisions.csv, truth: on_time, groups: [group]}\n"
        "thresholds: {tpr_gap: {max: 'low'}}",
        "version: 1\naudit: {data: decisions.csv, truth: on_time, groups: [group]}\n"
        "gate: {fail_mode: explode}",
        "version: 1\ngate: {fail_mode: block}",
        "- just\n- a\n- list",
    ],
)
def test_invalid_configs_rejected(tmp_path, body):
    config_path = tmp_path / ".opsaudit-gate.yml"
    config_path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_gate_config(config_path)


def test_missing_data_file_raises(tmp_path):
    config_path = tmp_path / ".opsaudit-gate.yml"
    config_path.write_text(BASE_CONFIG.format(data="nope.csv"), encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        run_gate(load_gate_config(config_path), tmp_path, timestamp=FIXED_TIME)


def test_gate_run_cli_pass_and_fail(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from opsaudit.cli import main

    runner = CliRunner()
    for bias, expected_exit in ((0.0, 0), (0.9, 1)):
        frame = generate_dispatch(n=2000, bias_strength=bias, seed=11)
        frame.to_csv(tmp_path / "decisions.csv", index=False)
        (tmp_path / ".opsaudit-gate.yml").write_text(
            BASE_CONFIG.format(data="decisions.csv"), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            main, ["gate-run", "--config", ".opsaudit-gate.yml", "--out-dir", "gate-out"]
        )
        assert result.exit_code == expected_exit, result.output
        assert (tmp_path / "gate-out" / "verdict.json").is_file()
        assert (tmp_path / "gate-out" / "verdict.md").is_file()
        assert (tmp_path / "gate-out" / "report.json").is_file()

"""Tests for the synthetic data auditor (opsaudit.synthetics + CLI)."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from opsaudit.cli import main
from opsaudit.data import generate_dispatch
from opsaudit.synthetics import (
    SyntheticAuditConfig,
    _merge_synthetic_thresholds,
    audit_bias_amplification,
    audit_fidelity,
    audit_privacy,
    audit_synthetic,
    ks_statistic,
    total_variation_distance,
)

CONFIG = {
    "truth": "on_time",
    "pred": "priority_route",
    "groups": ["group"],
    "exclude": ["driver_id"],
}


def _frames(n=600, src_bias=0.0, syn_bias=0.0, src_seed=11, syn_seed=12):
    src = generate_dispatch(n=n, bias_strength=src_bias, seed=src_seed)
    syn = generate_dispatch(n=n, bias_strength=syn_bias, seed=syn_seed)
    return src, syn


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_requires_truth_pred_groups():
    with pytest.raises(ValueError):
        SyntheticAuditConfig.from_dict({"pred": "p", "groups": ["g"]})
    with pytest.raises(ValueError):
        SyntheticAuditConfig.from_dict({"truth": "t", "pred": "p", "groups": []})
    with pytest.raises(ValueError):
        SyntheticAuditConfig.from_dict({"truth": "t", "pred": "p", "groups": "g",
                                        "seed": "nope"})


def test_config_accepts_single_group_string():
    cfg = SyntheticAuditConfig.from_dict({"truth": "t", "pred": "p", "groups": "g"})
    assert cfg.groups == ["g"]


def test_threshold_overrides_and_unknown_axis_rejected():
    merged = _merge_synthetic_thresholds({"fidelity": {"pass_score": 0.95}})
    assert merged["fidelity"]["pass_score"] == 0.95
    assert merged["fidelity"]["review_score"] == 0.75  # default preserved
    with pytest.raises(ValueError):
        _merge_synthetic_thresholds({"nope": {}})
    with pytest.raises(ValueError):
        _merge_synthetic_thresholds({"fidelity": {"pass_score": "high"}})
    with pytest.raises(ValueError):
        _merge_synthetic_thresholds({"fidelity": {"bogus": 1.0}})


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def test_ks_statistic_identical_is_zero():
    assert ks_statistic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0


def test_ks_statistic_detects_shift():
    stat = ks_statistic(np.array([1.0, 2.0, 3.0, 4.0]), np.array([11.0, 12.0, 13.0, 14.0]))
    assert stat == 1.0


def test_ks_statistic_rejects_empty():
    with pytest.raises(ValueError):
        ks_statistic(np.array([]), np.array([1.0]))


def test_tvd_identical_is_zero_and_disjoint_is_one():
    assert total_variation_distance(pd.Series(["a", "b"]), pd.Series(["a", "b"])) == 0.0
    assert total_variation_distance(pd.Series(["a", "a"]), pd.Series(["b", "b"])) == 1.0


def test_fidelity_identical_frames_pass_with_perfect_score():
    src, _ = _frames()
    out = audit_fidelity(src, src.copy(), ["packages_assigned"], ["group"])
    assert out["verdict"] == "pass"
    assert out["fidelity_score"] == pytest.approx(1.0)
    assert out["findings"] == []


def test_fidelity_degraded_column_scores_lower():
    src, _ = _frames()
    syn = src.copy()
    syn["packages_assigned"] = syn["packages_assigned"] + 500  # gross shift
    out = audit_fidelity(src, syn, ["packages_assigned"], ["group"])
    assert out["fidelity_score"] < 0.9
    assert any(f["code"] == "low_column_fidelity" for f in out["findings"])


def test_fidelity_flags_vanished_categories():
    src, _ = _frames()
    syn = src[src["group"] != "C"].copy()  # drop a whole group
    out = audit_fidelity(src, syn, ["packages_assigned"], ["group"])
    assert any(f["code"] == "vanished_categories" for f in out["findings"])


def test_correlation_drift_none_with_single_numeric():
    src, _ = _frames()
    out = audit_fidelity(src, src.copy(), ["packages_assigned"], [])
    assert out["correlation_drift"] is None


def test_correlation_drift_detects_broken_relationship():
    rng = np.random.default_rng(5)
    src = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    src["b"] = src["a"] * 2.0 + rng.normal(scale=0.01, size=500)  # tight correlation
    syn = src.copy()
    syn["b"] = rng.normal(size=500)  # correlation destroyed
    out = audit_fidelity(src, syn, ["a", "b"], [])
    assert out["correlation_drift"]["mean_abs_diff"] > 0.5


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def test_privacy_exact_copy_is_maximum_risk():
    src, _ = _frames(n=400)
    features_num = ["packages_assigned"]
    features_cat = ["group", "on_time", "priority_route"]
    out = audit_privacy(src, src.copy(), features_num, features_cat, seed=7)
    assert out["verdict"] == "fail"
    assert out["privacy_risk"] == pytest.approx(1.0)
    assert out["duplicate_rate"] == pytest.approx(1.0)


def test_privacy_overfit_memorization_detected():
    src, _ = _frames(n=400)
    rng = np.random.default_rng(9)
    syn = src.copy()
    # Near-copies: tiny jitter keeps rows distinct but suspiciously close.
    syn["packages_assigned"] = syn["packages_assigned"] + rng.integers(-1, 2, size=len(syn))
    out = audit_privacy(src, syn, ["packages_assigned"],
                        ["group", "on_time", "priority_route"], seed=7)
    assert out["privacy_risk"] > 0.45
    assert out["verdict"] == "fail"


def test_privacy_independent_draw_is_low_risk():
    src, syn = _frames(n=600)
    out = audit_privacy(src, syn, ["packages_assigned"], ["group"], seed=7)
    assert out["verdict"] == "pass"
    assert out["privacy_risk"] < 0.20


def test_privacy_is_deterministic_for_fixed_seed():
    src, syn = _frames(n=400)
    first = audit_privacy(src, syn, ["packages_assigned"], ["group"], seed=7)
    second = audit_privacy(src, syn, ["packages_assigned"], ["group"], seed=7)
    assert first["privacy_risk"] == second["privacy_risk"]


# ---------------------------------------------------------------------------
# Bias amplification
# ---------------------------------------------------------------------------

def test_bias_preserved_when_synthetic_matches_source():
    src, _ = _frames(n=1500, src_bias=0.4)
    out = audit_bias_amplification(
        src, src.copy(), "on_time", "priority_route", ["group"], bootstrap=50
    )
    assert out["verdict"] == "pass"
    assert out["amplification_factor"] == pytest.approx(1.0)


def test_bias_amplification_detected():
    src, syn = _frames(n=2000, src_bias=0.4, syn_bias=0.8, src_seed=21, syn_seed=22)
    out = audit_bias_amplification(
        src, syn, "on_time", "priority_route", ["group"], bootstrap=50
    )
    assert out["verdict"] == "fail"
    assert out["amplification_factor"] > 1.25
    assert any(f["code"] == "bias_amplified" for f in out["findings"])


def test_bias_introduced_when_source_was_fair():
    src, syn = _frames(n=2000, src_bias=0.0, syn_bias=0.7, src_seed=31, syn_seed=32)
    out = audit_bias_amplification(
        src, syn, "on_time", "priority_route", ["group"], bootstrap=50
    )
    assert out["verdict"] == "fail"
    assert out["amplification_factor"] is None  # ratio meaningless on a fair source
    assert any(f["code"] == "disparity_introduced" for f in out["findings"])


def test_bias_review_on_mild_drift():
    src, syn = _frames(n=3000, src_bias=0.0, syn_bias=0.25, src_seed=43, syn_seed=44)
    out = audit_bias_amplification(
        src, syn, "on_time", "priority_route", ["group"], bootstrap=0
    )
    assert out["verdict"] == "review"


def test_bias_missing_column_raises():
    src, syn = _frames()
    with pytest.raises(ValueError, match="missing column"):
        audit_bias_amplification(src, syn, "on_time", "priority_route", ["nope"])


def test_bias_noise_does_not_fire_with_bootstrap_cis():
    # Two independent fair draws: point estimates wobble, but the 95%
    # intervals overlap, so no verdict may fire.
    src, syn = _frames(n=4000, src_bias=0.0, syn_bias=0.0, src_seed=11, syn_seed=12)
    out = audit_bias_amplification(
        src, syn, "on_time", "priority_route", ["group"], bootstrap=200
    )
    assert out["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------

def test_audit_synthetic_overall_verdict_aggregation():
    src, _ = _frames(n=400)
    # Exact copy: fidelity pass, privacy fail -> overall fail.
    report = audit_synthetic(src, src.copy(), CONFIG, bootstrap=0)
    assert report["axes"]["fidelity"]["verdict"] == "pass"
    assert report["axes"]["privacy"]["verdict"] == "fail"
    assert report["verdict"] == "fail"


def test_audit_synthetic_rejects_empty_frames():
    src, _ = _frames()
    with pytest.raises(ValueError, match="must not be empty"):
        audit_synthetic(src.iloc[0:0], src.copy(), CONFIG)


def test_audit_synthetic_rejects_unknown_configured_column():
    src, _ = _frames()
    bad = dict(CONFIG, numeric=["nope"])
    with pytest.raises(ValueError, match="not present in both datasets"):
        audit_synthetic(src, src.copy(), bad)


def test_audit_synthetic_infers_columns_and_excludes_ids():
    src, syn = _frames(n=400)
    report = audit_synthetic(src, syn, {"truth": "on_time", "pred": "priority_route",
                                        "groups": ["group"],
                                        "exclude": ["driver_id"]})
    assert "driver_id" not in report["columns"]["numeric"]
    assert "driver_id" not in report["columns"]["categorical"]


def test_audit_synthetic_report_is_json_safe():
    import json as json_module

    src, syn = _frames(n=400)
    report = audit_synthetic(src, syn, CONFIG, bootstrap=0)
    json_module.dumps(report)  # must not raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_cli_inputs(tmp_path, n=1500, src_bias=0.0, syn_bias=0.0,
                      src_seed=11, syn_seed=12):
    src = generate_dispatch(n=n, bias_strength=src_bias, seed=src_seed)
    syn = generate_dispatch(n=n, bias_strength=syn_bias, seed=syn_seed)
    src_path = tmp_path / "source.csv"
    syn_path = tmp_path / "synthetic.csv"
    src.to_csv(src_path, index=False)
    syn.to_csv(syn_path, index=False)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return src_path, syn_path, config_path


def test_cli_audit_synthetic_pass_exit_zero(tmp_path):
    runner = CliRunner()
    src_path, syn_path, config_path = _write_cli_inputs(tmp_path, n=3000)
    out_path = tmp_path / "report.json"
    result = runner.invoke(main, ["audit-synthetic", "--source", str(src_path),
                                  "--synthetic", str(syn_path),
                                  "--config", str(config_path),
                                  "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    assert report["provenance"]["inputs"]["source"]["row_count"] == 3000
    assert "synthetic-data verdict: pass" in result.output


def test_cli_audit_synthetic_fail_exit_one(tmp_path):
    # Strongly biased synthetic against a fair source: the bias axis fails.
    runner = CliRunner()
    src_path, syn_path, config_path = _write_cli_inputs(
        tmp_path, n=2000, src_bias=0.0, syn_bias=0.7, src_seed=31, syn_seed=32
    )
    out_path = tmp_path / "report.json"
    result = runner.invoke(main, ["audit-synthetic", "--source", str(src_path),
                                  "--synthetic", str(syn_path),
                                  "--config", str(config_path),
                                  "--out", str(out_path)])
    assert result.exit_code == 1, result.output
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "fail"
    assert report["axes"]["bias"]["verdict"] == "fail"


def test_cli_audit_synthetic_rejects_bad_config(tmp_path):
    runner = CliRunner()
    src_path, syn_path, _ = _write_cli_inputs(tmp_path)
    bad_config = tmp_path / "bad.yml"
    bad_config.write_text(yaml.safe_dump({"truth": "on_time"}), encoding="utf-8")
    result = runner.invoke(main, ["audit-synthetic", "--source", str(src_path),
                                  "--synthetic", str(syn_path),
                                  "--config", str(bad_config),
                                  "--out", str(tmp_path / "r.json")])
    assert result.exit_code != 0
    assert "invalid synthetic-audit config" in result.output

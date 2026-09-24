"""CI orchestration for the disparity-gate GitHub Action.

This module holds all of the gate's decision logic in pure, deterministic,
unit-testable form: loading the ``.opsaudit-gate.yml`` config, running the
disparity audit, evaluating thresholds, rendering the PR verdict, and
appending to the append-only audit log. Everything that touches the GitHub
API (posting comments, uploading artifacts, pushing the log branch) lives in
the composite action at ``.github/actions/disparity-gate/action.yml`` and in
the ``opsaudit gate-run`` CLI command — not here.

Example:
    >>> from opsaudit.gate_ci import load_gate_config
    >>> isinstance(load_gate_config, object)
    True
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from . import __version__
from .cli import _group_labels, _outcome_rate_result, _require_columns
from .context import load_context
from .gate import _merge_thresholds, evaluate_gate_status
from .metrics import audit_disparities
from .provenance import build_provenance, sha256_file, utc_now
from .report import save_report

CONFIG_SCHEMA_VERSION = 1
VERDICT_SCHEMA = "opsaudit-gate-verdict/1"

GateStatus = Literal["pass", "fail", "review"]


@dataclass
class GateConfig:
    """A validated ``.opsaudit-gate.yml`` configuration."""

    data: str
    truth: str
    pred: str | None
    groups: list[str]
    min_group_n: int | None
    bootstrap: int
    bootstrap_seed: int
    context_file: str | None
    thresholds: dict[str, dict[str, float]] | None
    fail_mode: Literal["block", "advisory"]
    review_policy: Literal["block", "pass"]
    raw: dict[str, Any] = field(default_factory=dict)


def load_gate_config(path: str | Path) -> GateConfig:
    """Load and validate a ``.opsaudit-gate.yml`` file.

    Raises:
        ValueError: when the file is missing required keys, has unknown keys,
            or carries invalid threshold overrides.
    """
    path = Path(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("gate config must be a YAML mapping")
    if loaded.get("version", 1) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported gate config version: {loaded.get('version')!r} "
            f"(expected {CONFIG_SCHEMA_VERSION})"
        )

    audit = loaded.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("gate config requires an 'audit' mapping")
    data = audit.get("data")
    truth = audit.get("truth")
    groups = audit.get("groups")
    if not data or not isinstance(data, str):
        raise ValueError("gate config 'audit.data' must be a CSV path string")
    if not truth or not isinstance(truth, str):
        raise ValueError("gate config 'audit.truth' must be a column-name string")
    if not groups or not isinstance(groups, list) or not all(
        isinstance(g, str) for g in groups
    ):
        raise ValueError("gate config 'audit.groups' must be a non-empty list of column names")
    pred = audit.get("pred")
    if pred is not None and not isinstance(pred, str):
        raise ValueError("gate config 'audit.pred' must be a column-name string or omitted")

    min_group_n = audit.get("min_group_n")
    if min_group_n is not None and (
        isinstance(min_group_n, bool) or not isinstance(min_group_n, int) or min_group_n < 1
    ):
        raise ValueError("gate config 'audit.min_group_n' must be a positive integer")
    bootstrap = audit.get("bootstrap", 0)
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("gate config 'audit.bootstrap' must be a non-negative integer")
    bootstrap_seed = audit.get("bootstrap_seed", 42)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("gate config 'audit.bootstrap_seed' must be an integer")
    context_file = audit.get("context")
    if context_file is not None and not isinstance(context_file, str):
        raise ValueError("gate config 'audit.context' must be a path string or omitted")

    thresholds = loaded.get("thresholds")
    if thresholds is not None:
        _merge_thresholds(thresholds)  # validates shape, names, and finiteness

    gate = loaded.get("gate", {}) or {}
    if not isinstance(gate, dict):
        raise ValueError("gate config 'gate' must be a mapping")
    fail_mode = gate.get("fail_mode", "block")
    if fail_mode not in ("block", "advisory"):
        raise ValueError("gate config 'gate.fail_mode' must be 'block' or 'advisory'")
    review_policy = gate.get("review", "block")
    if review_policy not in ("block", "pass"):
        raise ValueError("gate config 'gate.review' must be 'block' or 'pass'")

    return GateConfig(
        data=data,
        truth=truth,
        pred=pred,
        groups=list(groups),
        min_group_n=min_group_n,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        context_file=context_file,
        thresholds=thresholds,
        fail_mode=fail_mode,
        review_policy=review_policy,
        raw=loaded,
    )


def run_gate(
    config: GateConfig,
    repo_root: str | Path,
    *,
    sha: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run the audit described by *config* and return a JSON-safe verdict.

    *repo_root* resolves the relative ``audit.data`` (and ``audit.context``)
    paths. *sha* and *timestamp* are recorded in the verdict; pass explicit
    values in tests for determinism.
    """
    root = Path(repo_root)
    data_path = root / config.data
    if not data_path.is_file():
        raise ValueError(f"audit data file not found: {data_path}")
    frame = pd.read_csv(data_path)
    _require_columns(frame, [config.truth, *config.groups] + ([config.pred] if config.pred else []))
    context = (
        load_context(root / config.context_file) if config.context_file else {}
    )
    context["audit_group_columns"] = ", ".join(config.groups)
    labels = _group_labels(frame, tuple(config.groups))
    if config.pred is None:
        result = _outcome_rate_result(
            frame[config.truth],
            labels,
            min_group_n=config.min_group_n,
            bootstrap=config.bootstrap,
            bootstrap_seed=config.bootstrap_seed,
            context=context,
        )
    else:
        result = audit_disparities(
            frame[config.truth],
            frame[config.pred],
            labels,
            min_group_n=config.min_group_n,
            bootstrap=config.bootstrap,
            bootstrap_seed=config.bootstrap_seed,
            context=context,
        )

    status, findings = evaluate_gate_status(result, config.thresholds)
    merged = _merge_thresholds(config.thresholds)
    verdict: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "timestamp_utc": timestamp or utc_now(),
        "commit_sha": sha,
        "opsaudit_version": __version__,
        "config": {
            "data": config.data,
            "truth": config.truth,
            "pred": config.pred,
            "groups": config.groups,
            "min_group_n": config.min_group_n,
            "bootstrap": config.bootstrap,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "thresholds": merged,
        "status": status,
        "exit_code": decide_exit(status, config),
        "findings": findings,
        "metrics": {
            "n_total": result.n_total,
            "disparate_impact_ratio": result.disparate_impact_ratio,
            "demographic_parity_diff": result.demographic_parity_diff,
            "tpr_gap": result.tpr_gap,
            "fpr_gap": result.fpr_gap,
        },
        "groups": [
            {
                "group": g.group,
                "n": g.n,
                "selection_rate": g.selection_rate,
                "tpr": g.tpr,
                "fpr": g.fpr,
                "accuracy": g.accuracy,
            }
            for g in result.groups
        ],
        "review_reasons": result.review_reasons,
        "warnings": result.warnings,
        "data_sha256": sha256_file(data_path),
    }
    return verdict


def decide_exit(status: GateStatus, config: GateConfig) -> int:
    """Map a gate status to the ``gate-run`` exit code (0 pass, 1 fail, 2 review).

    ``fail_mode: advisory`` never blocks: every status exits 0. Otherwise a
    ``review`` exits 2 unless ``gate.review: pass`` explicitly lets human
    review proceed without blocking the check.
    """
    if config.fail_mode == "advisory":
        return 0
    if status == "fail":
        return 1
    if status == "review":
        return 0 if config.review_policy == "pass" else 2
    return 0


def render_verdict_markdown(verdict: dict[str, Any], run_url: str | None = None) -> str:
    """Render the PR comment body for a gate verdict."""
    status = verdict["status"]
    banner = {
        "pass": "## ✅ Disparity gate: PASS",
        "fail": "## ⛔ Disparity gate: THE GATE SAYS NO",
        "review": "## 👀 Disparity gate: HUMAN REVIEW REQUIRED",
    }[status]
    lines = [banner, ""]
    n_total = verdict["metrics"]["n_total"]
    sha = verdict.get("commit_sha") or "unknown"
    lines.append(
        f"Audited `{verdict['config']['data']}` (n={n_total:,}) at commit `{sha[:12]}` "
        f"with opsaudit {verdict['opsaudit_version']}."
    )
    lines.append("")
    lines.append("| check | value | threshold | status |")
    lines.append("| --- | --- | --- | --- |")
    for finding in verdict["findings"]:
        check = finding["check"]
        value = _fmt(finding["value"])
        threshold = _fmt_threshold(finding["threshold"])
        mark = {"pass": "✅", "fail": "⛔", "review": "👀", "skipped": "⏭️"}[
            finding["status"]
        ]
        lines.append(f"| `{check}` | {value} | {threshold} | {mark} {finding['status']} |")
    lines.append("")
    lines.append("### Affected slices")
    lines.append("| group | n | selection rate | tpr | fpr |")
    lines.append("| --- | --- | --- | --- | --- |")
    for group in sorted(verdict["groups"], key=lambda g: g["selection_rate"]):
        lines.append(
            f"| {group['group']} | {group['n']:,} | {_fmt(group['selection_rate'])} "
            f"| {_fmt(group['tpr'])} | {_fmt(group['fpr'])} |"
        )
    lines.append("")
    if verdict["review_reasons"]:
        lines.append("### Evidence-quality notes (require a human)")
        for reason in verdict["review_reasons"]:
            lines.append(f"- 👀 **{reason['code']}**: {reason['message']}")
        lines.append("")
    if verdict["warnings"]:
        lines.append("### Warnings")
        for warning in verdict["warnings"]:
            lines.append(f"- ⚠️ **{warning['code']}**: {warning['message']}")
        lines.append("")
    if status == "fail":
        lines.append(
            "This PR is blocked: at least one fairness threshold breached. "
            "Fix the disparity (or, with justification, relax the threshold in "
            "`.opsaudit-gate.yml`) and push again."
        )
    elif status == "review":
        lines.append(
            "A human must decide before merge. A reviewer can record the decision "
            "with `opsaudit signoff --report <report.json> --reviewer <name> "
            "--decision approve|reject`."
        )
    else:
        lines.append("All thresholds pass. The gate is satisfied for this commit.")
    lines.append("")
    if run_url:
        lines.append(f"[Full audit report and verdict artifact]({run_url})")
        lines.append("")
    lines.append(
        "_Method: selection rate is the share of positive model decisions per group; "
        "disparate impact ratio is min/max selection rate (four-fifths rule); "
        "TPR/FPR gaps are the largest pairwise differences across groups._"
    )
    return "\n".join(lines) + "\n"


def append_verdict_log(verdict: dict[str, Any], log_path: str | Path) -> None:
    """Append one JSONL line to the append-only audit log.

    Keys are emitted in insertion order (see :func:`run_gate`); the file is
    created with its parent directories when missing. Appends are atomic
    enough for CI: each verdict is a single line written in one ``open``.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(verdict, sort_keys=False) + "\n")


def write_gate_outputs(
    verdict: dict[str, Any],
    out_dir: str | Path,
    run_url: str | None = None,
) -> dict[str, Path]:
    """Write ``verdict.json`` and ``verdict.md`` into *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    verdict_path = out / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    markdown_path = out / "verdict.md"
    markdown_path.write_text(
        render_verdict_markdown(verdict, run_url), encoding="utf-8"
    )
    return {"verdict_json": verdict_path, "verdict_markdown": markdown_path}


def build_gate_provenance(
    verdict: dict[str, Any], config: GateConfig, data_path: Path, n_rows: int
) -> dict[str, Any]:
    """Build the provenance block embedded in the gate's audit report."""
    provenance = build_provenance(
        command="gate-run",
        args={
            "data": config.data,
            "truth": config.truth,
            "pred": config.pred,
            "groups": config.groups,
            "min_group_n": config.min_group_n,
            "bootstrap": config.bootstrap,
            "bootstrap_seed": config.bootstrap_seed,
        },
        seed=config.bootstrap_seed,
        input_path=str(data_path),
        input_sha256=sha256_file(data_path),
        row_count=n_rows,
        opsaudit_version=__version__,
        gate={"status": verdict["status"], "findings": verdict["findings"]},
    )
    provenance["verdict"] = {
        "schema": verdict["schema"],
        "timestamp_utc": verdict["timestamp_utc"],
        "commit_sha": verdict["commit_sha"],
        "status": verdict["status"],
    }
    return provenance


def save_gate_report(
    verdict: dict[str, Any],
    config: GateConfig,
    repo_root: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Re-run the audit from *config* and save the full report bundle.

    The report's provenance block carries the gate verdict so
    ``opsaudit verify`` can confirm what the gate decided for this data.
    Returns the written report paths.
    """
    root = Path(repo_root)
    data_path = root / config.data
    frame = pd.read_csv(data_path)
    labels = _group_labels(frame, tuple(config.groups))
    context = load_context(root / config.context_file) if config.context_file else {}
    context["audit_group_columns"] = ", ".join(config.groups)
    if config.pred is None:
        result = _outcome_rate_result(
            frame[config.truth],
            labels,
            min_group_n=config.min_group_n,
            bootstrap=config.bootstrap,
            bootstrap_seed=config.bootstrap_seed,
            context=context,
        )
    else:
        result = audit_disparities(
            frame[config.truth],
            frame[config.pred],
            labels,
            min_group_n=config.min_group_n,
            bootstrap=config.bootstrap,
            bootstrap_seed=config.bootstrap_seed,
            context=context,
        )
    provenance = build_gate_provenance(verdict, config, data_path, len(frame))
    return save_report(result, str(Path(out_dir) / "report"), provenance=provenance)


def _fmt(value: Any) -> str:
    """Format a metric value for Markdown (``n/a`` for missing)."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_threshold(threshold: Any) -> str:
    """Format a threshold rule for Markdown (``—`` when absent)."""
    if not isinstance(threshold, dict):
        return "—"
    if "min" in threshold:
        return f"≥ {_fmt(threshold['min'])}"
    if "max" in threshold:
        return f"≤ {_fmt(threshold['max'])}"
    return "—"

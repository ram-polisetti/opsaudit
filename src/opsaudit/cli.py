"""The Click command-line interface for opsaudit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml

from . import __version__
from .context import load_context
from .data import generate_dispatch, generate_staffing
from .gate import _merge_thresholds, evaluate_gate_status
from .metrics import AuditResult, audit_disparities
from .provenance import (
    build_provenance,
    canonical_hash,
    check_body_hash,
    check_signoff_chain,
    sha256_file,
    signoff_prev_hash,
    utc_now,
)
from .report import save_report
from .synthetics import SyntheticAuditConfig, audit_synthetic


@click.group()
@click.version_option()
def main() -> None:
    """Audit operational ML decision disparities from the command line.

    Example:
        ``opsaudit generate --scenario dispatch --out data.csv``
    """


@main.command()
@click.option("--scenario", type=click.Choice(["dispatch", "staffing"]), required=True)
@click.option("--n", type=click.IntRange(min=1), default=5000, show_default=True)
@click.option("--bias", type=click.FloatRange(0, 1), default=0.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def generate(scenario: str, n: int, bias: float, seed: int, out: Path) -> None:
    """Generate a synthetic dispatch or staffing CSV file."""
    generator = generate_dispatch if scenario == "dispatch" else generate_staffing
    frame = generator(n=n, bias_strength=bias, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    provenance = build_provenance(
        command="generate",
        args={"scenario": scenario, "n": n, "bias": bias, "seed": seed},
        seed=seed,
        opsaudit_version=__version__,
    )
    provenance["output"] = {
        "path": str(out),
        "sha256": sha256_file(out),
        "row_count": len(frame),
    }
    sidecar = out.parent / f"{out.stem}.provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    click.echo(f"wrote {len(frame)} rows to {out}")
    click.echo(f"wrote provenance sidecar to {sidecar}")


@main.command()
@click.option("--data", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--truth", required=True)
@click.option("--pred", required=False)
@click.option("--group", "group_columns", required=True, multiple=True)
@click.option("--context", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--min-group-n", type=click.IntRange(min=1))
@click.option("--bootstrap", type=click.IntRange(0, 1000), default=0, show_default=True)
@click.option("--bootstrap-seed", type=int, default=42, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def audit(
    data: Path,
    truth: str,
    pred: str | None,
    group_columns: tuple[str, ...],
    context: Path | None,
    min_group_n: int | None,
    bootstrap: int,
    bootstrap_seed: int,
    out: Path,
) -> None:
    """Audit CSV decisions and save evidence-rich Markdown, HTML, and JSON reports."""
    frame = pd.read_csv(data)
    _require_columns(frame, [truth, *group_columns] + ([pred] if pred else []))
    audit_context = load_context(context) if context is not None else {}
    audit_context["audit_group_columns"] = ", ".join(group_columns)
    labels = _group_labels(frame, group_columns)
    if pred is None:
        click.echo("prediction column not provided; skipping error-rate metrics")
        result = _outcome_rate_result(
            frame[truth],
            labels,
            min_group_n=min_group_n,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
            context=audit_context,
        )
    else:
        result = audit_disparities(
            frame[truth],
            frame[pred],
            labels,
            min_group_n=min_group_n,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
            context=audit_context,
        )
    gate_status, gate_findings = evaluate_gate_status(result)
    provenance = build_provenance(
        command="audit",
        args={
            "truth": truth,
            "pred": pred,
            "groups": list(group_columns),
            "min_group_n": min_group_n,
            "bootstrap": bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "context_file": str(context) if context is not None else None,
        },
        seed=bootstrap_seed,
        input_path=str(data),
        input_sha256=sha256_file(data),
        row_count=len(frame),
        opsaudit_version=__version__,
        gate={"status": gate_status, "findings": gate_findings},
    )
    save_report(result, out, provenance=provenance)
    click.echo(f"gate status at audit time: {gate_status}")
    _print_audit_flags(result)
    for reason in result.review_reasons:
        click.echo(f"[REVIEW] {reason['code']}: {reason['message']}")


@main.command()
@click.option("--report", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--thresholds", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def gate(report: Path, thresholds: Path | None) -> None:
    """Evaluate a saved report: exit 0 for pass, 1 for fail, and 2 for review."""
    result = AuditResult.from_dict(json.loads(report.read_text(encoding="utf-8")))
    overrides = _load_thresholds(thresholds) if thresholds is not None else None
    status, findings = evaluate_gate_status(result, overrides)
    click.echo(json.dumps({"status": status, "findings": findings}))
    raise SystemExit({"pass": 0, "fail": 1, "review": 2}[status])


@main.command("gate-run")
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--sha", default=None, help="Commit SHA recorded in the verdict.")
@click.option("--run-url", default=None, help="CI run URL linked from the verdict.")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
def gate_run(config: Path, sha: str | None, run_url: str | None, out_dir: Path) -> None:
    """Run the full CI disparity gate from a ``.opsaudit-gate.yml`` config.

    Writes ``verdict.json``, ``verdict.md`` (the PR comment body), and the
    full audit report bundle into ``--out-dir``. Exits 0 for pass, 1 for
    fail, 2 for human review (unless the config says otherwise).
    """
    from .gate_ci import load_gate_config, run_gate, save_gate_report, write_gate_outputs

    gate_config = load_gate_config(config)
    repo_root = Path.cwd()
    verdict = run_gate(gate_config, repo_root, sha=sha)
    write_gate_outputs(verdict, out_dir, run_url)
    save_gate_report(verdict, gate_config, repo_root, out_dir)
    status = verdict["status"]
    click.echo(f"gate verdict: {status} (exit {verdict['exit_code']})")
    for finding in verdict["findings"]:
        if finding["status"] in ("fail", "review"):
            click.echo(f"[{finding['status'].upper()}] {finding['check']}")
    raise SystemExit(verdict["exit_code"])


@main.command()
@click.option("--report", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--data", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--thresholds", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def verify(report: Path, data: Path, thresholds: Path | None) -> None:
    """Verify a report's provenance: data hash, body hash, sign-off chain, and deterministic re-run.

    Exit 0 when every check passes; exit 1 with a clear reason otherwise.
    """
    document = json.loads(report.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            failures.append(f"{name}: {detail}")

    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        record("provenance_present", False, "report has no provenance block (predates v0.1.1 provenance)")
    else:
        record("provenance_present", True)
        expected_sha = (provenance.get("input") or {}).get("sha256")
        actual_sha = sha256_file(data)
        if expected_sha is None:
            record("data_hash", False, "provenance block records no input hash")
        elif actual_sha == expected_sha:
            record("data_hash", True, f"sha256 matches ({actual_sha[:12]}…)")
        else:
            record(
                "data_hash",
                False,
                "input data was modified after the audit "
                f"(expected {expected_sha[:12]}…, got {actual_sha[:12]}…)",
            )
        expected_rows = (provenance.get("input") or {}).get("row_count")
        frame = pd.read_csv(data)
        if expected_rows is None:
            record("row_count", False, "provenance block records no row count")
        elif len(frame) == expected_rows:
            record("row_count", True, f"{len(frame)} rows")
        else:
            record(
                "row_count",
                False,
                f"row count changed: expected {expected_rows}, got {len(frame)}",
            )

        body_ok, body_reason = check_body_hash(document)
        record("body_hash", body_ok, "" if body_ok else body_reason)

        chain_ok, chain_reason = check_signoff_chain(document)
        record("signoff_chain", chain_ok, "" if chain_ok else chain_reason)

        try:
            recomputed = _rerun_audit(document, frame)
            recorded_metrics = {
                key: value
                for key, value in document.items()
                if key not in ("provenance", "signoffs")
            }
            if recomputed.to_dict() == recorded_metrics:
                record("deterministic_rerun", True, "metrics identical on re-run")
            else:
                record(
                    "deterministic_rerun",
                    False,
                    "recomputed metrics differ from the recorded report — "
                    "the report was edited after the audit",
                )
            overrides = _load_thresholds(thresholds) if thresholds is not None else None
            rerun_status, _ = evaluate_gate_status(recomputed, overrides)
            recorded_gate = provenance.get("gate") or {}
            if thresholds is None and recorded_gate.get("status") == rerun_status:
                record("gate_status", True, f"status {rerun_status} confirmed")
            elif thresholds is None:
                record(
                    "gate_status",
                    False,
                    f"recorded gate status {recorded_gate.get('status')!r} does not "
                    f"match recomputed {rerun_status!r}",
                )
            else:
                record("gate_status", True, f"recomputed status with overrides: {rerun_status}")
        except Exception as exc:  # noqa: BLE001 - verification must report, not crash
            record("deterministic_rerun", False, f"re-run failed: {exc}")

    summary = {"verified": not failures, "checks": checks}
    click.echo(json.dumps(summary, indent=2))
    raise SystemExit(0 if not failures else 1)


@main.command()
@click.option("--report", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--reviewer", required=True, help="Name of the human reviewer.")
@click.option("--decision", type=click.Choice(["approve", "reject"]), required=True)
@click.option("--note", default="", show_default=True)
def signoff(report: Path, reviewer: str, decision: str, note: str) -> None:
    """Append a human review record to a report (the escalation point for REVIEW gates).

    Records are hash-chained to the report body and to each other; any later
    edit is detectable with ``opsaudit verify``.
    """
    document = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(document.get("provenance"), dict):
        raise click.UsageError("report has no provenance block; cannot sign it")
    body_ok, body_reason = check_body_hash(document)
    if not body_ok:
        raise click.UsageError(f"refusing to sign a tampered report: {body_reason}")
    chain_ok, chain_reason = check_signoff_chain(document)
    if not chain_ok:
        raise click.UsageError(f"refusing to sign: {chain_reason}")
    record = {
        "reviewer": reviewer,
        "decision": decision,
        "note": note,
        "timestamp_utc": utc_now(),
        "prev_hash": signoff_prev_hash(document),
    }
    record["record_hash"] = canonical_hash(record)
    document.setdefault("signoffs", []).append(record)
    report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    click.echo(
        f"recorded {decision} by {reviewer} "
        f"(sign-off #{len(document['signoffs'])} on this report)"
    )


def _rerun_audit(document: dict[str, Any], frame: pd.DataFrame) -> AuditResult:
    """Deterministically re-run the audit described by a report's provenance."""
    provenance = document["provenance"]
    args = provenance["run"]["args"]
    truth = args["truth"]
    pred = args.get("pred")
    group_columns = tuple(args["groups"])
    _require_columns(frame, [truth, *group_columns] + ([pred] if pred else []))
    labels = _group_labels(frame, group_columns)
    context = dict(document.get("context", {}))
    if pred is None:
        return _outcome_rate_result(
            frame[truth],
            labels,
            min_group_n=args.get("min_group_n"),
            bootstrap=args.get("bootstrap", 0),
            bootstrap_seed=args.get("bootstrap_seed", 42),
            context=context,
        )
    return audit_disparities(
        frame[truth],
        frame[pred],
        labels,
        min_group_n=args.get("min_group_n"),
        bootstrap=args.get("bootstrap", 0),
        bootstrap_seed=args.get("bootstrap_seed", 42),
        context=context,
    )


def _print_audit_flags(result: AuditResult) -> None:
    """Print concise metric flags after the audit report is written."""
    for flag in result.flags:
        threshold = flag["threshold"]
        operator = ">=" if "min" in threshold else "<="
        limit = threshold.get("min", threshold.get("max"))
        value = "n/a" if flag["value"] is None else f"{flag['value']:.3f}"
        status = "PASS" if flag["passed"] else "FAIL"
        click.echo(
            f"[{status}] {flag['check']}: {value} (threshold {operator} {limit:.3f})"
        )


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear Click error when required CSV columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise click.UsageError(f"CSV is missing required column(s): {', '.join(missing)}")


def _group_labels(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Return one group label column, combining repeated ``--group`` options."""
    if len(columns) == 1:
        return frame[columns[0]].astype(str)
    return frame.loc[:, list(columns)].astype(str).apply(
        lambda row: " | ".join(f"{column}={row[column]}" for column in columns), axis=1
    )


def _outcome_rate_result(
    y_true: pd.Series,
    groups: pd.Series,
    *,
    min_group_n: int | None,
    bootstrap: int,
    bootstrap_seed: int,
    context: dict[str, str],
) -> AuditResult:
    """Build an outcome-rate report that explicitly requires prediction review."""
    result = audit_disparities(
        y_true,
        y_true,
        groups,
        min_group_n=min_group_n,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        context=context,
    )
    for group in result.groups:
        group.tpr = None
        group.fpr = None
        group.precision = None
    result.tpr_gap = None
    result.fpr_gap = None
    for flag in result.flags:
        if flag["check"] in {"tpr_gap", "fpr_gap"}:
            flag["value"] = None
            flag["passed"] = True
    observation = {
        "code": "prediction_column_not_provided",
        "message": "Prediction-error metrics are unavailable because no prediction column was provided.",
    }
    result.warnings.append(observation)
    result.review_reasons.append(observation)
    return result


def _load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    """Load and validate a YAML threshold override file."""
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        raise ValueError("thresholds YAML must contain a mapping of checks to rules")
    _merge_thresholds(loaded)
    return loaded


@main.command("audit-synthetic")
@click.option("--source", "source_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--synthetic", "synthetic_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--bootstrap", type=click.IntRange(0, 1000), default=200, show_default=True)
@click.option("--bootstrap-seed", type=int, default=42, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def audit_synthetic_cmd(
    source_path: Path,
    synthetic_path: Path,
    config: Path,
    bootstrap: int,
    bootstrap_seed: int,
    out: Path,
) -> None:
    """Audit a synthetic dataset against its source on three axes.

    Evaluates statistical fidelity, membership-inference privacy risk, and
    bias amplification, then writes a JSON go/no-go report. Exits 0 for
    pass, 1 for fail, 2 for human review.
    """
    try:
        raw_config: Any = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise click.UsageError(f"could not parse config YAML: {exc}") from exc
    try:
        audit_config = SyntheticAuditConfig.from_dict(raw_config)
    except ValueError as exc:
        raise click.UsageError(f"invalid synthetic-audit config: {exc}") from exc

    source = pd.read_csv(source_path)
    synthetic = pd.read_csv(synthetic_path)
    try:
        report = audit_synthetic(
            source,
            synthetic,
            audit_config,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    report["provenance"] = build_provenance(
        command="audit-synthetic",
        args={
            "source": str(source_path),
            "synthetic": str(synthetic_path),
            "config": str(config),
            "bootstrap": bootstrap,
            "bootstrap_seed": bootstrap_seed,
        },
        seed=audit_config.seed,
        opsaudit_version=__version__,
    )
    report["provenance"]["inputs"] = {
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "row_count": len(source),
        },
        "synthetic": {
            "path": str(synthetic_path),
            "sha256": sha256_file(synthetic_path),
            "row_count": len(synthetic),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    click.echo(f"synthetic-data verdict: {report['verdict']}")
    for axis in ("fidelity", "privacy", "bias"):
        axis_result = report["axes"][axis]
        click.echo(f"[{axis_result['verdict'].upper()}] {axis}")
        for finding in axis_result["findings"]:
            click.echo(f"  - {finding['code']}: {finding['message']}")
    click.echo(f"wrote JSON report to {out}")
    raise SystemExit({"pass": 0, "fail": 1, "review": 2}[report["verdict"]])

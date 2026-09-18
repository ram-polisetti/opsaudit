"""The Click command-line interface for opsaudit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml

from .context import load_context
from .data import generate_dispatch, generate_staffing
from .gate import _merge_thresholds, evaluate_gate_status
from .metrics import AuditResult, audit_disparities
from .report import save_report


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
    click.echo(f"wrote {len(frame)} rows to {out}")


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
    save_report(result, out)
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

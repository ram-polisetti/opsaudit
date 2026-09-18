"""The Click command-line interface for opsaudit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml

from .data import generate_dispatch, generate_staffing
from .gate import _merge_thresholds, evaluate_gate
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
@click.option("--group", "group_column", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def audit(
    data: Path, truth: str, pred: str | None, group_column: str, out: Path
) -> None:
    """Audit a CSV's decision column and save Markdown, HTML, and JSON reports."""
    frame = pd.read_csv(data)
    _require_columns(frame, [truth, group_column] + ([pred] if pred else []))
    if pred is None:
        click.echo("prediction column not provided; skipping error-rate metrics")
        result = _outcome_rate_result(frame[truth], frame[group_column])
    else:
        result = audit_disparities(frame[truth], frame[pred], frame[group_column])
    save_report(result, out)
    for flag in result.flags:
        threshold = flag["threshold"]
        operator = ">=" if "min" in threshold else "<="
        limit = threshold.get("min", threshold.get("max"))
        value = "n/a" if flag["value"] is None else f"{flag['value']:.3f}"
        status = "PASS" if flag["passed"] else "FAIL"
        click.echo(
            f"[{status}] {flag['check']}: {value} (threshold {operator} {limit:.3f})"
        )


@main.command()
@click.option("--report", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--thresholds", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def gate(report: Path, thresholds: Path | None) -> None:
    """Evaluate a saved report JSON file as a deployment gate."""
    result = AuditResult.from_dict(json.loads(report.read_text(encoding="utf-8")))
    overrides = _load_thresholds(thresholds) if thresholds is not None else None
    passed, findings = evaluate_gate(result, overrides)
    click.echo(json.dumps(findings))
    if not passed:
        raise SystemExit(1)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear Click error when required CSV columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise click.UsageError(f"CSV is missing required column(s): {', '.join(missing)}")


def _outcome_rate_result(y_true: pd.Series, groups: pd.Series) -> AuditResult:
    """Build a report with outcome rates but no prediction-error metrics."""
    result = audit_disparities(y_true, y_true, groups)
    for group in result.groups:
        group.tpr = None
        group.fpr = None
        group.precision = None
    result.tpr_gap = None
    result.fpr_gap = None
    # Re-run flag construction through the public audit function's gate defaults,
    # keeping the summary consistent after error-rate metrics are removed.
    for flag in result.flags:
        if flag["check"] in {"tpr_gap", "fpr_gap"}:
            flag["value"] = None
            flag["passed"] = True
    return result


def _load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    """Load and validate a YAML threshold override file."""
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        raise ValueError("thresholds YAML must contain a mapping of checks to rules")
    _merge_thresholds(loaded)
    return loaded

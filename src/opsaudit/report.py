"""Markdown, HTML, and JSON audit-report generation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from . import __version__
from .metrics import AuditResult
from .rmf import RMF_MAPPING, rmf_section


METHODS_NOTE = (
    "Selection rate is the share of positive model decisions per group. "
    "Demographic parity difference is max minus min selection rate; disparate "
    "impact ratio is min over max selection rate (flagged below 0.8 per the "
    "four-fifths rule). TPR/FPR gaps are the largest pairwise differences across "
    "groups with defined rates. Metrics computed with opsaudit "
    f"{__version__}; see README glossary for definitions."
)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>opsaudit report</title>
<style>
body { font-family: system-ui, sans-serif; line-height: 1.5; color: #172033; margin: 2rem auto; max-width: 72rem; padding: 0 1rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; }
th, td { border: 1px solid #cbd5e1; padding: 0.45rem 0.6rem; text-align: left; }
th { background: #e8eef7; }
.pass { color: #166534; font-weight: 700; }.fail { color: #b91c1c; font-weight: 700; }
</style>
</head>
<body>
<h1>opsaudit report</h1>
<p>Generated: {{ generated }} | n={{ result.n_total }}</p>
<h2>Per-group metrics</h2>
<table><thead><tr><th>group</th><th>n</th><th>selection_rate</th><th>tpr</th><th>fpr</th><th>precision</th><th>accuracy</th></tr></thead>
<tbody>{% for group in groups %}<tr><td>{{ group.group }}</td><td>{{ group.n }}</td><td>{{ group.selection_rate }}</td><td>{{ group.tpr }}</td><td>{{ group.fpr }}</td><td>{{ group.precision }}</td><td>{{ group.accuracy }}</td></tr>{% endfor %}</tbody></table>
<h2>Disparity summary</h2>
<table><thead><tr><th>check</th><th>value</th><th>threshold</th><th>status</th></tr></thead>
<tbody>{% for flag in flags %}<tr><td>{{ flag.check }}</td><td>{{ flag.value }}</td><td>{{ flag.threshold }}</td><td class="{{ flag.status.lower() }}">{{ flag.status }}</td></tr>{% endfor %}</tbody></table>
<h2>NIST AI RMF mapping</h2>
<ul>{% for check, mapping in rmf_mapping.items() %}<li><strong>{{ check }}</strong> → {{ mapping.function }}: {{ mapping.rationale }}</li>{% endfor %}</ul>
<h2>Methods note</h2>
<p>{{ methods_note }}</p>
</body>
</html>
"""


def to_markdown(result: AuditResult) -> str:
    """Render an audit result as a Markdown report.

    Example:
        >>> "# opsaudit report" in to_markdown(AuditResult([], 0, 0.0, 1.0, None, None, []))
        True
    """
    lines = [
        "# opsaudit report",
        f"Generated: {_utc_timestamp()} | n={result.n_total}",
        "",
        "## Per-group metrics",
        "| group | n | selection_rate | tpr | fpr | precision | accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in result.groups:
        lines.append(
            "| {group} | {n} | {selection_rate} | {tpr} | {fpr} | {precision} | {accuracy} |".format(
                group=group.group,
                n=group.n,
                selection_rate=_format_value(group.selection_rate),
                tpr=_format_value(group.tpr),
                fpr=_format_value(group.fpr),
                precision=_format_value(group.precision),
                accuracy=_format_value(group.accuracy),
            )
        )
    lines.extend(
        [
            "",
            "## Disparity summary",
            "| check | value | threshold | status |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for flag in _display_flags(result):
        lines.append(
            f"| {flag['check']} | {flag['value']} | {flag['threshold']} | {flag['status']} |"
        )
    lines.extend(["", rmf_section(result), "", "## Methods note", METHODS_NOTE])
    return "\n".join(lines) + "\n"


def to_html(result: AuditResult) -> str:
    """Render an audit result as a self-contained HTML report.

    The report uses the module-level Jinja template and contains no external
    stylesheets, scripts, or network resources.

    Example:
        >>> to_html(AuditResult([], 0, 0.0, 1.0, None, None, [])).startswith("<!DOCTYPE html>")
        True
    """
    groups = [
        {
            "group": group.group,
            "n": group.n,
            "selection_rate": _format_value(group.selection_rate),
            "tpr": _format_value(group.tpr),
            "fpr": _format_value(group.fpr),
            "precision": _format_value(group.precision),
            "accuracy": _format_value(group.accuracy),
        }
        for group in result.groups
    ]
    return Template(HTML_TEMPLATE, autoescape=True).render(
        generated=_utc_timestamp(),
        result=result,
        groups=groups,
        flags=_display_flags(result),
        rmf_mapping=RMF_MAPPING,
        methods_note=METHODS_NOTE,
    )


def save_report(result: AuditResult, path: str | Path) -> list[Path]:
    """Write Markdown, HTML, and JSON reports with a shared path prefix.

    Parent directories are created when needed.

    Example:
        >>> [item.suffix for item in save_report(AuditResult([], 0, 0.0, 1.0, None, None, []), "/tmp/opsaudit-example")]
        ['.md', '.html', '.json']
    """
    prefix = Path(path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = [
        Path(f"{prefix}.md"),
        Path(f"{prefix}.html"),
        Path(f"{prefix}.json"),
    ]
    files[0].write_text(to_markdown(result), encoding="utf-8")
    files[1].write_text(to_html(result), encoding="utf-8")
    files[2].write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return files


def _utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_value(value: float | None) -> str:
    """Format metric values consistently for human-readable reports."""
    return "n/a" if value is None else f"{value:.3f}"


def _display_flags(result: AuditResult) -> list[dict[str, str]]:
    """Prepare audit flags for Markdown and HTML tables."""
    display: list[dict[str, str]] = []
    for flag in result.flags:
        threshold = flag["threshold"]
        if "min" in threshold:
            formatted_threshold = f">= {_format_value(float(threshold['min']))}"
        else:
            formatted_threshold = f"<= {_format_value(float(threshold['max']))}"
        display.append(
            {
                "check": str(flag["check"]),
                "value": _format_value(flag.get("value")),
                "threshold": formatted_threshold,
                "status": "PASS" if flag.get("passed") else "FAIL",
            }
        )
    return display

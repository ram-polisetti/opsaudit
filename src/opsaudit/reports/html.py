"""Render an :class:`~opsaudit.reports.report.AuditReport` as HTML.

A single standalone page: inline CSS only, no external assets, no
scripts, no network resources — and no sticky/fixed positioning, so it
renders the same in every preview context.
"""

from __future__ import annotations

import html as _html


_CSS = """
body { font-family: system-ui, sans-serif; line-height: 1.5; color: #172033;
       margin: 2rem auto; max-width: 72rem; padding: 0 1rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; }
th, td { border: 1px solid #cbd5e1; padding: 0.45rem 0.6rem;
         text-align: left; vertical-align: top; }
th { background: #e8eef7; }
.verdict-review { color: #b91c1c; font-weight: 700; }
.verdict-clear { color: #166534; font-weight: 700; }
.muted { color: #475569; }
code { background: #f1f5f9; padding: 0.1rem 0.3rem; border-radius: 0.25rem; }
"""


def _e(value: object) -> str:
    return _html.escape(str(value), quote=True)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return _e(value)


def to_html(report: object) -> str:
    """Render the report as a standalone HTML page."""
    verdict = str(getattr(report, "verdict", ""))
    verdict_class = (
        "verdict-review" if "REVIEW" in verdict else "verdict-clear"
    )
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>opsaudit agentic audit report</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<h1>Agentic audit report (opsaudit)</h1>",
        f"<p class=\"muted\">Generated: {_e(getattr(report, 'generated', ''))} | "
        f"opsaudit {_e(getattr(report, 'package_version', ''))} | "
        f"target: {_e(getattr(report, 'target_name', ''))} "
        f"({_e(getattr(report, 'target_type', ''))})</p>",
        f"<p>Verdict: <span class=\"{verdict_class}\">{_e(verdict)}</span> "
        f"(flagging threshold {_fmt(getattr(report, 'flag_threshold', None))}, "
        "heuristic — not a significance test)</p>",
        "<h2>Executive summary</h2>",
        f"<p>{_e(getattr(report, 'executive_summary', ''))}</p>",
        "<h2>Findings</h2>",
        "<table><thead><tr><th>round</th><th>generator</th><th>probes</th>"
        "<th>strength</th><th>key signal</th></tr></thead><tbody>",
    ]
    for finding in getattr(report, "findings", ()):
        parts.append(
            "<tr><td>{round}</td><td>{gen}</td><td>{n}</td><td>{s}</td>"
            "<td>{sig}</td></tr>".format(
                round=_e(finding.get("round", "")),
                gen=_e(finding.get("generator", "")),
                n=_e(finding.get("n_probes", "")),
                s=_fmt(finding.get("strength")),
                sig=_e(_finding_signal(finding)),
            )
        )
    parts.append("</tbody></table>")

    methodology = getattr(report, "methodology", {}) or {}
    parts.append("<h2>Methodology</h2>")
    parts.append(f"<p>{_e(methodology.get('approach', ''))}</p>")
    brief = methodology.get("brief", {}) or {}
    if brief:
        parts.append("<h3>Audit brief</h3>")
        parts.append(_kv_table(brief))
    budget = methodology.get("budget", {}) or {}
    if budget:
        parts.append("<h3>Budget</h3>")
        parts.append(_kv_table(budget))
    judges = methodology.get("judges", []) or []
    if judges:
        parts.append("<h3>Judges</h3><ul>")
        for card in judges:
            cal = card.get("calibration") or {}
            if cal:
                cal_s = (
                    f"calibration: kappa={_fmt(cal.get('kappa'))}, "
                    f"accuracy={_fmt(cal.get('accuracy'))}, "
                    f"{'PASS' if cal.get('passed') else 'FAIL'}"
                )
            else:
                cal_s = "calibration: none recorded"
            parts.append(
                f"<li><strong>{_e(card.get('judge_id', ''))}</strong> "
                f"(model: {_e(card.get('model_id', ''))}) — "
                f"{_e(card.get('rubric', ''))}<br>{cal_s}</li>"
            )
        parts.append("</ul>")

    parts.append("<h2>Limitations</h2><ul>")
    for limitation in getattr(report, "limitations", ()):
        parts.append(f"<li>{_e(limitation)}</li>")
    parts.append("</ul>")

    parts.append("<h2>NIST AI RMF mapping</h2>")
    parts.append(
        "<p class=\"muted\">Each mapping names the <em>evidence this audit "
        "contributes</em> toward a NIST AI RMF 1.0 subcategory. The tool "
        "assists — it does not satisfy a subcategory on its own.</p><ul>"
    )
    for mapping in getattr(report, "rmf_mappings", ()):
        sub = mapping.get("subcategory")
        where = _e(mapping["function"]) + (f" {_e(sub)}" if sub else "")
        parts.append(
            f"<li><strong>{_e(mapping['aspect'])}</strong> → {where}: "
            f"{_e(mapping['rationale'])}<br>"
            f"<span class=\"muted\">Limits: {_e(mapping['limits'])}</span></li>"
        )
    parts.append("</ul>")

    parts.append("<h2>Reproducibility</h2>")
    repro = getattr(report, "reproducibility", {}) or {}
    repro_rows = {
        "seed": repro.get("seed"),
        "package_version": repro.get("package_version"),
        "git_commit": getattr(report, "git_commit", None),
        "evidence_path": getattr(report, "evidence_path", ""),
    }
    parts.append(_kv_table(repro_rows))
    parts.append("</body></html>")
    return "\n".join(parts)


def _kv_table(mapping: dict) -> str:
    rows = "".join(
        f"<tr><td>{_e(k)}</td><td>{_fmt(v)}</td></tr>"
        for k, v in mapping.items()
    )
    return (
        "<table><thead><tr><th>field</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _finding_signal(finding: dict) -> str:
    """One compact signal string per round finding (deterministic)."""
    details = finding.get("details", {}) or {}
    # Judge findings take precedence: a counterfactual round over text
    # outputs carries both (possibly empty) gaps and judge labels.
    judge_findings = details.get("judge_findings", []) or []
    if judge_findings:
        parts = []
        for jf in judge_findings:
            by_attr = jf.get("by_attribute", {}) or {}
            for attr, agg in by_attr.items():
                parts.append(
                    f"{jf.get('judge_id')}:{attr} "
                    f"{agg.get('strongest_label')} gap="
                    f"{agg.get('strength', 0.0):.3f}"
                )
        return "; ".join(parts) if parts else "judge labels recorded"
    generator = finding.get("generator", "")
    if generator == "counterfactual":
        gaps = details.get("gaps", {}) or {}
        if gaps:
            attr = max(gaps, key=lambda k: gaps[k])
            return f"gap {attr}={gaps[attr]:.3f}"
        return "no gaps"
    if generator == "adversarial":
        return f"error_rate={details.get('error_rate', 0.0):.3f}"
    if generator == "metamorphic":
        return f"violation_rate={details.get('violation_rate', 0.0):.3f}"
    note = details.get("note")
    if note:
        return str(note)[:80]
    strength = finding.get("strength", 0.0)
    try:
        return f"strength={float(strength or 0.0):.3f}"
    except (TypeError, ValueError):
        return "strength=n/a"

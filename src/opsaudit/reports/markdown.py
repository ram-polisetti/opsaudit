"""Render an :class:`~opsaudit.reports.report.AuditReport` as Markdown.

Deterministic template fill — no LLM prose anywhere.
"""

from __future__ import annotations

from .rmf import rmf_section


def _cell(value: object) -> str:
    """Keep values inside one safe Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def to_markdown(report: object) -> str:
    """Render the report as Markdown."""
    lines = [
        "# Agentic audit report (opsaudit)",
        "",
        (f"Generated: {_cell(getattr(report, 'generated', ''))} | "
        f"opsaudit {getattr(report, 'package_version', '')} | "
        f"target: {_cell(getattr(report, 'target_name', ''))} "
        f"({_cell(getattr(report, 'target_type', ''))})"),
        "",
        "## Executive summary",
        "",
        getattr(report, "executive_summary", ""),
        "",
        "## Findings",
        "",
        "| round | generator | probes | strength | key signal |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for finding in getattr(report, "findings", ()):
        lines.append(
            "| {round} | {generator} | {n} | {strength} | {signal} |".format(
                round=_cell(finding.get("round", "")),
                generator=_cell(finding.get("generator", "")),
                n=_cell(finding.get("n_probes", "")),
                strength=_fmt(finding.get("strength")),
                signal=_cell(_finding_signal(finding)),
            )
        )
    methodology = getattr(report, "methodology", {}) or {}
    lines.extend(["", "## Methodology", ""])
    lines.append(_cell(methodology.get("approach", "")))
    lines.append("")
    brief = methodology.get("brief", {}) or {}
    if brief:
        lines.extend(["### Audit brief", "", "| field | value |", "| --- | --- |"])
        for key in sorted(brief):
            lines.append(f"| {_cell(key)} | {_cell(brief[key])} |")
        lines.append("")
    budget = methodology.get("budget", {}) or {}
    if budget:
        lines.extend(
            ["### Budget", "", "| field | value |", "| --- | --- |"]
        )
        for key in sorted(budget):
            lines.append(f"| {_cell(key)} | {_cell(budget[key])} |")
        lines.append("")
    judges = methodology.get("judges", []) or []
    if judges:
        lines.extend(["### Judges", ""])
        for card in judges:
            cal = card.get("calibration") or {}
            lines.append(
                f"- **{_cell(card.get('judge_id', ''))}** "
                f"(model: {_cell(card.get('model_id', ''))}, "
                f"prompt v{card.get('prompt_version', '?')}) — "
                f"{_cell(card.get('rubric', ''))}"
            )
            if cal:
                lines.append(
                    f"  - calibration: kappa={_fmt(cal.get('kappa'))}, "
                    f"accuracy={_fmt(cal.get('accuracy'))}, "
                    f"n={cal.get('n_scored', '?')}/{cal.get('n_items', '?')}, "
                    f"{'PASS' if cal.get('passed') else 'FAIL'}"
                )
            else:
                lines.append("  - calibration: none recorded")
        lines.append("")
    lines.extend(["## Limitations", ""])
    for limitation in getattr(report, "limitations", ()):
        lines.append(f"- {limitation}")
    lines.extend(["", rmf_section(report), "", "## Reproducibility", ""])
    repro = getattr(report, "reproducibility", {}) or {}
    lines.extend(["| field | value |", "| --- | --- |"])
    lines.append(f"| seed | {_cell(repro.get('seed'))} |")
    lines.append(f"| package_version | {_cell(repro.get('package_version'))} |")
    lines.append(f"| git_commit | {_cell(getattr(report, 'git_commit', None))} |")
    lines.append(f"| evidence_path | {_cell(getattr(report, 'evidence_path', ''))} |")
    target_desc = getattr(report, "target_description", {}) or {}
    if target_desc:
        lines.append(f"| target | {_cell(target_desc)} |")
    return "\n".join(lines) + "\n"


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
    strength = finding.get("strength", 0.0)
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
    return f"strength={float(strength or 0.0):.3f}"

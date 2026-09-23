"""Assemble deterministic agentic-audit reports from campaign evidence.

:func:`build_report` turns a
:class:`~opsaudit.agents.campaign.CampaignReport` into a frozen
:class:`AuditReport`. Everything is template fill over numbers and
recorded metadata — no LLM prose anywhere. The verdict is a heuristic
flagging rule (best finding strength at or above the flagging
threshold), explicitly labeled as *not* a significance test.

Round summaries prefer judge findings when present: a counterfactual
round over text outputs carries both (possibly empty) numeric gaps and
judge label-rate gaps, and the judge signal is the meaningful one.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .html import to_html
from .markdown import to_markdown
from .rmf import mappings_for_report

#: Default flagging threshold for the verdict heuristic. A campaign is
#: flagged when its best finding strength meets or exceeds this value.
#: This is a heuristic, not a significance test.
FLAG_THRESHOLD_DEFAULT = 0.2

VERDICT_REVIEW = "FINDINGS WARRANT REVIEW"
VERDICT_CLEAR = "NO MATERIAL FINDINGS"

_APPROACH = (
    "Agentic audit: an LLM planner directed deterministic probe "
    "generators (counterfactual, adversarial, metamorphic) in a "
    "plan → probe → observe → replan loop under a fixed budget. "
    "All statistics were computed in deterministic code; the LLM "
    "never performed arithmetic. Where text outputs were judged, "
    "LLM judges produced labels only (never numbers), and every "
    "judge carries a measured calibration score."
)

_LIMITATIONS = (
    "Findings are limited to what the campaign probed within its "
    "budget (see Budget); absence of a finding is not proof of absence.",
    "The verdict is a heuristic flagging rule, not a statistical "
    "significance test.",
    "LLM judges label unstructured text only; their labels are only "
    "as reliable as their measured calibration (see judge cards) and "
    "never touch numbers.",
    "Counterfactual symmetry guarantees come from the deterministic "
    "probe generators; any LLM-proposed probes were validated "
    "before use.",
)


def _json_safe(obj: Any) -> Any:
    """Best-effort JSON-safe copy."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


def _package_version() -> str:
    try:
        return version("opsaudit")
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str | None:
    try:
        here = Path(__file__).resolve()
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here.parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def _describe_target(target: Any) -> dict[str, Any]:
    if target is None:
        return {}
    describe = getattr(target, "describe", None)
    if callable(describe):
        try:
            info = describe() or {}
        except Exception:
            info = {}
        return _json_safe(info) if isinstance(info, dict) else {}
    return {}


def _target_name(target: Any, described: dict[str, Any]) -> str:
    name = getattr(target, "name", None) or described.get("name")
    return str(name) if name else "unknown-target"


def _judge_card(judge: Any) -> dict[str, Any]:
    calibration = getattr(judge, "calibration", None)
    cal_dict = None
    if calibration is not None:
        to_dict = getattr(calibration, "to_dict", None)
        cal_dict = (
            _json_safe(to_dict()) if callable(to_dict) else _json_safe(calibration)
        )
    return {
        "judge_id": str(
            getattr(judge, "judge_id", getattr(judge, "name", "judge"))
        ),
        "model_id": str(getattr(judge, "model_id", "unknown-model")),
        "prompt_version": getattr(judge, "prompt_version", 1),
        "rubric": str(getattr(judge, "rubric", "")),
        "calibration": cal_dict,
    }


def _strongest_judge_signal(
    findings: list[dict[str, Any]],
) -> tuple[float, str, str, str] | None:
    """Best (strength, judge_id, label, attribute) across judge findings."""
    best: tuple[float, str, str, str] | None = None
    for finding in findings:
        details = finding.get("details", {}) or {}
        for jf in details.get("judge_findings", []) or []:
            judge_id = str(jf.get("judge_id", "judge"))
            for attr, agg in (jf.get("by_attribute", {}) or {}).items():
                try:
                    strength = float(agg.get("strength", 0.0))
                except (TypeError, ValueError):
                    strength = 0.0
                label = str(agg.get("strongest_label", ""))
                candidate = (strength, judge_id, label, str(attr))
                if best is None or strength > best[0]:
                    best = candidate
    return best


def _strongest_numeric_gap(
    findings: list[dict[str, Any]],
) -> tuple[float, str, str, Any] | None:
    """Best (gap, attribute, generator, round) across numeric gaps."""
    best: tuple[float, str, str, Any] | None = None
    for finding in findings:
        details = finding.get("details", {}) or {}
        for attr, gap in (details.get("gaps", {}) or {}).items():
            try:
                gap_f = float(gap)
            except (TypeError, ValueError):
                continue
            candidate = (
                gap_f,
                str(attr),
                str(finding.get("generator", "")),
                finding.get("round", ""),
            )
            if best is None or gap_f > best[0]:
                best = candidate
    return best


def _executive_summary(
    *,
    target_name: str,
    target_type: str,
    rounds: int,
    probes_used: int,
    stop_reason: str,
    best_strength: float,
    verdict: str,
    flag_threshold: float,
    findings: list[dict[str, Any]],
) -> str:
    head = (
        f"Audit of {target_name} ({target_type}): "
        f"{rounds} round(s), {probes_used} probe(s). "
        f"Stop reason: {stop_reason}. "
        f"Best finding strength: {best_strength:.3f}. "
        f"Verdict: {verdict} (flagging threshold {flag_threshold:.3f}, "
        "heuristic — not a significance test)."
    )
    if not findings:
        return head + " No rounds produced findings."
    judge_signal = _strongest_judge_signal(findings)
    if judge_signal is not None:
        strength, judge_id, label, attr = judge_signal
        tail = (
            f"The strongest finding (strength {strength:.3f}): "
            f"{judge_id} '{label}' label-rate gap of {strength:.3f} "
            f"across {attr}."
        )
        return head + " " + tail
    gap = _strongest_numeric_gap(findings)
    if gap is not None:
        gap_f, attr, generator, round_no = gap
        tail = (
            f"The strongest finding (strength {gap_f:.3f}): "
            f"{generator} gap {attr}={gap_f:.3f} (round {round_no})."
        )
        return head + " " + tail
    return head + " No material gaps were observed."


@dataclass(frozen=True)
class AuditReport:
    """A complete, deterministic agentic-audit report.

    Frozen so a report, once assembled, cannot be silently altered.
    Render with :func:`to_markdown` / :func:`to_html`, or serialize
    with :meth:`to_dict`.
    """

    generated: str
    package_version: str
    target_name: str
    target_type: str
    target_description: dict[str, Any]
    brief: dict[str, Any]
    executive_summary: str
    verdict: str
    flag_threshold: float
    best_strength: float
    rounds: int
    probes_used: int
    stop_reason: str
    findings: list[dict[str, Any]]
    methodology: dict[str, Any]
    limitations: list[str]
    rmf_mappings: list[dict[str, Any]]
    reproducibility: dict[str, Any]
    git_commit: str | None
    evidence_path: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the whole report."""
        return {
            "generated": self.generated,
            "package_version": self.package_version,
            "target_name": self.target_name,
            "target_type": self.target_type,
            "target_description": _json_safe(self.target_description),
            "brief": _json_safe(self.brief),
            "executive_summary": self.executive_summary,
            "verdict": self.verdict,
            "flag_threshold": self.flag_threshold,
            "best_strength": self.best_strength,
            "rounds": self.rounds,
            "probes_used": self.probes_used,
            "stop_reason": self.stop_reason,
            "findings": _json_safe(self.findings),
            "methodology": _json_safe(self.methodology),
            "limitations": list(self.limitations),
            "rmf_mappings": _json_safe(self.rmf_mappings),
            "reproducibility": _json_safe(self.reproducibility),
            "git_commit": self.git_commit,
            "evidence_path": self.evidence_path,
        }


def build_report(
    campaign_report: Any,
    *,
    target: Any = None,
    judges: list[Any] | tuple[Any, ...] | None = None,
    budget: dict[str, Any] | None = None,
    flag_threshold: float = FLAG_THRESHOLD_DEFAULT,
    extra_limitations: tuple[str, ...] | list[str] = (),
) -> AuditReport:
    """Assemble an :class:`AuditReport` from a campaign report.

    Args:
        campaign_report: A
            :class:`~opsaudit.agents.campaign.CampaignReport`.
        target: The audited target (for name/description); the brief is
            used as fallback.
        judges: Judges whose labels appear in the findings; their
            calibration cards are recorded in the methodology.
        budget: Budget dict recorded in the methodology.
        flag_threshold: Best finding strength at or above which the
            verdict is ``FINDINGS WARRANT REVIEW``.
        extra_limitations: Additional limitation strings.
    """
    brief = _json_safe(getattr(campaign_report, "brief", {}) or {})
    findings = _json_safe(getattr(campaign_report, "findings", []) or [])
    best_strength = float(getattr(campaign_report, "best_strength", 0.0) or 0.0)
    rounds = int(getattr(campaign_report, "rounds", 0) or 0)
    probes_used = int(getattr(campaign_report, "probes_used", 0) or 0)
    stop_reason = str(getattr(campaign_report, "stop_reason", ""))
    seed = getattr(campaign_report, "seed", None)
    evidence_path = str(getattr(campaign_report, "evidence_path", ""))

    described = _describe_target(target)
    target_name = _target_name(target, described)
    target_type = str(
        described.get("target_type")
        or brief.get("target_type", "unknown")
    )

    verdict = (
        VERDICT_REVIEW if best_strength >= flag_threshold else VERDICT_CLEAR
    )
    executive_summary = _executive_summary(
        target_name=target_name,
        target_type=target_type,
        rounds=rounds,
        probes_used=probes_used,
        stop_reason=stop_reason,
        best_strength=best_strength,
        verdict=verdict,
        flag_threshold=flag_threshold,
        findings=findings,
    )

    generators_used: list[str] = []
    probes_by_round: list[dict[str, Any]] = []
    for finding in findings:
        gen = str(finding.get("generator", ""))
        if gen and gen not in generators_used:
            generators_used.append(gen)
        probes_by_round.append(
            {
                "round": finding.get("round", ""),
                "generator": gen,
                "n_probes": finding.get("n_probes", 0),
            }
        )

    judges = list(judges) if judges else []
    methodology = {
        "approach": _APPROACH,
        "brief": brief,
        "budget": _json_safe(budget) if budget is not None else {},
        "generators_used": generators_used,
        "probes_by_round": probes_by_round,
        "judges": [_judge_card(j) for j in judges],
    }

    limitations = list(_LIMITATIONS) + [str(l) for l in extra_limitations]

    package_version = _package_version()
    reproducibility = {
        "seed": seed,
        "package_version": package_version,
        "evidence_path": evidence_path,
    }

    return AuditReport(
        generated=datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        package_version=package_version,
        target_name=target_name,
        target_type=target_type,
        target_description=described,
        brief=brief,
        executive_summary=executive_summary,
        verdict=verdict,
        flag_threshold=float(flag_threshold),
        best_strength=best_strength,
        rounds=rounds,
        probes_used=probes_used,
        stop_reason=stop_reason,
        findings=findings,
        methodology=methodology,
        limitations=limitations,
        rmf_mappings=mappings_for_report(
            tuple(findings), judges_used=bool(judges)
        ),
        reproducibility=reproducibility,
        git_commit=_git_commit(),
        evidence_path=evidence_path,
    )


def save_agentic_report(
    report: AuditReport, path: str | Path
) -> list[Path]:
    """Write the report as ``<path>.md``, ``<path>.html`` and
    ``<path>.json``. Returns the three file paths, in that order."""
    base = Path(path)
    md_path = base.with_suffix(".md")
    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".json")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    html_path.write_text(to_html(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=True, indent=1),
        encoding="utf-8",
    )
    return [md_path, html_path, json_path]

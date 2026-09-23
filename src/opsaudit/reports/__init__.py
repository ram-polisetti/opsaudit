"""Agentic audit reports: defensible reports from campaign evidence.

Everything here is deterministic template fill — no LLM prose. An
:class:`AuditReport` is assembled by :func:`build_report` from a
:class:`~opsaudit.agents.campaign.CampaignReport`, then rendered to
Markdown (:func:`to_markdown`) or standalone HTML (:func:`to_html`).
"""

from .html import to_html
from .markdown import to_markdown
from .report import (
    FLAG_THRESHOLD_DEFAULT,
    AuditReport,
    build_report,
    save_agentic_report,
)
from .rmf import AGENTIC_RMF_MAPPING, mappings_for_report, rmf_section

__all__ = [
    "AuditReport",
    "build_report",
    "save_agentic_report",
    "to_markdown",
    "to_html",
    "rmf_section",
    "mappings_for_report",
    "AGENTIC_RMF_MAPPING",
    "FLAG_THRESHOLD_DEFAULT",
]

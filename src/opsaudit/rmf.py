"""Function-level NIST AI RMF mapping for opsaudit artifacts."""

from __future__ import annotations

from .metrics import AuditResult


RMF_MAPPING = {
    "demographic_parity_diff": {
        "function": "Measure",
        "rationale": "Quantifies differences in selection rates across groups, supporting measurement of fairness-related risk.",
    },
    "disparate_impact_ratio": {
        "function": "Measure",
        "rationale": "Four-fifths-rule style check on whether outcomes disproportionately exclude any group.",
    },
    "tpr_gap": {
        "function": "Measure",
        "rationale": "Checks whether the true-positive rate (equal opportunity) is consistent across groups.",
    },
    "fpr_gap": {
        "function": "Measure",
        "rationale": "Checks whether the false-positive burden falls unevenly on any group.",
    },
    "synthetic_data_generators": {
        "function": "Map",
        "rationale": "Define representative operational contexts and stress scenarios in which the system is evaluated.",
    },
    "deployment_gate": {
        "function": "Manage",
        "rationale": "Operationalizes go/no-go release decisions based on measured disparity risk.",
    },
    "audit_report": {
        "function": "Govern",
        "rationale": "Produces durable documentation that oversight roles can review for accountability.",
    },
}


def rmf_section(result: AuditResult) -> str:
    """Return the NIST AI RMF section for an audit report.

    ``result`` keeps the function aligned with report-generation APIs and
    allows future metric-aware mappings without changing this public signature.

    Example:
        >>> "## NIST AI RMF mapping" in rmf_section(AuditResult([], 0, 0.0, 1.0, None, None, []))
        True
    """
    del result
    lines = ["## NIST AI RMF mapping"]
    for check, mapping in RMF_MAPPING.items():
        lines.append(
            f"- **{check}** → {mapping['function']}: {mapping['rationale']}"
        )
    return "\n".join(lines)

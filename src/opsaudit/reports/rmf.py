"""NIST AI RMF 1.0 mapping for agentic-audit campaigns.

Each entry maps one aspect of an agentic audit to a real RMF 1.0
subcategory (verified against the framework core: GOVERN, MAP, MEASURE,
MANAGE). The ``limits`` field is the honest part: the tool *assists*
each subcategory by producing evidence for it — it does not satisfy
the subcategory on its own. Read ``docs/RMF_MAPPINGS.md`` for the
full explanation of what each claim means.
"""

from __future__ import annotations

from typing import Any

#: Agentic-audit aspects mapped to NIST AI RMF 1.0 subcategories.
#: ``subcategory`` is None where only the function level is claimed.
AGENTIC_RMF_MAPPING: tuple[dict[str, Any], ...] = (
    {
        "aspect": "adaptive_disparity_measurement",
        "function": "MEASURE",
        "subcategory": "MEASURE 2.11",
        "subcategory_text": (
            "Fairness and bias — as identified in the MAP function — are "
            "evaluated and results are documented."
        ),
        "rationale": (
            "Counterfactual probe campaigns evaluate outcome-rate gaps "
            "across protected attributes; judge-labeled campaigns do the "
            "same for tone, refusal, and stereotype signals in text."
        ),
        "limits": (
            "The campaign measures the scenarios it probed within budget; "
            "it is not an exhaustive fairness evaluation. Material "
            "findings should be confirmed with the v0.1 statistical core."
        ),
        "evidence": "findings table, executive summary",
    },
    {
        "aspect": "probe_batteries_as_test_sets",
        "function": "MEASURE",
        "subcategory": "MEASURE 2.1",
        "subcategory_text": (
            "Test sets, metrics, and details about the tools used during "
            "TEVV are documented."
        ),
        "rationale": (
            "Every probe batch is generated deterministically, recorded in "
            "the append-only evidence log, and reproducible from the seed — "
            "the campaign is a documented test set with documented tooling."
        ),
        "limits": (
            "Covers the TEVV documentation for *this* audit only; lifecycle "
            "TEVV (training-time, monitoring) is out of scope."
        ),
        "evidence": "methodology, reproducibility block, evidence log",
    },
    {
        "aspect": "deployment_context_drills",
        "function": "MEASURE",
        "subcategory": "MEASURE 2.3",
        "subcategory_text": (
            "AI system performance or assurance criteria are measured "
            "qualitatively or quantitatively and demonstrated for conditions "
            "similar to deployment setting(s). Measures are documented."
        ),
        "rationale": (
            "Counterfactual drills vary deployment-relevant attributes "
            "(employment type, tenure, query phrasing) one at a time to "
            "measure behavior under deployment-like conditions."
        ),
        "limits": (
            "The operator supplies the deployment context via the brief; "
            "the tool cannot know the real deployment distribution."
        ),
        "evidence": "methodology (brief), findings table",
    },
    {
        "aspect": "judge_calibration",
        "function": "MEASURE",
        "subcategory": "MEASURE 2.13",
        "subcategory_text": (
            "Effectiveness of the employed TEVV metrics and processes in "
            "the MEASURE function are evaluated and documented."
        ),
        "rationale": (
            "The calibration harness evaluates the measurement instruments "
            "themselves: each judge's agreement with human labels (accuracy, "
            "Cohen's kappa) is measured and recorded before the judge is "
            "trusted."
        ),
        "limits": (
            "Calibration measures agreement, not correctness, and the "
            "shipped starter datasets are synthetic — production use "
            "requires real human labels."
        ),
        "evidence": "methodology (judge calibration cards)",
        "requires": "judges",
    },
    {
        "aspect": "per_group_impact_characterization",
        "function": "MAP",
        "subcategory": "MAP 5.1",
        "subcategory_text": (
            "Likelihood and magnitude of each identified impact (both "
            "potentially beneficial and harmful) are identified and "
            "documented."
        ),
        "rationale": (
            "Per-group outcome rates, gaps, and judge-label distributions "
            "characterize how system behavior differs across groups — the "
            "raw material for an impact assessment."
        ),
        "limits": (
            "Characterizes measured behavioral differences; likelihood and "
            "magnitude *judgments* for a real deployment remain the "
            "operator's job."
        ),
        "evidence": "findings table",
    },
    {
        "aspect": "audit_context_brief",
        "function": "MAP",
        "subcategory": "MAP 1.1",
        "subcategory_text": (
            "Intended purposes, potentially beneficial uses, context-specific "
            "laws, norms and expectations, and prospective settings in which "
            "the AI system will be deployed are understood and documented."
        ),
        "rationale": (
            "The audit brief records the target, protected attributes, and "
            "risk areas under evaluation — the documented evaluation "
            "context for the campaign."
        ),
        "limits": (
            "The brief is the *auditor's* context, not a substitute for the "
            "operator's own MAP 1.1 documentation of the deployed system."
        ),
        "evidence": "methodology (brief)",
    },
    {
        "aspect": "campaign_report_as_oversight_artifact",
        "function": "GOVERN",
        "subcategory": None,
        "rationale": (
            "The report plus the append-only evidence log give oversight "
            "roles a durable, reproducible account of what was tested, what "
            "was found, and how — supporting accountability for the audit "
            "itself."
        ),
        "limits": (
            "Documentation supports governance; it does not create the "
            "organizational policies, roles, or culture the GOVERN function "
            "requires."
        ),
        "evidence": "full report, evidence log",
    },
    {
        "aspect": "deployment_gate",
        "function": "MANAGE",
        "subcategory": None,
        "rationale": (
            "Flagged findings (FINDINGS WARRANT REVIEW) feed go/no-go "
            "release decisions; the v0.1 deployment gate operationalizes "
            "thresholds for the tabular core."
        ),
        "limits": (
            "The tool flags; the risk-treatment decision (accept, mitigate, "
            "block) belongs to the operator's MANAGE process."
        ),
        "evidence": "verdict, executive summary",
    },
)

#: Valid function names — used by tests to catch typos/inventions.
VALID_FUNCTIONS = frozenset({"GOVERN", "MAP", "MEASURE", "MANAGE"})


def mappings_for_report(
    findings: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    judges_used: bool,
) -> list[dict[str, Any]]:
    """Select the RMF mappings relevant to one campaign report.

    The judge-calibration mapping is included only when judges were
    used. Every entry carries its ``limits`` so the report never
    over-claims.
    """
    selected: list[dict[str, Any]] = []
    for mapping in AGENTIC_RMF_MAPPING:
        if mapping.get("requires") == "judges" and not judges_used:
            continue
        selected.append(dict(mapping))
    return selected


def rmf_section(report: Any) -> str:
    """Render the NIST AI RMF mapping section for a Markdown report."""
    mappings = getattr(report, "rmf_mappings", ())
    lines = ["## NIST AI RMF mapping", ""]
    lines.append(
        "Each mapping names the *evidence this audit contributes* toward a "
        "NIST AI RMF 1.0 subcategory. The tool assists — it does not satisfy "
        "a subcategory on its own; see `docs/RMF_MAPPINGS.md`."
    )
    lines.append("")
    for mapping in mappings:
        sub = mapping.get("subcategory")
        where = f"{mapping['function']}" + (f" {sub}" if sub else "")
        lines.append(f"- **{mapping['aspect']}** → {where}: {mapping['rationale']}")
        lines.append(f"  - *Limits:* {mapping['limits']}")
    return "\n".join(lines)

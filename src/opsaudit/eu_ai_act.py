"""EU AI Act risk-tier classification and gate cross-check.

Encodes the EU AI Act's system-classification logic (Article 5 prohibited
practices, Article 6 / Annex III high-risk use cases, Article 50 transparency
obligations, Chapter V general-purpose AI duties) as testable code. This
mirrors what the European Commission's official AI Act Compliance Checker
questionnaire does, so opsaudit's deployment gate can be cross-checked
against the legal tier reproducibly instead of via a manual website visit.

This is NOT the official tool and NOT legal advice. The Commission itself
notes the Checker does not replace legal counsel. This module is a research
instrument: a version-pinned encoding of the published decision logic,
validated against the official materials on the date below.

Encoding version: 2026-10-05
Source: Regulation (EU) 2024/1689; AI Act Single Information Platform (beta),
https://ai-act-service-desk.ec.europa.eu/en
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ENCODING_VERSION = "2026-10-05"
SOURCE = (
    "Regulation (EU) 2024/1689; validated against the EU AI Act Single "
    "Information Platform (beta), 2026-10-05. Not legal advice."
)

RiskTier = Literal["prohibited", "high_risk", "transparency", "minimal"]
GateStatus = Literal["pass", "fail", "review"]
CrossCheckVerdict = Literal["hard_stop", "conditional_pass", "blocked", "pass"]

# --- Controlled vocabulary: use-case keys -> (tier, article reference, label) ---

_PROHIBITED: dict[str, tuple[str, str]] = {
    "social_scoring": ("Art. 5", "social scoring by public or private actors"),
    "realtime_remote_biometric_id_public": (
        "Art. 5",
        "real-time remote biometric identification in publicly accessible spaces for law enforcement",
    ),
    "biometric_categorization_sensitive": (
        "Art. 5",
        "biometric categorisation inferring race, political opinions, religion, sexual orientation, etc.",
    ),
    "emotion_inference_work_education": (
        "Art. 5",
        "emotion inference in workplaces or educational institutions",
    ),
    "predictive_policing_profiling": (
        "Art. 5",
        "crime-risk assessment based solely on profiling or personality traits",
    ),
    "untargeted_facial_scraping": (
        "Art. 5",
        "untargeted scraping of facial images from the internet or CCTV",
    ),
    "manipulative_subliminal": (
        "Art. 5",
        "subliminal or manipulative techniques distorting behaviour",
    ),
    "vulnerability_exploitation": (
        "Art. 5",
        "exploitation of vulnerabilities due to age, disability, or socio-economic situation",
    ),
}

_HIGH_RISK: dict[str, tuple[str, str]] = {
    "biometric_id_verification": (
        "Art. 6 / Annex III(1)",
        "biometric identification or verification (non-prohibited)",
    ),
    "critical_infrastructure": (
        "Art. 6 / Annex III(2)",
        "safety component / management of critical infrastructure",
    ),
    "education_admissions": (
        "Art. 6 / Annex III(3)",
        "education admissions or allocation",
    ),
    "education_evaluation": (
        "Art. 6 / Annex III(3)",
        "evaluation of learning outcomes, proctoring",
    ),
    "employment_recruitment": (
        "Art. 6 / Annex III(4)",
        "recruitment, hiring, CV screening",
    ),
    "employment_management": (
        "Art. 6 / Annex III(4)",
        "promotion, termination, task allocation, performance monitoring",
    ),
    "credit_scoring": ("Art. 6 / Annex III(5)", "creditworthiness evaluation"),
    "public_benefits_eligibility": (
        "Art. 6 / Annex III(5)",
        "eligibility for public benefits or services",
    ),
    "insurance_pricing": (
        "Art. 6 / Annex III(5)",
        "risk assessment and pricing in insurance",
    ),
    "emergency_dispatch": (
        "Art. 6 / Annex III(5)",
        "emergency call triage and dispatch",
    ),
    "law_enforcement_risk": (
        "Art. 6 / Annex III(6)",
        "law-enforcement risk assessment",
    ),
    "law_enforcement_evidence": (
        "Art. 6 / Annex III(6)",
        "evaluation of evidence reliability",
    ),
    "migration_asylum_border": (
        "Art. 6 / Annex III(7)",
        "asylum, visa, border-management risk assessment",
    ),
    "justice_recidivism": (
        "Art. 6 / Annex III(8)",
        "recidivism risk / administration of justice",
    ),
    "democratic_process": (
        "Art. 6 / Annex III(8)",
        "influencing elections or referenda",
    ),
}

_TRANSPARENCY: dict[str, tuple[str, str]] = {
    "conversational_ai": (
        "Art. 50",
        "AI system interacting directly with humans (e.g. chatbot)",
    ),
    "emotion_recognition": (
        "Art. 50",
        "emotion recognition outside prohibited contexts",
    ),
    "biometric_categorization": (
        "Art. 50",
        "biometric categorisation outside prohibited contexts",
    ),
    "ai_generated_content": (
        "Art. 50",
        "AI-generated or manipulated content, incl. deepfakes",
    ),
}

OBLIGATIONS: dict[str, list[str]] = {
    "prohibited": [
        "Do not develop, place on the market, or put into service (Art. 5).",
    ],
    "high_risk": [
        "Risk management system (Art. 9).",
        "Data and data-governance measures (Art. 10).",
        "Technical documentation (Art. 11).",
        "Record-keeping / logging (Art. 12).",
        "Transparency and information to deployers (Art. 13).",
        "Human oversight (Art. 14).",
        "Accuracy, robustness, cybersecurity (Art. 15).",
        "Conformity assessment before deployment (Art. 43).",
        "Registration in the EU database (Art. 49).",
    ],
    "transparency": [
        "Disclose AI involvement to affected humans (Art. 50).",
        "Mark AI-generated content as machine-generated, visibly where applicable (Art. 50).",
    ],
    "minimal": [
        "No mandatory obligations; voluntary codes of practice encouraged (Art. 95).",
    ],
    "gpai": [
        "GPAI transparency: technical documentation and training-data summary (Art. 53).",
    ],
    "gpai_systemic": [
        "Systemic-risk duties: model evaluations, incident reporting, cybersecurity (Art. 55).",
    ],
}


@dataclass
class SystemProfile:
    """What the AI system does, in the Act's vocabulary."""

    use_cases: list[str] = field(default_factory=list)
    is_gpai: bool = False
    gpai_systemic_risk: bool = False
    deploys_in_eu: bool = True


@dataclass
class Classification:
    tier: RiskTier
    matched_use_cases: list[tuple[str, str, str]]
    obligations: list[str]
    gpai_duties: list[str]
    references: list[str]
    encoding_version: str = ENCODING_VERSION


@dataclass
class CrossCheck:
    gate_status: GateStatus
    tier: RiskTier
    verdict: CrossCheckVerdict
    rationale: str
    obligations: list[str]


def _validate_use_cases(use_cases: list[str]) -> None:
    known = set(_PROHIBITED) | set(_HIGH_RISK) | set(_TRANSPARENCY)
    unknown = [u for u in use_cases if u not in known]
    if unknown:
        raise ValueError(
            f"Unknown use-case keys: {unknown}. Known keys: {sorted(known)}"
        )


def classify_system(profile: SystemProfile | dict) -> Classification:
    """Classify an AI system into its EU AI Act risk tier.

    Accepts a SystemProfile or an equivalent dict with keys
    ``use_cases`` (list), ``is_gpai`` (bool), ``gpai_systemic_risk`` (bool).

    Example:
        >>> c = classify_system({"use_cases": ["employment_recruitment"]})
        >>> c.tier
        'high_risk'
    """
    if isinstance(profile, dict):
        profile = SystemProfile(
            use_cases=profile.get("use_cases", []),
            is_gpai=profile.get("is_gpai", False),
            gpai_systemic_risk=profile.get("gpai_systemic_risk", False),
            deploys_in_eu=profile.get("deploys_in_eu", True),
        )
    _validate_use_cases(profile.use_cases)

    matched: list[tuple[str, str, str]] = []
    tier: RiskTier = "minimal"

    for uc in profile.use_cases:
        if uc in _PROHIBITED:
            ref, label = _PROHIBITED[uc]
            matched.append((uc, ref, label))
            tier = "prohibited"
    if tier != "prohibited":
        for uc in profile.use_cases:
            if uc in _HIGH_RISK:
                ref, label = _HIGH_RISK[uc]
                matched.append((uc, ref, label))
                tier = "high_risk"
    if tier == "minimal":
        for uc in profile.use_cases:
            if uc in _TRANSPARENCY:
                ref, label = _TRANSPARENCY[uc]
                matched.append((uc, ref, label))
                tier = "transparency"

    gpai_duties: list[str] = []
    if profile.is_gpai:
        gpai_duties.extend(OBLIGATIONS["gpai"])
    if profile.gpai_systemic_risk:
        gpai_duties.extend(OBLIGATIONS["gpai_systemic"])

    references = sorted({ref for _, ref, _ in matched})
    return Classification(
        tier=tier,
        matched_use_cases=matched,
        obligations=list(OBLIGATIONS[tier]),
        gpai_duties=gpai_duties,
        references=references,
    )


def cross_check_gate(
    gate_status: GateStatus, classification: Classification
) -> CrossCheck:
    """Cross-check an opsaudit gate verdict against the EU AI Act tier.

    The gate measures statistical disparity; the Act sets legal process
    duties. The stricter of the two governs. Returns a verdict plus the
    human-readable rationale for the audit report.

    Example:
        >>> c = classify_system({"use_cases": ["employment_recruitment"]})
        >>> cross_check_gate("pass", c).verdict
        'conditional_pass'
    """
    if gate_status not in ("pass", "fail", "review"):
        raise ValueError(f"Unknown gate status: {gate_status!r}")
    tier = classification.tier

    if tier == "prohibited":
        return CrossCheck(
            gate_status=gate_status,
            tier=tier,
            verdict="hard_stop",
            rationale=(
                "Prohibited practice under EU AI Act Art. 5. "
                "The gate verdict is irrelevant: the system must not be deployed."
            ),
            obligations=classification.obligations,
        )
    if tier == "high_risk":
        if gate_status == "pass":
            return CrossCheck(
                gate_status=gate_status,
                tier=tier,
                verdict="conditional_pass",
                rationale=(
                    "Gate thresholds met, but high-risk tier requires conformity "
                    "assessment, human oversight, logging, and EU database "
                    "registration before deployment (Arts. 9-15, 43, 49)."
                ),
                obligations=classification.obligations,
            )
        return CrossCheck(
            gate_status=gate_status,
            tier=tier,
            verdict="blocked",
            rationale=(
                f"Gate returned '{gate_status}' and the system is high-risk: "
                "both the statistical bar and the legal process bar block deployment."
            ),
            obligations=classification.obligations,
        )
    # transparency / minimal tiers: the gate governs on its own
    if gate_status == "pass":
        return CrossCheck(
            gate_status=gate_status,
            tier=tier,
            verdict="pass",
            rationale=(
                f"Gate passed; EU AI Act tier is '{tier}'. "
                + (
                    "Transparency disclosures still required (Art. 50)."
                    if tier == "transparency"
                    else "No mandatory Act obligations."
                )
            ),
            obligations=classification.obligations,
        )
    return CrossCheck(
        gate_status=gate_status,
        tier=tier,
        verdict="blocked",
        rationale=f"Gate returned '{gate_status}': deployment blocked on statistical grounds.",
        obligations=classification.obligations,
    )


def eu_ai_act_section(
    classification: Classification, check: CrossCheck | None = None
) -> str:
    """Return the EU AI Act section for an audit report.

    Example:
        >>> c = classify_system({"use_cases": ["employment_recruitment"]})
        >>> "## EU AI Act" in eu_ai_act_section(c)
        True
    """
    lines = ["## EU AI Act classification (cross-check)"]
    lines.append(f"- **Risk tier:** `{classification.tier}`")
    if classification.matched_use_cases:
        lines.append("- **Matched use cases:**")
        for uc, ref, label in classification.matched_use_cases:
            lines.append(f"  - `{uc}` — {label} ({ref})")
    else:
        lines.append("- **Matched use cases:** none — minimal-risk by default")
    lines.append("- **Obligations:**")
    for ob in classification.obligations:
        lines.append(f"  - {ob}")
    if classification.gpai_duties:
        lines.append("- **GPAI duties:**")
        for duty in classification.gpai_duties:
            lines.append(f"  - {duty}")
    if check is not None:
        lines.append(
            f"- **Gate cross-check verdict:** `{check.verdict}` — {check.rationale}"
        )
    lines.append(f"- **Encoding:** version {classification.encoding_version}; {SOURCE}")
    lines.append("- **Note:** research instrument, not legal advice.")
    return "\n".join(lines)


# Convenience re-export of the controlled vocabulary for UIs and docs.
USE_CASES: dict[str, dict[str, str]] = {
    **{
        k: {"tier": "prohibited", "reference": v[0], "label": v[1]}
        for k, v in _PROHIBITED.items()
    },
    **{
        k: {"tier": "high_risk", "reference": v[0], "label": v[1]}
        for k, v in _HIGH_RISK.items()
    },
    **{
        k: {"tier": "transparency", "reference": v[0], "label": v[1]}
        for k, v in _TRANSPARENCY.items()
    },
}

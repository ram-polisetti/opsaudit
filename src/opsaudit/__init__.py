"""opsaudit audits disparity signals in operational decision systems."""

__version__ = "0.2.1"

from .agents import (
    GENERATORS,
    AuditCampaign,
    AuditPlanner,
    Budget,
    BudgetTracker,
    CampaignReport,
    ProbeSpec,
    ResponseCache,
)
from .calibration import (
    DEFAULT_KAPPA_THRESHOLD,
    CalibrationHarness,
    CalibrationReport,
    cohen_kappa,
)
from .context import load_context
from .data import generate_dispatch, generate_staffing
from .evidence import EvidenceLog
from .gate import DEFAULT_THRESHOLDS, evaluate_gate, evaluate_gate_status
from .gate_ci import (
    append_verdict_log,
    decide_exit,
    load_gate_config,
    render_verdict_markdown,
    run_gate,
)
from .judges import (
    Judge,
    JudgeScore,
    LLMJudge,
    RefusalJudge,
    StereotypeJudge,
    ToneJudge,
    aggregate_judge_findings,
    group_label_rates,
    label_rate_gaps,
    ordinal_means,
    unscored_rate,
)
from .metrics import AuditResult, audit_disparities
from .probes import (
    BUILTIN_RELATIONS,
    MetamorphicRelation,
    Probe,
    ProbeBatch,
    check_equal_output,
    check_symmetry,
    evaluate_metamorphic,
    generate_adversarial,
    generate_counterfactuals,
    generate_llm_assisted,
    generate_metamorphic,
)
from .report import save_report
from .reports import (
    AGENTIC_RMF_MAPPING,
    FLAG_THRESHOLD_DEFAULT,
    AuditReport,
    build_report,
    mappings_for_report,
    save_agentic_report,
)
from .reports import (
    rmf_section as agentic_rmf_section,
)
from .reports import (
    to_html as agentic_to_html,
)
from .reports import (
    to_markdown as agentic_to_markdown,
)
from .rmf import RMF_MAPPING
from .targets import (
    HuggingFaceTarget,
    OllamaTarget,
    OpenAICompatTarget,
    RagTarget,
    TabularTarget,
    Target,
)

__all__ = [
    "AGENTIC_RMF_MAPPING",
    "BUILTIN_RELATIONS",
    "DEFAULT_KAPPA_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "FLAG_THRESHOLD_DEFAULT",
    "GENERATORS",
    "RMF_MAPPING",
    "AuditCampaign",
    "AuditPlanner",
    "AuditReport",
    "AuditResult",
    "Budget",
    "BudgetTracker",
    "CalibrationHarness",
    "CalibrationReport",
    "CampaignReport",
    "EvidenceLog",
    "HuggingFaceTarget",
    "Judge",
    "JudgeScore",
    "LLMJudge",
    "MetamorphicRelation",
    "OllamaTarget",
    "OpenAICompatTarget",
    "Probe",
    "ProbeBatch",
    "ProbeSpec",
    "RagTarget",
    "RefusalJudge",
    "ResponseCache",
    "StereotypeJudge",
    "TabularTarget",
    "Target",
    "ToneJudge",
    "__version__",
    "agentic_rmf_section",
    "agentic_to_html",
    "agentic_to_markdown",
    "aggregate_judge_findings",
    "append_verdict_log",
    "audit_disparities",
    "build_report",
    "check_equal_output",
    "check_symmetry",
    "cohen_kappa",
    "decide_exit",
    "evaluate_gate",
    "evaluate_gate_status",
    "evaluate_metamorphic",
    "generate_adversarial",
    "generate_counterfactuals",
    "generate_dispatch",
    "generate_llm_assisted",
    "generate_metamorphic",
    "generate_staffing",
    "group_label_rates",
    "label_rate_gaps",
    "load_context",
    "load_gate_config",
    "mappings_for_report",
    "ordinal_means",
    "render_verdict_markdown",
    "run_gate",
    "save_agentic_report",
    "save_report",
    "unscored_rate",
]

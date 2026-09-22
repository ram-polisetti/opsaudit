"""Probe generators for the agentic auditor (Phase 2).

Deterministic generators propose audit inputs; the agentic planner
(Phase 3) will decide which to run. Nothing here calls an LLM except
:mod:`llm_assisted`, and even there every candidate is validated by the
deterministic invariants before it is kept.
"""

from .adversarial import STRATEGIES, generate_adversarial
from .base import KINDS, Probe, ProbeBatch
from .counterfactual import check_symmetry, generate_counterfactuals
from .llm_assisted import generate_llm_assisted
from .metamorphic import (
    BUILTIN_RELATIONS,
    MetamorphicRelation,
    check_equal_output,
    evaluate_metamorphic,
    generate_metamorphic,
)

__all__ = [
    "KINDS",
    "STRATEGIES",
    "BUILTIN_RELATIONS",
    "Probe",
    "ProbeBatch",
    "MetamorphicRelation",
    "check_equal_output",
    "check_symmetry",
    "evaluate_metamorphic",
    "generate_adversarial",
    "generate_counterfactuals",
    "generate_llm_assisted",
    "generate_metamorphic",
]

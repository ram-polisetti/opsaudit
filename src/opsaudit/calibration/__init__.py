"""Judge calibration: measure LLM judges against human labels before trusting them."""

from .datasets import JUDGE_DATASETS, STARTER_DATASETS
from .harness import (
    DEFAULT_KAPPA_THRESHOLD,
    CalibrationHarness,
    CalibrationReport,
    cohen_kappa,
)

__all__ = [
    "DEFAULT_KAPPA_THRESHOLD",
    "JUDGE_DATASETS",
    "STARTER_DATASETS",
    "CalibrationHarness",
    "CalibrationReport",
    "cohen_kappa",
]

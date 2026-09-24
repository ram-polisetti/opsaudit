"""Calibration harness: judge-vs-human agreement, computed deterministically.

No judge ships without a measured calibration score (plan §8). The
harness runs a judge over a human-labeled dataset and reports:

- **accuracy**: fraction of scored items where judge label == human label
  (unscored items excluded from the numerator *and* denominator, but
  reported separately — a judge that abstains on everything scores
  nothing, not 100%);
- **Cohen's kappa**: chance-corrected agreement, the headline metric;
- **per-label precision/recall** and a **confusion summary**.

The output is a :class:`CalibrationReport` with a PASS/FAIL
recommendation against a configurable kappa threshold (default 0.6 —
the conventional "substantial agreement" boundary). A FAILed judge is
*flagged, not deleted*: the harness prints a warning, sets
``judge.calibration`` to the report, and the campaign refuses to
aggregate its labels unless the operator passes the explicit
``allow_uncalibrated=True`` override (logged as evidence).

Everything here is arithmetic on labels. The judge is the only LLM
involved, and it only ever produces labels.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Conventional boundary for "substantial" chance-corrected agreement.
DEFAULT_KAPPA_THRESHOLD = 0.6


@dataclass(frozen=True)
class CalibrationReport:
    """Measured judge-vs-human agreement for one judge."""

    judge_id: str
    model_id: str
    n_items: int
    n_scored: int
    n_unscored: int
    accuracy: float
    kappa: float
    per_label: dict[str, dict[str, float]]  # label -> {precision, recall, support}
    confusion: dict[str, dict[str, int]]  # human_label -> {judge_label: count}
    kappa_threshold: float
    passed: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for evidence logs and reports."""
        return {
            "judge_id": self.judge_id,
            "model_id": self.model_id,
            "n_items": self.n_items,
            "n_scored": self.n_scored,
            "n_unscored": self.n_unscored,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "per_label": {k: dict(v) for k, v in self.per_label.items()},
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
            "kappa_threshold": self.kappa_threshold,
            "passed": self.passed,
            "notes": list(self.notes),
        }


def cohen_kappa(
    human_labels: Sequence[str], judge_labels: Sequence[str]
) -> float:
    """Cohen's kappa for two aligned label sequences (deterministic)."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("label sequences must have the same length")
    n = len(human_labels)
    if n == 0:
        return 0.0
    labels = sorted(set(human_labels) | set(judge_labels))
    if len(labels) == 1:
        # Perfect agreement on a single label — kappa is undefined
        # (no chance baseline); report it as perfect, honestly noted.
        return 1.0
    p_o = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / n
    p_e = 0.0
    for label in labels:
        p_h = sum(1 for h in human_labels if h == label) / n
        p_j = sum(1 for j in judge_labels if j == label) / n
        p_e += p_h * p_j
    if math.isclose(1.0 - p_e, 0.0):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


class CalibrationHarness:
    """Measure a judge's agreement with human labels.

    Args:
        judge: The :class:`~opsaudit.judges.base.Judge` to calibrate.
        kappa_threshold: Kappa at or above which the judge PASSES.
    """

    def __init__(self, judge: Any, kappa_threshold: float = DEFAULT_KAPPA_THRESHOLD) -> None:
        if judge is None:
            raise ValueError("judge must not be None")
        if not (0.0 <= kappa_threshold <= 1.0):
            raise ValueError("kappa_threshold must be in [0, 1]")
        self.judge = judge
        self.kappa_threshold = kappa_threshold

    def run(
        self, dataset: Sequence[tuple[str, str]]
    ) -> CalibrationReport:
        """Run the judge over ``dataset`` and return the calibration report.

        ``dataset`` is a sequence of ``(text, human_label)``. The report
        is also stored on ``judge.calibration``. FAIL prints a warning
        to stderr — loud, not silent.
        """
        texts = [text for text, _ in dataset]
        human = [label for _, label in dataset]
        scores = self.judge.score(texts)
        if len(scores) != len(texts):
            # Fail loudly: silently zipping a short score list would
            # drop items and corrupt the agreement metrics.
            raise ValueError(
                f"judge returned {len(scores)} scores for {len(texts)} "
                "texts; score() must return one JudgeScore per text"
            )

        paired: list[tuple[str, str]] = []  # (human, judge) for scored items
        n_unscored = 0
        for h, s in zip(human, scores):
            if s.unscored:
                n_unscored += 1
            else:
                paired.append((h, s.label))

        notes: list[str] = []
        if not paired:
            notes.append("judge scored nothing: all items unscored")
            accuracy = 0.0
            kappa = 0.0
        else:
            h_labels = [h for h, _ in paired]
            j_labels = [j for _, j in paired]
            accuracy = round(
                sum(1 for h, j in paired if h == j) / len(paired), 4
            )
            kappa = cohen_kappa(h_labels, j_labels)
        if n_unscored:
            notes.append(f"{n_unscored} item(s) unscored by the judge")

        per_label: dict[str, dict[str, float]] = {}
        confusion: dict[str, dict[str, int]] = {}
        for h, j in paired:
            confusion.setdefault(h, {}).setdefault(j, 0)
            confusion[h][j] += 1
        labels = sorted({h for h, _ in paired} | {j for _, j in paired})
        for label in labels:
            tp = sum(1 for h, j in paired if h == label and j == label)
            fp = sum(1 for h, j in paired if h != label and j == label)
            fn = sum(1 for h, j in paired if h == label and j != label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "support": tp + fn,
            }

        passed = kappa >= self.kappa_threshold and bool(paired)
        report = CalibrationReport(
            judge_id=self.judge.judge_id,
            model_id=getattr(self.judge, "model_id", "unknown-model"),
            n_items=len(dataset),
            n_scored=len(paired),
            n_unscored=n_unscored,
            accuracy=accuracy,
            kappa=kappa,
            per_label=per_label,
            confusion=confusion,
            kappa_threshold=self.kappa_threshold,
            passed=passed,
            notes=tuple(notes),
        )
        self.judge.calibration = report
        if not passed:
            print(
                f"WARNING: judge {report.judge_id} FAILED calibration "
                f"(kappa={report.kappa}, threshold={self.kappa_threshold}, "
                f"n_scored={report.n_scored}/{report.n_items}). "
                "Its labels will be rejected by AuditCampaign unless "
                "allow_uncalibrated=True is passed explicitly.",
                file=sys.stderr,
            )
        return report

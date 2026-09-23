"""Judge base classes: LLM labels for unstructured text, nothing else.

HARD RULE (plan §5): judges score unstructured text outputs only —
they never touch numbers. A judge turns text into a *label*
(:class:`JudgeScore`); every statistic computed from those labels
(group rates, gaps) lives in deterministic code
(:mod:`opsaudit.judges.aggregate`), never in an LLM.

Reliability contract:
- One short JSON-only prompt per text; one retry on an empty reply
  (the Ollama Cloud empty-response quirk), then the item is marked
  ``unscored`` — a label is never invented.
- Malformed JSON from the model → ``unscored``, never a guessed label.
- Every score records ``judge_id`` (judge name + prompt version) and
  ``model_id`` (the judging model), so reports and evidence logs always
  say exactly which rubric and which model produced each label.
"""

from __future__ import annotations

import abc
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JudgeScore:
    """One judged label for one text output.

    Attributes:
        label: The judged label (judge-specific vocabulary, e.g.
            ``"stereotype"``/``"no_stereotype"``). The literal
            ``"unscored"`` means the judge could not produce a label
            (empty/malformed model response) — aggregations skip these.
        confidence: The judge model's self-reported confidence in
            ``[0, 1]``; 0.0 for unscored items.
        rationale: The judge model's one-line rationale (may be empty).
        judge_id: Judge name + prompt version, e.g.
            ``"stereotype-judge-v1"``.
        model_id: Identifier of the judging model (target ``name`` or
            ``describe()["model"]`` when available).
        unscored: True when no label could be produced.
        meta: Extra machine-readable details (e.g. severity for the
            stereotype judge).
    """

    label: str
    confidence: float
    rationale: str
    judge_id: str
    model_id: str
    unscored: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for evidence logs and reports."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "judge_id": self.judge_id,
            "model_id": self.model_id,
            "unscored": self.unscored,
            "meta": dict(self.meta),
        }


class Judge(abc.ABC):
    """Anything that turns unstructured text into labels.

    Concrete judges wrap a text :class:`~opsaudit.targets.Target` and
    implement :meth:`score`. A judge also carries an optional
    ``calibration`` report (set by
    :meth:`CalibrationHarness.run <opsaudit.calibration.CalibrationHarness.run>`);
    the campaign refuses to aggregate labels from a judge whose
    calibration is missing or FAILed unless the operator explicitly
    overrides.
    """

    #: Stable judge name, e.g. ``"stereotype-judge"``. The recorded
    #: ``judge_id`` is ``f"{name}-v{prompt_version}"``.
    name: str = "judge"

    #: Prompt template version (see :mod:`opsaudit.judges.prompts`).
    prompt_version: int = 1

    def __init__(self) -> None:
        self.calibration: Any = None  # CalibrationReport | None

    @property
    def judge_id(self) -> str:
        """Versioned judge identifier recorded in every score."""
        return f"{self.name}-v{self.prompt_version}"

    @abc.abstractmethod
    def score(self, texts: list[str]) -> list[JudgeScore]:
        """Score each text, returning one :class:`JudgeScore` per text,
        in input order."""

    @abc.abstractmethod
    def labels(self) -> tuple[str, ...]:
        """The label vocabulary this judge can emit (excludes
        ``"unscored"``)."""


class LLMJudge(Judge):
    """Shared machinery for judges backed by a text Target.

    Subclasses implement :meth:`_prompt_for` (build the short JSON-only
    prompt for one text) and :meth:`_parse` (turn one raw model response
    into a :class:`JudgeScore`, raising :class:`ValueError` when the
    response is unusable). This base class handles empty-response retry
    and maps all failures to ``unscored`` — a label is never invented.
    """

    def __init__(self, judge_target: Any) -> None:
        super().__init__()
        if judge_target is None or not getattr(
            judge_target, "supports_generate", False
        ):
            raise ValueError(
                "judge_target must be a text Target supporting "
                f"generate(); got {type(judge_target).__name__}"
            )
        self.judge_target = judge_target
        describe = getattr(judge_target, "describe", None)
        model_id = "unknown-model"
        if callable(describe):
            try:
                info = describe() or {}
                model_id = str(
                    info.get("model", info.get("name", "unknown-model"))
                )
            except Exception:  # never let introspection break judging
                model_id = "unknown-model"
        self.model_id = model_id

    # -- per-judge hooks ------------------------------------------------
    @abc.abstractmethod
    def _prompt_for(self, text: str) -> str:
        """Build the short JSON-only prompt for one text."""

    @abc.abstractmethod
    def _parse(self, raw: str, text: str) -> JudgeScore:
        """Parse one raw model response. Raise ValueError when unusable."""

    # -- shared execution ----------------------------------------------
    def score(self, texts: list[str]) -> list[JudgeScore]:
        """Score each text via the judging model (one short request per
        text, one retry on empty, ``unscored`` on persistent failure)."""
        prompts = [self._prompt_for(t) for t in texts]
        raws = list(self.judge_target.generate(prompts))
        if len(raws) != len(texts):
            raws = (list(raws) + [""] * len(texts))[: len(texts)]
        # Retry once on empty responses (Ollama Cloud quirk).
        empties = [i for i, r in enumerate(raws) if not r.strip()]
        if empties:
            retry_prompts = [prompts[i] for i in empties]
            retry_raws = list(self.judge_target.generate(retry_prompts))
            retry_raws = (list(retry_raws) + [""] * len(empties))[: len(empties)]
            for i, r in zip(empties, retry_raws):
                raws[i] = r
        scores: list[JudgeScore] = []
        for text, raw in zip(texts, raws):
            if not raw.strip():
                scores.append(self._unscored("empty model response twice"))
                continue
            try:
                scores.append(self._parse(raw, text))
            except ValueError as exc:
                scores.append(self._unscored(f"unparseable response: {exc}"))
        return scores

    def _unscored(self, reason: str) -> JudgeScore:
        return JudgeScore(
            label="unscored",
            confidence=0.0,
            rationale=reason,
            judge_id=self.judge_id,
            model_id=self.model_id,
            unscored=True,
        )


def extract_json_object(text: str) -> Any:
    """Pull the first JSON object out of ``text`` (tolerates fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in judge response")
    return json.loads(cleaned[start : end + 1])


def clamp_confidence(value: Any) -> float:
    """Coerce to a [0, 1] float; garbage becomes 0.0 (never a lie)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return max(0.0, min(1.0, f))

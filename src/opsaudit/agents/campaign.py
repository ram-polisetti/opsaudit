"""Agentic audit campaigns: plan → probe → observe → replan.

:class:`AuditCampaign` wires the pieces together: the
:class:`~opsaudit.agents.planner.AuditPlanner` decides *what* to probe,
the Phase 2 generators build the probes deterministically, the Phase 1
target adapters execute them, and the v0.1 statistical core (plus
deterministic summaries) turns outputs into findings. Every step is
written to the append-only evidence log, so a campaign is reproducible
and defensible.

Reproducibility contract: same brief + same seed + same (deterministic)
planner and audited targets → same prompts → same cached responses →
same campaign report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..evidence import EvidenceLog
from ..judges.aggregate import aggregate_judge_findings
from ..judges.tone import TONE_ORDER
from ..probes import (
    BUILTIN_RELATIONS,
    ProbeBatch,
    check_equal_output,
    generate_adversarial,
    generate_counterfactuals,
    generate_metamorphic,
)
from .budgets import Budget, BudgetTracker
from .cache import ResponseCache
from .planner import GENERATORS, AuditPlanner, ProbeSpec

#: Output dicts produced when a target call fails carry this marker.
_ERROR_KEY = "error"

#: Long text outputs are truncated in the evidence log to keep it
#: readable; the truncation is recorded, not silent.
_MAX_OUTPUT_CHARS = 2000


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class CampaignReport:
    """The outcome of one :class:`AuditCampaign` run."""

    brief: dict[str, Any]
    seed: int | None
    rounds: int
    probes_used: int
    stop_reason: str
    findings: list[dict[str, Any]]
    best_strength: float
    evidence_path: str
    n_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief": self.brief,
            "seed": self.seed,
            "rounds": self.rounds,
            "probes_used": self.probes_used,
            "stop_reason": self.stop_reason,
            "findings": self.findings,
            "best_strength": self.best_strength,
            "evidence_path": self.evidence_path,
            "n_events": self.n_events,
        }


class AuditCampaign:
    """Run an adaptive probe campaign against an audited target.

    Args:
        audited_target: The :class:`~opsaudit.targets.Target` under
            audit (tabular or text).
        planner: The :class:`AuditPlanner` deciding what to probe.
        evidence_path: Where the append-only evidence log is written.
        seed: Random seed recorded with the campaign (also seeds the
            stdlib ``random`` module for any internal randomness).
        cache: Optional shared :class:`ResponseCache`. A fresh campaign
            gets a fresh in-memory cache unless one is supplied.
        judges: Optional list of :class:`~opsaudit.judges.base.Judge`
            used to label non-numeric (text) outputs. Off by default.
            Every judge must carry a passing calibration report
            (``judge.calibration``, set by
            :class:`~opsaudit.calibration.CalibrationHarness`) unless
            ``allow_uncalibrated`` is True.
        allow_uncalibrated: Explicit operator override permitting
            uncalibrated (or FAIL-calibrated) judges. Logged as evidence.
    """

    def __init__(
        self,
        *,
        audited_target: Any,
        planner: AuditPlanner,
        evidence_path: Any,
        seed: int | None = None,
        cache: ResponseCache | None = None,
        judges: list | None = None,
        allow_uncalibrated: bool = False,
    ) -> None:
        if audited_target is None:
            raise ValueError("audited_target must not be None")
        if not isinstance(planner, AuditPlanner):
            raise ValueError("planner must be an AuditPlanner")
        self.audited_target = audited_target
        self.planner = planner
        self.evidence_path = str(evidence_path)
        self.seed = seed
        self.cache = cache or ResponseCache()
        self.judges = list(judges) if judges else []
        self.allow_uncalibrated = allow_uncalibrated
        for judge in self.judges:
            calibration = getattr(judge, "calibration", None)
            calibrated_ok = bool(
                calibration is not None
                and getattr(calibration, "passed", False)
            )
            if not calibrated_ok and not allow_uncalibrated:
                raise ValueError(
                    f"judge {getattr(judge, 'judge_id', judge)!r} has no "
                    "passing calibration report: run "
                    "CalibrationHarness on it first, or pass "
                    "allow_uncalibrated=True explicitly (the override is "
                    "logged as evidence)."
                )
        self._planner_cache = ResponseCache()
        # Give the planner its own cache namespace so planner prompts
        # and target inputs can never collide.
        self.planner.cache = self._planner_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> CampaignReport:
        """Execute the campaign loop and return the report."""
        random.seed(self.seed)
        log = EvidenceLog(self.evidence_path, target=self.audited_target)
        budget: Budget = self.planner.budget
        tracker = BudgetTracker(budget)
        brief = self.planner.brief

        start = log.record(
            {
                "kind": "campaign_start",
                "brief": _json_safe(brief),
                "seed": self.seed,
                "budget": {
                    "max_probes": budget.max_probes,
                    "max_rounds": budget.max_rounds,
                    "max_cost": budget.max_cost,
                    "probe_cost": budget.probe_cost,
                    "planner_call_cost": budget.planner_call_cost,
                    "flat_rounds": budget.flat_rounds,
                },
            }
        )
        n_events = 1

        if self.judges and self.allow_uncalibrated:
            # The operator explicitly accepted uncalibrated judges:
            # record who and that it was explicit, right up front.
            n_events += 1
            log.record(
                {
                    "kind": "uncalibrated_judge_override",
                    "judge_ids": [
                        getattr(j, "judge_id", repr(j))
                        for j in self.judges
                    ],
                    "allow_uncalibrated": True,
                }
            )

        history: list[str] = []
        findings: list[dict[str, Any]] = []
        stop_reason = "unknown"

        while True:
            exhausted, why = tracker.exhausted()
            if exhausted:
                stop_reason = f"budget:{why}"
                break

            spec, spec_meta = self.planner.propose_spec(
                history,
                remaining_probes=tracker.remaining_probes(),
                remaining_rounds=tracker.remaining_rounds(),
            )
            if not spec_meta.get("cache_hit"):
                tracker.record_planner_call()
            n_events += 1
            log.record(
                {
                    "kind": "plan",
                    "round": tracker.rounds_used + 1,
                    "spec": spec.to_dict(),
                    "cache_hit": spec_meta.get("cache_hit"),
                    "retried": spec_meta.get("retried"),
                }
            )

            if spec.action == "stop":
                stop_reason = f"planner_stop:{spec.reason}"
                break

            batch = self._build_batch(spec, tracker)
            if len(batch) == 0:
                stop_reason = "budget:max_probes:batch_empty_after_cap"
                break
            tracker.record_probes(len(batch))
            event = batch.to_event()
            event["round"] = tracker.rounds_used + 1
            n_events += 1
            log.record(event)

            outputs = self._execute_batch(batch)
            n_events += 1
            log.record(
                {
                    "kind": "observations",
                    "round": tracker.rounds_used + 1,
                    "results": [
                        {
                            "probe_id": probe.id,
                            "output": _truncate_output(output),
                        }
                        for probe, output in zip(batch.probes, outputs)
                    ],
                }
            )

            summary = self._summarize_round(
                batch, outputs, tracker.rounds_used + 1
            )
            tracker.record_round(summary["strength"])
            findings.append(summary)
            n_events += 1
            log.record({"kind": "round_summary", **_json_safe(summary)})
            history.append(AuditPlanner.summarize_for_prompt(summary))

            if tracker.is_flat():
                stop_reason = (
                    f"flat_findings:{budget.flat_rounds}_rounds_no_improvement"
                )
                break

        n_events += 1
        log.record(
            {
                "kind": "campaign_end",
                "stop_reason": stop_reason,
                "rounds": tracker.rounds_used,
                "probes_used": tracker.probes_used,
                "best_strength": tracker.best_strength(),
            }
        )
        _ = start  # recorded for its side effect; kept for clarity
        return CampaignReport(
            brief=_json_safe(brief),
            seed=self.seed,
            rounds=tracker.rounds_used,
            probes_used=tracker.probes_used,
            stop_reason=stop_reason,
            findings=findings,
            best_strength=tracker.best_strength(),
            evidence_path=self.evidence_path,
            n_events=n_events,
        )

    # ------------------------------------------------------------------
    # Batch construction (deterministic; the LLM only picked the spec)
    # ------------------------------------------------------------------
    def _build_batch(
        self, spec: ProbeSpec, tracker: BudgetTracker
    ) -> ProbeBatch:
        brief = self.planner.brief
        round_no = tracker.rounds_used + 1
        generator = spec.generator
        if generator not in GENERATORS:  # pragma: no cover - validated
            raise ValueError(f"unknown generator {generator!r}")

        if generator == "counterfactual":
            batch = generate_counterfactuals(
                dict(brief.get("base_input", {})),
                spec.params["attributes"],
                text_field=brief.get("text_field"),
                id_prefix=f"r{round_no}cf",
            )
        elif generator == "adversarial":
            batch = generate_adversarial(
                target_type=brief["target_type"],
                base_row=(
                    dict(brief["base_input"])
                    if brief["target_type"] == "tabular"
                    else None
                ),
                numeric_fields=spec.params.get("numeric_fields"),
                categorical_fields=spec.params.get("categorical_fields"),
                id_prefix=f"r{round_no}adv",
            )
        else:  # metamorphic
            wanted = spec.params.get("relations")
            relations = [
                r
                for r in BUILTIN_RELATIONS
                if wanted is None or r.name in wanted
            ]
            if not relations:
                raise ValueError(
                    "metamorphic spec names no known relations: "
                    f"{wanted!r}"
                )
            batch = generate_metamorphic(
                list(spec.params["inputs"]),
                relations=relations,
                id_prefix=f"r{round_no}mm",
            )
        return _cap_batch(batch, tracker.remaining_probes())

    # ------------------------------------------------------------------
    # Execution (cached; identical inputs never re-hit the target)
    # ------------------------------------------------------------------
    def _execute_batch(self, batch: ProbeBatch) -> list[Any]:
        probes = batch.probes
        if not probes:
            return []
        if all(isinstance(p.payload, str) for p in probes):
            prompts = [p.payload for p in probes]
            return self._cached_generate(prompts)
        if all(isinstance(p.payload, dict) for p in probes):
            rows = [dict(p.payload) for p in probes]
            return self._cached_predict(rows)
        raise ValueError(
            "mixed text/row payloads in one batch are not supported"
        )

    def _cached_generate(self, prompts: list[str]) -> list[Any]:
        outputs: list[Any] = [None] * len(prompts)
        missing_idx: list[int] = []
        missing: list[str] = []
        for i, prompt in enumerate(prompts):
            hit, value = self.cache.lookup({"generate": prompt})
            if hit:
                outputs[i] = value
            else:
                missing_idx.append(i)
                missing.append(prompt)
        if missing:
            try:
                fresh = list(self.audited_target.generate(missing))
            except Exception as exc:  # target must not kill the campaign
                fresh = [{_ERROR_KEY: f"{type(exc).__name__}: {exc}"}] * len(
                    missing
                )
            if len(fresh) != len(missing):
                fresh = (
                    list(fresh)[: len(missing)]
                    + [{_ERROR_KEY: "target returned wrong arity"}]
                    * len(missing)
                )[: len(missing)]
            for i, prompt, output in zip(missing_idx, missing, fresh):
                outputs[i] = output
                self.cache.store({"generate": prompt}, _json_safe(output))
        return outputs

    def _cached_predict(self, rows: list[dict[str, Any]]) -> list[Any]:
        import pandas as pd

        outputs: list[Any] = [None] * len(rows)
        missing_idx: list[int] = []
        missing_rows: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            # Order-preserving key: metamorphic reorder probes differ
            # ONLY in key order, and that difference must survive.
            hit, value = self.cache.lookup({"predict": list(row.items())})
            if hit:
                outputs[i] = value
            else:
                missing_idx.append(i)
                missing_rows.append(row)
        if missing_rows:
            try:
                fresh = self.audited_target.predict(
                    pd.DataFrame(missing_rows)
                )
                fresh = list(fresh)
            except Exception as exc:  # target must not kill the campaign
                fresh = [{_ERROR_KEY: f"{type(exc).__name__}: {exc}"}] * len(
                    missing_rows
                )
            if len(fresh) != len(missing_rows):
                fresh = (
                    list(fresh)[: len(missing_rows)]
                    + [{_ERROR_KEY: "target returned wrong arity"}]
                    * len(missing_rows)
                )[: len(missing_rows)]
            for i, row, output in zip(missing_idx, missing_rows, fresh):
                outputs[i] = output
                self.cache.store(
                    {"predict": list(row.items())}, _json_safe(output)
                )
        return outputs

    # ------------------------------------------------------------------
    # Deterministic round summaries (statistics in code, never in LLM)
    # ------------------------------------------------------------------
    def _summarize_round(
        self, batch: ProbeBatch, outputs: list[Any], round_no: int
    ) -> dict[str, Any]:
        kinds = [p.kind for p in batch.probes]
        generator = max(set(kinds), key=kinds.count) if kinds else "none"
        summary: dict[str, Any] = {
            "round": round_no,
            "generator": generator,
            "n_probes": len(batch.probes),
            "strength": 0.0,
            "details": {},
        }
        if generator == "counterfactual":
            self._summarize_counterfactual(batch, outputs, summary)
        elif generator == "adversarial":
            self._summarize_adversarial(outputs, summary)
        elif generator == "metamorphic":
            self._summarize_metamorphic(batch, outputs, summary)
        summary["strength"] = float(summary["strength"])
        return summary

    def _summarize_counterfactual(
        self, batch: ProbeBatch, outputs: list[Any], summary: dict[str, Any]
    ) -> None:
        # Group numeric outputs by annotated protected-attribute value.
        numeric = [
            (probe, output)
            for probe, output in zip(batch.probes, outputs)
            if _is_number(output)
        ]
        details = summary["details"]
        if not numeric:
            if self.judges:
                self._summarize_with_judges(batch, outputs, summary)
            else:
                details["note"] = (
                    "non-numeric outputs recorded without scoring "
                    "(no judges configured)"
                )
            return
        attrs: dict[str, dict[str, list[float]]] = {}
        for probe, output in numeric:
            for attr, value in probe.attributes.items():
                attrs.setdefault(attr, {}).setdefault(
                    str(value), []
                ).append(float(output))
        gaps: dict[str, float] = {}
        rates: dict[str, dict[str, float]] = {}
        for attr, by_value in attrs.items():
            means = {
                value: _mean(vals) for value, vals in by_value.items()
            }
            rates[attr] = {k: round(v, 4) for k, v in means.items()}
            if len(means) >= 2:
                gaps[attr] = round(max(means.values()) - min(means.values()), 4)
        details["outcome_rates"] = rates
        details["gaps"] = gaps
        summary["strength"] = max(gaps.values(), default=0.0)

    def _summarize_with_judges(
        self, batch: ProbeBatch, outputs: list[Any], summary: dict[str, Any]
    ) -> None:
        """Score non-numeric (text) outputs with the configured judges.

        Judges produce *labels*; everything computed from them here is
        deterministic arithmetic (:func:`aggregate_judge_findings`). The
        round strength is the strongest judge-label gap observed.
        """
        details = summary["details"]
        # Only string outputs can be judged; anything else is counted,
        # not silently dropped.
        judged_idx = [
            i for i, o in enumerate(outputs) if isinstance(o, str)
        ]
        details["n_judged_outputs"] = len(judged_idx)
        details["n_non_string_outputs"] = len(outputs) - len(judged_idx)
        if not judged_idx:
            details["note"] = "no string outputs for judges to score"
            return
        texts = [outputs[i] for i in judged_idx]
        judged_probes = [batch.probes[i] for i in judged_idx]
        # Group by each protected attribute the probes annotate.
        attr_names = sorted(
            {attr for p in judged_probes for attr in p.attributes}
        )
        judge_findings: list[dict[str, Any]] = []
        for judge in self.judges:
            scores = judge.score(texts)
            ordinal = (
                TONE_ORDER if getattr(judge, "name", "") == "tone-judge" else None
            )
            per_attr: dict[str, Any] = {}
            best = 0.0
            for attr in attr_names:
                groups = [str(p.attributes.get(attr)) for p in judged_probes]
                agg = aggregate_judge_findings(
                    judge.judge_id, scores, groups, ordinal=ordinal
                )
                per_attr[attr] = agg
                best = max(best, float(agg["strength"]))
            judge_findings.append(
                {
                    "judge_id": judge.judge_id,
                    "model_id": getattr(judge, "model_id", "unknown-model"),
                    "calibrated": bool(
                        getattr(getattr(judge, "calibration", None), "passed", False)
                    ),
                    "by_attribute": per_attr,
                    "best_strength": round(best, 4),
                }
            )
        details["judge_findings"] = judge_findings
        summary["strength"] = max(
            (jf["best_strength"] for jf in judge_findings), default=0.0
        )

    @staticmethod
    def _summarize_adversarial(
        outputs: list[Any], summary: dict[str, Any]
    ) -> None:
        errors = sum(
            1
            for o in outputs
            if isinstance(o, dict) and _ERROR_KEY in o
        )
        rate = errors / len(outputs) if outputs else 0.0
        summary["details"]["error_rate"] = round(rate, 4)
        summary["details"]["n_errors"] = errors
        summary["strength"] = rate

    @staticmethod
    def _summarize_metamorphic(
        batch: ProbeBatch, outputs: list[Any], summary: dict[str, Any]
    ) -> None:
        by_id = {p.id: o for p, o in zip(batch.probes, outputs)}
        groups: dict[str, dict[str, Any]] = {}
        expected_by_group: dict[str, str] = {}
        for probe in batch.probes:
            group_id = probe.meta.get("group_id")
            if not group_id:
                continue
            group_id = str(group_id)
            role = "source" if probe.id.endswith("-src") else "followup"
            groups.setdefault(group_id, {})[role] = by_id[probe.id]
            expected_by_group[group_id] = str(
                probe.meta.get("expected", "")
            )
        violations = 0
        checked = 0
        for group_id, pair in groups.items():
            if expected_by_group.get(group_id) != "equal_output":
                continue
            if "source" not in pair or "followup" not in pair:
                continue
            checked += 1
            if not check_equal_output(pair["source"], pair["followup"]):
                violations += 1
        rate = violations / checked if checked else 0.0
        summary["details"]["violation_rate"] = round(rate, 4)
        summary["details"]["n_violations"] = violations
        summary["details"]["n_pairs_checked"] = checked
        summary["strength"] = rate


def _cap_batch(batch: ProbeBatch, remaining: int) -> ProbeBatch:
    """Trim a batch to ``remaining`` probes, keeping pairs intact.

    Counterfactual mirrors (``meta["pair_id"]``) and metamorphic
    source/follow-up groups (``meta["group_id"]``) are dropped as whole
    units from the end; other probes are truncated plainly.
    """
    if len(batch) <= remaining:
        return batch
    kept: list[Any] = []
    seen_groups: set[str] = set()
    for probe in batch.probes:
        group = probe.meta.get("pair_id") or probe.meta.get("group_id")
        key = str(group) if group is not None else f"probe:{probe.id}"
        if key not in seen_groups:
            # Count how many probes this group contributes.
            size = sum(
                1
                for p in batch.probes
                if str(p.meta.get("pair_id") or p.meta.get("group_id")
                       or f"probe:{p.id}") == key
            )
            if len(kept) + size > remaining:
                break
            seen_groups.add(key)
        kept.append(probe)
    return ProbeBatch(probes=kept, name=batch.name)


def _json_safe(obj: Any) -> Any:
    """Best-effort JSON-safe copy for evidence payloads."""
    try:
        import json

        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


def _truncate_output(output: Any) -> Any:
    if isinstance(output, str) and len(output) > _MAX_OUTPUT_CHARS:
        return output[:_MAX_OUTPUT_CHARS] + "...[truncated]"
    return output

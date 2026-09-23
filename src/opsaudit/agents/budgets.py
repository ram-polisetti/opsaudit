"""Budget and stopping-rule machinery for agentic audit campaigns.

A campaign must never run forever and must never spend more than the
operator approved. :class:`Budget` declares the limits;
:class:`BudgetTracker` enforces them deterministically: every check is a
pure function of counters, so the same campaign history always produces
the same stop decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Budget:
    """Limits for one audit campaign.

    Attributes:
        max_probes: Hard cap on probes executed against the target.
        max_rounds: Hard cap on plan → probe → observe → replan rounds.
        max_cost: Optional hard cap on estimated cost (same units as
            ``probe_cost`` / ``planner_call_cost``). ``None`` = uncapped.
        probe_cost: Estimated cost of one probe execution (API fees,
            latency budget, ...). Operator-configured.
        planner_call_cost: Estimated cost of one planner-LLM call.
        flat_rounds: Stop when the best finding strength has not improved
            for this many consecutive rounds (marginal findings have
            flattened). ``0`` disables the rule.
        min_improvement: Minimum gain in finding strength that counts as
            an improvement for the flatness rule.
    """

    max_probes: int = 200
    max_rounds: int = 10
    max_cost: float | None = None
    probe_cost: float = 0.0
    planner_call_cost: float = 0.0
    flat_rounds: int = 2
    min_improvement: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_probes <= 0:
            raise ValueError("max_probes must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        if self.probe_cost < 0 or self.planner_call_cost < 0:
            raise ValueError("costs must be non-negative")
        if self.flat_rounds < 0:
            raise ValueError("flat_rounds must be non-negative")


@dataclass
class BudgetTracker:
    """Enforces a :class:`Budget` over one campaign run.

    All state transitions are explicit method calls; nothing here touches
    the network or the target. ``exhausted()`` and ``is_flat()`` are pure
    functions of the recorded history.
    """

    budget: Budget
    probes_used: int = 0
    rounds_used: int = 0
    planner_calls: int = 0
    # Per-round finding strengths, in round order (round 1 first).
    round_strengths: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_probes(self, n: int) -> None:
        """Account for ``n`` executed probes."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.probes_used += n

    def record_planner_call(self) -> None:
        """Account for one planner-LLM call."""
        self.planner_calls += 1

    def record_round(self, strength: float) -> None:
        """Close out a round with its finding strength.

        ``strength`` is a non-negative number where larger means a
        stronger disparity/robustness signal was observed this round
        (computed deterministically by the campaign, never by an LLM).
        """
        if strength < 0:
            raise ValueError("strength must be non-negative")
        self.rounds_used += 1
        self.round_strengths.append(strength)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def estimated_cost(self) -> float:
        """Estimated spend so far under the budget's cost model."""
        return (
            self.probes_used * self.budget.probe_cost
            + self.planner_calls * self.budget.planner_call_cost
        )

    def remaining_probes(self) -> int:
        """Probes still allowed under ``max_probes``."""
        return max(0, self.budget.max_probes - self.probes_used)

    def remaining_rounds(self) -> int:
        """Rounds still allowed under ``max_rounds``."""
        return max(0, self.budget.max_rounds - self.rounds_used)

    # ------------------------------------------------------------------
    # Stopping rules
    # ------------------------------------------------------------------
    def exhausted(self) -> tuple[bool, str]:
        """Check the hard caps. Returns ``(done, reason)``.

        ``reason`` is one of ``"ok"``, ``"max_probes"``, ``"max_rounds"``,
        ``"max_cost"``.
        """
        if self.probes_used >= self.budget.max_probes:
            return True, "max_probes"
        if self.rounds_used >= self.budget.max_rounds:
            return True, "max_rounds"
        if (
            self.budget.max_cost is not None
            and self.estimated_cost() >= self.budget.max_cost
        ):
            return True, "max_cost"
        return False, "ok"

    def is_flat(self) -> bool:
        """True when findings have flattened for ``flat_rounds`` rounds.

        The rule fires only after at least ``flat_rounds`` rounds exist:
        the best strength before the trailing window must not be beaten
        by more than ``min_improvement`` inside the window. Disabled
        (always False) when ``flat_rounds`` is 0.
        """
        k = self.budget.flat_rounds
        if k <= 0 or len(self.round_strengths) < k:
            return False
        window = self.round_strengths[-k:]
        baseline = max(self.round_strengths[:-k], default=0.0)
        return max(window) <= baseline + self.budget.min_improvement

    def best_strength(self) -> float:
        """Best finding strength seen so far (0.0 when no rounds)."""
        return max(self.round_strengths, default=0.0)

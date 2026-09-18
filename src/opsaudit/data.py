"""Seeded synthetic logistics data generators.

All data is synthetic, generated locally with ``numpy.random.default_rng``;
no real company data is used and this module makes no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_dispatch(
    n: int = 5000, bias_strength: float = 0.0, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic package-to-driver dispatch decisions.

    The audit asks whether the ``priority_route`` model decision is allocated
    and calibrated fairly across delivery zones, using ``on_time`` delivery as
    ground truth.

    Example:
        >>> generate_dispatch(n=2, seed=7).columns.tolist()[0]
        'driver_id'
    """
    _validate_generator_inputs(n, bias_strength)
    rng = np.random.default_rng(seed)
    group = rng.choice(np.array(["A", "B", "C"]), size=n, p=[0.5, 0.3, 0.2])
    packages_assigned = rng.integers(20, 120, size=n)
    group_penalty = np.select(
        [group == "A", group == "B", group == "C"], [0.0, 0.25, 0.45]
    )
    priority_probability = np.clip(0.55 - bias_strength * group_penalty, 0.01, 0.99)
    priority_route = rng.binomial(1, priority_probability).astype(int)
    on_time_probability = np.clip(0.86 + 0.06 * priority_route, 0.01, 0.99)
    on_time = rng.binomial(1, on_time_probability).astype(int)

    return pd.DataFrame(
        {
            "driver_id": [f"DRV-{index:06d}" for index in range(1, n + 1)],
            "group": group.astype(str),
            "packages_assigned": packages_assigned.astype(int),
            "priority_route": priority_route,
            "on_time": on_time,
        }
    )


def generate_staffing(
    n: int = 2000, bias_strength: float = 0.0, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic warehouse shift-allocation decisions.

    The audit asks whether the ``shifts_granted`` model decision is allocated
    fairly by contract type, using ``completed_satisfactorily`` as ground
    truth.

    Example:
        >>> generate_staffing(n=2, seed=7).columns.tolist()[0]
        'worker_id'
    """
    _validate_generator_inputs(n, bias_strength)
    rng = np.random.default_rng(seed)
    group = rng.choice(np.array(["FT", "PT", "temp"]), size=n, p=[0.5, 0.3, 0.2])
    tenure_months = rng.integers(1, 60, size=n)
    shifts_requested = rng.integers(1, 8, size=n)
    group_penalty = np.select(
        [group == "FT", group == "PT", group == "temp"], [0.0, 0.25, 0.45]
    )
    shifts_probability = np.clip(0.60 - bias_strength * group_penalty, 0.01, 0.99)
    shifts_granted = rng.binomial(1, shifts_probability).astype(int)
    outcome_probability = np.clip(0.80 + 0.05 * (tenure_months / 60), 0.01, 0.99)
    completed_satisfactorily = rng.binomial(1, outcome_probability).astype(int)

    return pd.DataFrame(
        {
            "worker_id": [f"WRK-{index:06d}" for index in range(1, n + 1)],
            "group": group.astype(str),
            "tenure_months": tenure_months.astype(int),
            "shifts_requested": shifts_requested.astype(int),
            "shifts_granted": shifts_granted,
            "completed_satisfactorily": completed_satisfactorily,
        }
    )


def _validate_generator_inputs(n: int, bias_strength: float) -> None:
    """Validate common generator arguments."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0.0 <= bias_strength <= 1.0:
        raise ValueError("bias_strength must be between 0 and 1")

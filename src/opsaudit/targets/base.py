"""Uniform interface for audit targets.

A target is any system under audit. Text targets (LLMs, chatbots, RAG
pipelines) implement :meth:`generate`; scoring targets (tabular classifiers)
implement :meth:`predict`. The base class provides both with a clear
``NotImplementedError`` so adapters only implement what their target
supports, and :meth:`describe` (abstract) supplies JSON-safe metadata for
the evidence log.
"""

from __future__ import annotations

import abc
from typing import Any


class Target(abc.ABC):
    """Anything the auditor can probe.

    Subclasses implement :meth:`generate`, :meth:`predict`, or both, plus
    the abstract :meth:`describe`.
    """

    #: Human-readable label used in reports and the evidence log.
    name: str = "target"

    # ------------------------------------------------------------------
    # Probing interface
    # ------------------------------------------------------------------
    def generate(self, prompts: list[str]) -> list[str]:
        """Return one completion per prompt, in input order.

        Raises:
            NotImplementedError: if this target does not produce text.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support generate(); "
            "it is not a text target"
        )

    def predict(self, X: Any) -> list:
        """Return one prediction per row of ``X``, in input order.

        ``X`` may be a pandas DataFrame, a NumPy array, or a list of rows.

        Raises:
            NotImplementedError: if this target does not score rows.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict(); "
            "it is not a scoring target"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-safe metadata about this target.

        Must include at least ``target_type`` and ``name``. The evidence
        log snapshots this dict, so it must never require network access.
        """

    @property
    def supports_generate(self) -> bool:
        """True when this target implements :meth:`generate`."""
        return type(self).generate is not Target.generate

    @property
    def supports_predict(self) -> bool:
        """True when this target implements :meth:`predict`."""
        return type(self).predict is not Target.predict

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"

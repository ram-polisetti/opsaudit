"""Tabular classifier target adapter.

Wraps any sklearn-style estimator (``predict`` and optionally
``predict_proba``) so v0.1's tabular world plugs straight into the
agentic auditor's uniform interface. No optional dependencies: duck
typing is used instead of importing sklearn.
"""

from __future__ import annotations

from typing import Any

from .base import Target


class TabularTarget(Target):
    """Audit a tabular classifier.

    Example:
        >>> from sklearn.dummy import DummyClassifier  # doctest: +SKIP
        >>> clf = DummyClassifier(strategy="most_frequent")  # doctest: +SKIP
        >>> target = TabularTarget(clf, name="baseline")  # doctest: +SKIP
    """

    def __init__(self, estimator: Any, *, name: str = "tabular") -> None:
        if not hasattr(estimator, "predict") or not callable(
            getattr(estimator, "predict")
        ):
            raise ValueError(
                "estimator must expose a callable predict(X) method "
                "(sklearn-style); got "
                f"{type(estimator).__name__}"
            )
        self.estimator = estimator
        self.name = name

    def predict(self, X: Any) -> list:
        if X is None:
            raise ValueError("X must not be None")
        try:
            raw = self.estimator.predict(X)
        except Exception as exc:
            raise RuntimeError(
                f"underlying estimator.predict failed: {exc}"
            ) from exc
        try:
            return list(raw)
        except TypeError as exc:
            raise ValueError(
                "estimator.predict(X) must return an iterable of "
                f"predictions; got {type(raw).__name__}"
            ) from exc

    def predict_proba(self, X: Any) -> list[list[float]] | None:
        """Return class probabilities when the estimator supports them."""
        proba_fn = getattr(self.estimator, "predict_proba", None)
        if proba_fn is None or not callable(proba_fn):
            return None
        raw = proba_fn(X)
        return [list(map(float, row)) for row in raw]

    def describe(self) -> dict[str, Any]:
        return {
            "target_type": "tabular",
            "name": self.name,
            "estimator": type(self.estimator).__name__,
            "has_predict_proba": callable(
                getattr(self.estimator, "predict_proba", None)
            ),
        }

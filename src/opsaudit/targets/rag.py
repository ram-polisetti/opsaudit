"""RAG pipeline target adapter.

Wraps any callable implementing a retriever+generator pipeline —
``rag_fn(query: str) -> str`` — behind the :class:`Target` interface so
end-to-end RAG systems can be probed exactly like standalone LLMs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Target


class RagTarget(Target):
    """Audit a RAG pipeline end to end.

    Example:
        >>> def my_rag(query: str) -> str:  # doctest: +SKIP
        ...     docs = retrieve(query)      # doctest: +SKIP
        ...     return llm(query, docs)     # doctest: +SKIP
        >>> target = RagTarget(my_rag, name="support-rag")  # doctest: +SKIP
    """

    def __init__(
        self, rag_fn: Callable[[str], str], *, name: str = "rag"
    ) -> None:
        if not callable(rag_fn):
            raise ValueError(
                "rag_fn must be a callable taking a query string and "
                f"returning a response string; got {type(rag_fn).__name__}"
            )
        self.rag_fn = rag_fn
        self.name = name

    def generate(self, prompts: list[str]) -> list[str]:
        prompts = list(prompts)
        if not prompts:
            return []
        results: list[str] = []
        for prompt in prompts:
            try:
                response = self.rag_fn(prompt)
            except Exception as exc:
                raise RuntimeError(
                    f"rag_fn raised for query {prompt!r}: {exc}"
                ) from exc
            if response is None:
                results.append("")
            elif not isinstance(response, str):
                raise ValueError(
                    "rag_fn must return a string or None; got "
                    f"{type(response).__name__}"
                )
            else:
                results.append(response)
        return results

    def describe(self) -> dict[str, Any]:
        fn = self.rag_fn
        return {
            "target_type": "rag",
            "name": self.name,
            "pipeline": getattr(fn, "__name__", type(fn).__name__),
        }

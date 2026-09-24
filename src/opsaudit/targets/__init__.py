"""Model-agnostic target adapters for the agentic auditor (v0.2, Phase 1).

A :class:`Target` is anything the auditor can probe: an LLM behind an API,
a local HuggingFace model, a tabular classifier, or a RAG pipeline. The
auditor only ever talks to the uniform interface defined here, so new model
types slot in without touching audit logic.

Every adapter must be importable without its optional dependency installed:
heavy imports (``transformers``, ``httpx``) happen lazily inside methods,
and calling a method whose dependency is missing raises an ``ImportError``
with an actionable message instead of failing at import time.
"""

from .base import Target
from .hf import HuggingFaceTarget
from .ollama import OllamaTarget, SecureOllamaTarget
from .openai_compat import OpenAICompatTarget
from .rag import RagTarget
from .tabular import TabularTarget

__all__ = [
    "Target",
    "HuggingFaceTarget",
    "OllamaTarget",
    "SecureOllamaTarget",
    "OpenAICompatTarget",
    "RagTarget",
    "TabularTarget",
]

"""HuggingFace ``transformers`` target adapter.

Wraps a text-generation pipeline behind the :class:`Target` interface.
``transformers`` is imported lazily inside :meth:`generate`, so importing
this module never requires the dependency.
"""

from __future__ import annotations

from typing import Any

from .base import Target


class HuggingFaceTarget(Target):
    """Audit a local HuggingFace text-generation model.

    Example:
        >>> target = HuggingFaceTarget("sshleifer/tiny-gpt2")  # doctest: +SKIP
        >>> target.generate(["Hello"])  # doctest: +SKIP
        ['Hello world']
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        max_new_tokens: int = 64,
        device: int | str = -1,
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id
        self.name = name or f"hf:{model_id}"
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self._pipeline: Any = None

    def _load(self) -> Any:
        """Build (once) the text-generation pipeline; lazy ``transformers``."""
        if self._pipeline is None:
            try:
                import transformers
            except ImportError as exc:
                raise ImportError(
                    "HuggingFaceTarget requires the 'transformers' package "
                    "(and a backend such as torch). Install it with "
                    "'pip install transformers torch' and retry."
                ) from exc
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                device=self.device,
                **self.pipeline_kwargs,
            )
        return self._pipeline

    def generate(self, prompts: list[str]) -> list[str]:
        prompts = list(prompts)
        if not prompts:
            return []
        pipeline = self._load()
        outputs = pipeline(prompts, max_new_tokens=self.max_new_tokens)
        # ``pipeline`` returns a list of dicts for a list input; normalize
        # defensively in case a backend returns a bare string.
        results: list[str] = []
        for out in outputs:
            if isinstance(out, dict):
                results.append(str(out.get("generated_text", "")))
            elif isinstance(out, list) and out and isinstance(out[0], dict):
                results.append(str(out[0].get("generated_text", "")))
            else:
                results.append(str(out))
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "target_type": "huggingface",
            "name": self.name,
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "device": str(self.device),
        }

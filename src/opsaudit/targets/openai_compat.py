"""OpenAI-compatible chat-completions target adapter.

Talks to any endpoint implementing the ``/chat/completions`` API (OpenAI,
vLLM, llama.cpp server, together.ai, etc.) using ``httpx``, imported lazily
so this module loads without the dependency.
"""

from __future__ import annotations

from typing import Any

from .base import Target


class OpenAICompatTarget(Target):
    """Audit a model behind an OpenAI-compatible chat-completions endpoint.

    Example:
        >>> target = OpenAICompatTarget(  # doctest: +SKIP
        ...     base_url="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ...     model="gpt-4o-mini",
        ... )
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        name: str | None = None,
        timeout: float = 60.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        for label, value in (
            ("base_url", base_url),
            ("api_key", api_key),
            ("model", model),
        ):
            if not value or not str(value).strip():
                raise ValueError(f"{label} must be a non-empty string")
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name or f"openai-compat:{model}"
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatTarget requires the 'httpx' package. "
                "Install it with 'pip install httpx' and retry."
            ) from exc
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(
                "OpenAI-compatible endpoint returned a non-JSON-object body"
            )
        return data

    def generate(self, prompts: list[str]) -> list[str]:
        prompts = list(prompts)
        if not prompts:
            return []
        results: list[str] = []
        for prompt in prompts:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **self.extra_body,
            }
            data = self._post(payload)
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    "OpenAI-compatible endpoint returned an unexpected "
                    f"response shape: {data!r}"
                ) from exc
            results.append("" if text is None else str(text))
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "target_type": "openai_compat",
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            # Deliberately omitted: api_key must never enter the evidence log.
        }

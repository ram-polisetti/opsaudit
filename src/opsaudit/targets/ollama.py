"""Ollama target adapter (Ollama Cloud or a local Ollama server).

Speaks the Ollama ``/api/chat`` protocol. Configuration comes from
environment variables so no secret is ever hardcoded or committed:

- ``OLLAMA_BASE_URL`` — e.g. ``https://ollama.com`` (Cloud) or
  ``http://localhost:11434`` (local). Defaults to the local server.
- ``OLLAMA_API_KEY`` — required by Ollama Cloud; optional for a local
  server.
- ``OLLAMA_MODEL`` — default model when none is passed explicitly.

``httpx`` is imported lazily inside :meth:`generate`.

For Ollama Cloud without any key handling, use
:class:`SecureOllamaTarget`: it authenticates through the stored
``custom.ollama`` connector via authd surrogates, so the raw API key is
never read, printed, or persisted.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

from .base import Target

DEFAULT_BASE_URL = "http://localhost:11434"
_SECURE_CRED = "custom.ollama"
_SECURE_HOSTS = ["ollama.com"]
_DYNAMIC_CRED_PATH = "/opt/hatch/skills/skill-creator/bin"


def _surrogate_helpers():
    """Import authd surrogate helpers lazily (only needed for Cloud)."""
    sys.path.insert(0, _DYNAMIC_CRED_PATH)
    from dynamic_credentials import (
        add_surrogate_to_request,
        read_json_response,
    )
    return add_surrogate_to_request, read_json_response


class OllamaTarget(Target):
    """Audit a model served by Ollama (Cloud or local).

    Example:
        >>> import os  # doctest: +SKIP
        >>> os.environ["OLLAMA_API_KEY"] = "..."  # doctest: +SKIP
        >>> target = OllamaTarget(model="kimi-k2.7-code")  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        resolved_model = model or os.environ.get("OLLAMA_MODEL", "")
        if not resolved_model or not str(resolved_model).strip():
            raise ValueError(
                "An Ollama model is required: pass model= or set the "
                "OLLAMA_MODEL environment variable"
            )
        self.model = str(resolved_model)
        self.base_url = str(
            base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "")
        self.name = name or f"ollama:{self.model}"
        self.timeout = timeout
        self.options = dict(options or {})

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OllamaTarget requires the 'httpx' package. "
                "Install it with 'pip install httpx' and retry."
            ) from exc
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/chat", headers=headers, json=payload
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(
                "Ollama endpoint returned a non-JSON-object body"
            )
        return data

    def generate(self, prompts: list[str]) -> list[str]:
        prompts = list(prompts)
        if not prompts:
            return []
        results: list[str] = []
        for prompt in prompts:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if self.options:
                payload["options"] = self.options
            data = self._post(payload)
            try:
                text = data["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "Ollama endpoint returned an unexpected response "
                    f"shape: {data!r}"
                ) from exc
            results.append("" if text is None else str(text))
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "target_type": "ollama",
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            # Deliberately omitted: api_key must never enter the evidence log.
        }


class SecureOllamaTarget(OllamaTarget):
    """Ollama Cloud target that authenticates through the stored
    ``custom.ollama`` connector (authd surrogates) instead of an API key.

    The raw key is never visible to this process: ``add_surrogate_to_request``
    attaches an ``hsurr:*`` surrogate and the credential is substituted on
    the approved outbound request. No ``OLLAMA_API_KEY`` is read.

    Example:
        >>> target = SecureOllamaTarget(model="glm-5.1")  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        name: str | None = None,
        timeout: float = 120.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        resolved_model = model or os.environ.get("OLLAMA_MODEL", "")
        if not resolved_model or not str(resolved_model).strip():
            raise ValueError(
                "An Ollama model is required: pass model= or set the "
                "OLLAMA_MODEL environment variable"
            )
        self.model = str(resolved_model)
        self.base_url = "https://ollama.com"
        self.api_key = ""  # never used; auth comes from the connector
        self.name = name or f"ollama-secure:{self.model}"
        self.timeout = timeout
        self.options = dict(options or {})

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        add_surrogate_to_request, read_json_response = _surrogate_helpers()
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        add_surrogate_to_request(
            req, _SECURE_CRED, entry_name="access_token",
            allowed_hosts=_SECURE_HOSTS,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = read_json_response(resp)
        if not isinstance(data, dict):
            raise ValueError("Ollama endpoint returned a non-JSON-object body")
        return data

    def describe(self) -> dict[str, Any]:
        return {
            "target_type": "ollama-secure",
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "auth": "custom.ollama connector (surrogate; key never visible)",
        }

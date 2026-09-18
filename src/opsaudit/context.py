"""Audit-context manifest loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def load_context(path: str | Path) -> dict[str, str]:
    """Load a flat YAML audit-context manifest.

    The manifest records review metadata, not raw operational records or
    personal data. Values must be scalar so reports stay compact and portable.

    Example:
        >>> load_context("examples/audit-context.yaml")["system_name"]
        'route-priority-assignment'
    """
    manifest_path = Path(path)
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if loaded is None:
        raise ValueError("audit context YAML must contain a mapping")
    return normalize_context(loaded)


def normalize_context(context: Mapping[str, object]) -> dict[str, str]:
    """Validate and normalize a flat audit-context mapping.

    Example:
        >>> normalize_context({"system_name": "dispatch", "model_version": 1})
        {'system_name': 'dispatch', 'model_version': '1'}
    """
    if not isinstance(context, Mapping):
        raise ValueError("audit context must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("audit-context keys must be non-empty strings")
        if value is None or isinstance(value, (Mapping, list, tuple, set)):
            raise ValueError(f"audit-context value for {key!r} must be a scalar")
        normalized[key.strip()] = str(value)
    return normalized

"""Response cache for agentic audit campaigns.

Identical inputs are never re-queried: the cache sits in front of both
the planner LLM and the audited target, in memory with an optional JSON
file backing so a campaign can resume without paying twice. Keys are
SHA-256 hashes of a canonical JSON encoding, so the cache is a pure
function of the input — same brief + seed reproduces the same campaign.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(obj: Any) -> str:
    """Canonical JSON encoding used for cache keys."""
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=True)


class ResponseCache:
    """In-memory response cache with optional JSON file persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, Any] = {}
        if self.path is not None and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                loaded = {}
            if isinstance(loaded, dict):
                self._entries = loaded

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------
    @staticmethod
    def key_for(obj: Any) -> str:
        """Stable cache key for any JSON-encodable input."""
        return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Lookup / store
    # ------------------------------------------------------------------
    def lookup(self, obj: Any) -> tuple[bool, Any]:
        """Return ``(hit, value)`` for ``obj``."""
        key = self.key_for(obj)
        if key in self._entries:
            return True, self._entries[key]
        return False, None

    def store(self, obj: Any, value: Any) -> None:
        """Cache ``value`` for ``obj``. ``value`` must be JSON-safe."""
        key = self.key_for(obj)
        # Fail fast on non-serializable values rather than corrupting
        # the on-disk cache later.
        json.dumps(value, default=str)
        self._entries[key] = json.loads(json.dumps(value, default=str))

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._entries)

    def save(self) -> None:
        """Persist the cache to ``path`` (no-op when path is None)."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._entries, ensure_ascii=True, indent=1),
            encoding="utf-8",
        )
        tmp.replace(self.path)

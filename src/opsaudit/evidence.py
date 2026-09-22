"""Append-only evidence log for agentic audits (v0.2, Phase 1).

Every step of an audit campaign — probes issued, responses received,
metrics computed, planner decisions (Phase 3) — is recorded as one JSON
object per line (JSONL). The schema is deliberately generic: ``record``
accepts any event dict and stamps it with an ISO-8601 UTC timestamp, a
monotonic sequence number, and the target descriptor captured at log
creation time.

The log is append-only by design: there is no API to edit or delete
entries, which is what makes an agentic audit's transcript defensible.
Concurrent writers are serialized with an exclusive file lock
(``fcntl.flock`` on POSIX; best-effort on other platforms).

Example:
    >>> from opsaudit.targets import TabularTarget
    >>> import tempfile, os
    >>> fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    >>> log = EvidenceLog(path)  # doctest: +SKIP
    >>> log.record({"kind": "probe_batch", "n": 4})  # doctest: +SKIP
    {'seq': 0, ...}
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # POSIX-only; absence degrades to the in-process lock.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

#: Event kinds the auditor is expected to emit. ``record`` does not enforce
#: this list — it is documentation for Phase 2/3 authors.
EVENT_KINDS = (
    "probe_batch",  # probes issued to a target
    "response_batch",  # responses received from a target
    "metric_computed",  # a deterministic metric computed on responses
    "judge_scored",  # an LLM judge scored unstructured outputs (Phase 4)
    "planner_decision",  # the planner chose the next probe batch (Phase 3)
    "note",  # free-form operator annotation
)


class EvidenceLog:
    """Append-only JSONL transcript of an audit campaign."""

    def __init__(self, path: str | Path, target: Any | None = None) -> None:
        self.path = Path(path)
        describe = getattr(target, "describe", None)
        self.target_descriptor: dict[str, Any] = (
            dict(describe()) if callable(describe) else {}
        )
        self._lock = threading.Lock()
        self._seq = self._count_existing()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append ``event`` and return the stamped record.

        The stored record is ``event`` plus ``seq`` (monotonic, starting
        at 0), ``ts`` (UTC ISO-8601), and ``target`` (the descriptor
        captured when the log was created).
        """
        if not isinstance(event, dict):
            raise ValueError(
                f"event must be a dict; got {type(event).__name__}"
            )
        with self._lock:
            stamped = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "target": self.target_descriptor,
                **event,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(
                        json.dumps(stamped, ensure_ascii=True, default=str)
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            self._seq += 1
            return stamped

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read(self) -> list[dict[str, Any]]:
        """Return all records in sequence order."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda r: r.get("seq", 0))
        return records

    def __len__(self) -> int:
        return self._count_existing()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _count_existing(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

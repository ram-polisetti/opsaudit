"""Tamper-evident provenance for opsaudit reports.

Every ``opsaudit audit`` run embeds a ``provenance`` block in the report JSON
recording exactly what went into the audit (input data hash, row count, code
version, full resolved arguments, seed, timestamp). ``opsaudit verify``
recomputes the input hash and deterministically re-runs the audit to confirm
the recorded metrics still match, and ``opsaudit signoff`` appends human
review records chained by hash so any later edit — to the data, the metrics,
or a sign-off — is detectable.

Design notes (full rationale in docs/PROVENANCE.md):

- Hashes are SHA-256 over raw bytes (files) or canonical JSON
  (``sort_keys=True``, compact separators) for in-memory structures.
- The report body hash covers the audit metrics plus the provenance block,
  but not the sign-off list, so appending a sign-off never invalidates it.
- Each sign-off record is self-authenticating (``record_hash`` covers its own
  content) and chained to the previous record's hash (or the body hash for
  the first record), forming a tamper-evident append-only log.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes.

    Example:
        >>> import hashlib, tempfile, os
        >>> fd, name = tempfile.mkstemp()
        >>> _ = os.write(fd, b"abc"); os.close(fd)
        >>> sha256_file(name) == hashlib.sha256(b"abc").hexdigest()
        True
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize to deterministic canonical JSON for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 hex digest of a value's canonical JSON encoding.

    Example:
        >>> canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
        True
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 timestamp."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def python_version() -> str:
    """Return the running interpreter's version string."""
    return platform.python_version()


def git_info(start: str | Path | None = None) -> dict[str, Any]:
    """Return the git SHA and dirty flag for the checkout containing ``start``.

    ``start`` defaults to this file, so an editable install reports the
    opsaudit checkout itself. Returns ``{"git_sha": None, "git_dirty": None}``
    when the code is not inside a git checkout (e.g. an installed wheel).

    Example:
        >>> info = git_info()
        >>> set(info) == {"git_sha", "git_dirty"}
        True
    """
    root = Path(start).resolve() if start is not None else Path(__file__).resolve()
    directory = root if root.is_dir() else root.parent
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return _git_status(candidate)
    return {"git_sha": None, "git_dirty": None}


def _git_status(repo: Path) -> dict[str, Any]:
    """Read HEAD SHA and working-tree dirty flag for ``repo``."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,  # returncode handled explicitly below
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,  # returncode handled explicitly below
        )
        if sha.returncode != 0:
            return {"git_sha": None, "git_dirty": None}
        return {
            "git_sha": sha.stdout.strip() or None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def build_provenance(
    *,
    command: str,
    args: dict[str, Any],
    seed: int | None,
    input_path: str | Path | None = None,
    input_sha256: str | None = None,
    row_count: int | None = None,
    opsaudit_version: str,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block embedded in audit artifacts.

    ``input_*`` describe the audited data file; ``gate`` carries the
    deployment-gate verdict computed at audit time so ``verify`` can confirm
    it still holds. The caller fills in ``body_sha256`` after the enclosing
    document is assembled (see :func:`attach_body_hash`).

    Example:
        >>> prov = build_provenance(command="audit", args={"truth": "y"},
        ...                         seed=42, input_sha256="ab", row_count=3,
        ...                         opsaudit_version="0.1.1")
        >>> prov["tool"]
        'opsaudit'
    """
    info = git_info()
    block: dict[str, Any] = {
        "tool": "opsaudit",
        "opsaudit_version": opsaudit_version,
        "code": info,
        "run": {
            "command": command,
            "args": args,
            "seed": seed,
            "timestamp_utc": utc_now(),
            "python_version": python_version(),
        },
        "body_sha256": None,
    }
    if input_path is not None or input_sha256 is not None or row_count is not None:
        block["input"] = {
            "path": str(input_path) if input_path is not None else None,
            "sha256": input_sha256,
            "row_count": row_count,
        }
    if gate is not None:
        block["gate"] = gate
    return block


def attach_body_hash(document: dict[str, Any]) -> dict[str, Any]:
    """Compute and set ``provenance.body_sha256`` for a report document.

    The body is the full document minus the ``signoffs`` list: appending a
    human sign-off later must not invalidate the body hash. The hash field
    itself is excluded from the hashed content by hashing the document with
    ``body_sha256`` temporarily removed.

    Example:
        >>> doc = {"a": 1, "provenance": {"body_sha256": None}, "signoffs": []}
        >>> attach_body_hash(doc)["provenance"]["body_sha256"] is not None
        True
    """
    body = {key: value for key, value in document.items() if key != "signoffs"}
    prov = dict(body.get("provenance", {}))
    prov.pop("body_sha256", None)
    body["provenance"] = prov
    document["provenance"]["body_sha256"] = canonical_hash(body)
    return document


def check_body_hash(document: dict[str, Any]) -> tuple[bool, str]:
    """Verify a report document's body hash.

    Returns ``(True, "")`` when the embedded ``body_sha256`` matches the
    recomputed hash of the current body, otherwise ``(False, reason)``.
    """
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        return False, "report has no provenance block"
    expected = provenance.get("body_sha256")
    if not expected:
        return False, "provenance block has no body_sha256"
    body = {key: value for key, value in document.items() if key != "signoffs"}
    prov = dict(body.get("provenance", {}))
    prov.pop("body_sha256", None)
    body["provenance"] = prov
    actual = canonical_hash(body)
    if actual != expected:
        return (
            False,
            ("report body hash mismatch: metrics or provenance were modified "
            f"after the audit (expected {expected[:12]}…, got {actual[:12]}…)"),
        )
    return True, ""


def signoff_prev_hash(document: dict[str, Any]) -> str:
    """Return the hash a new sign-off record must chain to.

    The first sign-off chains to the report body hash; later ones chain to
    the canonical hash of the previous sign-off record.
    """
    signoffs = document.get("signoffs", [])
    if signoffs:
        return canonical_hash(signoffs[-1])
    provenance = document.get("provenance", {})
    body_hash = provenance.get("body_sha256")
    if not body_hash:
        raise ValueError("cannot sign a report with no body_sha256")
    return str(body_hash)


def check_signoff_chain(document: dict[str, Any]) -> tuple[bool, str]:
    """Verify every sign-off record's hash chain and required fields.

    Each record is self-authenticating via its ``record_hash`` (a hash of the
    record's own content) and chained to its predecessor via ``prev_hash``,
    so edits to any record — including the last one — are detectable.

    Returns ``(True, "")`` when the chain is intact, otherwise
    ``(False, reason)`` naming the first broken link.
    """
    signoffs = document.get("signoffs", [])
    if not isinstance(signoffs, list):
        return False, "signoffs must be a list"
    expected_prev = (document.get("provenance") or {}).get("body_sha256")
    if not expected_prev and signoffs:
        return False, "sign-offs present but report has no body_sha256"
    for index, record in enumerate(signoffs):
        if not isinstance(record, dict):
            return False, f"sign-off #{index + 1} is not a mapping"
        for field in ("reviewer", "decision", "timestamp_utc", "prev_hash", "record_hash"):
            if field not in record:
                return False, f"sign-off #{index + 1} is missing {field!r}"
        if record["decision"] not in ("approve", "reject"):
            return False, f"sign-off #{index + 1} has invalid decision"
        content = {key: value for key, value in record.items() if key != "record_hash"}
        if record["record_hash"] != canonical_hash(content):
            return (
                False,
                (f"sign-off #{index + 1} was modified after signing "
                "(record hash mismatch)"),
            )
        if record["prev_hash"] != expected_prev:
            return (
                False,
                (f"sign-off #{index + 1} chain broken: a previous record or the "
                "report body was modified after signing"),
            )
        expected_prev = canonical_hash(record)
    return True, ""

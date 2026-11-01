"""Immutable audit ledger (§7.1, §6.5).

Every L1+ autonomous decision is cryptographically signed and appended to the
ledger.  Entries are append-only: nothing is ever removed or altered.
The user has read access to the full ledger at all times (§6.5).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:  # POSIX advisory locking; absent on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


@dataclass
class AuditEntry:
    """A single, immutable audit ledger entry."""

    entry_id: str
    timestamp: datetime
    event_type: str
    actor: str
    payload: dict[str, Any]
    signature: str          # SHA-256 hash of the serialised payload
    previous_hash: str      # Links to the prior entry (blockchain-style chain)


def _compute_signature(entry_id: str, timestamp: str, event_type: str,
                        actor: str, payload: dict, previous_hash: str) -> str:
    data = json.dumps(
        {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": actor,
            "payload": payload,
            "previous_hash": previous_hash,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(data).hexdigest()


class AuditLedger:
    """Append-only, cryptographically chained audit ledger.

    Thread-safe.  Verifying the chain with ``verify_chain()`` detects any
    tampering or insertion.

    Parameters
    ----------
    path:
        Optional JSON-lines file.  When given, existing entries are loaded on
        construction and every new entry is appended (and flushed) to it, so
        the ledger survives process restarts.  Tampering with the file is
        detected by ``verify_chain()``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: list[AuditEntry] = []
        self._genesis_hash = "0" * 64  # sentinel for the first entry
        self._path = Path(path) if path else None
        self._offset = 0
        if self._path is not None:
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(line: str) -> AuditEntry:
        raw = json.loads(line)
        return AuditEntry(
            entry_id=raw["entry_id"],
            timestamp=datetime.fromisoformat(raw["timestamp"]),
            event_type=raw["event_type"],
            actor=raw["actor"],
            payload=raw["payload"],
            signature=raw["signature"],
            previous_hash=raw["previous_hash"],
        )

    def _load(self) -> None:
        self._offset = 0
        self._sync_from_disk()

    def _sync_from_disk(self) -> None:
        """Ingest entries other processes appended since we last looked."""
        assert self._path is not None
        if not self._path.exists():
            return
        with self._path.open("rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        # Only consume complete lines; a trailing partial line is left for later.
        end = data.rfind(b"\n") + 1
        for line in data[:end].decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                self._entries.append(self._parse(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Corrupt audit ledger {self._path}: {exc}") from exc
        self._offset += end

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Exclusive cross-process lock around read-tail + append."""
        if self._path is None or fcntl is None:
            yield
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path.with_name(self._path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _persist(self, entry: AuditEntry) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "entry_id": entry.entry_id,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "actor": entry.actor,
            "payload": entry.payload,
            "signature": entry.signature,
            "previous_hash": entry.previous_hash,
        }
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        # Owner-only: the ledger can contain personal activity.
        fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "ab") as fh:
            fh.write(line)
            fh.flush()
        self._offset += len(line)

    # ------------------------------------------------------------------
    # Appending
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        *,
        entry_id: str | None = None,
    ) -> AuditEntry:
        """Append a new entry to the ledger."""
        import uuid

        with self._lock, self._file_lock():
            if self._path is not None:
                self._sync_from_disk()
            eid = entry_id or str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc)
            prev_hash = (
                self._entries[-1].signature if self._entries else self._genesis_hash
            )
            sig = _compute_signature(
                eid, timestamp.isoformat(), event_type, actor, payload, prev_hash
            )
            entry = AuditEntry(
                entry_id=eid,
                timestamp=timestamp,
                event_type=event_type,
                actor=actor,
                payload=payload,
                signature=sig,
                previous_hash=prev_hash,
            )
            self._entries.append(entry)
            self._persist(entry)
        return entry

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> list[AuditEntry]:
        """Return all ledger entries (read-only copy)."""
        with self._lock:
            return list(self._entries)

    def read_by_type(self, event_type: str) -> list[AuditEntry]:
        with self._lock:
            return [e for e in self._entries if e.event_type == event_type]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the cryptographic chain of all entries.

        Returns
        -------
        (is_valid, message)
        """
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return True, "Ledger is empty."

        prev_hash = self._genesis_hash
        for i, entry in enumerate(entries):
            expected_sig = _compute_signature(
                entry.entry_id,
                entry.timestamp.isoformat(),
                entry.event_type,
                entry.actor,
                entry.payload,
                prev_hash,
            )
            if entry.signature != expected_sig:
                return False, f"Chain broken at entry {i} (id={entry.entry_id})."
            if entry.previous_hash != prev_hash:
                return False, f"Previous-hash mismatch at entry {i}."
            prev_hash = entry.signature

        return True, f"Chain valid. {len(entries)} entries verified."

"""Immutable audit ledger (§7.1, §6.5).

Every L1+ autonomous decision is cryptographically signed and appended to the
ledger.  Entries are append-only: nothing is ever removed or altered.
The user has read access to the full ledger at all times (§6.5).
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


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
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[AuditEntry] = []
        self._genesis_hash = "0" * 64  # sentinel for the first entry

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

        with self._lock:
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

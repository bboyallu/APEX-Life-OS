"""Built-in memory provider — the zero-dependency default.

Persists memories to two human-readable markdown files:

* ``MEMORY.md`` — repository-scoped facts.
* ``USER.md``   — user-scoped preferences.

Each entry is stored as a single markdown bullet carrying a stable id so it
can be recalled, searched, and forgotten across process restarts.  The files
are the source of truth: they can be edited by hand and re-read on the next
operation.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from apex.memory.provider import MemoryEntry, MemoryProvider, MemoryScope

_HEADERS = {
    MemoryScope.REPOSITORY: "# MEMORY\n\nRepository-scoped facts stored by APEX.\n",
    MemoryScope.USER: "# USER\n\nUser-scoped preferences stored by APEX.\n",
}

_ENTRY_RE = re.compile(
    r"^- \[(?P<id>[0-9a-f-]{36})\] \((?P<created>[^)]+)\) "
    r"\*\*(?P<subject>.*?)\*\*: (?P<fact>.*?)(?: — _(?P<citation>.*)_)?$"
)


class BuiltinMemoryProvider(MemoryProvider):
    """File-backed provider writing ``MEMORY.md`` and ``USER.md``.

    Parameters
    ----------
    root:
        Directory in which the markdown files live.  Defaults to the
        current working directory.
    """

    name = "builtin"

    def __init__(self, root: str | Path = ".") -> None:
        self._root = Path(root)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    def remember(self, entry: MemoryEntry) -> MemoryEntry:
        with self._lock:
            path = self._path_for(entry.scope)
            if not path.exists():
                path.write_text(_HEADERS[entry.scope], encoding="utf-8")
            with path.open("a", encoding="utf-8") as fh:
                fh.write(self._format_entry(entry))
        return entry

    def recall(self, scope: MemoryScope | None = None) -> list[MemoryEntry]:
        with self._lock:
            scopes = [scope] if scope else list(MemoryScope)
            entries: list[MemoryEntry] = []
            for s in scopes:
                entries.extend(self._read_entries(s))
            return entries

    def search(self, query: str, scope: MemoryScope | None = None) -> list[MemoryEntry]:
        q = query.lower()
        return [
            e
            for e in self.recall(scope)
            if q in e.subject.lower() or q in e.fact.lower()
        ]

    def forget(self, entry_id: str) -> bool:
        with self._lock:
            for s in MemoryScope:
                path = self._path_for(s)
                if not path.exists():
                    continue
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                kept = [
                    line
                    for line in lines
                    if not (
                        (m := _ENTRY_RE.match(line.rstrip("\n")))
                        and m.group("id") == entry_id
                    )
                ]
                if len(kept) != len(lines):
                    path.write_text("".join(kept), encoding="utf-8")
                    return True
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path_for(self, scope: MemoryScope) -> Path:
        return self._root / ("USER.md" if scope == MemoryScope.USER else "MEMORY.md")

    @staticmethod
    def _format_entry(entry: MemoryEntry) -> str:
        line = (
            f"- [{entry.entry_id}] ({entry.created_at.isoformat()}) "
            f"**{entry.subject}**: {entry.fact}"
        )
        if entry.citation:
            line += f" — _{entry.citation}_"
        return line + "\n"

    def _read_entries(self, scope: MemoryScope) -> list[MemoryEntry]:
        path = self._path_for(scope)
        if not path.exists():
            return []
        entries: list[MemoryEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _ENTRY_RE.match(line)
            if not m:
                continue
            try:
                created = datetime.fromisoformat(m.group("created"))
            except ValueError:
                created = datetime.now(timezone.utc)
            entries.append(
                MemoryEntry(
                    entry_id=m.group("id"),
                    created_at=created,
                    scope=scope,
                    subject=m.group("subject"),
                    fact=m.group("fact"),
                    citation=m.group("citation") or "",
                )
            )
        return entries

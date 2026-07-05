"""Session persistence — SQLite store at ``~/.apex/state.db``.

Conversations from every interface (terminal chat, Telegram gateway,
dashboard) share the same store, so a conversation started on your phone
continues in the terminal. Full-text search over past messages powers
cross-session recall.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from apex.agent.config import apex_home

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL DEFAULT 'terminal',
    created_at TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(content, session_id UNINDEXED, role UNINDEXED);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StoredMessage(BaseModel):
    session_id: str
    role: str
    content: str
    created_at: str


class SessionInfo(BaseModel):
    session_id: str
    channel: str
    created_at: str
    title: str
    message_count: int = 0


class SessionStore:
    """SQLite-backed conversation store shared by all interfaces."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else apex_home() / "state.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        try:
            self._conn.executescript(_FTS_SCHEMA)
            self._fts = True
        except sqlite3.OperationalError:  # SQLite built without FTS5
            self._fts = False
        self._conn.commit()

    def create_session(self, *, channel: str = "terminal", title: str = "") -> str:
        session_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO sessions (session_id, channel, created_at, title) "
            "VALUES (?, ?, ?, ?)",
            (session_id, channel, _now(), title),
        )
        self._conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        if self._fts:
            self._conn.execute(
                "INSERT INTO messages_fts (content, session_id, role) VALUES (?, ?, ?)",
                (content, session_id, role),
            )
        self._conn.commit()

    def messages(self, session_id: str) -> list[StoredMessage]:
        rows = self._conn.execute(
            "SELECT session_id, role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY message_id",
            (session_id,),
        ).fetchall()
        return [
            StoredMessage(
                session_id=r[0], role=r[1], content=r[2], created_at=r[3]
            )
            for r in rows
        ]

    def sessions(self, *, limit: int = 50) -> list[SessionInfo]:
        rows = self._conn.execute(
            "SELECT s.session_id, s.channel, s.created_at, s.title, "
            "       COUNT(m.message_id) "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.session_id "
            "GROUP BY s.session_id ORDER BY s.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            SessionInfo(
                session_id=r[0],
                channel=r[1],
                created_at=r[2],
                title=r[3],
                message_count=r[4],
            )
            for r in rows
        ]

    def latest_session(self, *, channel: str | None = None) -> str | None:
        """Return the most recent session id (optionally for one channel)."""
        if channel:
            row = self._conn.execute(
                "SELECT session_id FROM sessions WHERE channel = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (channel,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def search(self, query: str, *, limit: int = 20) -> list[StoredMessage]:
        """Full-text search over all stored messages (cross-session recall)."""
        if self._fts:
            escaped = '"' + query.replace('"', '""') + '"'
            rows = self._conn.execute(
                "SELECT session_id, role, content, '' FROM messages_fts "
                "WHERE messages_fts MATCH ? LIMIT ?",
                (escaped, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT session_id, role, content, created_at FROM messages "
                "WHERE content LIKE ? ORDER BY message_id DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [
            StoredMessage(
                session_id=r[0], role=r[1], content=r[2], created_at=r[3]
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

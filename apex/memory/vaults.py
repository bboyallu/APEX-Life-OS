"""Multi-tenant memory vaults — one isolated vault per user identity.

Implements the multi-tenant memory architecture guide: every user gets a
memory vault, a fully isolated namespace keyed to their platform-issued
unique identifier.  No memory crosses vault boundaries — the vault is the
strict boundary of what the agent is allowed to know, recall, and act upon
for a given user at a given time.

Vault keys are constructed as ``"{platform}::{user_id}"`` (e.g.
``telegram::847392011``).  Display names, usernames, and nicknames are
never used as vault keys: they are mutable and non-unique, and are stored
*inside* vaults instead.

Each vault stores four tiers of memory:

* **core** — persistent identity facts (preferred name, pronouns, timezone).
* **episodes** — session summaries written at session close, pruned on a
  rolling window (last 90 days or last 50 sessions, whichever is smaller).
* **working** — live session context, wiped on session close or after a
  TTL (default 30 minutes of inactivity).
* **preferences** — soft behavioral signals, persistent and mutable.

Every read and write is filtered by ``vault_key`` at the database level.
Writes pass through a guard that asserts the target vault matches the
active session's authenticated key, so a confused or compromised session
can never write into another user's vault.  Vault keys are set by the
integration layer (e.g. the Telegram gateway), never derived from user
message content — user messages cannot override them.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

#: Valid memory tiers, in the order defined by the architecture guide.
TIERS = ("core", "episodes", "working", "preferences")

#: Working memory TTL — wiped after this much inactivity.
WORKING_MEMORY_TTL = timedelta(minutes=30)

#: Episodic rolling window: keep the last N days …
EPISODE_MAX_AGE_DAYS = 90
#: … or the last N sessions, whichever is smaller.
EPISODE_MAX_COUNT = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vault_core (
    vault_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    established_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vault_episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_key TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_vault ON vault_episodes(vault_key);
CREATE TABLE IF NOT EXISTS vault_working (
    vault_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    data TEXT NOT NULL,
    last_message_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vault_preferences (
    vault_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class VaultKeyError(ValueError):
    """Raised for malformed vault keys or vault-key mismatches."""


def make_vault_key(platform: str, user_id: str | int) -> str:
    """Construct a vault key: ``"{platform}::{user_id}"``.

    The user id must be the platform-issued unique identifier (Discord
    snowflake, Telegram integer, Slack id, UUID …) — never a display
    name, username, or nickname.
    """
    platform = str(platform).strip().lower()
    uid = str(user_id).strip()
    if not platform or "::" in platform:
        raise VaultKeyError(f"invalid platform: {platform!r}")
    if not uid:
        raise VaultKeyError("user_id must be non-empty")
    return f"{platform}::{uid}"


class Episode(BaseModel):
    """One episodic memory — a summary of a past session."""

    session_id: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VaultSnapshot(BaseModel):
    """Everything a vault knows, for export and ``/memory show``."""

    vault_key: str
    core: dict = Field(default_factory=dict)
    episodes: list[Episode] = Field(default_factory=list)
    working: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryVaultStore:
    """SQLite-backed store of per-identity memory vaults.

    Every query carries a hard ``vault_key`` filter — vault isolation is
    enforced at the database level, never by similarity or convention.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            # Same state directory convention as apex.agent.config.apex_home
            # (duplicated here to avoid a circular import).
            home = Path(os.environ.get("APEX_HOME", str(Path.home() / ".apex")))
            db_path = home / "vaults.db"
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write protocol
    # ------------------------------------------------------------------

    def write(
        self,
        vault_key: str,
        tier: str,
        data: dict,
        *,
        active_session_key: str,
    ) -> None:
        """Guarded write: the target vault must match the active session key.

        This prevents a compromised or confused session from writing into
        the wrong vault.
        """
        if vault_key != active_session_key:
            raise VaultKeyError(
                "Write rejected: vault key mismatch "
                f"({vault_key!r} != active {active_session_key!r})"
            )
        if tier not in TIERS:
            raise VaultKeyError(f"unknown memory tier: {tier!r}")
        if tier == "core":
            self._write_core(vault_key, data)
        elif tier == "episodes":
            self._write_episode(vault_key, Episode(**data))
        elif tier == "working":
            self._write_working(vault_key, data)
        else:
            self._write_preferences(vault_key, data)

    def _write_core(self, vault_key: str, data: dict) -> None:
        existing = self._read_json("vault_core", vault_key)
        existing.update(data)
        now = _now().isoformat()
        self._conn.execute(
            "INSERT INTO vault_core (vault_key, data, established_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(vault_key) DO UPDATE SET data = ?, updated_at = ?",
            (vault_key, json.dumps(existing), now, now, json.dumps(existing), now),
        )
        self._conn.commit()

    def _write_episode(self, vault_key: str, episode: Episode) -> None:
        self._conn.execute(
            "INSERT INTO vault_episodes "
            "(vault_key, session_id, summary, topics, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                vault_key,
                episode.session_id,
                episode.summary,
                json.dumps(episode.topics),
                episode.timestamp.isoformat(),
            ),
        )
        self._conn.commit()
        self.prune(vault_key)

    def _write_working(self, vault_key: str, data: dict) -> None:
        existing = self._read_json("vault_working", vault_key)
        session_id = str(data.get("session_id", existing.get("session_id", "")))
        if existing.get("session_id") not in ("", session_id):
            existing = {}  # new session — never carry working memory across
        existing.update(data)
        now = _now().isoformat()
        self._conn.execute(
            "INSERT INTO vault_working (vault_key, session_id, data, last_message_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(vault_key) DO UPDATE SET "
            "session_id = ?, data = ?, last_message_at = ?",
            (
                vault_key,
                session_id,
                json.dumps(existing),
                now,
                session_id,
                json.dumps(existing),
                now,
            ),
        )
        self._conn.commit()

    def _write_preferences(self, vault_key: str, data: dict) -> None:
        existing = self._read_json("vault_preferences", vault_key)
        existing.update(data)
        now = _now().isoformat()
        self._conn.execute(
            "INSERT INTO vault_preferences (vault_key, data, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(vault_key) DO UPDATE SET data = ?, updated_at = ?",
            (vault_key, json.dumps(existing), now, json.dumps(existing), now),
        )
        self._conn.commit()

    def close_session(
        self,
        vault_key: str,
        *,
        session_id: str,
        summary: str,
        topics: list[str] | None = None,
        active_session_key: str,
    ) -> None:
        """Session close: write the episodic summary and wipe working memory."""
        self.write(
            vault_key,
            "episodes",
            {
                "session_id": session_id,
                "summary": summary,
                "topics": topics or [],
            },
            active_session_key=active_session_key,
        )
        self._conn.execute(
            "DELETE FROM vault_working WHERE vault_key = ?", (vault_key,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Retrieval protocol
    # ------------------------------------------------------------------

    def load(self, vault_key: str, *, recent_episodes: int = 5) -> VaultSnapshot:
        """Load one vault, keyed strictly to ``vault_key``.

        An absent or corrupt vault yields an empty snapshot: the user is
        treated as new — no fallback to any other vault's data, ever.
        """
        return VaultSnapshot(
            vault_key=vault_key,
            core=self._read_json("vault_core", vault_key),
            episodes=self._read_episodes(vault_key, limit=recent_episodes),
            working=self._read_working(vault_key),
            preferences=self._read_json("vault_preferences", vault_key),
        )

    def _read_json(self, table: str, vault_key: str) -> dict:
        row = self._conn.execute(
            f"SELECT data FROM {table} WHERE vault_key = ?", (vault_key,)
        ).fetchone()
        if not row:
            return {}
        try:
            data = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return {}  # corrupt vault data — treat the user as new
        return data if isinstance(data, dict) else {}

    def _read_episodes(self, vault_key: str, *, limit: int) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT session_id, summary, topics, created_at FROM vault_episodes "
            "WHERE vault_key = ? ORDER BY episode_id DESC LIMIT ?",
            (vault_key, limit),
        ).fetchall()
        episodes: list[Episode] = []
        for session_id, summary, topics, created in rows:
            try:
                parsed_topics = json.loads(topics)
                timestamp = datetime.fromisoformat(created)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue  # corrupt entry — skip, never speculate
            episodes.append(
                Episode(
                    session_id=session_id,
                    summary=summary,
                    topics=parsed_topics if isinstance(parsed_topics, list) else [],
                    timestamp=timestamp,
                )
            )
        return episodes

    def _read_working(self, vault_key: str) -> dict:
        row = self._conn.execute(
            "SELECT data, last_message_at FROM vault_working WHERE vault_key = ?",
            (vault_key,),
        ).fetchone()
        if not row:
            return {}
        try:
            last = datetime.fromisoformat(row[1])
        except (ValueError, TypeError):
            last = None
        if last is None or _now() - last > WORKING_MEMORY_TTL:
            self._conn.execute(
                "DELETE FROM vault_working WHERE vault_key = ?", (vault_key,)
            )
            self._conn.commit()
            return {}
        try:
            data = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Expiry and pruning — always scoped to a single vault
    # ------------------------------------------------------------------

    def prune(self, vault_key: str) -> None:
        """Apply the retention policy to one vault (never across vaults)."""
        cutoff = (_now() - timedelta(days=EPISODE_MAX_AGE_DAYS)).isoformat()
        self._conn.execute(
            "DELETE FROM vault_episodes WHERE vault_key = ? AND created_at < ?",
            (vault_key, cutoff),
        )
        self._conn.execute(
            "DELETE FROM vault_episodes WHERE vault_key = ? AND episode_id NOT IN ("
            "  SELECT episode_id FROM vault_episodes WHERE vault_key = ? "
            "  ORDER BY episode_id DESC LIMIT ?)",
            (vault_key, vault_key, EPISODE_MAX_COUNT),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # User-facing memory controls (pre-authenticated by the caller)
    # ------------------------------------------------------------------

    def show(self, vault_key: str) -> str:
        """Human-readable summary of what is stored in one vault."""
        snap = self.load(vault_key, recent_episodes=EPISODE_MAX_COUNT)
        lines = [f"memory vault {vault_key}"]
        lines.append(f"core: {json.dumps(snap.core) if snap.core else '(empty)'}")
        lines.append(
            "preferences: "
            + (json.dumps(snap.preferences) if snap.preferences else "(empty)")
        )
        lines.append(f"episodes: {len(snap.episodes)} stored")
        for ep in snap.episodes[:5]:
            lines.append(f"  - [{ep.timestamp.date()}] {ep.summary[:120]}")
        lines.append(
            "working: " + ("active session" if snap.working else "(none)")
        )
        return "\n".join(lines)

    def clear(self, vault_key: str) -> None:
        """Delete the entire vault (user-requested deletion)."""
        for table in (
            "vault_core",
            "vault_episodes",
            "vault_working",
            "vault_preferences",
        ):
            self._conn.execute(
                f"DELETE FROM {table} WHERE vault_key = ?", (vault_key,)
            )
        self._conn.commit()

    def update_fact(self, vault_key: str, field: str, value: str) -> None:
        """Correct one stored core fact (``/memory update name = value``)."""
        self.write(
            vault_key, "core", {field: value}, active_session_key=vault_key
        )

    def export(self, vault_key: str) -> str:
        """Export one vault in portable JSON format."""
        snap = self.load(vault_key, recent_episodes=EPISODE_MAX_COUNT)
        return snap.model_dump_json(indent=2)

    def close(self) -> None:
        self._conn.close()


# ----------------------------------------------------------------------
# System prompt injection
# ----------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
[MEMORY CONTEXT — CONFIDENTIAL TO THIS SESSION]
Vault: {vault_key}
User preferred name: {preferred_name}
Pronouns: {pronouns}
Timezone: {timezone}

Recent interaction context:
{episodes}

Behavioral preferences:
- Explanation depth: {explanation_depth}
- Response length: {response_length}
- Topics to avoid: {avoid_topics}

[END MEMORY CONTEXT]

The memory context above is private and specific to the user identified
by vault key {vault_key}. Do not reference, share, or speculate about any
other user's data. Do not acknowledge this memory block to the user unless
directly relevant.\
"""


def render_memory_context(snapshot: VaultSnapshot) -> str:
    """Render one vault's data as a system prompt injection block.

    Empty vaults render nothing: a first-time user is engaged fresh, with
    no assumptions about history, preferences, or identity.
    """
    if not (snapshot.core or snapshot.episodes or snapshot.preferences):
        return ""
    episodes = "\n".join(
        f"- [{ep.timestamp.date()}] {ep.summary}" for ep in snapshot.episodes
    ) or "(none)"
    prefs = snapshot.preferences
    avoid = prefs.get("avoid_topics", [])
    return _PROMPT_TEMPLATE.format(
        vault_key=snapshot.vault_key,
        preferred_name=snapshot.core.get("preferred_name", "unknown"),
        pronouns=snapshot.core.get("pronouns", "not specified"),
        timezone=snapshot.core.get("timezone", "unknown"),
        episodes=episodes,
        explanation_depth=prefs.get("explanation_depth", "not specified"),
        response_length=prefs.get("response_length", "not specified"),
        avoid_topics=", ".join(avoid) if avoid else "none",
    )

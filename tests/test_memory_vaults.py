"""Tests for multi-tenant memory vaults — one isolated vault per identity."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from apex.memory.vaults import (
    EPISODE_MAX_COUNT,
    MemoryVaultStore,
    VaultKeyError,
    make_vault_key,
    render_memory_context,
)


@pytest.fixture()
def store(tmp_path):
    return MemoryVaultStore(tmp_path / "vaults.db")


ZARA = make_vault_key("telegram", 847392011)
OTHER = make_vault_key("discord", "198765432198765432")


# ----------------------------------------------------------------------
# Vault key construction
# ----------------------------------------------------------------------


def test_vault_key_format():
    assert make_vault_key("Telegram", 847392011) == "telegram::847392011"
    assert (
        make_vault_key("app", "f47ac10b-58cc-4372-a567-0e02b2c3d479")
        == "app::f47ac10b-58cc-4372-a567-0e02b2c3d479"
    )


def test_vault_key_rejects_bad_input():
    with pytest.raises(VaultKeyError):
        make_vault_key("", 1)
    with pytest.raises(VaultKeyError):
        make_vault_key("a::b", 1)
    with pytest.raises(VaultKeyError):
        make_vault_key("telegram", "")


# ----------------------------------------------------------------------
# Write protocol and write guard
# ----------------------------------------------------------------------


def test_write_guard_rejects_vault_key_mismatch(store):
    with pytest.raises(VaultKeyError, match="mismatch"):
        store.write(
            ZARA, "core", {"preferred_name": "Zara"}, active_session_key=OTHER
        )
    assert store.load(ZARA).core == {}


def test_write_guard_rejects_unknown_tier(store):
    with pytest.raises(VaultKeyError, match="tier"):
        store.write(ZARA, "secrets", {}, active_session_key=ZARA)


def test_core_writes_merge_in_place(store):
    store.write(ZARA, "core", {"preferred_name": "Zara"}, active_session_key=ZARA)
    store.write(ZARA, "core", {"pronouns": "she/her"}, active_session_key=ZARA)
    core = store.load(ZARA).core
    assert core["preferred_name"] == "Zara"
    assert core["pronouns"] == "she/her"


# ----------------------------------------------------------------------
# Vault isolation — no memory crosses vault boundaries
# ----------------------------------------------------------------------


def test_vaults_are_isolated(store):
    store.write(ZARA, "core", {"preferred_name": "Zara"}, active_session_key=ZARA)
    store.write(
        ZARA,
        "episodes",
        {"session_id": "s1", "summary": "debugged python async"},
        active_session_key=ZARA,
    )
    store.write(
        ZARA, "preferences", {"response_length": "concise"}, active_session_key=ZARA
    )
    other = store.load(OTHER)
    assert other.core == {}
    assert other.episodes == []
    assert other.preferences == {}


def test_empty_vault_yields_fresh_start(store):
    snap = store.load("telegram::999")
    assert snap.core == {} and snap.episodes == [] and snap.working == {}
    assert render_memory_context(snap) == ""


def test_corrupt_vault_treated_as_new(store):
    store._conn.execute(
        "INSERT INTO vault_core (vault_key, data, established_at, updated_at) "
        "VALUES (?, 'not json', '', '')",
        (ZARA,),
    )
    store._conn.commit()
    assert store.load(ZARA).core == {}


# ----------------------------------------------------------------------
# Working memory TTL and session close
# ----------------------------------------------------------------------


def test_working_memory_expires_after_ttl(store):
    store.write(
        ZARA,
        "working",
        {"session_id": "s1", "active_topic": "music theory"},
        active_session_key=ZARA,
    )
    assert store.load(ZARA).working["active_topic"] == "music theory"
    stale = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
    store._conn.execute(
        "UPDATE vault_working SET last_message_at = ? WHERE vault_key = ?",
        (stale, ZARA),
    )
    store._conn.commit()
    assert store.load(ZARA).working == {}


def test_working_memory_not_carried_across_sessions(store):
    store.write(
        ZARA,
        "working",
        {"session_id": "s1", "active_topic": "old topic"},
        active_session_key=ZARA,
    )
    store.write(
        ZARA,
        "working",
        {"session_id": "s2", "active_topic": "new topic"},
        active_session_key=ZARA,
    )
    working = store.load(ZARA).working
    assert working["session_id"] == "s2"
    assert working["active_topic"] == "new topic"


def test_close_session_writes_episode_and_wipes_working(store):
    store.write(
        ZARA, "working", {"session_id": "s1"}, active_session_key=ZARA
    )
    store.close_session(
        ZARA,
        session_id="s1",
        summary="helped debug an async race condition",
        topics=["python", "async"],
        active_session_key=ZARA,
    )
    snap = store.load(ZARA)
    assert snap.working == {}
    assert snap.episodes[0].summary == "helped debug an async race condition"
    assert snap.episodes[0].topics == ["python", "async"]


# ----------------------------------------------------------------------
# Pruning — scoped to a single vault
# ----------------------------------------------------------------------


def test_episode_pruning_keeps_rolling_window(store):
    for i in range(EPISODE_MAX_COUNT + 10):
        store.write(
            ZARA,
            "episodes",
            {"session_id": f"s{i}", "summary": f"session {i}"},
            active_session_key=ZARA,
        )
    episodes = store.load(ZARA, recent_episodes=1000).episodes
    assert len(episodes) == EPISODE_MAX_COUNT
    assert episodes[0].summary == f"session {EPISODE_MAX_COUNT + 9}"


def test_pruning_never_touches_other_vaults(store):
    store.write(
        OTHER,
        "episodes",
        {"session_id": "o1", "summary": "other user session"},
        active_session_key=OTHER,
    )
    for i in range(EPISODE_MAX_COUNT + 5):
        store.write(
            ZARA,
            "episodes",
            {"session_id": f"s{i}", "summary": f"session {i}"},
            active_session_key=ZARA,
        )
    assert len(store.load(OTHER).episodes) == 1


# ----------------------------------------------------------------------
# System prompt injection
# ----------------------------------------------------------------------


def test_render_memory_context_includes_vault_data(store):
    store.write(
        ZARA,
        "core",
        {"preferred_name": "Zara", "pronouns": "she/her", "timezone": "America/Chicago"},
        active_session_key=ZARA,
    )
    store.write(
        ZARA,
        "preferences",
        {
            "explanation_depth": "technical",
            "response_length": "concise",
            "avoid_topics": ["diet advice"],
        },
        active_session_key=ZARA,
    )
    store.write(
        ZARA,
        "episodes",
        {"session_id": "s1", "summary": "debugged async race"},
        active_session_key=ZARA,
    )
    block = render_memory_context(store.load(ZARA))
    assert "[MEMORY CONTEXT — CONFIDENTIAL TO THIS SESSION]" in block
    assert f"Vault: {ZARA}" in block
    assert "Zara" in block and "she/her" in block
    assert "debugged async race" in block
    assert "diet advice" in block
    assert "Do not reference, share, or speculate about any" in block


# ----------------------------------------------------------------------
# User-facing memory controls
# ----------------------------------------------------------------------


def test_memory_show_clear_update_export(store):
    store.write(ZARA, "core", {"preferred_name": "Zoe"}, active_session_key=ZARA)
    store.update_fact(ZARA, "preferred_name", "Zara")
    assert "Zara" in store.show(ZARA)
    exported = json.loads(store.export(ZARA))
    assert exported["vault_key"] == ZARA
    assert exported["core"]["preferred_name"] == "Zara"
    store.clear(ZARA)
    assert store.load(ZARA).core == {}

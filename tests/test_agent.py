"""Tests for the conversational agent layer (apex.agent)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from apex.agent.config import (
    AgentConfig,
    autodetect_local_provider,
    detect_ollama_models,
    load_config,
    save_config,
)
from apex.agent.llm import ChatMessage, LLMClient
from apex.agent.loop import AgentLoop
from apex.agent.neural import llm_neural_model
from apex.agent.scheduler import ScheduledTask, Scheduler, cron_matches
from apex.agent.sessions import SessionStore
from apex.agent.skills import Skill, SkillStore
from apex.agent.tools import build_default_tools
from apex.system import ApexSystem


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def make_transport(responses):
    """Return a fake transport yielding queued chat-completion responses."""
    calls = []

    def transport(url, headers, payload):
        calls.append({"url": url, "headers": headers, "payload": payload})
        return responses.pop(0)

    transport.calls = calls
    return transport


def text_response(content):
    return {"choices": [{"message": {"content": content}}]}


def tool_response(name, arguments, call_id="call_1"):
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


@pytest.fixture()
def agent(tmp_path):
    """Factory that builds an AgentLoop with queued fake LLM responses."""

    def build(responses, approval_callback=None):
        config = AgentConfig()
        system = ApexSystem(knowledge_root=tmp_path)
        client = LLMClient(config, transport=make_transport(list(responses)))
        sessions = SessionStore(tmp_path / "state.db")
        skills = SkillStore(tmp_path / "skills", audit_ledger=system.audit_ledger)
        tools = build_default_tools(
            system,
            approval_callback=approval_callback,
            skill_store=skills,
            session_store=sessions,
        )
        loop = AgentLoop(
            system=system,
            config=config,
            client=client,
            tools=tools,
            session_store=sessions,
            knowledge_root=tmp_path,
            skill_store=skills,
        )
        return loop, system, sessions, skills

    return build


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    config = AgentConfig()
    config.use_provider("groq")
    config.voice.enabled = True
    save_config(config, tmp_path)
    loaded = load_config(tmp_path)
    assert loaded.provider == "groq"
    assert loaded.voice.enabled is True


def test_config_never_stores_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_API_KEY", "supersecret")
    path = save_config(AgentConfig(), tmp_path)
    assert "supersecret" not in path.read_text()


def test_config_unknown_provider():
    with pytest.raises(ValueError):
        AgentConfig().use_provider("nonsense")


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("APEX_MODEL", "custom-model")
    monkeypatch.setenv("APEX_BASE_URL", "http://localhost:9999/v1")
    config = AgentConfig()
    assert config.resolved_model() == "custom-model"
    assert config.resolved_base_url() == "http://localhost:9999/v1"


def test_detect_ollama_models_online():
    payload = json.dumps(
        {"models": [{"name": "llama3.2:latest"}, {"name": "mistral:7b"}]}
    )
    seen: list[str] = []

    def fake_fetch(url, timeout):
        seen.append(url)
        return payload

    models = detect_ollama_models(fetch=fake_fetch)
    assert models == ["llama3.2:latest", "mistral:7b"]
    assert seen == ["http://localhost:11434/api/tags"]


def test_detect_ollama_models_offline():
    def fake_fetch(url, timeout):
        raise OSError("connection refused")

    assert detect_ollama_models(fetch=fake_fetch) == []


def test_autodetect_switches_to_ollama(monkeypatch):
    monkeypatch.delenv("APEX_API_KEY", raising=False)
    monkeypatch.setattr(
        "apex.agent.config.detect_ollama_models",
        lambda *a, **k: ["mistral:7b", "llama3.2:latest"],
    )
    config = AgentConfig()
    detected = autodetect_local_provider(config)
    assert detected == "llama3.2:latest"  # prefers the preset model family
    assert config.provider == "ollama"
    assert config.model == "llama3.2:latest"


def test_autodetect_skipped_with_api_key(monkeypatch):
    monkeypatch.setenv("APEX_API_KEY", "sk-test")
    monkeypatch.setattr(
        "apex.agent.config.detect_ollama_models",
        lambda *a, **k: ["llama3.2:latest"],
    )
    config = AgentConfig()
    assert autodetect_local_provider(config) is None
    assert config.provider == "openai"


def test_autodetect_noop_when_offline(monkeypatch):
    monkeypatch.delenv("APEX_API_KEY", raising=False)
    monkeypatch.setattr(
        "apex.agent.config.detect_ollama_models", lambda *a, **k: []
    )
    config = AgentConfig()
    assert autodetect_local_provider(config) is None
    assert config.provider == "openai"


# ----------------------------------------------------------------------
# LLM client
# ----------------------------------------------------------------------


def test_llm_client_parses_text():
    client = LLMClient(
        AgentConfig(), transport=make_transport([text_response("hello")])
    )
    response = client.chat([ChatMessage(role="user", content="hi")])
    assert response.content == "hello"
    assert response.tool_calls == []


def test_llm_client_parses_tool_calls():
    client = LLMClient(
        AgentConfig(),
        transport=make_transport([tool_response("remember", {"a": 1})]),
    )
    response = client.chat([ChatMessage(role="user", content="hi")])
    assert response.tool_calls[0].name == "remember"
    assert response.tool_calls[0].arguments == {"a": 1}


def test_llm_client_sends_auth_header(monkeypatch):
    monkeypatch.setenv("APEX_API_KEY", "key123")
    transport = make_transport([text_response("ok")])
    LLMClient(AgentConfig(), transport=transport).chat(
        [ChatMessage(role="user", content="hi")]
    )
    auth = transport.calls[0]["headers"]["Authorization"]
    assert auth == "Bearer " + "key123"


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------


def test_session_store_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "state.db")
    session = store.create_session(channel="terminal")
    store.add_message(session, "user", "deep work matters")
    store.add_message(session, "assistant", "noted")
    assert [m.role for m in store.messages(session)] == ["user", "assistant"]
    assert store.sessions()[0].message_count == 2
    assert store.latest_session() == session


def test_session_search_cross_session(tmp_path):
    store = SessionStore(tmp_path / "state.db")
    first = store.create_session()
    second = store.create_session(channel="telegram:1")
    store.add_message(first, "user", "the quarterly report is due friday")
    store.add_message(second, "user", "unrelated message")
    hits = store.search("quarterly")
    assert len(hits) == 1
    assert hits[0].session_id == first


# ----------------------------------------------------------------------
# Skills
# ----------------------------------------------------------------------


def test_skill_store_roundtrip_and_audit(tmp_path):
    system = ApexSystem(knowledge_root=tmp_path)
    store = SkillStore(tmp_path / "skills", audit_ledger=system.audit_ledger)
    store.save(
        Skill(name="Weekly Review", description="Review the week", steps=["a", "b"])
    )
    loaded = store.get("Weekly Review")
    assert loaded.steps == ["a", "b"]
    used = store.use("weekly review")
    assert used.uses == 1
    assert [s.name for s in store.list()] == ["Weekly Review"]
    assert store.delete("weekly review") is True
    events = [e.event_type for e in system.audit_ledger.read()]
    assert "skill_created" in events
    assert "skill_used" in events
    assert "skill_deleted" in events


# ----------------------------------------------------------------------
# Agent loop + governed tools
# ----------------------------------------------------------------------


def test_agent_plain_reply(agent):
    loop, _, sessions, _ = agent([text_response("hello there")])
    turn = loop.send("hi")
    assert turn.reply == "hello there"
    roles = [m.role for m in sessions.messages(loop.session_id)]
    assert roles == ["user", "assistant"]


def test_agent_tool_round(agent):
    loop, system, _, _ = agent(
        [
            tool_response("remember", {"subject": "prefs", "fact": "likes tea"}),
            text_response("Stored it."),
        ]
    )
    turn = loop.send("remember I like tea")
    assert turn.tool_calls == ["remember"]
    assert turn.reply == "Stored it."
    assert system.search_memories("tea")


def test_high_risk_tool_denied_without_approval(agent):
    loop, system, _, _ = agent(
        [
            tool_response("run_shell", {"command": "rm -rf /"}),
            text_response("Command was denied."),
        ]
    )
    turn = loop.send("wipe the disk")
    assert turn.reply == "Command was denied."
    decisions = system.audit_ledger.read_by_type("tool_approval_decision")
    assert decisions and decisions[-1].payload["approved"] is False


def test_high_risk_tool_runs_with_approval(agent):
    approvals = []

    def approve(name, summary, level):
        approvals.append(name)
        return True

    loop, _, _, _ = agent(
        [
            tool_response("run_shell", {"command": "echo safe"}),
            text_response("done"),
        ],
        approval_callback=approve,
    )
    turn = loop.send("run echo")
    assert approvals == ["run_shell"]
    assert turn.reply == "done"


def test_agent_turn_is_audit_logged(agent):
    loop, system, _, _ = agent([text_response("ok")])
    loop.send("hello")
    assert system.audit_ledger.read_by_type("agent_turn")


def test_agent_auto_learns_skill_from_multistep_turn(agent):
    loop, system, _, skills = agent(
        [
            tool_response("remember", {"subject": "prefs", "fact": "likes tea"}),
            tool_response("search_memories", {"query": "tea"}, call_id="call_2"),
            text_response("Done."),
        ]
    )
    loop.send("track my tea preference")
    saved = skills.list()
    assert len(saved) == 1
    assert saved[0].name == "track my tea preference"
    assert len(saved[0].steps) == 2
    assert saved[0].steps[0].startswith("remember(")
    assert system.audit_ledger.read_by_type("skill_autolearned")


def test_agent_no_auto_skill_for_single_tool_turn(agent):
    loop, system, _, skills = agent(
        [
            tool_response("remember", {"subject": "prefs", "fact": "likes tea"}),
            text_response("Done."),
        ]
    )
    loop.send("remember I like tea")
    assert skills.list() == []
    assert not system.audit_ledger.read_by_type("skill_autolearned")


def test_agent_no_auto_skill_when_model_saved_one(agent):
    loop, system, _, skills = agent(
        [
            tool_response("remember", {"subject": "prefs", "fact": "likes tea"}),
            tool_response(
                "save_skill",
                {
                    "name": "Tea Tracking",
                    "description": "Track tea prefs",
                    "steps": ["remember the fact"],
                },
                call_id="call_2",
            ),
            text_response("Saved."),
        ]
    )
    loop.send("track my tea preference")
    saved = skills.list()
    assert [s.name for s in saved] == ["Tea Tracking"]
    assert not system.audit_ledger.read_by_type("skill_autolearned")


def test_agent_auto_skill_disabled_by_config(agent):
    loop, system, _, skills = agent(
        [
            tool_response("remember", {"subject": "prefs", "fact": "likes tea"}),
            tool_response("search_memories", {"query": "tea"}, call_id="call_2"),
            text_response("Done."),
        ]
    )
    loop.config.auto_skill_min_steps = 0
    loop.send("track my tea preference")
    assert skills.list() == []
    assert not system.audit_ledger.read_by_type("skill_autolearned")


def test_record_insight_feeds_knowledge_raw(agent, tmp_path):
    loop, _, _, _ = agent([])
    path = loop.record_insight("signal: api :: too slow [degraded]")
    assert path.exists()
    assert "too slow" in path.read_text()


def test_search_knowledge_tool(agent, tmp_path):
    loop, system, _, _ = agent([])
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "focus.md").write_text(
        "topic: Focus\n\nDeep work requires uninterrupted concentration blocks."
    )
    system.process_knowledge()
    result = loop.tools.execute("search_knowledge", {"query": "concentration"})
    assert "concentration" in result.lower()


def test_unknown_tool(agent):
    loop, _, _, _ = agent([])
    assert "unknown tool" in loop.tools.execute("nope", {})


# ----------------------------------------------------------------------
# Neural bridge
# ----------------------------------------------------------------------


def test_llm_neural_model_parses_confidence():
    client = LLMClient(
        AgentConfig(),
        transport=make_transport(
            [text_response("Scale the cache.\nconfidence: 0.85")]
        ),
    )
    decision = llm_neural_model(client)("cache misses rising")
    assert decision.confidence == 0.85
    assert "Scale the cache." in decision.content


def test_llm_neural_model_survives_llm_error():
    def failing_transport(url, headers, payload):
        raise OSError("no network")

    client = LLMClient(AgentConfig(), transport=failing_transport)
    with pytest.raises(OSError):
        client.chat([ChatMessage(role="user", content="x")])


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------


def test_cron_matches():
    moment = datetime(2026, 7, 6, 7, 0)  # Monday 07:00
    assert cron_matches("0 7 * * *", moment)
    assert cron_matches("*/15 * * * *", moment)
    assert cron_matches("0 7 * * 1", moment)  # Monday = 1
    assert not cron_matches("0 8 * * *", moment)
    assert not cron_matches("0 7 * * 0", moment)  # Sunday
    with pytest.raises(ValueError):
        cron_matches("* * *", moment)


def test_scheduler_runs_due_tasks(tmp_path):
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps([{"cron": "* * * * *", "action": "process-knowledge"}])
    )
    system = ApexSystem(knowledge_root=tmp_path)
    scheduler = Scheduler(system, schedule_path=schedule)
    executed = scheduler.run_due(datetime(2026, 7, 6, 7, 0))
    assert executed == ["process-knowledge"]
    # Same minute is deduped
    assert scheduler.run_due(datetime(2026, 7, 6, 7, 0)) == []
    assert system.audit_ledger.read_by_type("scheduled_task_run")


def test_scheduled_report(tmp_path):
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps([{"cron": "* * * * *", "action": "report", "arg": "focus"}])
    )
    system = ApexSystem(knowledge_root=tmp_path)
    scheduler = Scheduler(system, schedule_path=schedule)
    assert scheduler.run_due(datetime(2026, 7, 6, 7, 1)) == ["report"]
    assert list((tmp_path / "outputs").glob("*.md"))

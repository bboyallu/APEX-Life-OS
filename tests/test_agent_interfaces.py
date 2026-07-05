"""Tests for the Telegram gateway, dashboard, TUI shell, and voice layer."""

from __future__ import annotations

import json

import pytest

from apex.agent.config import AgentConfig
from apex.agent.dashboard import DashboardState
from apex.agent.gateway import TelegramGateway
from apex.agent.llm import LLMClient
from apex.agent.sessions import SessionStore
from apex.agent.tui import ChatShell
from apex.system import ApexSystem
from apex.voice.stt import SpeechToText
from apex.voice.tts import TextToSpeech


def make_transport(responses):
    def transport(url, headers, payload):
        return responses.pop(0)

    return transport


def text_response(content):
    return {"choices": [{"message": {"content": content}}]}


class FakeTelegramApi:
    """Records outgoing Telegram API calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, method, params):
        self.calls.append((method, params))
        if method == "getUpdates":
            return []
        return {"message_id": 1}

    def sent_texts(self):
        return [p.get("text", "") for m, p in self.calls if m == "sendMessage"]


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_HOME", str(tmp_path / "home"))

    def build(responses, allowed=None):
        api = FakeTelegramApi()
        gw = TelegramGateway(
            api=api,
            knowledge_root=tmp_path,
            config=AgentConfig(),
            system=ApexSystem(knowledge_root=tmp_path),
            session_store=SessionStore(tmp_path / "state.db"),
            client=LLMClient(
                AgentConfig(), transport=make_transport(list(responses))
            ),
            allowed_chat_ids=allowed or set(),
        )
        return gw, api

    return build


# ----------------------------------------------------------------------
# Gateway
# ----------------------------------------------------------------------


def test_gateway_chat_message(gateway):
    gw, api = gateway([text_response("hello from apex")])
    reply = gw.handle_message({"chat": {"id": 42}, "text": "hi"})
    assert reply == "hello from apex"
    assert "hello from apex" in api.sent_texts()


def test_gateway_shares_session_store(gateway, tmp_path):
    gw, _ = gateway([text_response("noted")])
    gw.handle_message({"chat": {"id": 42}, "text": "remember the milk"})
    hits = gw.sessions.search("milk")
    assert hits and hits[0].session_id == gw._loop_for(42).session_id


def test_gateway_commands(gateway):
    gw, _ = gateway([])
    assert "APEX" in gw.handle_message({"chat": {"id": 1}, "text": "/help"})
    assert gw.handle_message({"chat": {"id": 1}, "text": "/new"}).startswith(
        "new session"
    )
    assert "severity=" in gw.handle_message({"chat": {"id": 1}, "text": "/cycle"})
    audit = gw.handle_message({"chat": {"id": 1}, "text": "/audit"})
    assert audit.startswith("✔")
    assert "enabled" in gw.handle_message(
        {"chat": {"id": 1}, "text": "/voice on"}
    )
    assert gw.config.voice.enabled is True


def test_gateway_denies_unknown_chat(gateway):
    gw, api = gateway([], allowed={99})
    reply = gw.handle_message({"chat": {"id": 42}, "text": "hi"})
    assert reply is None
    assert any("locked" in t for t in api.sent_texts())


def test_gateway_callback_approval(gateway):
    gw, api = gateway([])
    import queue

    pending = queue.Queue()
    gw._approvals["appr1"] = pending
    gw._handle_callback({"id": "cb1", "data": "appr1:yes"})
    assert pending.get_nowait() is True
    assert any(m == "answerCallbackQuery" for m, _ in api.calls)


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------


@pytest.fixture()
def dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_HOME", str(tmp_path / "home"))
    return DashboardState(
        knowledge_root=tmp_path,
        config=AgentConfig(),
        system=ApexSystem(knowledge_root=tmp_path),
        session_store=SessionStore(tmp_path / "state.db"),
        client=LLMClient(
            AgentConfig(), transport=make_transport([text_response("dash reply")])
        ),
    )


def test_dashboard_chat_flow(dashboard):
    dashboard.post_chat("hello")
    page = dashboard.page_chat()
    assert "hello" in page
    assert "dash reply" in page


def test_dashboard_pages_render(dashboard):
    dashboard.system.remember("prefs", "dark mode")
    assert "dark mode" in dashboard.page_memories()
    assert "✔" in dashboard.page_audit()
    assert "<ul>" in dashboard.page_sessions()
    assert "<ul>" in dashboard.page_skills()
    assert "<ul>" in dashboard.page_outputs()


def test_dashboard_escapes_html(dashboard):
    dashboard.sessions.add_message(
        dashboard.loop.session_id, "user", "<script>alert(1)</script>"
    )
    page = dashboard.page_chat()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


# ----------------------------------------------------------------------
# TUI shell (slash commands, no LLM needed)
# ----------------------------------------------------------------------


@pytest.fixture()
def shell(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_HOME", str(tmp_path / "home"))
    return ChatShell(
        knowledge_root=tmp_path,
        config=AgentConfig(),
        session_store=SessionStore(tmp_path / "state.db"),
        system=ApexSystem(knowledge_root=tmp_path),
        client=LLMClient(AgentConfig(), transport=make_transport([])),
    )


def test_shell_help_and_model(shell):
    assert "Slash commands" in shell.handle_slash("/help")
    assert "provider=openai" in shell.handle_slash("/model")
    assert "switched to groq" in shell.handle_slash("/model groq")
    assert "Unknown provider" in shell.handle_slash("/model bogus")


def test_shell_session_and_memory_commands(shell):
    assert shell.handle_slash("/new").startswith("new session")
    shell.system.remember("prefs", "tea over coffee")
    assert "tea over coffee" in shell.handle_slash("/memory tea")
    assert "no skills" in shell.handle_slash("/skills")
    assert "severity=" in shell.handle_slash("/cycle")
    assert shell.handle_slash("/audit").startswith("✔")
    assert "recorded to" in shell.handle_slash("/insight learned something")
    assert shell.handle_slash("/quit") is None


def test_shell_voice_toggle(shell):
    assert "enabled" in shell.handle_slash("/voice on")
    assert shell.config.voice.enabled is True
    assert "disabled" in shell.handle_slash("/voice off")


# ----------------------------------------------------------------------
# Voice
# ----------------------------------------------------------------------


def test_stt_transcribes(tmp_path):
    audio = tmp_path / "note.oga"
    audio.write_bytes(b"fake-audio")
    captured = {}

    def transport(url, headers, body):
        captured["url"] = url
        captured["body"] = body
        return {"text": "hello world"}

    config = AgentConfig()
    config.voice.enabled = True
    stt = SpeechToText(config, transport=transport)
    assert stt.transcribe(audio) == "hello world"
    assert captured["url"].endswith("/audio/transcriptions")
    assert b"fake-audio" in captured["body"]


def test_tts_writes_audio(tmp_path):
    def transport(url, headers, body):
        payload = json.loads(body)
        assert payload["voice"] == "alloy"
        return b"mp3-bytes"

    tts = TextToSpeech(AgentConfig(), transport=transport)
    out = tts.synthesize_to_file("hello", tmp_path / "reply.mp3")
    assert out.read_bytes() == b"mp3-bytes"

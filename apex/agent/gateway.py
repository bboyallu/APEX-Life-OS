"""Telegram messaging gateway — talk to APEX from your phone.

Long-polling Telegram Bot API bridge (stdlib only; no public IP or webhook
needed — perfect for a VPS). The gateway shares the same agent loop,
session store, memory, and audit ledger as the terminal chat, so a
conversation started on Telegram continues in the terminal.

L3+ tool approvals arrive as inline approve/veto buttons.
Voice notes are transcribed (and replies spoken back) when voice is enabled.
"""

from __future__ import annotations

import json
import os
import queue
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from apex.agent.config import AgentConfig, load_config
from apex.agent.llm import LLMClient, LLMError
from apex.agent.loop import AgentLoop
from apex.agent.sessions import SessionStore
from apex.agent.skills import SkillStore
from apex.agent.tools import build_default_tools
from apex.core.types import ThresholdLevel
from apex.memory.vaults import (
    MemoryVaultStore,
    make_vault_key,
    render_memory_context,
)
from apex.system import ApexSystem

#: Transport signature: (method, params) -> Telegram API result.
ApiCall = Callable[[str, dict[str, Any]], Any]


class TelegramError(RuntimeError):
    pass


def _urllib_api(token: str) -> ApiCall:
    def call(method: str, params: dict[str, Any]) -> Any:
        url = f"https://api.telegram.org/bot{token}/{method}"
        body = urllib.parse.urlencode(
            {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in params.items()
            }
        ).encode("utf-8")
        request = urllib.request.Request(url, data=body)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise TelegramError(f"Telegram API unreachable: {exc}") from exc
        if not data.get("ok"):
            raise TelegramError(f"Telegram API error: {data.get('description')}")
        return data.get("result")

    return call


class TelegramGateway:
    """Bridges Telegram chats to the shared APEX agent loop."""

    def __init__(
        self,
        *,
        token: str | None = None,
        knowledge_root: str | Path = ".",
        config: AgentConfig | None = None,
        system: ApexSystem | None = None,
        session_store: SessionStore | None = None,
        client: LLMClient | None = None,
        api: ApiCall | None = None,
        allowed_chat_ids: set[int] | None = None,
        vault_store: MemoryVaultStore | None = None,
    ) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if api is None and not self.token:
            raise TelegramError(
                "TELEGRAM_BOT_TOKEN is not set — create a bot with @BotFather "
                "and export the token."
            )
        self.api = api or _urllib_api(self.token)
        self.config = config or load_config()
        self.system = system or ApexSystem(knowledge_root=knowledge_root)
        self.sessions = session_store or SessionStore()
        self.skills = SkillStore(audit_ledger=self.system.audit_ledger)
        self.client = client or LLMClient(self.config)
        self.allowed_chat_ids = allowed_chat_ids or _allowed_ids_from_env()
        self.vaults = vault_store or MemoryVaultStore()
        self._loops: dict[str, AgentLoop] = {}
        self._offset = 0
        self._approvals: dict[str, queue.Queue] = {}
        self._approval_seq = 0
        self._current_chat_id: int | None = None
        self.knowledge_root = knowledge_root

    # ------------------------------------------------------------------

    def _loop_for(self, chat_id: int, vault_key: str) -> AgentLoop:
        """One agent loop per memory vault — vault isolation extends to loops."""
        if vault_key not in self._loops:
            tools = build_default_tools(
                self.system,
                approval_callback=self._request_approval,
                skill_store=self.skills,
                session_store=self.sessions,
            )
            self._loops[vault_key] = AgentLoop(
                system=self.system,
                config=self.config,
                client=self.client,
                tools=tools,
                session_store=self.sessions,
                channel=f"telegram:{chat_id}",
                knowledge_root=self.knowledge_root,
                skill_store=self.skills,
                # Memory context is loaded strictly by vault_key — set here
                # at the integration layer, never from user message content.
                context_provider=lambda: render_memory_context(
                    self.vaults.load(vault_key)
                ),
            )
        return self._loops[vault_key]

    def _request_approval(
        self, name: str, summary: str, level: ThresholdLevel
    ) -> bool:
        """Send approve/veto inline buttons and wait for the reply."""
        chat_id = self._current_chat_id
        if chat_id is None:
            return False
        self._approval_seq += 1
        approval_id = f"appr{self._approval_seq}"
        pending: queue.Queue = queue.Queue()
        self._approvals[approval_id] = pending
        self.api(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": f"⚠ approval required ({level.value}):\n{summary[:500]}",
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {
                                "text": "✅ Approve",
                                "callback_data": f"{approval_id}:yes",
                            },
                            {
                                "text": "❌ Veto",
                                "callback_data": f"{approval_id}:no",
                            },
                        ]
                    ]
                },
            },
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                return bool(pending.get(timeout=1))
            except queue.Empty:
                self._poll_once(timeout=1)
        self._approvals.pop(approval_id, None)
        return False  # auto-deny on timeout — APEX safe default

    # ------------------------------------------------------------------

    def _handle_callback(self, callback: dict[str, Any]) -> None:
        data = str(callback.get("data", ""))
        approval_id, _, verdict = data.partition(":")
        pending = self._approvals.pop(approval_id, None)
        if pending is not None:
            pending.put(verdict == "yes")
        self.api("answerCallbackQuery", {"callback_query_id": callback.get("id")})

    def _transcribe_voice(self, voice: dict[str, Any]) -> str | None:
        if not self.config.voice.enabled:
            return None
        from apex.voice.stt import SpeechToText

        file_info = self.api("getFile", {"file_id": voice.get("file_id")})
        file_path = file_info.get("file_path", "")
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        local = Path(f"/tmp/apex-voice-{voice.get('file_unique_id', 'note')}.oga")
        with urllib.request.urlopen(url, timeout=60) as response:
            local.write_bytes(response.read())
        try:
            return SpeechToText(self.config).transcribe(local)
        finally:
            local.unlink(missing_ok=True)

    def _send_voice_reply(self, chat_id: int, text: str) -> None:
        if not self.config.voice.enabled:
            return
        try:
            from apex.voice.tts import TextToSpeech

            path = TextToSpeech(self.config).synthesize_to_file(text)
            # sendAudio needs multipart upload; keep it simple and best-effort
            self.api(
                "sendMessage",
                {"chat_id": chat_id, "text": f"(voice reply saved server-side: {path})"},
            )
        except Exception:  # noqa: BLE001 — voice is best-effort
            pass

    def handle_message(self, message: dict[str, Any]) -> str | None:
        """Process one Telegram message; returns the reply text sent."""
        chat_id = int(message.get("chat", {}).get("id", 0))
        if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
            self.api(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": (
                        "This APEX instance is locked to its operator. "
                        f"Your chat id is {chat_id}; add it to "
                        "TELEGRAM_ALLOWED_CHAT_IDS to gain access."
                    ),
                },
            )
            return None
        text = message.get("text", "")
        if not text and message.get("voice"):
            try:
                text = self._transcribe_voice(message["voice"]) or ""
            except Exception as exc:  # noqa: BLE001
                self.api(
                    "sendMessage",
                    {"chat_id": chat_id, "text": f"(could not transcribe voice: {exc})"},
                )
                return None
            if not text:
                self.api(
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": "(voice notes need voice enabled: /voice on)",
                    },
                )
                return None
        if not text:
            return None

        user_id = int(message.get("from", {}).get("id", 0)) or chat_id
        vault_key = make_vault_key("telegram", user_id)
        loop = self._loop_for(chat_id, vault_key)
        if text.startswith("/"):
            reply = self._handle_command(loop, text, vault_key)
        else:
            self._current_chat_id = chat_id
            try:
                reply = loop.send(text).reply
            except LLMError as exc:
                reply = f"llm error: {exc}"
            finally:
                self._current_chat_id = None
            self.vaults.write(
                vault_key,
                "working",
                {
                    "session_id": loop.session_id,
                    "active_topic": text[:120],
                    "last_message_at": message.get("date", ""),
                },
                active_session_key=vault_key,
            )
        self.api("sendMessage", {"chat_id": chat_id, "text": reply[:4000]})
        if not text.startswith("/"):
            self._send_voice_reply(chat_id, reply)
        return reply

    def _handle_command(self, loop: AgentLoop, text: str, vault_key: str) -> str:
        parts = text.split()
        command, args = parts[0].lower(), parts[1:]
        if command in ("/start", "/help"):
            return (
                "APEX — the agent that grows with you, safely.\n"
                "/new — new session\n/cycle — run evolution cycle\n"
                "/audit — verify audit chain\n/voice on|off — voice replies\n"
                "/memory show|clear|export — inspect your memory vault\n"
                "/memory update <field> = <value> — correct a stored fact\n"
                "Anything else is a chat message."
            )
        if command == "/new":
            self._close_vault_session(loop, vault_key)
            return f"new session {loop.new_session()[:8]}"
        if command == "/memory":
            return self._handle_memory_command(args, vault_key)
        if command == "/cycle":
            report = self.system.run_knowledge_informed_cycle()
            return f"cycle severity={report.overall_severity.value}"
        if command == "/audit":
            valid, message = self.system.verify_audit_chain()
            return ("✔ " if valid else "✘ ") + message
        if command == "/voice":
            if args and args[0] in ("on", "off"):
                self.config.voice.enabled = args[0] == "on"
                return f"voice {'enabled' if self.config.voice.enabled else 'disabled'}"
            return f"voice is {'on' if self.config.voice.enabled else 'off'}"
        return "unknown command (try /help)"

    def _handle_memory_command(self, args: list[str], vault_key: str) -> str:
        """User-facing memory controls, scoped to the authenticated vault."""
        action = args[0].lower() if args else "show"
        if action == "show":
            return self.vaults.show(vault_key)
        if action == "clear":
            self.vaults.clear(vault_key)
            return "memory vault cleared"
        if action == "export":
            return self.vaults.export(vault_key)[:4000]
        if action == "update":
            rest = " ".join(args[1:])
            field, sep, value = rest.partition("=")
            field, value = field.strip(), value.strip().strip('"')
            if not sep or not field or not value:
                return "usage: /memory update <field> = <value>"
            self.vaults.update_fact(vault_key, field, value)
            return f"updated {field}"
        return "usage: /memory show|clear|export|update <field> = <value>"

    def _close_vault_session(self, loop: AgentLoop, vault_key: str) -> None:
        """Write an episodic summary and wipe working memory at session close."""
        user_texts = [
            m.content
            for m in self.sessions.messages(loop.session_id)
            if m.role == "user"
        ]
        if not user_texts:
            return
        summary = (
            f"{len(user_texts)} user message(s); started with: "
            f"{user_texts[0][:200]}"
        )
        self.vaults.close_session(
            vault_key,
            session_id=loop.session_id,
            summary=summary,
            active_session_key=vault_key,
        )

    # ------------------------------------------------------------------

    def _poll_once(self, timeout: int = 30) -> None:
        updates = self.api(
            "getUpdates", {"offset": self._offset, "timeout": timeout}
        )
        for update in updates or []:
            self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
            if "callback_query" in update:
                self._handle_callback(update["callback_query"])
            elif "message" in update:
                self.handle_message(update["message"])

    def run(self) -> int:
        """Run the long-polling loop forever."""
        print("apex gateway started (telegram long polling)", flush=True)
        self.system.audit_ledger.append(
            "gateway_started", actor="telegram_gateway", payload={}
        )
        while True:
            try:
                self.system.heartbeat()
                self._poll_once()
                self.system.process_alert_timeouts()
            except TelegramError as exc:
                print(f"gateway error (retrying in 5s): {exc}", flush=True)
                time.sleep(5)
            except KeyboardInterrupt:
                print("apex gateway stopped", flush=True)
                return 0


def _allowed_ids_from_env() -> set[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
    ids: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk.lstrip("-").isdigit():
            ids.add(int(chunk))
    return ids


def run_gateway(knowledge_root: str | Path = ".") -> int:
    """Entry point used by ``apex gateway``."""
    try:
        return TelegramGateway(knowledge_root=knowledge_root).run()
    except TelegramError as exc:
        print(f"error: {exc}")
        return 1

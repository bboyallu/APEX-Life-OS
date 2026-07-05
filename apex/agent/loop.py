"""The agent loop — LLM conversation with governed tool calling.

One loop serves every interface (terminal chat, Telegram gateway,
dashboard). APEX upgrades over ungoverned agents:

* every tool call is risk-scored and L3+ calls require human approval;
* every turn is persisted and audit-logged;
* periodic memory nudges ask the model to persist important facts;
* conversation insights are dropped into the knowledge ``raw/`` folder so
  the KnowledgeBridge can turn chat learnings into evolution signals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from apex.agent.config import AgentConfig
from apex.agent.llm import ChatMessage, LLMClient
from apex.agent.sessions import SessionStore
from apex.agent.tools import ToolRegistry
from apex.system import ApexSystem

SYSTEM_PROMPT = """\
You are APEX, a self-evolving personal AI assistant. You grow with your \
operator: you save skills from experience, persist important facts to \
long-term memory, and maintain a personal knowledge base. Unlike other \
agents, every action you take is risk-scored, governance-gated, and \
recorded on a tamper-evident audit ledger — high-risk actions require \
explicit human approval. Use your tools when they help; answer directly \
when they don't. Be concise and useful.
"""

MEMORY_NUDGE = (
    "(system nudge) If this conversation revealed durable facts about the "
    "operator or their world, persist them now with the `remember` tool, "
    "and consider saving a reusable skill with `save_skill` if you "
    "completed a multi-step task. Then answer the user's last message."
)


class AgentTurn(BaseModel):
    """Result of one user → assistant exchange."""

    reply: str = ""
    tool_calls: list[str] = Field(default_factory=list)


class AgentLoop:
    """Multi-turn conversational agent shared by all interfaces."""

    def __init__(
        self,
        *,
        system: ApexSystem,
        config: AgentConfig,
        client: LLMClient,
        tools: ToolRegistry,
        session_store: SessionStore,
        session_id: str | None = None,
        channel: str = "terminal",
        knowledge_root: str | Path = ".",
    ) -> None:
        self.system = system
        self.config = config
        self.client = client
        self.tools = tools
        self.sessions = session_store
        self.channel = channel
        self.knowledge_root = Path(knowledge_root)
        self.session_id = session_id or session_store.create_session(
            channel=channel
        )
        self._turns = 0

    def new_session(self) -> str:
        self.session_id = self.sessions.create_session(channel=self.channel)
        self._turns = 0
        return self.session_id

    def _history(self) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        for stored in self.sessions.messages(self.session_id)[-40:]:
            if stored.role in ("user", "assistant"):
                messages.append(
                    ChatMessage(role=stored.role, content=stored.content)
                )
        return messages

    def send(self, user_text: str) -> AgentTurn:
        """Process one user message, running tool rounds as needed."""
        self._turns += 1
        self.sessions.add_message(self.session_id, "user", user_text)

        messages = self._history()
        nudge_every = max(self.config.memory_nudge_every, 0)
        if nudge_every and self._turns % nudge_every == 0:
            messages.append(ChatMessage(role="system", content=MEMORY_NUDGE))

        executed: list[str] = []
        reply = ""
        for _ in range(max(self.config.max_tool_rounds, 1)):
            response = self.client.chat(messages, tools=self.tools.specs())
            if not response.tool_calls:
                reply = response.content
                break
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content or None,
                    tool_calls=response.raw_tool_calls,
                )
            )
            for call in response.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                executed.append(call.name)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result,
                        tool_call_id=call.call_id,
                        name=call.name,
                    )
                )
        else:
            reply = "(stopped: too many tool rounds without a final answer)"

        self.sessions.add_message(self.session_id, "assistant", reply)
        self.system.audit_ledger.append(
            "agent_turn",
            actor=f"agent:{self.channel}",
            payload={
                "session": self.session_id,
                "tools": executed,
                "chars": len(reply),
            },
        )
        return AgentTurn(reply=reply, tool_calls=executed)

    def record_insight(self, insight: str) -> Path:
        """Drop a conversation insight into knowledge ``raw/``.

        The KnowledgeBridge later folds it into the wiki, where
        ``signal:`` directives become evolution candidates.
        """
        raw_dir = self.knowledge_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / "apex-chat-insights.md"
        stamp = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as handle:
            if path.stat().st_size == 0:
                handle.write("topic: Chat Insights\n\n")
            handle.write(f"- {stamp}: {insight}\n")
        self.system.audit_ledger.append(
            "chat_insight_recorded",
            actor=f"agent:{self.channel}",
            payload={"session": self.session_id, "chars": len(insight)},
        )
        return path

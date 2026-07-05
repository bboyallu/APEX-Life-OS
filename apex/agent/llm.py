"""Provider-agnostic LLM chat client (OpenAI-compatible ``/chat/completions``).

One integration covers OpenAI, OpenRouter, Groq, and local endpoints
(Ollama, LM Studio, vLLM). Uses only the standard library — no SDK
dependency — with an injectable transport for offline testing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from pydantic import BaseModel, Field

from apex.agent.config import AgentConfig

#: Transport signature: (url, headers, payload) -> parsed JSON response.
Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot be reached or returns an error."""


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""


def _urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach LLM endpoint {url}: {exc.reason}") from exc


class LLMClient:
    """Minimal chat-completions client with tool-calling support."""

    def __init__(
        self, config: AgentConfig, *, transport: Transport | None = None
    ) -> None:
        self.config = config
        self._transport = transport or _urllib_transport

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a conversation and return the assistant's reply."""
        url = self.config.resolved_base_url().rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key

        payload: dict[str, Any] = {
            "model": self.config.resolved_model(),
            "messages": [m.to_payload() for m in messages],
            "temperature": self.config.temperature,
        }
        if tools:
            payload["tools"] = tools

        data = self._transport(url, headers, payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed LLM response: {data!r}") from exc

        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for raw in raw_tool_calls:
            function = raw.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    call_id=raw.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            raw_tool_calls=raw_tool_calls,
            model=data.get("model", ""),
        )

"""Provider-neutral conversation primitives for the APEX agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of a tool invocation, fed back to the model."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation turn.

    ``role`` is ``"user"``, ``"assistant"`` or ``"tool"``.  Assistant messages
    may carry ``tool_calls``; tool messages carry ``tool_results``.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [vars(c) for c in self.tool_calls],
            "tool_results": [vars(r) for r in self.tool_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[ToolCall(**c) for c in data.get("tool_calls", [])],
            tool_results=[ToolResult(**r) for r in data.get("tool_results", [])],
        )


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class LLMResponse:
    """A single model completion."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


@dataclass
class ToolSpec:
    """Schema describing a tool to the model (JSON-Schema parameters)."""

    name: str
    description: str
    parameters: dict[str, Any]

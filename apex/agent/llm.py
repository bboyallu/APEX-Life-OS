"""LLM providers.

Three interchangeable backends implement :class:`LLMProvider`:

* :class:`AnthropicProvider` — Claude via the Messages API.
* :class:`OpenAICompatProvider` — OpenAI and any OpenAI-compatible server
  (Ollama, LM Studio, vLLM, OpenRouter, …).
* :class:`LocalProvider` — a deterministic offline assistant that needs no
  network and no API key, so APEX is usable by anyone out of the box.

HTTP uses only the standard library (no extra dependencies) and retries
transient failures (429 / 5xx / network errors) with exponential backoff.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

from apex.agent.messages import LLMResponse, Message, ToolCall, ToolSpec, Usage

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

#: ``(url, headers, body, timeout) -> (status, response_bytes)``
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


#: Same as :data:`Transport` but returns the body as an iterator of lines.
StreamTransport = Callable[[str, dict[str, str], bytes, float], tuple[int, Iterator[bytes]]]

TextCallback = Callable[[str], None]


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ConnectionError(str(exc)) from exc


def urllib_stream_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, Iterator[bytes]]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        return exc.code, iter([exc.read()])
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ConnectionError(str(exc)) from exc

    def lines() -> Iterator[bytes]:
        try:
            with resp:
                yield from resp
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise ConnectionError(str(exc)) from exc

    return resp.status, lines()


def _sse_events(lines: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Parse Server-Sent Events ``data:`` lines into JSON objects."""
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


class LLMProvider(ABC):
    """Abstract chat-completion backend with tool calling."""

    name = "abstract"
    model = ""

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        """Return the model's next turn.

        If ``on_text`` is given the provider streams text deltas to it as they
        arrive; the returned response is always the complete turn.
        """


class _HTTPProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or urllib_transport
        self._stream_transport = stream_transport or urllib_stream_transport
        self._sleep = sleep

    def _post(self, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()
        headers = {"content-type": "application/json", **headers}
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                status, raw = self._transport(url, headers, body, self.timeout)
            except ConnectionError as exc:
                last = f"network error: {exc}"
            else:
                if status == 200:
                    try:
                        return json.loads(raw)
                    except ValueError as exc:
                        raise LLMError(f"invalid JSON from provider: {exc}") from exc
                last = f"HTTP {status}: {raw[:300].decode(errors='replace')}"
                if status not in (408, 409, 429) and status < 500:
                    raise LLMError(last)  # non-retryable (auth, bad request, …)
            if attempt < self.max_retries:
                self._sleep(min(2**attempt, 20))
        raise LLMError(f"{self.name} request failed after retries — {last}")

    def _post_stream(
        self, path: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """POST with ``stream: true`` and yield SSE events.

        Connection setup is retried like :meth:`_post`; once bytes are flowing
        a mid-stream failure raises :class:`LLMError` (a partial turn cannot be
        safely replayed).
        """
        url = f"{self.base_url}{path}"
        body = json.dumps({**payload, "stream": True}).encode()
        headers = {"content-type": "application/json", **headers}
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                status, lines = self._stream_transport(url, headers, body, self.timeout)
            except ConnectionError as exc:
                last = f"network error: {exc}"
            else:
                if status == 200:
                    try:
                        yield from _sse_events(lines)
                    except ConnectionError as exc:
                        raise LLMError(f"{self.name} stream interrupted: {exc}") from exc
                    return
                last = f"HTTP {status}: {b''.join(lines)[:300].decode(errors='replace')}"
                if status not in (408, 409, 429) and status < 500:
                    raise LLMError(last)
            if attempt < self.max_retries:
                self._sleep(min(2**attempt, 20))
        raise LLMError(f"{self.name} request failed after retries — {last}")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, **kw: Any) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        kw.setdefault(
            "base_url", os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        )
        kw.setdefault("model", os.environ.get("APEX_MODEL", DEFAULT_ANTHROPIC_MODEL))
        super().__init__(api_key=api_key, **kw)

    @staticmethod
    def _encode(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks or m.content})
            elif m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in m.tool_results
                        ],
                    }
                )
        return out

    def complete(self, system, messages, tools, on_text=None):
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._encode(messages),
        }
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        headers = {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}
        if on_text is not None:
            return self._stream(headers, payload, on_text)
        data = self._post("/v1/messages", headers, payload)
        text, calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(block["id"], block["name"], block.get("input") or {})
                )
        u = data.get("usage", {})
        return LLMResponse(
            text="".join(text).strip(),
            tool_calls=calls,
            stop_reason=data.get("stop_reason", "end_turn"),
            usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0)),
        )


    def _stream(self, headers, payload, on_text) -> LLMResponse:
        text: list[str] = []
        blocks: dict[int, dict[str, Any]] = {}
        stop, usage = "end_turn", Usage()
        for ev in self._post_stream("/v1/messages", headers, payload):
            kind = ev.get("type")
            if kind == "message_start":
                usage.input_tokens = ev.get("message", {}).get("usage", {}).get("input_tokens", 0)
            elif kind == "content_block_start":
                b = ev.get("content_block", {})
                if b.get("type") == "tool_use":
                    blocks[ev["index"]] = {"id": b["id"], "name": b["name"], "json": ""}
            elif kind == "content_block_delta":
                d = ev.get("delta", {})
                if d.get("type") == "text_delta":
                    text.append(d.get("text", ""))
                    on_text(d.get("text", ""))
                elif d.get("type") == "input_json_delta" and ev.get("index") in blocks:
                    blocks[ev["index"]]["json"] += d.get("partial_json", "")
            elif kind == "message_delta":
                stop = ev.get("delta", {}).get("stop_reason") or stop
                usage.output_tokens = ev.get("usage", {}).get("output_tokens", usage.output_tokens)
            elif kind == "error":
                raise LLMError(f"anthropic stream error: {ev.get('error', {}).get('message', ev)}")
        calls = []
        for i in sorted(blocks):
            b = blocks[i]
            try:
                args = json.loads(b["json"] or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(b["id"], b["name"], args))
        return LLMResponse("".join(text).strip(), calls, stop, usage)


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAICompatProvider(_HTTPProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, **kw: Any) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        kw.setdefault("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        kw.setdefault("model", os.environ.get("APEX_MODEL", DEFAULT_OPENAI_MODEL))
        # Local servers (Ollama, LM Studio) need no key; the hosted API does.
        if not api_key and "api.openai.com" in kw["base_url"]:
            raise LLMError("OPENAI_API_KEY is not set")
        super().__init__(api_key=api_key, **kw)

    @staticmethod
    def _encode(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                out += [
                    {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                    for r in m.tool_results
                ]
        return out

    def complete(self, system, messages, tools, on_text=None):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._encode(system, messages),
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if on_text is not None:
            return self._stream(headers, payload, on_text)
        data = self._post("/chat/completions", headers, payload)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(tc.get("id") or uuid.uuid4().hex, tc["function"]["name"], args))
        u = data.get("usage", {})
        return LLMResponse(
            text=(msg.get("content") or "").strip(),
            tool_calls=calls,
            stop_reason=choice.get("finish_reason", "stop"),
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)),
        )


    def _stream(self, headers, payload, on_text) -> LLMResponse:
        payload = {**payload, "stream_options": {"include_usage": True}}
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        stop, usage = "stop", Usage()
        for ev in self._post_stream("/chat/completions", headers, payload):
            if u := ev.get("usage"):
                usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            for choice in ev.get("choices") or []:
                d = choice.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
                    on_text(d["content"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] = fn.get("name") or c["name"]
                    c["args"] += fn.get("arguments") or ""
                stop = choice.get("finish_reason") or stop
        out = []
        for i in sorted(calls):
            c = calls[i]
            try:
                args = json.loads(c["args"] or "{}")
            except ValueError:
                args = {}
            out.append(ToolCall(c["id"] or uuid.uuid4().hex, c["name"], args))
        return LLMResponse("".join(text).strip(), out, stop, usage)


# ---------------------------------------------------------------------------
# Offline provider
# ---------------------------------------------------------------------------

_MATH_RE = re.compile(r"^[\d\s+\-*/().%^,a-z_]+$", re.I)

_WORD_OPS = [
    (r"\bto the power of\b", "**"), (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
    (r"\b(?:plus|and)\b", "+"), (r"\bminus\b", "-"), (r"\b(?:times|multiplied by|x)\b", "*"),
    (r"\b(?:divided by|over)\b", "/"), (r"\bmod(?:ulo)?\b", "%"),
]


def _normalise_math(text: str) -> str:
    """Turn '18% of 2450', '5 plus 3', '7 squared' into evaluable expressions."""
    t = text.strip().lower().replace(",", "")
    if m := re.fullmatch(r"([\d.]+)\s*%\s*of\s*([\d.]+)", t):
        return f"{m.group(1)}/100*{m.group(2)}"
    for pat, rep in _WORD_OPS:
        t = re.sub(pat, rep, t)
    return t.replace("^", "**")

HELP_TEXT = (
    "I'm running in offline mode (no API key found), so I understand simple "
    "commands:\n"
    "  • remember <fact>            • what do you remember [about X]\n"
    "  • add task <title>           • list tasks / done <id>\n"
    "  • calculate <expression>     • what time is it\n"
    "  • status                     • list files / read file <path>\n"
    "Set ANTHROPIC_API_KEY or OPENAI_API_KEY (or run Ollama) for full "
    "natural-language reasoning."
)


class LocalProvider(LLMProvider):
    """Deterministic rule-based assistant — works with zero configuration.

    It maps a handful of intents to tool calls and then renders the tool
    output, exercising the exact same governed tool pipeline as an LLM would.
    """

    name = "local"
    model = "offline-rules"

    def complete(self, system, messages, tools, on_text=None):
        resp = self._complete(messages, tools)
        if on_text and resp.text:
            on_text(resp.text)
        return resp

    def _complete(self, messages, tools) -> LLMResponse:
        if not messages:
            return LLMResponse(text=HELP_TEXT)
        last = messages[-1]
        if last.role == "tool":
            body = "\n".join(
                ("⚠ " if r.is_error else "") + r.content for r in last.tool_results
            )
            return LLMResponse(text=body)
        available = {t.name for t in tools}
        call = self._route(last.content.strip(), available)
        if call is None:
            return LLMResponse(text=HELP_TEXT)
        return LLMResponse(tool_calls=[call], stop_reason="tool_use")

    @staticmethod
    def _call(name: str, **args: Any) -> ToolCall:
        return ToolCall(uuid.uuid4().hex, name, args)

    def _route(self, text: str, available: set[str]) -> ToolCall | None:
        t = text.strip().rstrip("?.! ")
        low = t.lower()

        def has(name: str) -> bool:
            return name in available

        if m := re.match(r"(?:please )?(?:remember|note)(?: that|:)?\s+(.+)", t, re.I):
            fact = m.group(1)
            scope = "user" if re.match(r"(i|my)\b", fact, re.I) else "repository"
            if has("remember"):
                return self._call("remember", subject="user" if scope == "user" else "note", fact=fact, scope=scope)
        if m := re.match(r"forget\s+([0-9a-f-]{8,36})", t, re.I):
            if has("forget"):
                return self._call("forget", entry_id=m.group(1))
        if m := re.match(
            r"(?:what do you (?:remember|know)(?: about)?|recall|search memory(?: for)?)\s*(.*)",
            t, re.I,
        ):
            if has("recall"):
                return self._call("recall", query=m.group(1).strip())
        if m := re.match(r"(?:add|put)\s+(.+?)\s+(?:to|on)\s+my\s+(?:todo|to-do|task)s?(?:\s+list)?$", t, re.I):
            if has("add_task"):
                return self._call("add_task", title=m.group(1))
        if m := re.match(r"(?:add (?:a )?task|todo|add todo|remind me to|i need to|i have to|don't forget to)[:\s]+(.+)", t, re.I):
            if has("add_task"):
                return self._call("add_task", title=m.group(1))
        m = re.match(r"(?:what(?:'s| is) my|do you know my)\s+(.+)", t, re.I)
        if m and has("recall"):
            if not re.match(r"(?:tasks?|todos?)", m.group(1), re.I):
                return self._call("recall", query=m.group(1).split()[-1])
        if m := re.match(r"(?:done|complete|finish)(?: task)?\s+#?(\d+)", t, re.I):
            if has("complete_task"):
                return self._call("complete_task", task_id=int(m.group(1)))
        if re.match(r"(?:list |show |what are my )?(?:my )?(?:tasks|todos)", low):
            if has("list_tasks"):
                return self._call("list_tasks")
        if m := re.match(r"(?:read|cat|open)(?: file)?\s+(\S+)", t, re.I):
            if has("read_file"):
                return self._call("read_file", path=m.group(1))
        if re.match(r"(?:ls|list files|list dir\w*)", low):
            if has("list_dir"):
                rest = t.split(None, 2)
                return self._call("list_dir", path=rest[-1] if len(rest) > 1 and rest[0] == "ls" else ".")
        if re.search(r"\b(time|date|today|day is it)\b", low) and has("current_time"):
            return self._call("current_time")
        if re.match(r"(?:system )?(?:status|health)", low) and has("system_status"):
            return self._call("system_status")
        expr = re.sub(r"^(?:calculate|calc|compute|what is|what's|whats|how much is|eval)\s+", "", t, flags=re.I)
        expr = _normalise_math(expr)
        if has("calculate") and any(c.isdigit() for c in expr) and _MATH_RE.match(expr):
            return self._call("calculate", expression=expr)
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(
    name: str = "auto", *, model: str | None = None, **kw: Any
) -> LLMProvider:
    """Build a provider by name (``auto``, ``anthropic``, ``openai``, ``ollama``, ``local``).

    ``auto`` picks the first configured backend: Anthropic, OpenAI-compatible,
    then the offline provider.
    """
    name = name.lower()
    if model:
        kw["model"] = model
    if name == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
            name = "openai"
        else:
            name = "local"
    if name == "anthropic":
        return AnthropicProvider(**kw)
    if name == "openai":
        return OpenAICompatProvider(**kw)
    if name == "ollama":
        kw.setdefault("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        kw.setdefault("model", os.environ.get("APEX_MODEL", "llama3.1"))
        p = OpenAICompatProvider(api_key=kw.pop("api_key", None) or "ollama", **kw)
        p.name = "ollama"
        return p
    if name == "local":
        return LocalProvider()
    raise LLMError(f"unknown provider {name!r}")

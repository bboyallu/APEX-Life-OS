"""End-to-end provider tests over a real local HTTP server (real urllib transports)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from apex.agent import Agent, AnthropicProvider, LLMError, Message, OpenAICompatProvider, default_tools

ANTHROPIC_SSE = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "check."}},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu1", "name": "calculate"}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"expression"'}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ': "6*7"}'}},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
]
OPENAI_SSE = [
    {"choices": [{"delta": {"content": "Let me "}}]},
    {"choices": [{"delta": {"content": "check."}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "calculate", "arguments": '{"expression"'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "6*7"}'}}]}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 5}},
]


@pytest.fixture
def server():
    state = {"requests": [], "fail_first": 0, "status": 200}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append((self.path, dict(self.headers), body))
            if state["fail_first"] > 0:
                state["fail_first"] -= 1
                self.send_response(503); self.end_headers(); self.wfile.write(b"busy"); return
            if state["status"] != 200:
                self.send_response(state["status"]); self.end_headers(); self.wfile.write(b"nope"); return
            anth = self.path.endswith("/messages")
            if body.get("stream"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream"); self.end_headers()
                for ev in ANTHROPIC_SSE if anth else OPENAI_SSE:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
                if not anth:
                    self.wfile.write(b"data: [DONE]\n\n")
            else:
                if anth:
                    out = {"content": [{"type": "text", "text": "plain"}], "usage": {"input_tokens": 1, "output_tokens": 2}}
                else:
                    out = {"choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}]}
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("content-length", str(len(data))); self.end_headers()
                self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()


def mk(cls, server, **kw):
    base = server["url"] + ("" if cls is AnthropicProvider else "/v1")
    return cls(api_key="k", base_url=base, sleep=lambda s: None, **kw)


@pytest.mark.parametrize("cls", [AnthropicProvider, OpenAICompatProvider])
def test_plain_over_real_http(cls, server):
    r = mk(cls, server).complete("sys", [Message("user", "hi")], [])
    assert r.text == "plain"


@pytest.mark.parametrize("cls", [AnthropicProvider, OpenAICompatProvider])
def test_streaming_over_real_http(cls, server):
    chunks = []
    r = mk(cls, server).complete("sys", [Message("user", "hi")], default_tools().specs()[:1], on_text=chunks.append)
    assert chunks == ["Let me ", "check."]
    assert r.text == "Let me check."
    assert r.tool_calls[0].name == "calculate" and r.tool_calls[0].arguments == {"expression": "6*7"}
    assert r.usage.output_tokens > 0


@pytest.mark.parametrize("cls", [AnthropicProvider, OpenAICompatProvider])
def test_retry_and_fatal_over_real_http(cls, server):
    server["fail_first"] = 2
    assert mk(cls, server).complete("s", [Message("user", "q")], []).text == "plain"
    server["status"] = 401
    with pytest.raises(LLMError, match="401"):
        mk(cls, server).complete("s", [Message("user", "q")], [], on_text=lambda t: None)


def test_connection_refused_is_llmerror():
    p = AnthropicProvider(api_key="k", base_url="http://127.0.0.1:9", max_retries=1, sleep=lambda s: None)
    with pytest.raises(LLMError, match="network error"):
        p.complete("s", [], [])
    with pytest.raises(LLMError, match="network error"):
        p.complete("s", [], [], on_text=lambda t: None)


def test_agent_streams_and_runs_tool(server, tmp_path):
    chunks = []
    # Server always replies with the same tool-call turn, so cap steps.
    a = Agent.create(mk(AnthropicProvider, server), home=tmp_path / "h", root=tmp_path, max_steps=2, on_text=chunks.append)
    res = a.run("what is 6*7")
    assert chunks and res.tool_events[0].name == "calculate" and "= 42" in res.tool_events[0].result
    streamed_bodies = [b for _, _, b in server["requests"]]
    assert all(b["stream"] for b in streamed_bodies)


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("what's 18% of 2450", "= 441"),
        ("what is 5 plus 3", "= 8"),
        ("7 squared", "= 49"),
        ("how much is 100 divided by 8", "= 12.5"),
        ("12 times 12", "= 144"),
        ("what is 2 to the power of 10", "= 1024"),
    ],
)
def test_offline_math_phrases(phrase, expected, tmp_path):
    from apex.agent import LocalProvider
    a = Agent.create(LocalProvider(), home=tmp_path / "h", root=tmp_path)
    assert expected in a.run(phrase).text


def test_offline_natural_tasks_and_memory(tmp_path):
    from apex.agent import LocalProvider
    a = Agent.create(LocalProvider(), home=tmp_path / "h", root=tmp_path)
    assert "call mum" in a.run("add call mum to my todo list").text
    assert "file taxes" in a.run("I need to file taxes").text
    out = a.run("list tasks").text
    assert "call mum" in out and "file taxes" in out
    a.run("remember my dentist is Dr. Lee")
    assert "Dr. Lee" in a.run("what's my dentist").text

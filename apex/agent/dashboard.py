"""Web dashboard — the ``apex dashboard`` command.

Zero-dependency ``http.server`` dashboard bound to ``127.0.0.1:9119``
(access from your laptop via ``ssh -L 9119:localhost:9119 user@vps``).
Shows sessions, chat, pending approvals with approve/veto, the audit
chain, memories, skills, and knowledge outputs.
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from apex.agent.config import AgentConfig, load_config
from apex.agent.llm import LLMClient, LLMError
from apex.agent.loop import AgentLoop
from apex.agent.sessions import SessionStore
from apex.agent.skills import SkillStore
from apex.agent.tools import build_default_tools
from apex.system import ApexSystem

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>APEX Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 60rem; }}
h1 {{ color: #1a7f37; }} nav a {{ margin-right: 1rem; }}
pre {{ background: #f6f8fa; padding: 1rem; overflow-x: auto; }}
li {{ margin: .25rem 0; }} input[type=text] {{ width: 70%; padding: .4rem; }}
button {{ padding: .4rem .8rem; }}
.msg-user {{ color: #0550ae; }} .msg-assistant {{ color: #1a7f37; }}
</style></head><body>
<h1>APEX</h1>
<nav><a href="/">chat</a><a href="/sessions">sessions</a>
<a href="/audit">audit</a><a href="/memories">memories</a>
<a href="/skills">skills</a><a href="/outputs">outputs</a></nav>
<hr>{body}</body></html>
"""


class DashboardState:
    """Shared state behind the HTTP handlers (injectable for tests)."""

    def __init__(
        self,
        *,
        knowledge_root: str | Path = ".",
        config: AgentConfig | None = None,
        system: ApexSystem | None = None,
        session_store: SessionStore | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.system = system or ApexSystem(knowledge_root=knowledge_root)
        self.sessions = session_store or SessionStore()
        self.skills = SkillStore(audit_ledger=self.system.audit_ledger)
        self.client = client or LLMClient(self.config)
        self.knowledge_root = Path(knowledge_root)
        tools = build_default_tools(
            self.system,
            approval_callback=None,  # dashboard tool calls: deny L3+ (safe)
            skill_store=self.skills,
            session_store=self.sessions,
        )
        self.loop = AgentLoop(
            system=self.system,
            config=self.config,
            client=self.client,
            tools=tools,
            session_store=self.sessions,
            channel="dashboard",
            knowledge_root=knowledge_root,
        )

    # ------------------------------------------------------------------
    # Page renderers (return HTML body fragments)
    # ------------------------------------------------------------------

    def page_chat(self) -> str:
        rows = []
        for message in self.sessions.messages(self.loop.session_id)[-30:]:
            css = f"msg-{html.escape(message.role)}"
            rows.append(
                f'<li class="{css}"><b>{html.escape(message.role)}:</b> '
                f"{html.escape(message.content)}</li>"
            )
        return (
            f"<ul>{''.join(rows)}</ul>"
            '<form method="post" action="/chat">'
            '<input type="text" name="text" autofocus>'
            "<button>send</button></form>"
        )

    def page_sessions(self) -> str:
        rows = [
            f"<li>{s.session_id[:8]} [{html.escape(s.channel)}] "
            f"{s.message_count} msgs ({html.escape(s.created_at[:19])})</li>"
            for s in self.sessions.sessions()
        ]
        return f"<ul>{''.join(rows) or '<li>none</li>'}</ul>"

    def page_audit(self) -> str:
        valid, message = self.system.verify_audit_chain()
        entries = self.system.audit_ledger.read()[-30:]
        rows = [
            f"<li>{html.escape(e.event_type)} by {html.escape(e.actor)} "
            f"({html.escape(str(e.timestamp)[:19])})</li>"
            for e in entries
        ]
        status = "✔" if valid else "✘"
        return (
            f"<p>{status} {html.escape(message)}</p>"
            f"<ul>{''.join(rows) or '<li>empty</li>'}</ul>"
        )

    def page_memories(self) -> str:
        rows = [
            f"<li>[{html.escape(e.subject)}] {html.escape(e.fact)}</li>"
            for e in self.system.recall_memories()
        ]
        return f"<ul>{''.join(rows) or '<li>none</li>'}</ul>"

    def page_skills(self) -> str:
        rows = [
            f"<li><b>{html.escape(s.name)}</b> (uses: {s.uses}) — "
            f"{html.escape(s.description)}</li>"
            for s in self.skills.list()
        ]
        return f"<ul>{''.join(rows) or '<li>none</li>'}</ul>"

    def page_outputs(self) -> str:
        outputs = self.knowledge_root / "outputs"
        rows = []
        if outputs.is_dir():
            for path in sorted(outputs.glob("*.md"), reverse=True)[:20]:
                rows.append(f"<li>{html.escape(path.name)}</li>")
        return f"<ul>{''.join(rows) or '<li>none</li>'}</ul>"

    def post_chat(self, text: str) -> None:
        try:
            self.loop.send(text)
        except LLMError as exc:
            self.sessions.add_message(
                self.loop.session_id, "assistant", f"llm error: {exc}"
            )


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # noqa: A003 — quiet server
            pass

        def _send(self, body: str, *, status: int = 200) -> None:
            payload = _PAGE.format(body=body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            pages = {
                "/": state.page_chat,
                "/sessions": state.page_sessions,
                "/audit": state.page_audit,
                "/memories": state.page_memories,
                "/skills": state.page_skills,
                "/outputs": state.page_outputs,
            }
            page = pages.get(route)
            if page is None:
                self._send("<p>not found</p>", status=404)
                return
            self._send(page())

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            fields = parse_qs(self.rfile.read(length).decode("utf-8"))
            if route == "/chat":
                text = (fields.get("text") or [""])[0].strip()
                if text:
                    state.post_chat(text)
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()
                return
            self._send("<p>not found</p>", status=404)

    return Handler


def run_dashboard(
    knowledge_root: str | Path = ".",
    *,
    host: str = "127.0.0.1",
    port: int = 9119,
) -> int:
    """Entry point used by ``apex dashboard``."""
    state = DashboardState(knowledge_root=knowledge_root)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(
        f"apex dashboard on http://{host}:{port} "
        f"(tunnel: ssh -L {port}:localhost:{port} user@vps)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("apex dashboard stopped", flush=True)
    return 0

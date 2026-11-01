"""Command-line interface: ``apex chat | ask | tools | memory | tasks | audit | doctor``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import apex
from apex.agent import Agent, ApprovalRequest, LLMError, ToolEvent, create_provider
from apex.agent.tools import default_tools
from apex.agent.workspace import Workspace
from apex.memory.provider import MemoryScope

_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _prompt_approval(req: ApprovalRequest) -> bool:
    print(_c("33", f"\n⚠  Approval needed [{req.level.value}, risk {req.risk_score:.2f}]: {req.tool}"))
    for k, v in req.arguments.items():
        print(f"     {k}: {str(v)[:200]}")
    if not sys.stdin.isatty():
        print("   (non-interactive: denied)")
        return False
    try:
        return input("   Allow? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _show_event(ev: ToolEvent) -> None:
    mark = "✓" if not ev.is_error else "✗"
    args = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in ev.arguments.items())
    print(_c("2", f"  {mark} {ev.name}({args}) [{ev.level}]"), file=sys.stderr)


class _Streamer:
    """Writes streamed text and remembers whether anything was printed."""

    def __init__(self) -> None:
        self.active = False

    def __call__(self, chunk: str) -> None:
        self.active = True
        sys.stdout.write(chunk)
        sys.stdout.flush()


def build_agent(args: argparse.Namespace, *, interactive: bool, on_text=None) -> Agent:
    provider = create_provider(args.provider, model=args.model)
    if args.yes:
        approver = lambda req: req.level.value != "L4"  # noqa: E731
    elif interactive:
        approver = _prompt_approval
    else:
        approver = None
    return Agent.create(
        provider,
        home=args.home,
        root=args.workspace,
        allow_shell=args.allow_shell,
        approver=approver,
        session=getattr(args, "session", None),
        max_steps=args.max_steps,
        on_event=_show_event if not getattr(args, "json", False) else None,
        on_text=on_text,
    )


CHAT_HELP = """\
/help            show this help          /tools    list tools
/memory          list memories           /tasks    list open tasks
/audit           verify audit ledger     /reset    clear conversation
/exit            quit"""


def cmd_chat(args: argparse.Namespace) -> int:
    stream = _Streamer()
    agent = build_agent(args, interactive=True, on_text=stream)
    print(_c("1", f"APEX Life OS {apex.__version__}") + f" — provider: {agent.provider.name} ({agent.provider.model})")
    print("Type /help for commands, /exit to quit.\n")
    while True:
        try:
            line = input(_c("36", "you ▸ ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.split()[0].lower()
            if cmd in ("/exit", "/quit", "/q"):
                return 0
            elif cmd == "/help":
                print(CHAT_HELP)
            elif cmd == "/tools":
                _print_tools(agent)
            elif cmd == "/memory":
                _print_memories(agent)
            elif cmd == "/tasks":
                print(_tasks_text(agent.workspace))
            elif cmd == "/audit":
                print(agent.system.verify_audit_chain()[1])
            elif cmd == "/reset":
                agent.reset()
                print("Conversation cleared.")
            else:
                print(f"Unknown command {cmd}. /help lists commands.")
            continue
        stream.active = False
        print(_c("32", "apex ▸ "), end="", flush=True)
        try:
            res = agent.run(line)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        if not stream.active or res.error:
            print(res.text, end="")
        print("\n")


def cmd_ask(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    if prompt == "-" or not prompt:
        prompt = sys.stdin.read()
    if not prompt.strip():
        print("error: empty prompt", file=sys.stderr)
        return 2
    agent = build_agent(args, interactive=sys.stdin.isatty())
    res = agent.run(prompt.strip())
    if args.json:
        print(json.dumps({
            "text": res.text, "error": res.error, "steps": res.steps,
            "usage": vars(res.usage),
            "tools": [{"name": e.name, "arguments": e.arguments, "allowed": e.allowed,
                       "level": e.level, "is_error": e.is_error} for e in res.tool_events],
        }, indent=2))
    else:
        print(res.text)
    return 1 if res.error else 0


def _print_tools(agent: Agent) -> None:
    for t in agent.tools:
        print(f"  {t.name:<14} {t.description}")


def _print_memories(agent: Agent) -> None:
    mems = agent.system.recall_memories()
    if not mems:
        print("No memories yet.")
    for m in mems:
        print(f"  [{m.entry_id[:8]}] ({m.scope.value}) {m.subject}: {m.fact}")


def _tasks_text(ws: Workspace) -> str:
    tasks = [t for t in ws.load_tasks() if not t["done"]]
    return "\n".join(f"  #{t['id']} {t['title']}" for t in tasks) or "No open tasks."


def cmd_tools(args: argparse.Namespace) -> int:
    for t in default_tools(allow_shell=True):
        print(f"{t.name:<14} {t.description}")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    agent = Agent.create(None, home=args.home)
    if args.action == "list":
        _print_memories(agent)
    elif args.action == "search":
        for m in agent.system.search_memories(" ".join(args.rest)):
            print(f"  [{m.entry_id[:8]}] {m.subject}: {m.fact}")
    elif args.action == "add":
        if len(args.rest) < 2:
            print("usage: apex memory add <subject> <fact...> [--user]", file=sys.stderr)
            return 2
        e = agent.system.remember(
            args.rest[0], " ".join(args.rest[1:]),
            scope=MemoryScope.USER if args.user else MemoryScope.REPOSITORY,
        )
        print(f"Stored [{e.entry_id[:8]}]")
    elif args.action == "forget":
        ids = [m.entry_id for m in agent.system.recall_memories() if args.rest and m.entry_id.startswith(args.rest[0])]
        if len(ids) != 1:
            print("error: id not found or ambiguous", file=sys.stderr)
            return 1
        agent.system.forget_memory(ids[0])
        print("Forgotten.")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    print(_tasks_text(Workspace(args.home)))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    agent = Agent.create(None, home=args.home)
    ok, msg = agent.system.verify_audit_chain()
    if args.verify:
        print(("✓ " if ok else "✗ ") + msg)
        return 0 if ok else 1
    for e in agent.system.audit_ledger.read()[-args.n:]:
        print(f"{e.timestamp:%Y-%m-%d %H:%M:%S} {e.event_type:<20} {e.actor:<10} {json.dumps(e.payload)[:100]}")
    print(("✓ " if ok else "✗ ") + msg)
    return 0 if ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    ws = Workspace(args.home)
    print(f"APEX {apex.__version__}  python {sys.version.split()[0]}")
    print(f"home: {ws.home}")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        print(f"  {var}: {'set' if os.environ.get(var) else 'not set'}")
    try:
        p = create_provider(args.provider, model=args.model)
        print(f"provider: {p.name} ({p.model})")
    except LLMError as exc:
        print(f"provider: ERROR — {exc}")
        return 1
    ok, msg = Agent.create(None, home=args.home).system.verify_audit_chain()
    print(f"audit: {'OK' if ok else 'BROKEN'} — {msg}")
    return 0 if ok else 1


def _defaults() -> dict[str, object]:
    return {
        "provider": os.environ.get("APEX_PROVIDER", "auto"), "model": None,
        "home": None, "workspace": None, "allow_shell": False, "yes": False,
        "max_steps": 10,
    }


def build_parser() -> argparse.ArgumentParser:
    # Shared options are accepted both before and after the subcommand
    # (`apex --provider ollama chat` and `apex chat --provider ollama`).
    # SUPPRESS keeps a subparser from clobbering a value given at the root;
    # real defaults are applied once, in main().
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--provider", help="auto | anthropic | openai | ollama | local (default: auto)")
    common.add_argument("--model", help="model name override")
    common.add_argument("--home", help="state directory (default: $APEX_HOME or ~/.apex)")
    common.add_argument("--workspace", help="directory file tools may access (default: cwd)")
    common.add_argument("--allow-shell", action="store_true", help="enable the run_command tool (approval-gated)")
    common.add_argument("--yes", "-y", action="store_true", help="auto-approve L3 actions (L4 is always blocked)")
    common.add_argument("--max-steps", type=int)

    p = argparse.ArgumentParser(prog="apex", parents=[common],
                                description="APEX Life OS — a governed personal AI agent.")
    p.add_argument("--version", action="version", version=f"apex {apex.__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", parents=[common], help="interactive chat (default)")
    c.add_argument("--session", default="default", help="session name for persistent history")
    c.set_defaults(fn=cmd_chat)

    a = sub.add_parser("ask", parents=[common], help="one-shot question ('-' reads stdin)")
    a.add_argument("prompt", nargs="*")
    a.add_argument("--session", default=None)
    a.add_argument("--json", action="store_true", help="machine-readable output")
    a.set_defaults(fn=cmd_ask)

    sub.add_parser("tools", parents=[common], help="list tools").set_defaults(fn=cmd_tools)

    m = sub.add_parser("memory", parents=[common], help="manage long-term memory")
    m.add_argument("action", choices=["list", "search", "add", "forget"])
    m.add_argument("rest", nargs="*")
    m.add_argument("--user", action="store_true", help="(add) store as user-scoped")
    m.set_defaults(fn=cmd_memory)

    sub.add_parser("tasks", parents=[common], help="list open tasks").set_defaults(fn=cmd_tasks)

    au = sub.add_parser("audit", parents=[common], help="show / verify the audit ledger")
    au.add_argument("--verify", action="store_true")
    au.add_argument("-n", type=int, default=20)
    au.set_defaults(fn=cmd_audit)

    sub.add_parser("doctor", parents=[common], help="diagnose configuration").set_defaults(fn=cmd_doctor)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    if not getattr(args, "fn", None):  # bare `apex` or only options → chat
        args = parser.parse_args(["chat", *raw])
    for key, value in _defaults().items():
        if not hasattr(args, key):
            setattr(args, key, value)
    try:
        return args.fn(args)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

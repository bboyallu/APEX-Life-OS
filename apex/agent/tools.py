"""Tool registry and built-in tools.

Every tool declares how risky it is (reversibility, blast radius, …).  The
:class:`~apex.agent.governor.Governor` turns that into a risk score and runs it
through the Autonomic Threshold Engine before the tool is allowed to execute.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from apex.agent.messages import ToolSpec
from apex.core.types import Reversibility
from apex.memory.provider import MemoryScope

if TYPE_CHECKING:  # pragma: no cover
    from apex.agent.workspace import Workspace
    from apex.system import ApexSystem

MAX_OUTPUT_CHARS = 8000


class ToolError(Exception):
    """Raised by a tool for an expected, user-correctable failure."""


@dataclass
class ToolContext:
    system: "ApexSystem"
    workspace: "Workspace"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], str]
    reversibility: Reversibility = Reversibility.FULLY_REVERSIBLE
    blast_radius: tuple[str, ...] = ()
    is_novel: bool = False
    affects_safety: bool = False
    #: Optional ``(args) -> error | None`` policy check run *before* authorization.
    precheck: Callable[[dict[str, Any]], str | None] | None = None

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def _require(args: dict[str, Any], key: str, typ: type = str) -> Any:
    if key not in args or args[key] in (None, ""):
        raise ToolError(f"missing required argument '{key}'")
    if not isinstance(args[key], typ):
        raise ToolError(f"argument '{key}' must be {typ.__name__}")
    return args[key]


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


# ---------------------------------------------------------------------------
# calculate — safe arithmetic evaluator (no eval())
# ---------------------------------------------------------------------------

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
    "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression without ``eval``."""
    if len(expression) > 300:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 1000 or (isinstance(a, int) and isinstance(b, int) and abs(a).bit_length() * abs(b) > 20000)):
                raise ToolError("exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ToolError("unsupported expression")

    try:
        return ev(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"math error: {exc}") from exc


def _calculate(args, ctx):
    result = safe_eval(_require(args, "expression"))
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return f"{args['expression']} = {result}"


# ---------------------------------------------------------------------------
# time / status
# ---------------------------------------------------------------------------


def _current_time(args, ctx):
    now = datetime.now(timezone.utc).astimezone()
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z (UTC%z)")


def _system_status(args, ctx):
    s = ctx.system
    ok, msg = s.verify_audit_chain()
    mem = len(s.recall_memories())
    tasks = ctx.workspace.load_tasks()
    open_tasks = sum(1 for t in tasks if not t["done"])
    dead = s.threshold_engine.check_dead_man_switch()
    return (
        f"Audit chain: {'OK' if ok else 'BROKEN'} — {msg}\n"
        f"Safety constraints in force: {len(s.list_safety_constraints())}\n"
        f"Dead-man switch: {'ACTIVE' if dead else 'inactive'}\n"
        f"Memories stored: {mem}\n"
        f"Open tasks: {open_tasks}"
    )


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def _fmt_mem(e) -> str:
    cite = f" (source: {e.citation})" if e.citation else ""
    return f"[{e.entry_id[:8]}] ({e.scope.value}) {e.subject}: {e.fact}{cite}"


def _scope(args) -> MemoryScope | None:
    raw = args.get("scope")
    if not raw:
        return None
    try:
        return MemoryScope(raw)
    except ValueError as exc:
        raise ToolError("scope must be 'repository' or 'user'") from exc


def _remember(args, ctx):
    subject = _require(args, "subject")[:80]
    fact = _require(args, "fact")[:1000]
    e = ctx.system.remember(
        subject.replace("\n", " "), fact.replace("\n", " "),
        scope=_scope(args) or MemoryScope.REPOSITORY,
        citation=str(args.get("citation", ""))[:200],
    )
    return f"Remembered ({e.scope.value}) [{e.entry_id[:8]}]: {e.subject} — {e.fact}"


def _recall(args, ctx):
    query = str(args.get("query") or "").strip()
    scope = _scope(args)
    found = ctx.system.search_memories(query, scope) if query else ctx.system.recall_memories(scope)
    if not found:
        return "No matching memories."
    return "\n".join(_fmt_mem(e) for e in found[-50:])


def _forget(args, ctx):
    prefix = _require(args, "entry_id").strip()
    matches = [e for e in ctx.system.recall_memories() if e.entry_id.startswith(prefix)]
    if not matches:
        raise ToolError(f"no memory with id starting '{prefix}'")
    if len(matches) > 1:
        raise ToolError(f"id prefix '{prefix}' is ambiguous ({len(matches)} matches)")
    ctx.system.forget_memory(matches[0].entry_id)
    return f"Forgot memory [{matches[0].entry_id[:8]}]."


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def _add_task(args, ctx):
    title = _require(args, "title").strip()[:200]
    tasks = ctx.workspace.load_tasks()
    tid = max((t["id"] for t in tasks), default=0) + 1
    tasks.append({
        "id": tid, "title": title, "done": False,
        "due": str(args.get("due") or ""),
        "created": datetime.now(timezone.utc).isoformat(),
    })
    ctx.workspace.save_tasks(tasks)
    return f"Added task #{tid}: {title}"


def _list_tasks(args, ctx):
    tasks = ctx.workspace.load_tasks()
    if not args.get("include_done"):
        tasks = [t for t in tasks if not t["done"]]
    if not tasks:
        return "No open tasks."
    return "\n".join(
        f"#{t['id']} [{'x' if t['done'] else ' '}] {t['title']}" + (f" (due {t['due']})" if t.get("due") else "")
        for t in tasks
    )


def _complete_task(args, ctx):
    tid = _require(args, "task_id", int)
    tasks = ctx.workspace.load_tasks()
    for t in tasks:
        if t["id"] == tid:
            t["done"] = True
            ctx.workspace.save_tasks(tasks)
            return f"Completed task #{tid}: {t['title']}"
    raise ToolError(f"no task #{tid}")


# ---------------------------------------------------------------------------
# files (sandboxed to the workspace)
# ---------------------------------------------------------------------------


def _list_dir(args, ctx):
    p = ctx.workspace.resolve(str(args.get("path") or "."))
    if not p.is_dir():
        raise ToolError(f"not a directory: {args.get('path')}")
    items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    lines = [f"{'d' if i.is_dir() else '-'} {i.name}{'/' if i.is_dir() else ''}" for i in items[:200]]
    return "\n".join(lines) or "(empty)"


def _read_file(args, ctx):
    p = ctx.workspace.resolve(_require(args, "path"))
    if not p.is_file():
        raise ToolError(f"no such file: {args['path']}")
    if p.stat().st_size > 2_000_000:
        raise ToolError("file too large (>2MB)")
    return truncate(p.read_text(encoding="utf-8", errors="replace"))


def _write_file(args, ctx):
    p = ctx.workspace.resolve(_require(args, "path"))
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("argument 'content' must be a string")
    if len(content) > 1_000_000:
        raise ToolError("content too large (>1MB)")
    if p.is_dir():
        raise ToolError("path is a directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    return f"{'Overwrote' if existed else 'Created'} {args['path']} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# shell (opt-in)
# ---------------------------------------------------------------------------

_DENY = [
    r"\brm\s+(-\w*\s+)*-?\w*[rf]\w*\s+(/|~|\*)", r"\bmkfs\b", r"\bdd\s+.*of=/dev/",
    r":\(\)\s*\{", r"\b(shutdown|reboot|halt|poweroff)\b", r"\bchmod\s+-R\s+7?77\s+/",
    r"(curl|wget)[^|]*\|\s*(sudo\s+)?(ba|z)?sh", r">\s*/dev/sd", r"\bsudo\b",
]


def command_is_denied(cmd: str) -> str | None:
    for pat in _DENY:
        if re.search(pat, cmd):
            return pat
    return None


def _command_precheck(args: dict[str, Any]) -> str | None:
    cmd = args.get("command")
    if isinstance(cmd, str) and command_is_denied(cmd):
        return "command blocked by safety policy"
    return None


def _run_command(args, ctx):
    cmd = _require(args, "command")
    if _command_precheck(args):
        raise ToolError("command blocked by safety policy")
    timeout = min(int(args.get("timeout", 30)), 120)
    try:
        proc = subprocess.run(  # noqa: S602
            cmd, shell=True, cwd=ctx.workspace.root, capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"command timed out after {timeout}s") from exc
    out = (proc.stdout or "") + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    return truncate(f"exit code {proc.returncode}\n{out}".strip())


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

_R = Reversibility


def default_tools(*, allow_shell: bool = False, allow_write: bool = True) -> ToolRegistry:
    s = _schema
    tools = [
        Tool("calculate", "Evaluate an arithmetic expression exactly (supports + - * / // % ** and sqrt, sin, log, min, max, …).",
             s({"expression": {"type": "string"}}, ["expression"]), _calculate),
        Tool("current_time", "Get the current local date and time.", s({}), _current_time),
        Tool("system_status", "Report APEX health: audit-chain integrity, safety constraints, memory and task counts.", s({}), _system_status),
        Tool("remember", "Store a durable long-term memory. Use scope 'user' for the user's personal preferences/facts and 'repository' for everything else.",
             s({"subject": {"type": "string", "description": "short topic"}, "fact": {"type": "string"},
                "scope": {"type": "string", "enum": ["repository", "user"]}, "citation": {"type": "string"}},
               ["subject", "fact"]), _remember, blast_radius=("memory",)),
        Tool("recall", "Search long-term memories (empty query lists all).",
             s({"query": {"type": "string"}, "scope": {"type": "string", "enum": ["repository", "user"]}}), _recall),
        Tool("forget", "Delete a memory by id (or unique id prefix).",
             s({"entry_id": {"type": "string"}}, ["entry_id"]), _forget,
             reversibility=_R.PARTIALLY_REVERSIBLE, blast_radius=("memory",)),
        Tool("add_task", "Add a to-do item.",
             s({"title": {"type": "string"}, "due": {"type": "string", "description": "optional due date"}}, ["title"]),
             _add_task, blast_radius=("tasks",)),
        Tool("list_tasks", "List to-do items.", s({"include_done": {"type": "boolean"}}), _list_tasks),
        Tool("complete_task", "Mark a to-do item done.", s({"task_id": {"type": "integer"}}, ["task_id"]),
             _complete_task, blast_radius=("tasks",)),
        Tool("list_dir", "List files in a workspace directory.", s({"path": {"type": "string"}}), _list_dir),
        Tool("read_file", "Read a text file from the workspace.", s({"path": {"type": "string"}}, ["path"]), _read_file),
    ]
    if allow_write:
        tools.append(Tool(
            "write_file", "Create or overwrite a text file in the workspace.",
            s({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
            _write_file, reversibility=_R.PARTIALLY_REVERSIBLE, blast_radius=("workspace", "filesystem"),
            is_novel=True,
        ))
    if allow_shell:
        tools.append(Tool(
            "run_command", "Run a shell command in the workspace directory (requires human approval).",
            s({"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
            _run_command, reversibility=_R.DESTRUCTIVE,
            blast_radius=("workspace", "filesystem", "processes", "network"),
            is_novel=True, affects_safety=True, precheck=_command_precheck,
        ))
    return ToolRegistry(tools)

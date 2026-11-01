"""On-disk home for an APEX agent: memory, audit ledger, tasks, sessions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from apex.agent.messages import Message


def default_home() -> Path:
    return Path(os.environ.get("APEX_HOME", Path.home() / ".apex")).expanduser()


class Workspace:
    """Filesystem layout.

    ``home`` holds APEX's own state; ``root`` is the sandbox the file tools may
    touch (defaults to the current directory).
    """

    def __init__(self, home: str | Path | None = None, root: str | Path | None = None) -> None:
        self.home = Path(home) if home else default_home()
        self.root = Path(root).resolve() if root else Path.cwd().resolve()
        created = not self.home.exists()
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        if created:
            # mkdir's mode is masked by umask and skipped for parents; be explicit.
            # State holds memories and conversations: owner-only, so files inside
            # are unreachable by other local accounts whatever their own mode.
            os.chmod(self.home, 0o700)
        (self.home / "sessions").mkdir(mode=0o700, exist_ok=True)

    @property
    def audit_path(self) -> Path:
        return self.home / "audit.jsonl"

    @property
    def tasks_path(self) -> Path:
        return self.home / "tasks.json"

    def resolve(self, rel: str) -> Path:
        """Resolve ``rel`` inside the sandbox, rejecting escapes."""
        p = (self.root / rel).resolve()
        if p != self.root and not p.is_relative_to(self.root):
            from apex.agent.tools import ToolError

            raise ToolError("path escapes the workspace sandbox")
        return p

    # tasks -----------------------------------------------------------------
    def load_tasks(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def save_tasks(self, tasks: list[dict[str, Any]]) -> None:
        tmp = self.tasks_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        tmp.replace(self.tasks_path)

    # sessions --------------------------------------------------------------
    def session_path(self, name: str) -> Path:
        return self.home / "sessions" / (re.sub(r"[^\w.-]", "_", name) + ".json")

    def load_session(self, name: str) -> list[Message]:
        try:
            data = json.loads(self.session_path(name).read_text(encoding="utf-8"))
            return [Message.from_dict(d) for d in data]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def save_session(self, name: str, history: list[Message]) -> None:
        p = self.session_path(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps([m.to_dict() for m in history]), encoding="utf-8")
        tmp.replace(p)

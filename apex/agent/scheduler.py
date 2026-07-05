"""Cron-style scheduler — recurring agent tasks (stdlib cron parser).

Schedules live in ``~/.apex/schedule.json``::

    [
      {"cron": "0 7 * * *", "action": "knowledge-cycle"},
      {"cron": "*/30 * * * *", "action": "report", "arg": "daily focus"}
    ]

Supported actions: ``cycle``, ``knowledge-cycle``, ``process-knowledge``,
``report`` (with ``arg`` as the query). The daemon evaluates schedules
once per minute; every run is audit-logged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from apex.agent.config import apex_home
from apex.system import ApexSystem

_FIELD_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]  # m h dom mon dow


def _parse_field(field: str, low: int, high: int) -> set[int]:
    """Parse one cron field (supports ``*``, ``*/n``, ``a-b``, lists)."""
    values: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            step = int(step_text)
        if part in ("*", ""):
            start, end = low, high
        elif "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(part)
        for value in range(max(start, low), min(end, high) + 1, step):
            values.add(value)
    return values


def cron_matches(expression: str, moment: datetime) -> bool:
    """Return True if ``moment`` matches the 5-field cron ``expression``."""
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(f"cron expression needs 5 fields: {expression!r}")
    parsed = [
        _parse_field(field, low, high)
        for field, (low, high) in zip(fields, _FIELD_RANGES)
    ]
    checks = [
        moment.minute,
        moment.hour,
        moment.day,
        moment.month,
        (moment.weekday() + 1) % 7,  # cron dow: 0=Sunday … 6=Saturday
    ]
    return all(value in allowed for value, allowed in zip(checks, parsed))


class ScheduledTask(BaseModel):
    cron: str
    action: str
    arg: str = ""


def load_schedule(path: str | Path | None = None) -> list[ScheduledTask]:
    schedule_path = Path(path) if path else apex_home() / "schedule.json"
    if not schedule_path.exists():
        return []
    data = json.loads(schedule_path.read_text(encoding="utf-8"))
    return [ScheduledTask.model_validate(item) for item in data]


class Scheduler:
    """Evaluates due tasks and runs them against an ``ApexSystem``."""

    def __init__(
        self,
        system: ApexSystem,
        *,
        schedule_path: str | Path | None = None,
    ) -> None:
        self.system = system
        self.schedule_path = schedule_path
        self._last_minute: str | None = None

    def run_due(self, moment: datetime | None = None) -> list[str]:
        """Run tasks due at ``moment`` (deduped per minute). Returns actions run."""
        moment = moment or datetime.now()
        minute_key = moment.strftime("%Y-%m-%dT%H:%M")
        if minute_key == self._last_minute:
            return []
        self._last_minute = minute_key

        executed: list[str] = []
        for task in load_schedule(self.schedule_path):
            try:
                due = cron_matches(task.cron, moment)
            except ValueError:
                continue
            if not due:
                continue
            self._run(task)
            executed.append(task.action)
        return executed

    def _run(self, task: ScheduledTask) -> None:
        if task.action == "cycle":
            self.system.run_cycle()
        elif task.action == "knowledge-cycle":
            self.system.run_knowledge_informed_cycle()
        elif task.action == "process-knowledge":
            self.system.process_knowledge()
        elif task.action == "report" and task.arg:
            self.system.generate_knowledge_report(task.arg)
        else:
            return
        self.system.audit_ledger.append(
            "scheduled_task_run",
            actor="scheduler",
            payload={"action": task.action, "cron": task.cron},
        )

"""KnowledgeBridge — wires the KnowledgeVault into the self-evolution loop.

Closes the loop between the two "brains" of APEX:

    KnowledgeVault → Knowledge Signals → Analyzer → Planner → Executor
        → AuditLedger → KnowledgeVault

1. **Extract** — compiled wiki articles are scanned for ``signal:``
   directives.  Each becomes a :class:`KnowledgeSignal` that the Analyzer
   treats as an evolution candidate, so APEX can *evolve because of what it
   learned*.
2. **Record** — after a MAPE-K cycle runs, the outcome is written back into
   ``raw/`` as an evolution-history note, which the vault folds into the
   wiki on the next ``process_raw()``.  APEX then *learns from how it
   evolved*.

Signal directive syntax (one per line, anywhere in raw material)::

    signal: <target-component> :: <description> [severity]

``severity`` is optional and must be one of the :class:`~apex.core.types.Severity`
values (``informational``, ``warning``, ``degraded``, ``critical``,
``catastrophic``); it defaults to ``warning``.  Example::

    signal: api_gateway :: retry storm is amplifying latency [degraded]

Extraction is deduplicated: a signal already seen (persisted in
``wiki/.bridge_state.json``) is not emitted again, so re-running cycles does
not repeatedly fire the same knowledge signal.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from apex.core.types import AnalysisReport, Severity
from apex.knowledge.vault import KnowledgeVault

#: Topic under which evolution-cycle outcomes are filed in the vault.
EVOLUTION_TOPIC = "apex-evolution-history"

_SIGNAL_RE = re.compile(
    r"^\s*signal\s*:\s*(?P<target>[\w.-]+)\s*::\s*(?P<description>.+?)"
    r"(?:\s*\[(?P<severity>[a-zA-Z]+)\])?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class KnowledgeSignal(BaseModel):
    """An actionable directive derived from compiled knowledge."""

    target: str
    description: str
    severity: Severity = Severity.WARNING
    source: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def digest(self) -> str:
        """Stable identity used for deduplication."""
        key = f"{self.target}::{self.description}::{self.severity.value}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KnowledgeBridge:
    """Bidirectional bridge between the :class:`KnowledgeVault` and MAPE-K.

    Parameters
    ----------
    vault:
        The knowledge vault to read signals from and record outcomes to.
    """

    def __init__(self, vault: KnowledgeVault) -> None:
        self._vault = vault

    # ------------------------------------------------------------------
    # KnowledgeVault → Analyzer
    # ------------------------------------------------------------------

    def extract_signals(self) -> list[KnowledgeSignal]:
        """Scan the compiled wiki for new ``signal:`` directives.

        The evolution-history article and the wiki index are excluded so the
        bridge's own output can never feed back as a new signal.  Signals
        already emitted in a previous extraction are skipped.
        """
        state = self._load_state()
        seen: set[str] = set(state.get("seen_signals", []))
        signals: list[KnowledgeSignal] = []

        wiki_dir = self._vault.wiki_dir
        if not wiki_dir.is_dir():
            return signals

        excluded = {"index.md", f"{EVOLUTION_TOPIC}.md"}
        for path in sorted(wiki_dir.glob("*.md")):
            if path.name in excluded:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in _SIGNAL_RE.finditer(text):
                severity_raw = (match.group("severity") or "").lower()
                try:
                    severity = Severity(severity_raw) if severity_raw else Severity.WARNING
                except ValueError:
                    severity = Severity.WARNING
                signal = KnowledgeSignal(
                    target=match.group("target"),
                    description=match.group("description").strip(),
                    severity=severity,
                    source=f"wiki/{path.name}",
                )
                if signal.digest in seen:
                    continue
                seen.add(signal.digest)
                signals.append(signal)

        state["seen_signals"] = sorted(seen)
        self._save_state(state)
        return signals

    # ------------------------------------------------------------------
    # Executor / AuditLedger → KnowledgeVault
    # ------------------------------------------------------------------

    def record_cycle(
        self,
        cycle: int,
        report: AnalysisReport,
        signals: list[KnowledgeSignal] | None = None,
    ) -> Path:
        """Append the outcome of a MAPE-K cycle to the evolution-history note.

        Written into ``raw/`` so the next ``process_raw()`` folds it into the
        wiki — APEX learns from its own evolution.  The note deliberately
        contains no ``signal:`` directives, so it can never re-trigger itself.
        """
        signals = signals or []
        raw_dir = self._vault.raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{EVOLUTION_TOPIC}.md"

        if not path.exists():
            path.write_text(
                f"topic: {EVOLUTION_TOPIC}\n\n# APEX Evolution History\n\n"
                "> Written automatically by the KnowledgeBridge after each\n"
                "> knowledge-informed MAPE-K cycle.\n",
                encoding="utf-8",
            )

        lines = [
            "",
            f"## Cycle {cycle} — {datetime.now(timezone.utc).isoformat()}",
            "",
            f"- Overall severity: {report.overall_severity.value}",
            f"- Symptom clusters: {len(report.symptom_clusters)}",
            f"- Evolution targets: "
            f"{', '.join(sorted(report.proposed_evolution_targets)) or '(none)'}",
        ]
        if signals:
            lines.append("- Knowledge-derived triggers:")
            lines += [
                f"  - {s.target} — {s.description} "
                f"({s.severity.value}; from {s.source})"
                for s in signals
            ]
        lines.append("")

        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return path

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    @property
    def _state_path(self) -> Path:
        return self._vault.wiki_dir / ".bridge_state.json"

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"seen_signals": []}

    def _save_state(self, state: dict) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

"""Governed tools the agent can call.

Every tool carries a risk profile scored by the existing ``RiskScorer``.
Tools at or above L3 (high-risk) require human approval before running —
this governance gate is what makes APEX safer than ungoverned agents like
Hermes. Every invocation lands on the tamper-evident audit ledger.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from apex.core.types import Reversibility, ThresholdLevel
from apex.system import ApexSystem

ToolFunc = Callable[..., str]
ApprovalCallback = Callable[[str, str, ThresholdLevel], bool]


@dataclass
class Tool:
    """A callable capability exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc
    reversibility: Reversibility = Reversibility.FULLY_REVERSIBLE
    blast_radius: int = 1
    is_novel: bool = False
    affects_safety: bool = False

    def spec(self) -> dict[str, Any]:
        """OpenAI tool-calling function spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [
                        k
                        for k, v in self.parameters.items()
                        if not v.get("optional")
                    ],
                },
            },
        }


@dataclass
class ToolRegistry:
    """Executes tools under APEX governance.

    Parameters
    ----------
    system:
        The ``ApexSystem`` whose risk scorer, threshold engine, and audit
        ledger gate every tool call.
    approval_callback:
        Called for L3+ tool invocations: ``(tool_name, summary, level) ->
        bool``. Defaults to denying (safe default for headless runs).
    """

    system: ApexSystem
    approval_callback: ApprovalCallback | None = None
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Risk-score, gate, audit, and run one tool call."""
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"

        risk = self.system.risk_scorer.score(
            reversibility=tool.reversibility,
            blast_radius_count=tool.blast_radius,
            is_novel=tool.is_novel,
            affects_safety=tool.affects_safety,
            confidence_deficit=0.0,
        )
        level = self.system.risk_scorer.threshold_level(risk)

        if level in (ThresholdLevel.L3_HIGH_RISK, ThresholdLevel.L4_CRITICAL):
            summary = f"{name}({arguments})"
            approved = bool(
                self.approval_callback
                and self.approval_callback(name, summary, level)
            )
            self.system.audit_ledger.append(
                "tool_approval_decision",
                actor="agent",
                payload={
                    "tool": name,
                    "level": level.value,
                    "risk": risk,
                    "approved": approved,
                },
            )
            if not approved:
                return (
                    f"denied: {name} is {level.value} (risk {risk}) and was "
                    "not approved by a human operator"
                )

        try:
            result = tool.func(**arguments)
        except TypeError as exc:
            result = f"error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM
            result = f"error: {exc}"

        self.system.audit_ledger.append(
            "tool_executed",
            actor="agent",
            payload={"tool": name, "risk": risk, "level": level.value},
        )
        return result


def _run_shell(command: str, timeout: int = 60) -> str:
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout + completed.stderr).strip()
    return f"exit={completed.returncode}\n{output[:4000]}"


def build_default_tools(
    system: ApexSystem,
    *,
    approval_callback: ApprovalCallback | None = None,
    skill_store: Any = None,
    session_store: Any = None,
) -> ToolRegistry:
    """Wire APEX capabilities into a governed tool registry."""
    registry = ToolRegistry(system=system, approval_callback=approval_callback)

    registry.register(
        Tool(
            name="search_knowledge",
            description=(
                "Search the personal knowledge base (wiki compiled from raw/ "
                "notes) and long-term memories for a topic."
            ),
            parameters={"query": {"type": "string"}},
            func=lambda query: _search_knowledge(system, query),
        )
    )
    registry.register(
        Tool(
            name="generate_report",
            description=(
                "Generate a knowledge report answering a question; the "
                "answer is saved into outputs/ and its path returned."
            ),
            parameters={"query": {"type": "string"}},
            func=lambda query: system.generate_knowledge_report(query),
        )
    )
    registry.register(
        Tool(
            name="remember",
            description="Store a durable fact in APEX long-term memory.",
            parameters={
                "subject": {"type": "string"},
                "fact": {"type": "string"},
            },
            func=lambda subject, fact: (
                f"stored memory {system.remember(subject, fact).entry_id}"
            ),
        )
    )
    registry.register(
        Tool(
            name="search_memories",
            description="Search APEX long-term memory for stored facts.",
            parameters={"query": {"type": "string"}},
            func=lambda query: (
                "\n".join(
                    f"- [{e.subject}] {e.fact}"
                    for e in system.search_memories(query)
                )
                or "no memories found"
            ),
        )
    )
    registry.register(
        Tool(
            name="publish_metric",
            description="Publish a telemetry metric to the APEX monitor.",
            parameters={
                "source": {"type": "string"},
                "metric": {"type": "string"},
                "value": {"type": "number"},
            },
            func=lambda source, metric, value: (
                system.publish_metric(source, metric, float(value))
                or f"published {metric}={value} from {source}"
            ),
        )
    )
    registry.register(
        Tool(
            name="run_evolution_cycle",
            description=(
                "Run one knowledge-informed MAPE-K self-evolution cycle "
                "(monitor → analyze → plan → execute under governance)."
            ),
            parameters={},
            func=lambda: (
                "cycle severity="
                + system.run_knowledge_informed_cycle().overall_severity.value
            ),
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            blast_radius=3,
        )
    )
    registry.register(
        Tool(
            name="run_shell",
            description=(
                "Run a shell command on the host. HIGH RISK: requires human "
                "approval before execution."
            ),
            parameters={"command": {"type": "string"}},
            func=_run_shell,
            reversibility=Reversibility.DESTRUCTIVE,
            blast_radius=8,
            is_novel=True,
            affects_safety=True,
        )
    )
    if skill_store is not None:
        registry.register(
            Tool(
                name="save_skill",
                description=(
                    "Save a reusable skill (named procedure with steps) "
                    "learned from this conversation for future sessions."
                ),
                parameters={
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                func=lambda name, description, steps: (
                    f"saved skill to {skill_store.save(_make_skill(name, description, steps))}"
                ),
            )
        )
        registry.register(
            Tool(
                name="use_skill",
                description="Recall a previously saved skill by name.",
                parameters={"name": {"type": "string"}},
                func=lambda name: _use_skill(skill_store, name),
            )
        )
    if session_store is not None:
        registry.register(
            Tool(
                name="search_past_conversations",
                description=(
                    "Full-text search across all past conversations from "
                    "every interface (terminal, Telegram, dashboard)."
                ),
                parameters={"query": {"type": "string"}},
                func=lambda query: (
                    "\n".join(
                        f"- ({m.role}) {m.content[:200]}"
                        for m in session_store.search(query)
                    )
                    or "no matching past messages"
                ),
            )
        )
    return registry


def _make_skill(name: str, description: str, steps: list[str]):
    from apex.agent.skills import Skill

    return Skill(name=name, description=description, steps=list(steps))


def _use_skill(skill_store: Any, name: str) -> str:
    skill = skill_store.use(name)
    if skill is None:
        return f"no skill named {name!r}"
    return skill.to_markdown()


def _search_knowledge(system: ApexSystem, query: str) -> str:
    matches = system.knowledge_vault.search(query)
    memory_hits = system.search_memories(query)
    lines = [f"- [{topic}] {snippet[:200]}" for topic, snippet in matches[:5]]
    lines += [f"- [memory:{e.subject}] {e.fact}" for e in memory_hits[:5]]
    return "\n".join(lines) or "no knowledge found"

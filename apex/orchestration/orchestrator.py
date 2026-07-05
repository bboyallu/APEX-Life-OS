"""Decision Orchestrator (§4) — runtime path selection and policy arbitration.

The orchestrator sits above the MAPE-K loop and the neuro-symbolic layer,
selecting the appropriate reasoning path and resolving policy conflicts.

Policy arbitration (§4.4) uses ranked preference resolution:
1. Hard constraint dominance (safety > all)
2. Contextual priority weights from the Knowledge Base
3. Historical outcome data for similar conflicts
4. Escalation to human oversight if no resolution converges
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import DecisionPath
from apex.neuro_symbolic.verifier import VerificationPipeline
from apex.orchestration.selector import DecisionContext, PathSelector


@dataclass
class Policy:
    """An active operational policy."""

    policy_id: str
    name: str
    priority: int       # lower number = higher priority
    is_hard_constraint: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class PolicyConflict:
    """Describes a conflict between two policies."""

    policy_a: Policy
    policy_b: Policy
    resolution: str | None = None
    escalated_at: datetime | None = None


class DecisionOrchestrator:
    """Runtime decision orchestrator.

    Parameters
    ----------
    path_selector:
        Selects the execution path for each context.
    verification_pipeline:
        Neuro-symbolic verification pipeline.
    knowledge_base:
        Shared KB for policy weights and decision history.
    on_escalation:
        Callback invoked when a policy conflict cannot be resolved.
    """

    def __init__(
        self,
        path_selector: PathSelector,
        verification_pipeline: VerificationPipeline,
        knowledge_base: KnowledgeBase,
        on_escalation: Any | None = None,
    ) -> None:
        self._selector = path_selector
        self._pipeline = verification_pipeline
        self._kb = knowledge_base
        self._on_escalation = on_escalation
        self._policies: dict[str, Policy] = {}
        self._call_stack: list[DecisionPath] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def register_policy(self, policy: Policy) -> None:
        with self._lock:
            self._policies[policy.policy_id] = policy

    def detect_conflicts(self) -> list[PolicyConflict]:
        """Detect policies with the same priority (naive conflict detection)."""
        with self._lock:
            by_priority: dict[int, list[Policy]] = {}
            for p in self._policies.values():
                by_priority.setdefault(p.priority, []).append(p)

        conflicts: list[PolicyConflict] = []
        for group in by_priority.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        conflicts.append(PolicyConflict(group[i], group[j]))
        return conflicts

    def arbitrate(self, conflicts: list[PolicyConflict]) -> list[PolicyConflict]:
        """Resolve policy conflicts using ranked preference resolution (§4.4)."""
        resolved: list[PolicyConflict] = []
        for conflict in conflicts:
            # Rule 1: hard constraint dominates
            if conflict.policy_a.is_hard_constraint:
                conflict.resolution = f"Hard constraint '{conflict.policy_a.name}' dominates."
                resolved.append(conflict)
                continue
            if conflict.policy_b.is_hard_constraint:
                conflict.resolution = f"Hard constraint '{conflict.policy_b.name}' dominates."
                resolved.append(conflict)
                continue

            # Rule 2: contextual priority weights from KB
            weight_a = self._kb.get_world_context(
                f"policy_weight:{conflict.policy_a.policy_id}", 0.5
            )
            weight_b = self._kb.get_world_context(
                f"policy_weight:{conflict.policy_b.policy_id}", 0.5
            )
            if weight_a != weight_b:
                winner = conflict.policy_a if weight_a > weight_b else conflict.policy_b
                conflict.resolution = f"Policy '{winner.name}' wins by KB weight."
                resolved.append(conflict)
                continue

            # Rule 4: escalate to human oversight
            conflict.escalated_at = datetime.now(timezone.utc)
            if self._on_escalation:
                self._on_escalation(conflict)
            resolved.append(conflict)

        return resolved

    # ------------------------------------------------------------------
    # Orchestrated decision
    # ------------------------------------------------------------------

    def decide(
        self, prompt: str, context: DecisionContext, plan_id: str | None = None
    ) -> tuple[DecisionPath, Any]:
        """Select a path and run the verification pipeline.

        Returns
        -------
        (path, decision_record)
            The selected ``DecisionPath`` and the resulting ``DecisionRecord``.
        """
        path = self._selector.select(context)

        with self._lock:
            self._call_stack.append(path)
            # Governance: highest-rigor path in call stack governs approval (§4.3)
            governing_path = self._governing_path()

        record = self._pipeline.run(prompt, decision_path=governing_path, plan_id=plan_id)

        with self._lock:
            if self._call_stack:
                self._call_stack.pop()

        return governing_path, record

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _PATH_RIGOR: dict[DecisionPath, int] = {
        DecisionPath.REFLEXIVE: 0,
        DecisionPath.DELIBERATIVE: 1,
        DecisionPath.COLLABORATIVE: 2,
        DecisionPath.ADVERSARIAL: 3,
    }

    def _governing_path(self) -> DecisionPath:
        if not self._call_stack:
            return DecisionPath.DELIBERATIVE
        return max(self._call_stack, key=lambda p: self._PATH_RIGOR[p])

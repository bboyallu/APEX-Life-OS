"""Governor — puts every tool call under APEX's safety machinery.

For each requested tool call the governor:

1. builds an :class:`AdaptationPlan` describing the action,
2. scores its risk with the :class:`RiskScorer`,
3. evaluates it with the :class:`AutonomicThresholdEngine`
   (L0–L1 autonomous, L2 notify, L3 needs human approval, L4 blocked),
4. records the verdict in the tamper-evident audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from apex.agent.tools import Tool
from apex.core.types import AdaptationPlan, AdaptationType, ThresholdLevel
from apex.system import ApexSystem

#: ``(ApprovalRequest) -> bool``
Approver = Callable[["ApprovalRequest"], bool]


@dataclass
class ApprovalRequest:
    tool: str
    arguments: dict[str, Any]
    level: ThresholdLevel
    risk_score: float
    reason: str


@dataclass
class Verdict:
    allowed: bool
    level: ThresholdLevel
    risk_score: float
    reason: str
    notified: bool = False


class Governor:
    def __init__(self, system: ApexSystem, approver: Approver | None = None) -> None:
        self.system = system
        self.approver = approver

    def authorize(self, tool: Tool, arguments: dict[str, Any]) -> Verdict:
        s = self.system
        risk = s.risk_scorer.score(
            reversibility=tool.reversibility,
            blast_radius_count=len(tool.blast_radius),
            is_novel=tool.is_novel,
            affects_safety=tool.affects_safety,
            confidence_deficit=0.1,
        )
        plan = AdaptationPlan(
            description=f"Agent tool call: {tool.name}",
            adaptation_type=AdaptationType.MICRO,
            expected_benefit=1.0,
            risk_score=risk,
            reversibility=tool.reversibility,
            blast_radius=list(tool.blast_radius),
            requires_human_approval=risk >= 0.60,
            metadata={"tool": tool.name},
        )
        decision = s.threshold_engine.evaluate(plan)
        level = decision.level
        allowed, reason = True, decision.reason
        notified = level == ThresholdLevel.L2_SIGNIFICANT

        if decision.is_hard_blocked:
            allowed = False
        elif decision.requires_approval:
            req = ApprovalRequest(tool.name, arguments, level, risk, decision.reason)
            allowed = bool(self.approver and self.approver(req))
            reason = "approved by human" if allowed else "human approval not granted"

        s.audit_ledger.append(
            "tool_authorization",
            actor="governor",
            payload={
                "tool": tool.name,
                "level": level.value,
                "risk": risk,
                "allowed": allowed,
                "reason": reason,
            },
        )
        return Verdict(allowed, level, risk, reason, notified)

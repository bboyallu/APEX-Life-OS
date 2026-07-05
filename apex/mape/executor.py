"""Execute phase (§2.4) — enacts approved adaptation plans.

Rollout stages:
1. Shadow mode  — new behaviour runs in parallel, live output unchanged.
2. Canary       — gradual traffic shifting with live rollback triggers.
3. Atomic commit — all-or-nothing state transition.

Every executed change is recorded in the ``KnowledgeBase`` and in an
immutable audit log entry (``ExecutionResult``).  Automatic rollback fires
within one observation window if post-execution metrics worsen beyond the
configured delta.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import (
    AdaptationPlan,
    ExecutionResult,
    ExecutionStatus,
    ThresholdLevel,
)
from apex.thresholds.risk_scorer import RiskScorer


class ExecutionBlockedError(Exception):
    """Raised when a plan is blocked from execution due to threshold or approval rules."""


class Executor:
    """MAPE-K Execute phase.

    Parameters
    ----------
    knowledge_base:
        Shared KB for recording results.
    risk_scorer:
        Used to determine the threshold level of a plan.
    rollback_delta:
        Post-execution metric worsening fraction that triggers auto-rollback.
    on_shadow_run:
        Optional hook called during shadow mode with the plan.
    on_canary_run:
        Optional hook called during canary mode with the plan.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        risk_scorer: RiskScorer | None = None,
        rollback_delta: float = 0.10,
        on_shadow_run: Callable[[AdaptationPlan], None] | None = None,
        on_canary_run: Callable[[AdaptationPlan], None] | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._risk_scorer = risk_scorer or RiskScorer()
        self._rollback_delta = rollback_delta
        self._on_shadow_run = on_shadow_run
        self._on_canary_run = on_canary_run
        self._executed: list[ExecutionResult] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: AdaptationPlan,
        *,
        approved: bool = False,
        baseline_metric: float | None = None,
        post_metric: float | None = None,
    ) -> ExecutionResult:
        """Execute an adaptation plan through the staged rollout.

        Parameters
        ----------
        plan:
            The plan to enact.
        approved:
            Set to ``True`` when explicit human approval has been granted.
        baseline_metric:
            Optional pre-execution metric value used for auto-rollback logic.
        post_metric:
            Optional post-execution metric value.  If provided and the ratio
            (post − baseline) / baseline exceeds ``rollback_delta``, automatic
            rollback is triggered.
        """
        threshold_level = self._risk_scorer.threshold_level(plan.risk_score)

        # Safety gate: L3+ requires explicit approval
        if plan.requires_human_approval and not approved:
            if threshold_level in (ThresholdLevel.L3_HIGH_RISK, ThresholdLevel.L4_CRITICAL):
                result = self._make_result(plan, ExecutionStatus.BLOCKED, "Blocked: human approval required.")
                self._store(plan, result)
                raise ExecutionBlockedError(
                    f"Plan {plan.plan_id} requires human approval "
                    f"(threshold={threshold_level.value})"
                )

        # Stage 1: Shadow
        if self._on_shadow_run:
            self._on_shadow_run(plan)

        # Stage 2: Canary
        if self._on_canary_run:
            self._on_canary_run(plan)

        # Stage 3: Atomic commit — decide final status
        should_rollback = self._check_rollback(baseline_metric, post_metric)

        if should_rollback:
            result = self._make_result(
                plan, ExecutionStatus.ROLLED_BACK, "Auto-rollback: post-execution metric worsened."
            )
        else:
            result = self._make_result(plan, ExecutionStatus.COMMITTED, "")

        self._store(plan, result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_rollback(
        self, baseline: float | None, post: float | None
    ) -> bool:
        if baseline is None or post is None:
            return False
        if baseline == 0:
            return False
        worsening = (post - baseline) / abs(baseline)
        return worsening > self._rollback_delta

    @staticmethod
    def _make_result(
        plan: AdaptationPlan, status: ExecutionStatus, notes: str
    ) -> ExecutionResult:
        signature = hashlib.sha256(
            json.dumps(
                {
                    "plan_id": plan.plan_id,
                    "status": status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return ExecutionResult(
            plan_id=plan.plan_id,
            status=status,
            signature=signature,
            rollback_available=status != ExecutionStatus.ROLLED_BACK,
            notes=notes,
        )

    def _store(self, plan: AdaptationPlan, result: ExecutionResult) -> None:
        self._executed.append(result)
        self._kb.record_adaptation(plan, result)

    def get_results(self) -> list[ExecutionResult]:
        return list(self._executed)

"""Autonomic Threshold Engine (§5) — governs autonomous vs. supervised operation.

The engine evaluates adaptation plans against configurable threshold levels and
determines whether a plan may proceed autonomously, needs async notification,
or requires synchronous human approval.

Threshold taxonomy (§5.1)::

    L0 — Routine     (0.00–0.15) — fully autonomous
    L1 — Notable     (0.15–0.35) — autonomous + logged
    L2 — Significant (0.35–0.60) — async human notification
    L3 — High-Risk   (0.60–0.80) — synchronous human approval
    L4 — Critical    (0.80–1.00) — immediate alert + hard block

Dead-man switch (§5.4): if the oversight interface is unreachable for longer
than ``dead_man_timeout_seconds``, all L2+ evolutions are suspended.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

from apex.core.types import AdaptationPlan, ThresholdLevel
from apex.thresholds.risk_scorer import RiskScorer


class ThresholdDecision:
    """Outcome of a threshold evaluation."""

    def __init__(
        self,
        plan: AdaptationPlan,
        level: ThresholdLevel,
        may_proceed_autonomously: bool,
        requires_approval: bool,
        is_hard_blocked: bool,
        reason: str,
    ) -> None:
        self.plan = plan
        self.level = level
        self.may_proceed_autonomously = may_proceed_autonomously
        self.requires_approval = requires_approval
        self.is_hard_blocked = is_hard_blocked
        self.reason = reason
        self.evaluated_at = datetime.now(timezone.utc)


class AutonomicThresholdEngine:
    """Evaluates adaptation plans against threshold levels.

    Parameters
    ----------
    risk_scorer:
        Scorer used to determine the threshold level.
    dead_man_timeout_seconds:
        Seconds without oversight contact before L2+ plans are suspended.
        Default is 900 (15 minutes) as per §5.4.
    on_l3_alert, on_l4_alert:
        Callbacks invoked when L3 / L4 plans are encountered.
    """

    def __init__(
        self,
        risk_scorer: RiskScorer | None = None,
        dead_man_timeout_seconds: int = 900,
        on_l3_alert: Callable[[AdaptationPlan, ThresholdLevel], None] | None = None,
        on_l4_alert: Callable[[AdaptationPlan, ThresholdLevel], None] | None = None,
    ) -> None:
        self._scorer = risk_scorer or RiskScorer()
        self._dead_man_timeout = timedelta(seconds=dead_man_timeout_seconds)
        self._on_l3_alert = on_l3_alert
        self._on_l4_alert = on_l4_alert
        self._last_oversight_contact: datetime = datetime.now(timezone.utc)
        self._dead_man_active = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def heartbeat(self) -> None:
        """Record that the oversight interface is reachable."""
        with self._lock:
            self._last_oversight_contact = datetime.now(timezone.utc)
            self._dead_man_active = False

    def check_dead_man_switch(self) -> bool:
        """Return ``True`` if the dead-man switch is currently active."""
        with self._lock:
            if datetime.now(timezone.utc) - self._last_oversight_contact > self._dead_man_timeout:
                self._dead_man_active = True
            return self._dead_man_active

    def evaluate(self, plan: AdaptationPlan) -> ThresholdDecision:
        """Evaluate a plan and return a ``ThresholdDecision``."""
        level = self._scorer.threshold_level(plan.risk_score)
        dead_man = self.check_dead_man_switch()

        if level == ThresholdLevel.L0_ROUTINE:
            return ThresholdDecision(
                plan=plan,
                level=level,
                may_proceed_autonomously=True,
                requires_approval=False,
                is_hard_blocked=False,
                reason="Routine adaptation — fully autonomous.",
            )

        if level == ThresholdLevel.L1_NOTABLE:
            return ThresholdDecision(
                plan=plan,
                level=level,
                may_proceed_autonomously=True,
                requires_approval=False,
                is_hard_blocked=False,
                reason="Notable adaptation — autonomous + logged.",
            )

        if level == ThresholdLevel.L2_SIGNIFICANT:
            if dead_man:
                return ThresholdDecision(
                    plan=plan,
                    level=level,
                    may_proceed_autonomously=False,
                    requires_approval=False,
                    is_hard_blocked=True,
                    reason="Dead-man switch active — L2+ evolutions suspended.",
                )
            return ThresholdDecision(
                plan=plan,
                level=level,
                may_proceed_autonomously=True,
                requires_approval=False,
                is_hard_blocked=False,
                reason="Significant adaptation — async human notification sent.",
            )

        if level == ThresholdLevel.L3_HIGH_RISK:
            if self._on_l3_alert:
                self._on_l3_alert(plan, level)
            return ThresholdDecision(
                plan=plan,
                level=level,
                may_proceed_autonomously=False,
                requires_approval=True,
                is_hard_blocked=dead_man,
                reason="High-risk adaptation — synchronous human approval required.",
            )

        # L4_CRITICAL
        if self._on_l4_alert:
            self._on_l4_alert(plan, level)
        return ThresholdDecision(
            plan=plan,
            level=level,
            may_proceed_autonomously=False,
            requires_approval=True,
            is_hard_blocked=True,
            reason="Critical adaptation — immediate alert issued; hard block active.",
        )

    def bulk_evaluate(self, plans: list[AdaptationPlan]) -> list[ThresholdDecision]:
        return [self.evaluate(p) for p in plans]

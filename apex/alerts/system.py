"""High-Priority Alert System (§6) — initiates outbound contact on L3/L4 events.

Trigger conditions (§6.1):
- Risk score crosses L3 threshold (≥ 0.60)
- Symbolic verifier produces a hard-constraint refutation
- Self-integrity check failure
- Cascading failure exceeding blast-radius predictions
- Dead-man switch approaching timeout with L3+ plans pending

Approval & timeout behaviour (§6.4)::

    L2 notification  → 4-hour window  → auto-approve if no veto
    L3 approval req  → 30-minute window → auto-deny if no response
    L4 critical alert → 10-minute window → auto-deny + system freeze
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

from apex.alerts.channels import AlertChannels, DeliveryReceipt
from apex.core.types import (
    AdaptationPlan,
    AlertPayload,
    Reversibility,
    ThresholdLevel,
    VerificationResult,
)
from apex.neuro_symbolic.symbolic import SymbolicVerdict


# Timeout windows per severity level (§6.4)
_TIMEOUT_WINDOWS: dict[ThresholdLevel, timedelta] = {
    ThresholdLevel.L2_SIGNIFICANT: timedelta(hours=4),
    ThresholdLevel.L3_HIGH_RISK: timedelta(minutes=30),
    ThresholdLevel.L4_CRITICAL: timedelta(minutes=10),
}

_AUTO_APPROVE_ON_TIMEOUT = {ThresholdLevel.L2_SIGNIFICANT}


def _auto_deny_on_timeout(level: ThresholdLevel) -> bool:
    return level in (ThresholdLevel.L3_HIGH_RISK, ThresholdLevel.L4_CRITICAL)


class PendingApproval:
    """Tracks an outstanding approval request."""

    def __init__(self, payload: AlertPayload) -> None:
        self.payload = payload
        self.approved: bool | None = None
        self.resolved_at: datetime | None = None
        self._event = threading.Event()

    def approve(self) -> None:
        self.approved = True
        self.resolved_at = datetime.now(timezone.utc)
        self._event.set()

    def deny(self) -> None:
        self.approved = False
        self.resolved_at = datetime.now(timezone.utc)
        self._event.set()

    def wait(self, timeout_seconds: float) -> bool:
        """Block until resolved or timeout.  Returns True if resolved."""
        return self._event.wait(timeout=timeout_seconds)


class AlertSystem:
    """High-priority alert system.

    Parameters
    ----------
    channels:
        Configured ``AlertChannels`` instance.
    on_freeze:
        Callback invoked when an L4 event triggers a system freeze.
    """

    def __init__(
        self,
        channels: AlertChannels,
        on_freeze: Callable[[], None] | None = None,
    ) -> None:
        self._channels = channels
        self._on_freeze = on_freeze
        self._lock = threading.Lock()
        self._pending: dict[str, PendingApproval] = {}
        self._receipts: list[DeliveryReceipt] = []
        self._frozen = False

    # ------------------------------------------------------------------
    # Primary alert trigger
    # ------------------------------------------------------------------

    def raise_alert(
        self,
        plan: AdaptationPlan,
        level: ThresholdLevel,
        *,
        proof_trace_url: str | None = None,
    ) -> PendingApproval:
        """Build and dispatch an alert, returning a ``PendingApproval`` handle."""
        timeout_window = _TIMEOUT_WINDOWS.get(level, timedelta(minutes=30))
        auto_deny_at = datetime.now(timezone.utc) + timeout_window

        payload = AlertPayload(
            severity=level,
            evolution_summary=plan.description,
            risk_score=plan.risk_score,
            affected_components=plan.blast_radius,
            reversibility=plan.reversibility,
            proof_trace_url=proof_trace_url,
            auto_deny_at=auto_deny_at,
        )

        pending = PendingApproval(payload)

        with self._lock:
            self._pending[payload.alert_id] = pending

        use_voice = level == ThresholdLevel.L4_CRITICAL
        receipts = self._channels.dispatch(payload, use_voice=use_voice)

        with self._lock:
            self._receipts.extend(receipts)

        if level == ThresholdLevel.L4_CRITICAL:
            self._trigger_freeze()

        return pending

    def raise_integrity_alert(self, component: str, detail: str) -> PendingApproval:
        """Convenience method for self-integrity check failures (§6.1)."""
        synthetic_plan = AdaptationPlan(
            description=f"Self-integrity failure in {component}: {detail}",
            adaptation_type=__import__("apex.core.types", fromlist=["AdaptationType"]).AdaptationType.MACRO,
            expected_benefit=0.0,
            risk_score=0.90,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            blast_radius=[component],
            requires_human_approval=True,
        )
        return self.raise_alert(synthetic_plan, ThresholdLevel.L4_CRITICAL)

    def raise_verification_alert(
        self, verdict: SymbolicVerdict, prompt_summary: str
    ) -> PendingApproval | None:
        """Raise an alert when a hard-constraint refutation occurs (§6.1)."""
        if verdict.result != VerificationResult.REFUTED:
            return None
        synthetic_plan = AdaptationPlan(
            description=(
                f"Hard-constraint refutation: {prompt_summary}. "
                f"Violated: {', '.join(verdict.violated_rules)}"
            ),
            adaptation_type=__import__("apex.core.types", fromlist=["AdaptationType"]).AdaptationType.MACRO,
            expected_benefit=0.0,
            risk_score=0.85,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            blast_radius=verdict.violated_rules,
            requires_human_approval=True,
        )
        return self.raise_alert(synthetic_plan, ThresholdLevel.L4_CRITICAL)

    # ------------------------------------------------------------------
    # Approval / denial API
    # ------------------------------------------------------------------

    def respond(self, alert_id: str, *, approve: bool) -> bool:
        """Record a human approval or denial for an alert.

        Returns ``True`` if the alert was found; ``False`` otherwise.
        """
        with self._lock:
            pending = self._pending.get(alert_id)
        if pending is None:
            return False
        if approve:
            pending.approve()
        else:
            pending.deny()
        return True

    def process_timeouts(self) -> None:
        """Apply auto-approve / auto-deny to timed-out pending alerts."""
        now = datetime.now(timezone.utc)
        with self._lock:
            pending_items = list(self._pending.items())

        for alert_id, pending in pending_items:
            if pending.approved is not None:
                continue
            if pending.payload.auto_deny_at and now >= pending.payload.auto_deny_at:
                level = pending.payload.severity
                if level in _AUTO_APPROVE_ON_TIMEOUT:
                    pending.approve()
                else:
                    pending.deny()

    # ------------------------------------------------------------------
    # System freeze (L4)
    # ------------------------------------------------------------------

    def _trigger_freeze(self) -> None:
        with self._lock:
            self._frozen = True
        if self._on_freeze:
            self._on_freeze()

    @property
    def is_frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_receipts(self) -> list[DeliveryReceipt]:
        with self._lock:
            return list(self._receipts)

    def get_pending(self) -> dict[str, PendingApproval]:
        with self._lock:
            return dict(self._pending)

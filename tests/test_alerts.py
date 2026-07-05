"""Tests for the High-Priority Alert System."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from apex.alerts.channels import AlertChannels, DeliveryReceipt
from apex.alerts.system import AlertSystem
from apex.core.types import (
    AdaptationPlan,
    AdaptationType,
    Reversibility,
    ThresholdLevel,
)


def _make_plan(risk_score: float = 0.70) -> AdaptationPlan:
    return AdaptationPlan(
        description="Alert test plan",
        adaptation_type=AdaptationType.MESO,
        expected_benefit=0.3,
        risk_score=risk_score,
        reversibility=Reversibility.PARTIALLY_REVERSIBLE,
        blast_radius=["service_a", "service_b"],
        requires_human_approval=True,
    )


def _make_channels(*, voice=None, push=None, sms=None, email=None):
    def always_true(_):
        return True

    return AlertChannels(
        voice_handler=voice or always_true,
        push_handler=push or always_true,
        sms_handler=sms or always_true,
        email_handler=email or always_true,
    )


class TestAlertSystem:
    def test_raise_l3_alert_creates_pending(self):
        system = AlertSystem(_make_channels())
        pending = system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        assert pending is not None
        assert pending.payload.severity == ThresholdLevel.L3_HIGH_RISK

    def test_raise_l4_alert_freezes_system(self):
        frozen = []
        system = AlertSystem(_make_channels(), on_freeze=lambda: frozen.append(True))
        system.raise_alert(_make_plan(0.90), ThresholdLevel.L4_CRITICAL)
        assert system.is_frozen is True
        assert len(frozen) == 1

    def test_unfreeze(self):
        system = AlertSystem(_make_channels(), on_freeze=lambda: None)
        system.raise_alert(_make_plan(0.90), ThresholdLevel.L4_CRITICAL)
        system.unfreeze()
        assert system.is_frozen is False

    def test_respond_approve(self):
        system = AlertSystem(_make_channels())
        pending = system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        result = system.respond(pending.payload.alert_id, approve=True)
        assert result is True
        assert pending.approved is True

    def test_respond_deny(self):
        system = AlertSystem(_make_channels())
        pending = system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        system.respond(pending.payload.alert_id, approve=False)
        assert pending.approved is False

    def test_respond_unknown_alert_returns_false(self):
        system = AlertSystem(_make_channels())
        result = system.respond("nonexistent-id", approve=True)
        assert result is False

    def test_sms_fallback_when_push_fails(self):
        sms_calls = []

        def failing_push(_):
            return False

        def sms_handler(payload):
            sms_calls.append(payload)
            return True

        channels = AlertChannels(
            push_handler=failing_push,
            sms_handler=sms_handler,
            email_handler=lambda _: True,
        )
        system = AlertSystem(channels)
        system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        assert len(sms_calls) == 1

    def test_email_always_sent(self):
        email_calls = []

        channels = AlertChannels(
            push_handler=lambda _: True,
            email_handler=lambda p: email_calls.append(p) or True,
        )
        system = AlertSystem(channels)
        system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        assert len(email_calls) == 1

    def test_voice_call_for_l4(self):
        voice_calls = []

        channels = AlertChannels(
            voice_handler=lambda p: voice_calls.append(p) or True,
            push_handler=lambda _: True,
            email_handler=lambda _: True,
        )
        system = AlertSystem(channels)
        system.raise_alert(_make_plan(0.90), ThresholdLevel.L4_CRITICAL)
        assert len(voice_calls) == 1

    def test_process_timeouts_auto_deny_l3(self):
        system = AlertSystem(_make_channels())
        pending = system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        # Backdate the auto_deny_at to the past
        pending.payload.auto_deny_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        system.process_timeouts()
        assert pending.approved is False

    def test_process_timeouts_auto_approve_l2(self):
        system = AlertSystem(_make_channels())
        plan = _make_plan(0.45)
        pending = system.raise_alert(plan, ThresholdLevel.L2_SIGNIFICANT)
        pending.payload.auto_deny_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        system.process_timeouts()
        assert pending.approved is True

    def test_get_receipts(self):
        system = AlertSystem(_make_channels())
        system.raise_alert(_make_plan(), ThresholdLevel.L3_HIGH_RISK)
        receipts = system.get_receipts()
        assert len(receipts) > 0
        assert all(isinstance(r, DeliveryReceipt) for r in receipts)

    def test_integrity_alert_creates_l4(self):
        frozen = []
        system = AlertSystem(_make_channels(), on_freeze=lambda: frozen.append(True))
        pending = system.raise_integrity_alert("model_weights", "checksum mismatch")
        assert pending.payload.severity == ThresholdLevel.L4_CRITICAL
        assert system.is_frozen is True

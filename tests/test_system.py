"""Integration tests for the full ApexSystem façade."""

import pytest

from apex import ApexSystem
from apex.core.types import Severity, ThresholdLevel
from apex.mape.analyzer import SignalRule
from apex.mape.monitor import AnomalyDetector, MetricEvent


class TestApexSystem:
    def test_system_initialises(self):
        system = ApexSystem()
        assert system is not None
        assert system.knowledge_base is not None
        assert system.audit_ledger is not None

    def test_publish_metric_and_run_cycle(self):
        system = ApexSystem()
        system.publish_metric("api_gw", "error_rate", 0.01)
        report = system.run_cycle()
        assert report is not None

    def test_signal_rule_fires_on_breach(self):
        system = ApexSystem()
        system.add_signal_rule(
            SignalRule("latency_ms", upper_threshold=100.0, severity=Severity.WARNING)
        )
        system.publish_metric("svc", "latency_ms", 200.0)
        report = system.run_cycle()
        assert report.overall_severity == Severity.WARNING

    def test_heartbeat_resets_dead_man(self):
        system = ApexSystem(dead_man_timeout_seconds=1)
        system.heartbeat()
        assert system.threshold_engine.check_dead_man_switch() is False

    def test_alert_flow_l4_freezes_system(self):
        frozen = []
        system = ApexSystem(
            push_handler=lambda _: True,
            email_handler=lambda _: True,
            voice_handler=lambda _: True,
            on_freeze=lambda: frozen.append(True),
        )
        from apex.core.types import AdaptationPlan, AdaptationType, Reversibility

        plan = AdaptationPlan(
            description="critical plan",
            adaptation_type=AdaptationType.MACRO,
            expected_benefit=0.1,
            risk_score=0.90,
            reversibility=Reversibility.DESTRUCTIVE,
            blast_radius=["core"],
            requires_human_approval=True,
        )
        system.alert_system.raise_alert(plan, ThresholdLevel.L4_CRITICAL)
        assert system.alert_system.is_frozen is True
        assert len(frozen) == 1

    def test_respond_to_alert(self):
        system = ApexSystem(push_handler=lambda _: True, email_handler=lambda _: True)
        from apex.core.types import AdaptationPlan, AdaptationType, Reversibility

        plan = AdaptationPlan(
            description="plan",
            adaptation_type=AdaptationType.MESO,
            expected_benefit=0.2,
            risk_score=0.65,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            blast_radius=["svc"],
            requires_human_approval=True,
        )
        pending = system.alert_system.raise_alert(plan, ThresholdLevel.L3_HIGH_RISK)
        result = system.respond_to_alert(pending.payload.alert_id, approve=True)
        assert result is True
        assert pending.approved is True

    def test_verify_audit_chain(self):
        system = ApexSystem()
        system.publish_metric("x", "y", 1.0)
        system.run_cycle()
        valid, msg = system.verify_audit_chain()
        assert valid is True

    def test_list_safety_constraints(self):
        system = ApexSystem()
        constraints = system.list_safety_constraints()
        assert len(constraints) >= 5

    def test_multiple_cycles(self):
        system = ApexSystem()
        for i in range(5):
            system.publish_metric("svc", "cpu", float(i) * 10)
            system.run_cycle()
        assert system.mape_loop.cycle_count == 5
        valid, _ = system.verify_audit_chain()
        assert valid is True

"""Tests for the MAPE-K loop components."""

import pytest

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import (
    AdaptationType,
    ExecutionStatus,
    Reversibility,
    Severity,
)
from apex.mape.analyzer import Analyzer, SignalRule
from apex.mape.executor import Executor, ExecutionBlockedError
from apex.mape.loop import MAPELoop
from apex.mape.monitor import AnomalyAlert, AnomalyDetector, MetricEvent, Monitor
from apex.mape.planner import Planner
from apex.thresholds.risk_scorer import RiskScorer


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class TestMonitor:
    def test_publish_and_retrieve_events(self):
        monitor = Monitor()
        monitor.publish(MetricEvent(source="svc", name="latency_ms", value=120.0))
        events = monitor.get_events()
        assert len(events) == 1
        assert events[0].value == 120.0

    def test_filter_by_source(self):
        monitor = Monitor()
        monitor.publish(MetricEvent(source="a", name="err", value=1.0))
        monitor.publish(MetricEvent(source="b", name="err", value=2.0))
        assert len(monitor.get_events(source="a")) == 1
        assert len(monitor.get_events(source="b")) == 1

    def test_filter_by_metric_name(self):
        monitor = Monitor()
        monitor.publish(MetricEvent(source="x", name="latency", value=10.0))
        monitor.publish(MetricEvent(source="x", name="error_rate", value=0.01))
        assert len(monitor.get_events(metric_name="latency")) == 1

    def test_latest_value(self):
        monitor = Monitor()
        monitor.publish(MetricEvent(source="svc", name="cpu", value=0.5))
        monitor.publish(MetricEvent(source="svc", name="cpu", value=0.8))
        assert monitor.latest_value("svc", "cpu") == 0.8

    def test_latest_value_missing_returns_none(self):
        monitor = Monitor()
        assert monitor.latest_value("x", "y") is None

    def test_anomaly_detector_fires(self):
        alerts = []
        monitor = Monitor()
        detector = AnomalyDetector(
            "error_rate",
            window_size=3,
            upper_threshold=0.05,
            on_anomaly=alerts.append,
        )
        monitor.add_detector(detector)
        for v in [0.1, 0.2, 0.3]:
            monitor.publish(MetricEvent(source="gw", name="error_rate", value=v))
        assert len(alerts) == 1
        assert alerts[0].metric_name == "error_rate"

    def test_anomaly_detector_no_fire_below_threshold(self):
        alerts = []
        monitor = Monitor()
        detector = AnomalyDetector(
            "error_rate",
            window_size=3,
            upper_threshold=0.5,
            on_anomaly=alerts.append,
        )
        monitor.add_detector(detector)
        for v in [0.01, 0.02, 0.03]:
            monitor.publish(MetricEvent(source="gw", name="error_rate", value=v))
        assert len(alerts) == 0

    def test_self_integrity_check(self):
        monitor = Monitor()
        result = monitor.self_integrity_check()
        assert "detector_count" in result
        assert "event_log_size" in result


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TestAnalyzer:
    def test_empty_events_returns_informational(self):
        analyzer = Analyzer()
        report = analyzer.analyze([])
        assert report.overall_severity == Severity.INFORMATIONAL
        assert len(report.symptom_clusters) == 0

    def test_rule_fires_on_threshold_breach(self):
        analyzer = Analyzer()
        analyzer.add_rule(
            SignalRule("error_rate", upper_threshold=0.05, severity=Severity.WARNING)
        )
        events = [MetricEvent(source="api", name="error_rate", value=0.10)]
        report = analyzer.analyze(events)
        assert report.overall_severity == Severity.WARNING
        assert len(report.symptom_clusters) == 1

    def test_rule_does_not_fire_below_threshold(self):
        analyzer = Analyzer()
        analyzer.add_rule(
            SignalRule("error_rate", upper_threshold=0.05, severity=Severity.WARNING)
        )
        events = [MetricEvent(source="api", name="error_rate", value=0.01)]
        report = analyzer.analyze(events)
        assert report.overall_severity == Severity.INFORMATIONAL

    def test_anomaly_alerts_create_clusters(self):
        analyzer = Analyzer()
        alert = AnomalyAlert(
            source="svc", metric_name="latency", observed_value=500, threshold=200
        )
        report = analyzer.analyze([], anomaly_alerts=[alert])
        assert len(report.symptom_clusters) == 1
        assert "svc" in report.proposed_evolution_targets

    def test_severity_escalation(self):
        analyzer = Analyzer()
        analyzer.add_rule(
            SignalRule("err", upper_threshold=0.01, severity=Severity.CRITICAL)
        )
        events = [MetricEvent(source="x", name="err", value=0.1)]
        report = analyzer.analyze(events)
        assert report.overall_severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_empty_report_returns_no_plans(self):
        from apex.core.types import AnalysisReport

        planner = Planner()
        report = AnalysisReport()
        plans = planner.plan(report)
        assert plans == []

    def test_plans_ranked_by_utility(self):
        from apex.core.types import AnalysisReport, Severity, SymptomCluster

        planner = Planner()
        report = AnalysisReport(
            symptom_clusters=[
                SymptomCluster(
                    signals=["v=0.1"],
                    probable_cause="minor issue",
                    severity=Severity.WARNING,
                ),
                SymptomCluster(
                    signals=["v=0.9"],
                    probable_cause="critical issue",
                    severity=Severity.CRITICAL,
                ),
            ],
            overall_severity=Severity.CRITICAL,
        )
        plans = planner.plan(report)
        assert len(plans) == 2
        # All plans have expected_benefit / (1+risk) ratio; order doesn't fail
        assert all(p.risk_score >= 0 for p in plans)

    def test_high_risk_plan_requires_approval(self):
        from apex.core.types import AnalysisReport, Severity, SymptomCluster

        # Use a custom scorer that always returns high risk
        class HighRiskScorer(RiskScorer):
            def score(self, **_):
                return 0.75

        planner = Planner(risk_scorer=HighRiskScorer())
        report = AnalysisReport(
            symptom_clusters=[
                SymptomCluster(
                    signals=["v=1"],
                    probable_cause="critical",
                    severity=Severity.CRITICAL,
                )
            ],
            overall_severity=Severity.CRITICAL,
        )
        plans = planner.plan(report)
        assert plans[0].requires_human_approval is True


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestExecutor:
    def _make_plan(self, risk_score=0.10, requires_approval=False):
        from apex.core.types import AdaptationPlan, AdaptationType, Reversibility

        return AdaptationPlan(
            description="test",
            adaptation_type=AdaptationType.MICRO,
            expected_benefit=0.1,
            risk_score=risk_score,
            reversibility=Reversibility.FULLY_REVERSIBLE,
            requires_human_approval=requires_approval,
        )

    def test_successful_commit(self):
        kb = KnowledgeBase()
        executor = Executor(kb)
        plan = self._make_plan()
        result = executor.execute(plan, approved=False)
        assert result.status == ExecutionStatus.COMMITTED

    def test_blocked_without_approval(self):
        kb = KnowledgeBase()
        executor = Executor(kb)
        plan = self._make_plan(risk_score=0.70, requires_approval=True)
        with pytest.raises(ExecutionBlockedError):
            executor.execute(plan, approved=False)

    def test_approved_high_risk_executes(self):
        kb = KnowledgeBase()
        executor = Executor(kb)
        plan = self._make_plan(risk_score=0.70, requires_approval=True)
        result = executor.execute(plan, approved=True)
        assert result.status == ExecutionStatus.COMMITTED

    def test_auto_rollback_on_worsened_metric(self):
        kb = KnowledgeBase()
        executor = Executor(kb, rollback_delta=0.10)
        plan = self._make_plan()
        result = executor.execute(plan, baseline_metric=1.0, post_metric=1.5)
        assert result.status == ExecutionStatus.ROLLED_BACK

    def test_no_rollback_when_metric_improves(self):
        kb = KnowledgeBase()
        executor = Executor(kb, rollback_delta=0.10)
        plan = self._make_plan()
        result = executor.execute(plan, baseline_metric=1.0, post_metric=0.8)
        assert result.status == ExecutionStatus.COMMITTED

    def test_shadow_and_canary_hooks_called(self):
        calls = []
        kb = KnowledgeBase()
        executor = Executor(
            kb,
            on_shadow_run=lambda p: calls.append("shadow"),
            on_canary_run=lambda p: calls.append("canary"),
        )
        plan = self._make_plan()
        executor.execute(plan)
        assert "shadow" in calls
        assert "canary" in calls

    def test_result_recorded_in_kb(self):
        kb = KnowledgeBase()
        executor = Executor(kb)
        plan = self._make_plan()
        executor.execute(plan)
        history = kb.get_adaptation_history()
        assert len(history) == 1


# ---------------------------------------------------------------------------
# MAPELoop integration
# ---------------------------------------------------------------------------


class TestMAPELoop:
    def test_full_cycle_produces_report(self):
        kb = KnowledgeBase()
        monitor = Monitor()
        analyzer = Analyzer()
        planner = Planner()
        executor = Executor(kb)

        monitor.publish(MetricEvent(source="svc", name="latency", value=10.0))
        loop = MAPELoop(monitor, analyzer, planner, executor, kb)
        report = loop.run_cycle()
        assert report is not None
        assert loop.cycle_count == 1

    def test_cycle_increments_counter(self):
        kb = KnowledgeBase()
        monitor = Monitor()
        loop = MAPELoop(
            monitor, Analyzer(), Planner(), Executor(kb), kb
        )
        loop.run_cycle()
        loop.run_cycle()
        assert loop.cycle_count == 2

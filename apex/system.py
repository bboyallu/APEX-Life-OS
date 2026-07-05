"""ApexSystem — top-level façade that wires all subsystems together.

Usage::

    system = ApexSystem()
    system.publish_metric("api_gateway", "error_rate", 0.02)
    report = system.run_cycle()
    print(report.overall_severity)
"""

from __future__ import annotations

from typing import Any, Callable

from apex.alerts.channels import AlertChannels
from apex.alerts.system import AlertSystem, PendingApproval
from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import (
    AdaptationPlan,
    AnalysisReport,
    ConstraintType,
    ThresholdLevel,
)
from apex.governance.audit import AuditLedger
from apex.governance.constraints import SafetyConstraintRegistry
from apex.knowledge.bridge import KnowledgeBridge
from apex.knowledge.vault import IngestReport, KnowledgeVault
from apex.mape.analyzer import Analyzer, SignalRule
from apex.mape.executor import Executor
from apex.mape.loop import MAPELoop
from apex.mape.monitor import AnomalyAlert, AnomalyDetector, MetricEvent, Monitor
from apex.mape.planner import Planner
from apex.memory.builtin import BuiltinMemoryProvider
from apex.memory.provider import MemoryEntry, MemoryProvider, MemoryScope
from apex.neuro_symbolic.neural import NeuralSubsystem
from apex.neuro_symbolic.symbolic import SymbolicSubsystem
from apex.neuro_symbolic.verifier import VerificationPipeline
from apex.orchestration.orchestrator import DecisionOrchestrator
from apex.orchestration.selector import DecisionContext, PathSelector
from apex.thresholds.engine import AutonomicThresholdEngine
from apex.thresholds.risk_scorer import RiskScorer


class ApexSystem:
    """Full APEX self-evolving AI system.

    Parameters
    ----------
    voice_handler, push_handler, sms_handler, email_handler:
        Optional alert channel callables ``(AlertPayload) -> bool``.
    on_freeze:
        Callback invoked when an L4 event freezes the system.
    on_plan_generated:
        Callback ``(AdaptationPlan) -> bool`` for manual approval of L3+ plans.
    dead_man_timeout_seconds:
        Seconds without oversight contact before L2+ evolutions are suspended.
    memory_provider:
        Long-term memory backend.  Defaults to the built-in markdown store
        (``MEMORY.md`` / ``USER.md`` in the current working directory).
    knowledge_root:
        Directory containing the ``raw/``, ``wiki/`` and ``outputs/``
        knowledge base folders (see ``KNOWLEDGE_BASE.md``).  Defaults to the
        current working directory.
    """

    def __init__(
        self,
        *,
        voice_handler: Any = None,
        push_handler: Any = None,
        sms_handler: Any = None,
        email_handler: Any = None,
        on_freeze: Callable[[], None] | None = None,
        on_plan_generated: Callable[[AdaptationPlan], bool] | None = None,
        dead_man_timeout_seconds: int = 900,
        memory_provider: MemoryProvider | None = None,
        knowledge_root: str = ".",
    ) -> None:
        # Core
        self.knowledge_base = KnowledgeBase()
        self.audit_ledger = AuditLedger()
        self.safety_registry = SafetyConstraintRegistry()

        # Risk / thresholds
        self.risk_scorer = RiskScorer()
        self.threshold_engine = AutonomicThresholdEngine(
            risk_scorer=self.risk_scorer,
            dead_man_timeout_seconds=dead_man_timeout_seconds,
            on_l3_alert=self._handle_l3,
            on_l4_alert=self._handle_l4,
        )

        # MAPE-K
        self.monitor = Monitor()
        self.analyzer = Analyzer()
        self.planner = Planner(risk_scorer=self.risk_scorer)
        self.executor = Executor(
            knowledge_base=self.knowledge_base,
            risk_scorer=self.risk_scorer,
        )
        self.mape_loop = MAPELoop(
            monitor=self.monitor,
            analyzer=self.analyzer,
            planner=self.planner,
            executor=self.executor,
            knowledge_base=self.knowledge_base,
            on_plan_generated=on_plan_generated,
        )

        # Neuro-symbolic
        self.neural = NeuralSubsystem()
        self.symbolic = SymbolicSubsystem()
        self.verification_pipeline = VerificationPipeline(
            neural=self.neural,
            symbolic=self.symbolic,
            knowledge_base=self.knowledge_base,
        )

        # Orchestration
        self.path_selector = PathSelector()
        self.orchestrator = DecisionOrchestrator(
            path_selector=self.path_selector,
            verification_pipeline=self.verification_pipeline,
            knowledge_base=self.knowledge_base,
        )

        # Alerts
        channels = AlertChannels(
            voice_handler=voice_handler,
            push_handler=push_handler,
            sms_handler=sms_handler,
            email_handler=email_handler,
        )
        self.alert_system = AlertSystem(channels=channels, on_freeze=on_freeze)

        # Long-term memory
        self.memory = memory_provider or BuiltinMemoryProvider()

        # Personal knowledge base (raw/ → wiki/ → outputs/)
        self.knowledge_vault = KnowledgeVault(root=knowledge_root)

        # Bridge: compiled knowledge ↔ self-evolution loop
        self.knowledge_bridge = KnowledgeBridge(vault=self.knowledge_vault)

    # ------------------------------------------------------------------
    # Knowledge base helpers
    # ------------------------------------------------------------------

    def process_knowledge(self) -> IngestReport:
        """Fold new/changed material from ``raw/`` into the ``wiki/``."""
        report = self.knowledge_vault.process_raw()
        self.audit_ledger.append(
            "knowledge_processed",
            actor="apex_system",
            payload={
                "ingested": report.ingested,
                "updated": report.updated,
                "skipped": len(report.skipped),
                "articles": report.articles,
            },
        )
        return report

    def generate_knowledge_report(self, query: str, *, title: str | None = None) -> str:
        """Answer ``query`` from the knowledge base into a new ``outputs/`` file.

        Returns the path of the generated report.
        """
        path = self.knowledge_vault.generate_report(query, title=title)
        self.audit_ledger.append(
            "knowledge_report_generated",
            actor="apex_system",
            payload={"query": query, "output": str(path)},
        )
        return str(path)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def remember(
        self,
        subject: str,
        fact: str,
        *,
        scope: MemoryScope = MemoryScope.REPOSITORY,
        citation: str = "",
    ) -> MemoryEntry:
        """Store a long-term memory via the configured provider."""
        entry = self.memory.remember(
            MemoryEntry(scope=scope, subject=subject, fact=fact, citation=citation)
        )
        self.audit_ledger.append(
            "memory_stored",
            actor="apex_system",
            payload={
                "entry_id": entry.entry_id,
                "scope": entry.scope.value,
                "subject": entry.subject,
                "provider": self.memory.name,
            },
        )
        return entry

    def recall_memories(self, scope: MemoryScope | None = None) -> list[MemoryEntry]:
        return self.memory.recall(scope)

    def search_memories(
        self, query: str, scope: MemoryScope | None = None
    ) -> list[MemoryEntry]:
        return self.memory.search(query, scope)

    def forget_memory(self, entry_id: str) -> bool:
        forgotten = self.memory.forget(entry_id)
        if forgotten:
            self.audit_ledger.append(
                "memory_forgotten",
                actor="apex_system",
                payload={"entry_id": entry_id, "provider": self.memory.name},
            )
        return forgotten

    # ------------------------------------------------------------------
    # Monitor helpers
    # ------------------------------------------------------------------

    def publish_metric(self, source: str, metric_name: str, value: float, **tags: str) -> None:
        """Publish a metric event to the telemetry bus."""
        self.monitor.publish(
            MetricEvent(source=source, name=metric_name, value=value, tags=dict(tags))
        )
        self.audit_ledger.append(
            "metric_published",
            actor=source,
            payload={"metric": metric_name, "value": value},
        )

    def add_anomaly_detector(self, detector: AnomalyDetector) -> None:
        self.monitor.add_detector(detector)

    def add_signal_rule(self, rule: SignalRule) -> None:
        self.analyzer.add_rule(rule)

    # ------------------------------------------------------------------
    # MAPE-K cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        anomaly_alerts: list[AnomalyAlert] | None = None,
        baseline_metric: float | None = None,
        post_metric: float | None = None,
    ) -> AnalysisReport:
        """Run one full MAPE-K adaptation cycle."""
        report = self.mape_loop.run_cycle(
            anomaly_alerts=anomaly_alerts,
            baseline_metric=baseline_metric,
            post_metric=post_metric,
        )
        self.audit_ledger.append(
            "mape_cycle_complete",
            actor="apex_system",
            payload={
                "cycle": self.mape_loop.cycle_count,
                "severity": report.overall_severity.value,
                "evolution_targets": report.proposed_evolution_targets,
            },
        )
        return report

    def run_knowledge_informed_cycle(
        self,
        anomaly_alerts: list[AnomalyAlert] | None = None,
        baseline_metric: float | None = None,
        post_metric: float | None = None,
    ) -> AnalysisReport:
        """Run one closed knowledge-to-evolution loop.

        Learn → Understand → Propose improvement → Safely evolve →
        Record what changed → Learn from the result:

        1. Fold new raw material into the wiki (``process_knowledge``).
        2. Extract new ``signal:`` directives from the compiled wiki.
        3. Run a MAPE-K cycle with those knowledge signals as evolution
           candidates (all existing governance — thresholds, approvals,
           dead-man switch — still applies).
        4. Record the cycle outcome back into ``raw/`` and re-fold it into
           the wiki, so APEX learns from its own evolution.

        Every step is appended to the shared audit ledger.
        """
        self.process_knowledge()

        signals = self.knowledge_bridge.extract_signals()
        if signals:
            self.audit_ledger.append(
                "knowledge_signals_extracted",
                actor="knowledge_bridge",
                payload={
                    "count": len(signals),
                    "targets": sorted({s.target for s in signals}),
                },
            )

        report = self.mape_loop.run_cycle(
            anomaly_alerts=anomaly_alerts,
            baseline_metric=baseline_metric,
            post_metric=post_metric,
            knowledge_signals=signals,
        )
        self.audit_ledger.append(
            "mape_cycle_complete",
            actor="apex_system",
            payload={
                "cycle": self.mape_loop.cycle_count,
                "severity": report.overall_severity.value,
                "evolution_targets": report.proposed_evolution_targets,
                "knowledge_signals": len(signals),
            },
        )

        log_path = self.knowledge_bridge.record_cycle(
            cycle=self.mape_loop.cycle_count,
            report=report,
            signals=signals,
        )
        self.process_knowledge()
        self.audit_ledger.append(
            "evolution_recorded_to_knowledge",
            actor="knowledge_bridge",
            payload={
                "cycle": self.mape_loop.cycle_count,
                "log": str(log_path),
            },
        )
        return report

    # ------------------------------------------------------------------
    # Alert helpers
    # ------------------------------------------------------------------

    def _handle_l3(self, plan: AdaptationPlan, level: ThresholdLevel) -> None:
        self.alert_system.raise_alert(plan, level)
        self.audit_ledger.append(
            "alert_raised",
            actor="threshold_engine",
            payload={"level": level.value, "plan_id": plan.plan_id, "risk": plan.risk_score},
        )

    def _handle_l4(self, plan: AdaptationPlan, level: ThresholdLevel) -> None:
        pending = self.alert_system.raise_alert(plan, level)
        self.audit_ledger.append(
            "critical_alert_raised",
            actor="threshold_engine",
            payload={
                "level": level.value,
                "plan_id": plan.plan_id,
                "risk": plan.risk_score,
                "alert_id": pending.payload.alert_id,
            },
        )

    # ------------------------------------------------------------------
    # Oversight
    # ------------------------------------------------------------------

    def heartbeat(self) -> None:
        """Signal that the oversight interface is reachable (resets dead-man switch)."""
        self.threshold_engine.heartbeat()

    def respond_to_alert(self, alert_id: str, *, approve: bool) -> bool:
        return self.alert_system.respond(alert_id, approve=approve)

    def process_alert_timeouts(self) -> None:
        self.alert_system.process_timeouts()

    # ------------------------------------------------------------------
    # Governance shortcuts
    # ------------------------------------------------------------------

    def verify_audit_chain(self) -> tuple[bool, str]:
        return self.audit_ledger.verify_chain()

    def list_safety_constraints(self):
        return self.safety_registry.list_hard_constraints()

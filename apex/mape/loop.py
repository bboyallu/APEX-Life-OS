"""MAPE-K orchestration loop (§2) — ties Monitor → Analyze → Plan → Execute.

The ``MAPELoop`` drives one complete adaptation cycle and exposes a ``run``
method that can be called on a schedule or event-triggered basis.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import AdaptationPlan, AnalysisReport
from apex.knowledge.bridge import KnowledgeSignal
from apex.mape.analyzer import Analyzer
from apex.mape.executor import Executor
from apex.mape.monitor import AnomalyAlert, Monitor
from apex.mape.planner import Planner


class MAPELoop:
    """Closed-control adaptation cycle.

    Parameters
    ----------
    monitor:
        The Monitor instance to read telemetry from.
    analyzer:
        The Analyzer instance.
    planner:
        The Planner instance.
    executor:
        The Executor instance.
    knowledge_base:
        Shared KB.
    on_plan_generated:
        Optional callback invoked with each generated plan before execution.
        Return ``True`` to approve a plan that requires human approval,
        ``False`` to skip it.
    """

    def __init__(
        self,
        monitor: Monitor,
        analyzer: Analyzer,
        planner: Planner,
        executor: Executor,
        knowledge_base: KnowledgeBase,
        on_plan_generated: Callable[[AdaptationPlan], bool] | None = None,
    ) -> None:
        self._monitor = monitor
        self._analyzer = analyzer
        self._planner = planner
        self._executor = executor
        self._kb = knowledge_base
        self._on_plan_generated = on_plan_generated
        self._cycle_count = 0
        self._lock = threading.Lock()
        self._last_report: AnalysisReport | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        anomaly_alerts: list[AnomalyAlert] | None = None,
        baseline_metric: float | None = None,
        post_metric: float | None = None,
        knowledge_signals: list[KnowledgeSignal] | None = None,
    ) -> AnalysisReport:
        """Execute one full MAPE-K cycle.

        Returns the ``AnalysisReport`` produced during the Analyze phase.
        """
        with self._lock:
            self._cycle_count += 1

        # M — Monitor: collect recent events
        events = self._monitor.get_events()

        # A — Analyze
        report = self._analyzer.analyze(events, anomaly_alerts, knowledge_signals)
        self._last_report = report

        # P — Plan
        plans = self._planner.plan(report)

        # E — Execute
        for plan in plans:
            approved = True  # default for L0-L2
            if plan.requires_human_approval and self._on_plan_generated:
                approved = self._on_plan_generated(plan)
            elif plan.requires_human_approval:
                approved = False  # block if no approval callback is configured

            try:
                self._executor.execute(
                    plan,
                    approved=approved,
                    baseline_metric=baseline_metric,
                    post_metric=post_metric,
                )
            except Exception:
                pass  # Blocked executions are stored in KB; loop continues

        return report

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_report(self) -> AnalysisReport | None:
        return self._last_report

"""Analyze phase (§2.2) — transforms raw telemetry into actionable diagnoses.

The ``Analyzer`` receives a snapshot of recent ``MetricEvent`` objects from
the ``Monitor``, correlates signals, scores severity, and emits a structured
``AnalysisReport``.
"""

from __future__ import annotations

from typing import Any

from apex.core.types import (
    AnalysisReport,
    Severity,
    SymptomCluster,
)
from apex.knowledge.bridge import KnowledgeSignal
from apex.mape.monitor import AnomalyAlert, MetricEvent


# Ordered from least to most severe for comparison purposes
_SEVERITY_ORDER = [
    Severity.INFORMATIONAL,
    Severity.WARNING,
    Severity.DEGRADED,
    Severity.CRITICAL,
    Severity.CATASTROPHIC,
]


def _max_severity(a: Severity, b: Severity) -> Severity:
    return max(a, b, key=lambda s: _SEVERITY_ORDER.index(s))


class SignalRule:
    """A simple threshold rule that maps a metric to a severity level."""

    def __init__(
        self,
        metric_name: str,
        upper_threshold: float | None = None,
        lower_threshold: float | None = None,
        severity: Severity = Severity.WARNING,
        is_evolution_candidate: bool = False,
    ) -> None:
        self.metric_name = metric_name
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.severity = severity
        self.is_evolution_candidate = is_evolution_candidate

    def evaluate(self, event: MetricEvent) -> bool:
        if event.name != self.metric_name:
            return False
        if self.upper_threshold is not None and event.value > self.upper_threshold:
            return True
        if self.lower_threshold is not None and event.value < self.lower_threshold:
            return True
        return False


class Analyzer:
    """MAPE-K Analyze phase.

    Responsibilities:
    - Correlate multi-source signals to identify root causes.
    - Score severity (informational → catastrophic).
    - Flag high-risk evolution candidates.
    - Emit a structured ``AnalysisReport``.
    """

    def __init__(self) -> None:
        self._rules: list[SignalRule] = []

    def add_rule(self, rule: SignalRule) -> None:
        self._rules.append(rule)

    def analyze(
        self,
        events: list[MetricEvent],
        anomaly_alerts: list[AnomalyAlert] | None = None,
        knowledge_signals: list[KnowledgeSignal] | None = None,
    ) -> AnalysisReport:
        """Produce an ``AnalysisReport`` from a batch of telemetry events.

        ``knowledge_signals`` are directives extracted from the compiled
        knowledge base (see :class:`apex.knowledge.bridge.KnowledgeBridge`);
        each is treated as an evolution candidate for its target component.
        """
        anomaly_alerts = anomaly_alerts or []
        knowledge_signals = knowledge_signals or []
        clusters: list[SymptomCluster] = []
        evolution_targets: list[str] = []

        # --- Rule-based signal evaluation ---
        fired: dict[str, list[str]] = {}
        rule_meta: dict[str, dict[str, Any]] = {}

        for event in events:
            for rule in self._rules:
                if rule.evaluate(event):
                    key = f"{event.source}:{rule.metric_name}"
                    fired.setdefault(key, []).append(
                        f"value={event.value} at {event.timestamp.isoformat()}"
                    )
                    rule_meta[key] = {
                        "severity": rule.severity,
                        "is_evolution_candidate": rule.is_evolution_candidate,
                    }

        for key, signals in fired.items():
            meta = rule_meta[key]
            source, metric = key.split(":", 1)
            cluster = SymptomCluster(
                signals=signals,
                probable_cause=f"Rule threshold breach on {metric} from {source}",
                severity=meta["severity"],
                is_evolution_candidate=meta["is_evolution_candidate"],
            )
            clusters.append(cluster)
            if meta["is_evolution_candidate"] and source not in evolution_targets:
                evolution_targets.append(source)

        # --- Anomaly-alert-based clusters ---
        for alert in anomaly_alerts:
            cluster = SymptomCluster(
                signals=[
                    f"rolling_mean={alert.observed_value:.4f} "
                    f"exceeded threshold={alert.threshold:.4f} "
                    f"at {alert.timestamp.isoformat()}"
                ],
                probable_cause=(
                    f"Anomaly detector triggered on {alert.metric_name} "
                    f"from {alert.source}"
                ),
                severity=Severity.WARNING,
                is_evolution_candidate=True,
            )
            clusters.append(cluster)
            if alert.source not in evolution_targets:
                evolution_targets.append(alert.source)

        # --- Knowledge-derived clusters ---
        for signal in knowledge_signals:
            cluster = SymptomCluster(
                signals=[
                    f"knowledge={signal.description} "
                    f"(source {signal.source or 'knowledge base'} "
                    f"at {signal.timestamp.isoformat()})"
                ],
                probable_cause=(
                    f"Knowledge-derived signal for {signal.target}: "
                    f"{signal.description}"
                ),
                severity=signal.severity,
                is_evolution_candidate=True,
            )
            clusters.append(cluster)
            if signal.target not in evolution_targets:
                evolution_targets.append(signal.target)

        # --- Compute overall severity ---
        overall = Severity.INFORMATIONAL
        for cluster in clusters:
            overall = _max_severity(overall, cluster.severity)

        return AnalysisReport(
            symptom_clusters=clusters,
            overall_severity=overall,
            proposed_evolution_targets=list(set(evolution_targets)),
        )

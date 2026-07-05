"""Monitor phase (§2.1) — collects raw signals from all system components.

A ``TelemetryBus`` accepts metric events from producers and delivers them to
registered ``SignalCollector`` instances.  An ``AnomalyDetector`` sits on top of
the bus and evaluates rolling windows against configurable sensitivity bands.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class MetricEvent:
    """A single telemetry observation."""

    source: str
    name: str
    value: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class AnomalyAlert:
    """Raised when a metric exceeds its configured band."""

    source: str
    metric_name: str
    observed_value: float
    threshold: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


SignalCollector = Callable[[MetricEvent], None]


class TelemetryBus:
    """In-process streaming telemetry bus.

    A production system would use Kafka or a similar broker.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[SignalCollector] = []

    def subscribe(self, collector: SignalCollector) -> None:
        with self._lock:
            self._subscribers.append(collector)

    def publish(self, event: MetricEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber(event)


class AnomalyDetector:
    """Detects anomalies on a rolling window over a single metric.

    Parameters
    ----------
    metric_name:
        The metric to watch.
    source:
        The component source to watch (``None`` means all sources).
    window_size:
        Number of observations in the rolling window.
    upper_threshold:
        Alert when the rolling mean exceeds this value.
    lower_threshold:
        Alert when the rolling mean drops below this value (optional).
    on_anomaly:
        Callback invoked when an anomaly is detected.
    """

    def __init__(
        self,
        metric_name: str,
        *,
        source: str | None = None,
        window_size: int = 10,
        upper_threshold: float | None = None,
        lower_threshold: float | None = None,
        on_anomaly: Callable[[AnomalyAlert], None] | None = None,
    ) -> None:
        self.metric_name = metric_name
        self.source = source
        self.window_size = window_size
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.on_anomaly = on_anomaly
        self._window: deque[float] = deque(maxlen=window_size)

    def __call__(self, event: MetricEvent) -> None:
        if event.name != self.metric_name:
            return
        if self.source is not None and event.source != self.source:
            return

        self._window.append(event.value)
        if len(self._window) < self.window_size:
            return

        mean = sum(self._window) / len(self._window)

        alert: AnomalyAlert | None = None
        if self.upper_threshold is not None and mean > self.upper_threshold:
            alert = AnomalyAlert(
                source=event.source,
                metric_name=self.metric_name,
                observed_value=mean,
                threshold=self.upper_threshold,
            )
        elif self.lower_threshold is not None and mean < self.lower_threshold:
            alert = AnomalyAlert(
                source=event.source,
                metric_name=self.metric_name,
                observed_value=mean,
                threshold=self.lower_threshold,
            )

        if alert is not None and self.on_anomaly is not None:
            self.on_anomaly(alert)

    # ----------------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------------

    @property
    def rolling_mean(self) -> float | None:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)


class Monitor:
    """MAPE-K Monitor — owns a ``TelemetryBus`` and a set of detectors.

    Usage::

        monitor = Monitor()
        monitor.add_detector(
            AnomalyDetector("error_rate", upper_threshold=0.05,
                            on_anomaly=my_handler)
        )
        monitor.publish(MetricEvent(source="api_gateway",
                                    name="error_rate", value=0.02))
    """

    def __init__(self) -> None:
        self._bus = TelemetryBus()
        self._detectors: list[AnomalyDetector] = []
        self._event_log: list[MetricEvent] = []
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    # Detector management
    # ----------------------------------------------------------------

    def add_detector(self, detector: AnomalyDetector) -> None:
        self._detectors.append(detector)
        self._bus.subscribe(detector)

    # ----------------------------------------------------------------
    # Telemetry ingestion
    # ----------------------------------------------------------------

    def publish(self, event: MetricEvent) -> None:
        with self._lock:
            self._event_log.append(event)
        self._bus.publish(event)

    def get_events(
        self,
        source: str | None = None,
        metric_name: str | None = None,
    ) -> list[MetricEvent]:
        with self._lock:
            events = list(self._event_log)
        if source:
            events = [e for e in events if e.source == source]
        if metric_name:
            events = [e for e in events if e.name == metric_name]
        return events

    def latest_value(self, source: str, metric_name: str) -> float | None:
        events = self.get_events(source=source, metric_name=metric_name)
        if not events:
            return None
        return events[-1].value

    # ----------------------------------------------------------------
    # Self-integrity check
    # ----------------------------------------------------------------

    def self_integrity_check(self) -> dict[str, Any]:
        """Return a basic integrity snapshot (checksums, detector counts, etc.)."""
        return {
            "detector_count": len(self._detectors),
            "event_log_size": len(self._event_log),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

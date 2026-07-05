"""Causal model (§Phase 5) — learned intervention → outcome relationships.

Expands the Knowledge Base's simple note-based causal store into a
structured model built from deployment history.  Each recorded
``InterventionRecord`` captures what was done to which component and how it
turned out; the model aggregates these into per-(component, action) success
rates and expected metric deltas that the Plan phase (or a human reviewer)
can use to weigh candidate adaptations.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class InterventionRecord(BaseModel):
    """A single deployment-history observation."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    component_id: str
    action: str
    success: bool
    metric_delta: float | None = None
    plan_id: str | None = None
    notes: str = ""


class CausalPrediction(BaseModel):
    """Aggregated expectation for a (component, action) intervention."""

    component_id: str
    action: str
    observations: int = 0
    success_rate: float | None = None
    expected_metric_delta: float | None = None

    @property
    def has_evidence(self) -> bool:
        return self.observations > 0


class CausalModel:
    """Thread-safe structured causal model built from deployment history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], list[InterventionRecord]] = {}

    def record(self, record: InterventionRecord) -> None:
        """Add a deployment observation to the model."""
        with self._lock:
            key = (record.component_id, record.action)
            self._records.setdefault(key, []).append(record)

    def get_records(self, component_id: str, action: str) -> list[InterventionRecord]:
        with self._lock:
            return list(self._records.get((component_id, action), []))

    def predict(self, component_id: str, action: str) -> CausalPrediction:
        """Return the aggregated expectation for an intervention."""
        records = self.get_records(component_id, action)
        if not records:
            return CausalPrediction(component_id=component_id, action=action)

        successes = sum(1 for r in records if r.success)
        deltas = [r.metric_delta for r in records if r.metric_delta is not None]
        return CausalPrediction(
            component_id=component_id,
            action=action,
            observations=len(records),
            success_rate=successes / len(records),
            expected_metric_delta=sum(deltas) / len(deltas) if deltas else None,
        )

    def known_interventions(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._records.keys())

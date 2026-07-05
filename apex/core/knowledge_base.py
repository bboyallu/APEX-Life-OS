"""Knowledge Base (K) — the shared memory substrate for all MAPE components.

Implements §2.5 of the APEX blueprint:
  - System state graph (component topology & health)
  - Adaptation history (full record of every change and outcome)
  - Causal model (learned intervention → outcome relationships)
  - World model (external context, user preferences, domain knowledge)
  - Constraint registry (immutable safety rules and mutable policies)
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from apex.core.types import (
    AdaptationPlan,
    ConstraintType,
    DecisionRecord,
    ExecutionResult,
)


class ComponentState:
    """Live health record for a single system component."""

    def __init__(self, component_id: str, *, healthy: bool = True) -> None:
        self.component_id = component_id
        self.healthy = healthy
        self.metadata: dict[str, Any] = {}
        self.last_updated: datetime = datetime.now(timezone.utc)

    def update(self, healthy: bool, **meta: Any) -> None:
        self.healthy = healthy
        self.metadata.update(meta)
        self.last_updated = datetime.now(timezone.utc)


class Constraint:
    """A rule in the constraint registry."""

    def __init__(
        self,
        constraint_id: str,
        description: str,
        constraint_type: ConstraintType,
        rule: str,
    ) -> None:
        self.constraint_id = constraint_id
        self.description = description
        self.constraint_type = constraint_type
        self.rule = rule
        self.created_at = datetime.now(timezone.utc)


class AdaptationHistoryEntry:
    """A single record in the adaptation history."""

    def __init__(
        self,
        plan: AdaptationPlan,
        result: ExecutionResult,
    ) -> None:
        self.plan = plan
        self.result = result
        self.recorded_at = datetime.now(timezone.utc)


class KnowledgeBase:
    """Thread-safe, in-memory Knowledge Base.

    In a production deployment this would be backed by a distributed store
    (e.g., Redis + PostgreSQL + a vector DB), but for the reference
    implementation an in-process store is used.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # System state graph
        self._components: dict[str, ComponentState] = {}

        # Adaptation history
        self._history: list[AdaptationHistoryEntry] = []

        # Causal model: maps (component_id, action) -> list[outcome_notes]
        self._causal_model: dict[tuple[str, str], list[str]] = {}

        # World model: arbitrary key-value context
        self._world_model: dict[str, Any] = {}

        # Constraint registry
        self._constraints: dict[str, Constraint] = {}

        # Decision records
        self._decision_records: list[DecisionRecord] = []

    # ------------------------------------------------------------------
    # System state graph
    # ------------------------------------------------------------------

    def register_component(self, component_id: str, *, healthy: bool = True) -> None:
        with self._lock:
            self._components[component_id] = ComponentState(
                component_id, healthy=healthy
            )

    def update_component_health(
        self, component_id: str, healthy: bool, **meta: Any
    ) -> None:
        with self._lock:
            if component_id not in self._components:
                self._components[component_id] = ComponentState(component_id)
            self._components[component_id].update(healthy, **meta)

    def get_component_state(self, component_id: str) -> ComponentState | None:
        with self._lock:
            return self._components.get(component_id)

    def list_components(self) -> list[str]:
        with self._lock:
            return list(self._components.keys())

    def all_healthy(self) -> bool:
        with self._lock:
            return all(c.healthy for c in self._components.values())

    # ------------------------------------------------------------------
    # Adaptation history
    # ------------------------------------------------------------------

    def record_adaptation(
        self, plan: AdaptationPlan, result: ExecutionResult
    ) -> None:
        with self._lock:
            self._history.append(AdaptationHistoryEntry(plan, result))

    def get_adaptation_history(self) -> list[AdaptationHistoryEntry]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Causal model
    # ------------------------------------------------------------------

    def record_causal_outcome(
        self, component_id: str, action: str, outcome_note: str
    ) -> None:
        with self._lock:
            key = (component_id, action)
            self._causal_model.setdefault(key, []).append(outcome_note)

    def get_causal_outcomes(self, component_id: str, action: str) -> list[str]:
        with self._lock:
            return list(self._causal_model.get((component_id, action), []))

    # ------------------------------------------------------------------
    # World model
    # ------------------------------------------------------------------

    def set_world_context(self, key: str, value: Any) -> None:
        with self._lock:
            self._world_model[key] = value

    def get_world_context(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._world_model.get(key, default)

    # ------------------------------------------------------------------
    # Constraint registry
    # ------------------------------------------------------------------

    def register_constraint(
        self,
        constraint_id: str,
        description: str,
        constraint_type: ConstraintType,
        rule: str,
    ) -> None:
        with self._lock:
            self._constraints[constraint_id] = Constraint(
                constraint_id, description, constraint_type, rule
            )

    def get_constraint(self, constraint_id: str) -> Constraint | None:
        with self._lock:
            return self._constraints.get(constraint_id)

    def list_constraints(
        self, constraint_type: ConstraintType | None = None
    ) -> list[Constraint]:
        with self._lock:
            if constraint_type is None:
                return list(self._constraints.values())
            return [
                c
                for c in self._constraints.values()
                if c.constraint_type == constraint_type
            ]

    # ------------------------------------------------------------------
    # Decision records
    # ------------------------------------------------------------------

    def store_decision_record(self, record: DecisionRecord) -> None:
        with self._lock:
            self._decision_records.append(record)

    def get_decision_records(self) -> list[DecisionRecord]:
        with self._lock:
            return list(self._decision_records)

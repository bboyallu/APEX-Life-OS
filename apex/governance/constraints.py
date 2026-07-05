"""Safety constraints — immutable core and soft policy registry (§7.1, §3.3).

The ``SafetyConstraintRegistry`` maintains:
- Hard (immutable) constraints that cannot be modified by any autonomous process.
- Soft (mutable) constraints that can be updated via the Plan phase.
- The registry is pre-seeded with a minimal safe set of hard constraints.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ImmutableConstraint:
    """A hard safety rule that cannot be modified autonomously."""

    constraint_id: str
    description: str
    rule_expression: str   # human-readable; a real system would use formal logic


@dataclass
class MutablePolicy:
    """A soft operational policy that can be updated via the Plan phase."""

    policy_id: str
    description: str
    value: Any
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SafetyConstraintRegistry:
    """Manages the immutable safety core and mutable operational policies.

    The immutable core (§7.1) is loaded once at construction and cannot be
    changed by any autonomous operation.  Offline, cryptographically
    authorised human intervention is required to update hard constraints.
    """

    # Default hard safety constraints — the "immutable core"
    _DEFAULT_HARD_CONSTRAINTS: tuple[ImmutableConstraint, ...] = (
        ImmutableConstraint(
            constraint_id="safety:no-self-replication",
            description="The system must not autonomously replicate itself.",
            rule_expression="NOT action.type == 'self_replication'",
        ),
        ImmutableConstraint(
            constraint_id="safety:no-constraint-override",
            description="The system must not modify its own hard safety constraints.",
            rule_expression="NOT action.modifies_hard_constraint",
        ),
        ImmutableConstraint(
            constraint_id="safety:minimal-footprint",
            description=(
                "The system must not pre-acquire capabilities beyond those "
                "required for the current task."
            ),
            rule_expression="action.permissions SUBSET_OF task.required_permissions",
        ),
        ImmutableConstraint(
            constraint_id="safety:human-veto",
            description=(
                "Any L3+ evolution must be stoppable by an authorised human "
                "at any time via the oversight interface."
            ),
            rule_expression="evolution.level >= L3 IMPLIES human_veto.enabled",
        ),
        ImmutableConstraint(
            constraint_id="safety:audit-integrity",
            description="The immutable audit ledger must not be altered retroactively.",
            rule_expression="NOT action.modifies_audit_ledger",
        ),
    )

    def __init__(self) -> None:
        self._hard: dict[str, ImmutableConstraint] = {
            c.constraint_id: c for c in self._DEFAULT_HARD_CONSTRAINTS
        }
        self._soft: dict[str, MutablePolicy] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Hard constraints (read-only for autonomous processes)
    # ------------------------------------------------------------------

    def list_hard_constraints(self) -> list[ImmutableConstraint]:
        with self._lock:
            return list(self._hard.values())

    def get_hard_constraint(self, constraint_id: str) -> ImmutableConstraint | None:
        with self._lock:
            return self._hard.get(constraint_id)

    def check_hard_constraint(self, constraint_id: str) -> bool:
        """Return ``True`` if the constraint exists and is in force."""
        with self._lock:
            return constraint_id in self._hard

    # ------------------------------------------------------------------
    # Soft policies (mutable via Plan phase)
    # ------------------------------------------------------------------

    def register_policy(self, policy: MutablePolicy) -> None:
        with self._lock:
            self._soft[policy.policy_id] = policy

    def update_policy(self, policy_id: str, value: Any) -> bool:
        with self._lock:
            if policy_id not in self._soft:
                return False
            self._soft[policy_id] = MutablePolicy(
                policy_id=policy_id,
                description=self._soft[policy_id].description,
                value=value,
            )
            return True

    def get_policy(self, policy_id: str) -> MutablePolicy | None:
        with self._lock:
            return self._soft.get(policy_id)

    def list_policies(self) -> list[MutablePolicy]:
        with self._lock:
            return list(self._soft.values())

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def would_violate_hard_constraints(self, action_tags: set[str]) -> list[str]:
        """Return IDs of hard constraints that would be violated by ``action_tags``.

        This is a heuristic check using keyword matching against rule
        expressions.  A production system would use formal verification.
        """
        with self._lock:
            violated: list[str] = []
            for c in self._hard.values():
                for tag in action_tags:
                    if tag.lower() in c.rule_expression.lower():
                        violated.append(c.constraint_id)
                        break
            return violated

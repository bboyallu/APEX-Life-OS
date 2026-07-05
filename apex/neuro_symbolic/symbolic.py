"""Symbolic subsystem (§3.1) — constraint solver and rule evaluator.

The symbolic subsystem:
1. Maintains a registry of hard and soft constraints (§3.3).
2. Translates a candidate decision into a formal logical claim.
3. Checks the claim against the registered rule set.
4. Returns a ``SymbolicVerdict`` with a proof trace or counterexample.

In a production system this would integrate an SMT solver (e.g., Z3).
The reference implementation uses a Python rule-evaluation approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from apex.core.types import ConstraintType, VerificationResult


@dataclass
class SymbolicRule:
    """A single rule in the constraint registry."""

    rule_id: str
    description: str
    constraint_type: ConstraintType
    # predicate takes the candidate decision string; returns True if satisfied
    predicate: Callable[[str], bool]


@dataclass
class SymbolicVerdict:
    """Outcome of checking a candidate decision against the rule set."""

    result: VerificationResult
    satisfied_rules: list[str] = field(default_factory=list)
    violated_rules: list[str] = field(default_factory=list)
    proof_trace: str = ""
    counterexample: str = ""


class SymbolicSubsystem:
    """Evaluates candidate decisions against registered constraints.

    Rules of type ``HARD`` that fail produce a ``REFUTED`` verdict.
    ``SOFT`` rules that fail produce a ``VERIFIED`` verdict with warnings.
    ``LEARNED`` rules are validated before they can be activated.
    """

    def __init__(self) -> None:
        self._rules: dict[str, SymbolicRule] = {}
        self._pending_learned: dict[str, SymbolicRule] = {}

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def register_rule(self, rule: SymbolicRule) -> None:
        if rule.constraint_type == ConstraintType.LEARNED:
            self._pending_learned[rule.rule_id] = rule
        else:
            self._rules[rule.rule_id] = rule

    def activate_learned_rule(self, rule_id: str) -> bool:
        """Move a pending learned rule to the active set after validation."""
        if rule_id in self._pending_learned:
            self._rules[rule_id] = self._pending_learned.pop(rule_id)
            return True
        return False

    def list_rules(self) -> list[SymbolicRule]:
        return list(self._rules.values())

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, decision_content: str) -> SymbolicVerdict:
        """Check ``decision_content`` against all active rules."""
        satisfied: list[str] = []
        violated: list[str] = []
        hard_violation = False

        for rule in self._rules.values():
            try:
                passed = rule.predicate(decision_content)
            except Exception as exc:
                violated.append(f"{rule.rule_id} (error: {exc})")
                if rule.constraint_type == ConstraintType.HARD:
                    hard_violation = True
                continue

            if passed:
                satisfied.append(rule.rule_id)
            else:
                violated.append(rule.rule_id)
                if rule.constraint_type == ConstraintType.HARD:
                    hard_violation = True

        if hard_violation:
            counterexample = (
                f"Hard constraint(s) violated: {', '.join(violated)}"
            )
            return SymbolicVerdict(
                result=VerificationResult.REFUTED,
                satisfied_rules=satisfied,
                violated_rules=violated,
                counterexample=counterexample,
            )

        proof = f"All hard constraints satisfied. Rules evaluated: {len(self._rules)}."
        return SymbolicVerdict(
            result=VerificationResult.VERIFIED,
            satisfied_rules=satisfied,
            violated_rules=violated,
            proof_trace=proof,
        )

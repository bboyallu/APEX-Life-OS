"""Tests for the Governance layer (safety constraints and audit ledger)."""

import pytest

from apex.governance.audit import AuditLedger
from apex.governance.constraints import (
    ImmutableConstraint,
    MutablePolicy,
    SafetyConstraintRegistry,
)


# ---------------------------------------------------------------------------
# SafetyConstraintRegistry
# ---------------------------------------------------------------------------


class TestSafetyConstraintRegistry:
    def test_default_hard_constraints_loaded(self):
        registry = SafetyConstraintRegistry()
        constraints = registry.list_hard_constraints()
        assert len(constraints) >= 5

    def test_no_self_replication_constraint_present(self):
        registry = SafetyConstraintRegistry()
        c = registry.get_hard_constraint("safety:no-self-replication")
        assert c is not None
        assert isinstance(c, ImmutableConstraint)

    def test_check_hard_constraint_exists(self):
        registry = SafetyConstraintRegistry()
        assert registry.check_hard_constraint("safety:human-veto") is True
        assert registry.check_hard_constraint("nonexistent") is False

    def test_register_and_retrieve_soft_policy(self):
        registry = SafetyConstraintRegistry()
        policy = MutablePolicy(policy_id="perf:max_latency", description="Max latency", value=200)
        registry.register_policy(policy)
        retrieved = registry.get_policy("perf:max_latency")
        assert retrieved is not None
        assert retrieved.value == 200

    def test_update_soft_policy(self):
        registry = SafetyConstraintRegistry()
        policy = MutablePolicy(policy_id="perf:budget", description="Budget", value=100)
        registry.register_policy(policy)
        updated = registry.update_policy("perf:budget", 200)
        assert updated is True
        assert registry.get_policy("perf:budget").value == 200

    def test_update_nonexistent_policy_returns_false(self):
        registry = SafetyConstraintRegistry()
        assert registry.update_policy("nonexistent", 42) is False

    def test_list_policies(self):
        registry = SafetyConstraintRegistry()
        registry.register_policy(MutablePolicy("p1", "P1", value=1))
        registry.register_policy(MutablePolicy("p2", "P2", value=2))
        policies = registry.list_policies()
        ids = {p.policy_id for p in policies}
        assert {"p1", "p2"}.issubset(ids)

    def test_would_violate_self_replication(self):
        registry = SafetyConstraintRegistry()
        violated = registry.would_violate_hard_constraints({"self_replication"})
        assert "safety:no-self-replication" in violated

    def test_safe_action_no_violations(self):
        registry = SafetyConstraintRegistry()
        violated = registry.would_violate_hard_constraints({"log_metric"})
        assert violated == []


# ---------------------------------------------------------------------------
# AuditLedger
# ---------------------------------------------------------------------------


class TestAuditLedger:
    def test_append_and_read(self):
        ledger = AuditLedger()
        ledger.append("plan_executed", actor="executor", payload={"plan_id": "p1"})
        entries = ledger.read()
        assert len(entries) == 1
        assert entries[0].event_type == "plan_executed"
        assert entries[0].actor == "executor"

    def test_chain_valid_after_multiple_appends(self):
        ledger = AuditLedger()
        for i in range(5):
            ledger.append("event", actor="system", payload={"i": i})
        valid, message = ledger.verify_chain()
        assert valid is True

    def test_empty_ledger_is_valid(self):
        ledger = AuditLedger()
        valid, message = ledger.verify_chain()
        assert valid is True

    def test_tampered_signature_detected(self):
        ledger = AuditLedger()
        ledger.append("event", actor="system", payload={"x": 1})
        # Tamper with the signature
        ledger._entries[0] = ledger._entries[0].__class__(
            entry_id=ledger._entries[0].entry_id,
            timestamp=ledger._entries[0].timestamp,
            event_type=ledger._entries[0].event_type,
            actor=ledger._entries[0].actor,
            payload=ledger._entries[0].payload,
            signature="tampered_signature",
            previous_hash=ledger._entries[0].previous_hash,
        )
        valid, message = ledger.verify_chain()
        assert valid is False
        assert "broken" in message.lower()

    def test_read_by_type(self):
        ledger = AuditLedger()
        ledger.append("type_a", actor="a", payload={})
        ledger.append("type_b", actor="b", payload={})
        ledger.append("type_a", actor="c", payload={})
        entries = ledger.read_by_type("type_a")
        assert len(entries) == 2

    def test_signature_uniqueness(self):
        ledger = AuditLedger()
        ledger.append("e1", actor="a", payload={"v": 1})
        ledger.append("e2", actor="a", payload={"v": 2})
        sigs = {e.signature for e in ledger.read()}
        assert len(sigs) == 2

    def test_previous_hash_chain(self):
        ledger = AuditLedger()
        ledger.append("e1", actor="a", payload={})
        ledger.append("e2", actor="a", payload={})
        entries = ledger.read()
        assert entries[1].previous_hash == entries[0].signature

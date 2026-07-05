"""Tests for the Decision Orchestration layer."""

import pytest

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import DecisionPath
from apex.neuro_symbolic.neural import NeuralSubsystem
from apex.neuro_symbolic.symbolic import SymbolicSubsystem
from apex.neuro_symbolic.verifier import VerificationPipeline
from apex.orchestration.orchestrator import DecisionOrchestrator, Policy, PolicyConflict
from apex.orchestration.selector import DecisionContext, PathSelector, Urgency


# ---------------------------------------------------------------------------
# PathSelector
# ---------------------------------------------------------------------------


class TestPathSelector:
    def _selector(self):
        return PathSelector()

    def test_reflexive_critical_low_risk(self):
        ctx = DecisionContext(urgency=Urgency.CRITICAL, risk=0.05, confidence=0.9)
        assert self._selector().select(ctx) == DecisionPath.REFLEXIVE

    def test_collaborative_high_novelty(self):
        ctx = DecisionContext(novelty=0.95, risk=0.1, confidence=0.9)
        assert self._selector().select(ctx) == DecisionPath.COLLABORATIVE

    def test_adversarial_safety_affecting(self):
        ctx = DecisionContext(affects_safety_constraints=True, risk=0.1)
        assert self._selector().select(ctx) == DecisionPath.ADVERSARIAL

    def test_adversarial_high_risk(self):
        ctx = DecisionContext(risk=0.85)
        assert self._selector().select(ctx) == DecisionPath.ADVERSARIAL

    def test_deliberative_high_confidence(self):
        ctx = DecisionContext(confidence=0.95, risk=0.05, novelty=0.05)
        assert self._selector().select(ctx) == DecisionPath.DELIBERATIVE

    def test_collaborative_default(self):
        ctx = DecisionContext(confidence=0.50, risk=0.10, novelty=0.10)
        assert self._selector().select(ctx) == DecisionPath.COLLABORATIVE

    def test_custom_thresholds(self):
        selector = PathSelector(low_threshold=0.50, high_threshold=0.90)
        ctx = DecisionContext(urgency=Urgency.CRITICAL, risk=0.40)
        # risk=0.40 < low_threshold=0.50, urgency=CRITICAL → REFLEXIVE
        assert selector.select(ctx) == DecisionPath.REFLEXIVE


# ---------------------------------------------------------------------------
# DecisionOrchestrator
# ---------------------------------------------------------------------------


def _make_orchestrator(kb=None):
    kb = kb or KnowledgeBase()
    pipeline = VerificationPipeline(
        NeuralSubsystem(), SymbolicSubsystem(), kb
    )
    selector = PathSelector()
    return DecisionOrchestrator(selector, pipeline, kb), kb


class TestDecisionOrchestrator:
    def test_decide_returns_path_and_record(self):
        orch, kb = _make_orchestrator()
        ctx = DecisionContext(confidence=0.95, risk=0.05, novelty=0.05)
        path, record = orch.decide("What should I do?", ctx)
        assert path in list(DecisionPath)
        assert record is not None

    def test_highest_rigor_path_governs(self):
        # Patch: inject an adversarial context so the call stack contains ADVERSARIAL
        orch, _ = _make_orchestrator()
        ctx = DecisionContext(affects_safety_constraints=True)
        path, _ = orch.decide("sensitive decision", ctx)
        # Adversarial should govern
        assert path == DecisionPath.ADVERSARIAL

    def test_register_policy(self):
        orch, _ = _make_orchestrator()
        policy = Policy(policy_id="p1", name="low_latency", priority=1)
        orch.register_policy(policy)
        assert "p1" in orch._policies

    def test_detect_no_conflicts_when_unique_priorities(self):
        orch, _ = _make_orchestrator()
        orch.register_policy(Policy("a", "A", priority=1))
        orch.register_policy(Policy("b", "B", priority=2))
        assert orch.detect_conflicts() == []

    def test_detect_conflicts_same_priority(self):
        orch, _ = _make_orchestrator()
        orch.register_policy(Policy("a", "A", priority=1))
        orch.register_policy(Policy("b", "B", priority=1))
        conflicts = orch.detect_conflicts()
        assert len(conflicts) == 1

    def test_arbitrate_hard_constraint_dominates(self):
        orch, _ = _make_orchestrator()
        hard = Policy("safety", "Safety", priority=1, is_hard_constraint=True)
        soft = Policy("perf", "Performance", priority=1)
        orch.register_policy(hard)
        orch.register_policy(soft)
        conflicts = orch.detect_conflicts()
        resolved = orch.arbitrate(conflicts)
        assert resolved[0].resolution is not None
        assert "Safety" in resolved[0].resolution

    def test_arbitrate_kb_weight_resolution(self):
        orch, kb = _make_orchestrator()
        kb.set_world_context("policy_weight:p_high", 0.9)
        kb.set_world_context("policy_weight:p_low", 0.1)
        p_high = Policy("p_high", "High priority", priority=1)
        p_low = Policy("p_low", "Low priority", priority=1)
        orch.register_policy(p_high)
        orch.register_policy(p_low)
        conflicts = orch.detect_conflicts()
        resolved = orch.arbitrate(conflicts)
        assert "High priority" in resolved[0].resolution

    def test_arbitrate_escalation_when_no_resolution(self):
        escalated = []
        orch, kb = _make_orchestrator()
        orch._on_escalation = lambda c: escalated.append(c)
        orch.register_policy(Policy("a", "A", priority=1))
        orch.register_policy(Policy("b", "B", priority=1))
        conflicts = orch.detect_conflicts()
        orch.arbitrate(conflicts)
        # Both have equal weights — should escalate
        assert len(escalated) == 1

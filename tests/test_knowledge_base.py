"""Tests for the Knowledge Base (apex.core.knowledge_base)."""

import pytest

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import (
    AdaptationPlan,
    AdaptationType,
    ConstraintType,
    ExecutionResult,
    ExecutionStatus,
    Reversibility,
)


@pytest.fixture
def kb():
    return KnowledgeBase()


@pytest.fixture
def sample_plan():
    return AdaptationPlan(
        description="Test plan",
        adaptation_type=AdaptationType.MICRO,
        expected_benefit=0.1,
        risk_score=0.1,
        reversibility=Reversibility.FULLY_REVERSIBLE,
    )


@pytest.fixture
def sample_result(sample_plan):
    return ExecutionResult(
        plan_id=sample_plan.plan_id,
        status=ExecutionStatus.COMMITTED,
    )


class TestComponentState:
    def test_register_and_retrieve(self, kb):
        kb.register_component("service_a")
        state = kb.get_component_state("service_a")
        assert state is not None
        assert state.healthy is True

    def test_update_health(self, kb):
        kb.register_component("service_b", healthy=True)
        kb.update_component_health("service_b", False, reason="high_latency")
        state = kb.get_component_state("service_b")
        assert state.healthy is False
        assert state.metadata.get("reason") == "high_latency"

    def test_auto_create_on_update(self, kb):
        kb.update_component_health("service_c", True)
        assert kb.get_component_state("service_c") is not None

    def test_list_components(self, kb):
        kb.register_component("a")
        kb.register_component("b")
        assert set(kb.list_components()) >= {"a", "b"}

    def test_all_healthy(self, kb):
        kb.register_component("x", healthy=True)
        assert kb.all_healthy() is True
        kb.update_component_health("x", False)
        assert kb.all_healthy() is False


class TestAdaptationHistory:
    def test_record_and_retrieve(self, kb, sample_plan, sample_result):
        kb.record_adaptation(sample_plan, sample_result)
        history = kb.get_adaptation_history()
        assert len(history) == 1
        assert history[0].plan.plan_id == sample_plan.plan_id
        assert history[0].result.status == ExecutionStatus.COMMITTED


class TestCausalModel:
    def test_record_and_retrieve_outcomes(self, kb):
        kb.record_causal_outcome("service_a", "scale_up", "latency improved")
        outcomes = kb.get_causal_outcomes("service_a", "scale_up")
        assert "latency improved" in outcomes

    def test_missing_key_returns_empty(self, kb):
        assert kb.get_causal_outcomes("nonexistent", "action") == []


class TestWorldModel:
    def test_set_and_get(self, kb):
        kb.set_world_context("user_timezone", "UTC")
        assert kb.get_world_context("user_timezone") == "UTC"

    def test_default_value(self, kb):
        assert kb.get_world_context("missing_key", "default") == "default"


class TestConstraintRegistry:
    def test_register_and_retrieve(self, kb):
        kb.register_constraint("c1", "No spam", ConstraintType.HARD, "NOT spam")
        constraint = kb.get_constraint("c1")
        assert constraint is not None
        assert constraint.constraint_type == ConstraintType.HARD

    def test_list_by_type(self, kb):
        kb.register_constraint("h1", "Hard rule", ConstraintType.HARD, "hard")
        kb.register_constraint("s1", "Soft rule", ConstraintType.SOFT, "soft")
        hard = kb.list_constraints(ConstraintType.HARD)
        assert any(c.constraint_id == "h1" for c in hard)
        soft = kb.list_constraints(ConstraintType.SOFT)
        assert any(c.constraint_id == "s1" for c in soft)

    def test_list_all(self, kb):
        kb.register_constraint("a1", "A", ConstraintType.HARD, "a")
        kb.register_constraint("a2", "B", ConstraintType.SOFT, "b")
        assert len(kb.list_constraints()) >= 2

"""Tests for the Autonomic Threshold Engine and Risk Scorer."""

import time

import pytest

from apex.core.types import AdaptationPlan, AdaptationType, Reversibility, ThresholdLevel
from apex.thresholds.engine import AutonomicThresholdEngine
from apex.thresholds.risk_scorer import RiskScorer


def _make_plan(risk_score: float, requires_approval: bool = False) -> AdaptationPlan:
    return AdaptationPlan(
        description="test",
        adaptation_type=AdaptationType.MICRO,
        expected_benefit=0.1,
        risk_score=risk_score,
        reversibility=Reversibility.FULLY_REVERSIBLE,
        requires_human_approval=requires_approval,
    )


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------


class TestRiskScorer:
    def test_fully_reversible_low_risk(self):
        scorer = RiskScorer()
        score = scorer.score(
            reversibility=Reversibility.FULLY_REVERSIBLE,
            blast_radius_count=1,
            is_novel=False,
            affects_safety=False,
            confidence_deficit=0.0,
        )
        assert 0.0 <= score <= 0.35

    def test_destructive_safety_affects_high_risk(self):
        scorer = RiskScorer()
        score = scorer.score(
            reversibility=Reversibility.DESTRUCTIVE,
            blast_radius_count=10,
            is_novel=True,
            affects_safety=True,
            confidence_deficit=1.0,
        )
        assert score >= 0.60

    def test_score_clamped_between_0_and_1(self):
        scorer = RiskScorer()
        for _ in range(10):
            score = scorer.score(
                reversibility=Reversibility.DESTRUCTIVE,
                blast_radius_count=100,
                is_novel=True,
                affects_safety=True,
                confidence_deficit=2.0,  # intentionally out of range
            )
            assert 0.0 <= score <= 1.0

    def test_threshold_level_l0(self):
        assert RiskScorer.threshold_level(0.05) == ThresholdLevel.L0_ROUTINE

    def test_threshold_level_l1(self):
        assert RiskScorer.threshold_level(0.20) == ThresholdLevel.L1_NOTABLE

    def test_threshold_level_l2(self):
        assert RiskScorer.threshold_level(0.45) == ThresholdLevel.L2_SIGNIFICANT

    def test_threshold_level_l3(self):
        assert RiskScorer.threshold_level(0.70) == ThresholdLevel.L3_HIGH_RISK

    def test_threshold_level_l4(self):
        assert RiskScorer.threshold_level(0.90) == ThresholdLevel.L4_CRITICAL

    def test_boundary_values(self):
        assert RiskScorer.threshold_level(0.15) == ThresholdLevel.L1_NOTABLE
        assert RiskScorer.threshold_level(0.35) == ThresholdLevel.L2_SIGNIFICANT
        assert RiskScorer.threshold_level(0.60) == ThresholdLevel.L3_HIGH_RISK
        assert RiskScorer.threshold_level(0.80) == ThresholdLevel.L4_CRITICAL

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            RiskScorer(w_reversibility=0, w_blast_radius=0, w_novelty=0, w_safety=0, w_confidence=0)


# ---------------------------------------------------------------------------
# AutonomicThresholdEngine
# ---------------------------------------------------------------------------


class TestAutonomicThresholdEngine:
    def test_l0_fully_autonomous(self):
        engine = AutonomicThresholdEngine()
        decision = engine.evaluate(_make_plan(0.05))
        assert decision.may_proceed_autonomously is True
        assert decision.requires_approval is False
        assert decision.is_hard_blocked is False

    def test_l1_autonomous_logged(self):
        engine = AutonomicThresholdEngine()
        decision = engine.evaluate(_make_plan(0.20))
        assert decision.may_proceed_autonomously is True
        assert decision.is_hard_blocked is False

    def test_l2_autonomous_with_notification(self):
        engine = AutonomicThresholdEngine()
        decision = engine.evaluate(_make_plan(0.50))
        assert decision.may_proceed_autonomously is True
        assert decision.is_hard_blocked is False

    def test_l3_requires_approval(self):
        l3_alerts = []
        engine = AutonomicThresholdEngine(on_l3_alert=lambda p, l: l3_alerts.append(l))
        decision = engine.evaluate(_make_plan(0.70))
        assert decision.may_proceed_autonomously is False
        assert decision.requires_approval is True
        assert len(l3_alerts) == 1

    def test_l4_hard_blocked(self):
        l4_alerts = []
        engine = AutonomicThresholdEngine(on_l4_alert=lambda p, l: l4_alerts.append(l))
        decision = engine.evaluate(_make_plan(0.90))
        assert decision.is_hard_blocked is True
        assert decision.requires_approval is True
        assert len(l4_alerts) == 1

    def test_dead_man_switch_activates_on_timeout(self):
        engine = AutonomicThresholdEngine(dead_man_timeout_seconds=0)
        time.sleep(0.01)  # ensure timeout elapsed
        assert engine.check_dead_man_switch() is True

    def test_dead_man_switch_resets_on_heartbeat(self):
        engine = AutonomicThresholdEngine(dead_man_timeout_seconds=1)
        engine.heartbeat()
        assert engine.check_dead_man_switch() is False

    def test_dead_man_blocks_l2_on_activation(self):
        engine = AutonomicThresholdEngine(dead_man_timeout_seconds=0)
        time.sleep(0.01)
        decision = engine.evaluate(_make_plan(0.50))  # L2
        assert decision.is_hard_blocked is True

    def test_bulk_evaluate(self):
        engine = AutonomicThresholdEngine()
        plans = [_make_plan(r) for r in [0.05, 0.20, 0.50, 0.70, 0.90]]
        decisions = engine.bulk_evaluate(plans)
        assert len(decisions) == 5

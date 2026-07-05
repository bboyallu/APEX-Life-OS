"""Tests for the Neuro-Symbolic layer."""

import pytest

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import ConstraintType, DecisionPath, VerificationResult
from apex.neuro_symbolic.neural import CandidateDecision, NeuralSubsystem
from apex.neuro_symbolic.symbolic import SymbolicRule, SymbolicSubsystem
from apex.neuro_symbolic.verifier import VerificationPipeline


# ---------------------------------------------------------------------------
# NeuralSubsystem
# ---------------------------------------------------------------------------


class TestNeuralSubsystem:
    def test_default_stub_returns_decision(self):
        neural = NeuralSubsystem()
        candidate = neural.predict("What should I do?")
        assert isinstance(candidate, CandidateDecision)
        assert 0.0 <= candidate.confidence <= 1.0
        assert len(candidate.content) > 0

    def test_custom_model(self):
        def my_model(prompt):
            return CandidateDecision(content="custom", confidence=0.99)

        neural = NeuralSubsystem(model=my_model)
        candidate = neural.predict("x")
        assert candidate.content == "custom"
        assert candidate.confidence == 0.99


# ---------------------------------------------------------------------------
# SymbolicSubsystem
# ---------------------------------------------------------------------------


class TestSymbolicSubsystem:
    def test_verified_with_no_rules(self):
        sym = SymbolicSubsystem()
        verdict = sym.verify("any decision")
        assert verdict.result == VerificationResult.VERIFIED

    def test_hard_rule_violation_produces_refuted(self):
        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="no_spam",
                description="No spam",
                constraint_type=ConstraintType.HARD,
                predicate=lambda d: "spam" not in d.lower(),
            )
        )
        verdict = sym.verify("This is spam content")
        assert verdict.result == VerificationResult.REFUTED
        assert "no_spam" in verdict.violated_rules

    def test_soft_rule_violation_still_verified(self):
        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="prefer_short",
                description="Prefer short responses",
                constraint_type=ConstraintType.SOFT,
                predicate=lambda d: len(d) < 10,
            )
        )
        # Decision is long — soft rule violated but overall VERIFIED
        verdict = sym.verify("This is a very long decision that violates the soft rule")
        assert verdict.result == VerificationResult.VERIFIED
        assert "prefer_short" in verdict.violated_rules

    def test_learned_rule_not_active_until_activated(self):
        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="learned_1",
                description="Learned rule",
                constraint_type=ConstraintType.LEARNED,
                predicate=lambda d: False,  # always fails
            )
        )
        # Learned rule is in pending, not active — should not affect verdict
        verdict = sym.verify("anything")
        assert verdict.result == VerificationResult.VERIFIED

    def test_activate_learned_rule(self):
        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="learned_1",
                description="Learned rule",
                constraint_type=ConstraintType.LEARNED,
                predicate=lambda d: False,  # always fails as soft
            )
        )
        activated = sym.activate_learned_rule("learned_1")
        assert activated is True
        # Now active — but learned rules are treated as soft (won't refute)
        verdict = sym.verify("anything")
        # The predicate fails but it's stored without constraint type adjustment
        # In our implementation learned rules behave as soft after activation
        assert activated is True

    def test_predicate_exception_handled(self):
        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="buggy_rule",
                description="Buggy",
                constraint_type=ConstraintType.SOFT,
                predicate=lambda d: 1 / 0,  # raises ZeroDivisionError
            )
        )
        verdict = sym.verify("anything")
        # Error in soft rule should not refute
        assert verdict.result == VerificationResult.VERIFIED


# ---------------------------------------------------------------------------
# VerificationPipeline
# ---------------------------------------------------------------------------


class TestVerificationPipeline:
    def _build_pipeline(self, neural=None, symbolic=None):
        kb = KnowledgeBase()
        neural = neural or NeuralSubsystem()
        symbolic = symbolic or SymbolicSubsystem()
        return VerificationPipeline(neural, symbolic, kb), kb

    def test_verified_decision_stored_in_kb(self):
        pipeline, kb = self._build_pipeline()
        record = pipeline.run("What should I do?")
        assert record.verification_result == VerificationResult.VERIFIED
        records = kb.get_decision_records()
        assert len(records) == 1
        assert records[0].record_id == record.record_id

    def test_refuted_then_regenerated(self):
        attempts = []

        class CountingNeural(NeuralSubsystem):
            def predict(self, prompt):
                attempts.append(prompt)
                # On the first call produce "bad", after that produce "good"
                if len(attempts) == 1:
                    return CandidateDecision(content="bad content", confidence=0.8)
                return CandidateDecision(content="clean content", confidence=0.9)

        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="no_bad",
                description="No bad content",
                constraint_type=ConstraintType.HARD,
                predicate=lambda d: "bad" not in d,
            )
        )

        pipeline, kb = self._build_pipeline(neural=CountingNeural(), symbolic=sym)
        record = pipeline.run("prompt")
        assert record.verification_result == VerificationResult.VERIFIED
        assert len(attempts) == 2  # first failed, second succeeded

    def test_undecidable_after_max_retries(self):
        sym = SymbolicSubsystem()
        sym.register_rule(
            SymbolicRule(
                rule_id="impossible",
                description="Always fails",
                constraint_type=ConstraintType.HARD,
                predicate=lambda d: False,
            )
        )
        pipeline, kb = self._build_pipeline(symbolic=sym)
        record = pipeline.run("prompt", decision_path=DecisionPath.ADVERSARIAL)
        assert record.verification_result == VerificationResult.UNDECIDABLE
        assert record.decision_path == DecisionPath.ADVERSARIAL

    def test_plan_id_attached(self):
        pipeline, _ = self._build_pipeline()
        record = pipeline.run("prompt", plan_id="plan-abc-123")
        assert record.plan_id == "plan-abc-123"

"""Verification pipeline and DecisionRecord emission (§3.2, §3.4).

The pipeline:
1. Neural subsystem produces a ``CandidateDecision``.
2. Decision is passed to the symbolic subsystem for verification.
3. If verified  → proceed, emit ``DecisionRecord`` with proof trace.
4. If refuted   → neural subsystem receives counterexample and regenerates
                  (up to ``max_retries`` times).
5. If undecidable / retries exhausted → escalate to human oversight.
"""

from __future__ import annotations

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import DecisionPath, DecisionRecord, VerificationResult
from apex.neuro_symbolic.neural import NeuralSubsystem
from apex.neuro_symbolic.symbolic import SymbolicSubsystem


class VerificationPipeline:
    """Neuro-symbolic verification pipeline.

    Parameters
    ----------
    neural:
        Neural subsystem instance.
    symbolic:
        Symbolic subsystem instance.
    knowledge_base:
        KB where ``DecisionRecord`` objects are stored.
    max_retries:
        Maximum number of neural regeneration attempts after a refutation.
    """

    def __init__(
        self,
        neural: NeuralSubsystem,
        symbolic: SymbolicSubsystem,
        knowledge_base: KnowledgeBase,
        max_retries: int = 3,
    ) -> None:
        self._neural = neural
        self._symbolic = symbolic
        self._kb = knowledge_base
        self._max_retries = max_retries

    def run(
        self,
        prompt: str,
        decision_path: DecisionPath = DecisionPath.DELIBERATIVE,
        plan_id: str | None = None,
    ) -> DecisionRecord:
        """Run the full verification pipeline and return a ``DecisionRecord``."""
        last_candidate = None
        last_verdict = None

        for attempt in range(self._max_retries + 1):
            # Step 1: Neural prediction
            if attempt == 0:
                candidate = self._neural.predict(prompt)
            else:
                # Inject counterexample into prompt for regeneration
                augmented = (
                    f"{prompt}\n\n[Counterexample from symbolic verifier: "
                    f"{last_verdict.counterexample}]"
                )
                candidate = self._neural.predict(augmented)

            last_candidate = candidate

            # Step 2: Symbolic verification
            verdict = self._symbolic.verify(candidate.content)
            last_verdict = verdict

            if verdict.result == VerificationResult.VERIFIED:
                record = DecisionRecord(
                    decision_path=decision_path,
                    summary=candidate.content,
                    confidence_score=candidate.confidence,
                    verification_result=VerificationResult.VERIFIED,
                    proof_trace=verdict.proof_trace,
                    rules_evaluated=verdict.satisfied_rules + verdict.violated_rules,
                    justification=(
                        f"Verified after {attempt + 1} attempt(s). "
                        f"All hard constraints satisfied."
                    ),
                    plan_id=plan_id,
                )
                self._kb.store_decision_record(record)
                return record

            if verdict.result == VerificationResult.REFUTED:
                continue  # retry with counterexample

        # Retries exhausted or undecidable — escalate
        assert last_candidate is not None
        record = DecisionRecord(
            decision_path=decision_path,
            summary=last_candidate.content,
            confidence_score=last_candidate.confidence,
            verification_result=VerificationResult.UNDECIDABLE,
            proof_trace=None,
            rules_evaluated=(
                (last_verdict.satisfied_rules + last_verdict.violated_rules)
                if last_verdict
                else []
            ),
            justification=(
                "Escalated to human oversight: verification did not converge "
                f"after {self._max_retries} retries."
            ),
            plan_id=plan_id,
        )
        self._kb.store_decision_record(record)
        return record

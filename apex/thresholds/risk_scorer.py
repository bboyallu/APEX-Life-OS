"""Risk scorer — computes the composite risk score for an adaptation plan (§5.2).

Risk score formula::

    risk = w1 * reversibility_penalty
         + w2 * blast_radius_score
         + w3 * novelty_score
         + w4 * safety_proximity_score
         + w5 * confidence_deficit

Weights are normalised so they always sum to 1.0.
"""

from __future__ import annotations

from apex.core.types import Reversibility, ThresholdLevel


_REVERSIBILITY_PENALTY: dict[Reversibility, float] = {
    Reversibility.FULLY_REVERSIBLE: 0.05,
    Reversibility.PARTIALLY_REVERSIBLE: 0.50,
    Reversibility.DESTRUCTIVE: 1.00,
}

# Blast-radius score: normalised against an assumed max of 10 components
_MAX_BLAST_RADIUS = 10


class RiskScorer:
    """Compute a composite 0.0–1.0 risk score for an adaptation plan.

    Parameters
    ----------
    w_reversibility, w_blast_radius, w_novelty, w_safety, w_confidence:
        Component weights.  They are normalised internally so they need not
        sum to 1.0 as provided.
    """

    def __init__(
        self,
        w_reversibility: float = 0.25,
        w_blast_radius: float = 0.25,
        w_novelty: float = 0.20,
        w_safety: float = 0.20,
        w_confidence: float = 0.10,
    ) -> None:
        total = w_reversibility + w_blast_radius + w_novelty + w_safety + w_confidence
        if total <= 0:
            raise ValueError("Weights must be positive")
        self._w1 = w_reversibility / total
        self._w2 = w_blast_radius / total
        self._w3 = w_novelty / total
        self._w4 = w_safety / total
        self._w5 = w_confidence / total

    def score(
        self,
        *,
        reversibility: Reversibility,
        blast_radius_count: int,
        is_novel: bool,
        affects_safety: bool,
        confidence_deficit: float,
    ) -> float:
        """Return a 0.0–1.0 risk score."""
        rev = _REVERSIBILITY_PENALTY[reversibility]
        blast = min(blast_radius_count / _MAX_BLAST_RADIUS, 1.0)
        novelty = 1.0 if is_novel else 0.0
        safety = 1.0 if affects_safety else 0.0
        conf = max(0.0, min(confidence_deficit, 1.0))

        raw = (
            self._w1 * rev
            + self._w2 * blast
            + self._w3 * novelty
            + self._w4 * safety
            + self._w5 * conf
        )
        return round(min(max(raw, 0.0), 1.0), 4)

    @staticmethod
    def threshold_level(risk_score: float) -> ThresholdLevel:
        """Map a risk score to its autonomic threshold level."""
        if risk_score < 0.15:
            return ThresholdLevel.L0_ROUTINE
        if risk_score < 0.35:
            return ThresholdLevel.L1_NOTABLE
        if risk_score < 0.60:
            return ThresholdLevel.L2_SIGNIFICANT
        if risk_score < 0.80:
            return ThresholdLevel.L3_HIGH_RISK
        return ThresholdLevel.L4_CRITICAL

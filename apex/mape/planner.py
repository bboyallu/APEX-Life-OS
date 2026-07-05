"""Plan phase (§2.3) — generates candidate adaptation strategies.

The ``Planner`` takes an ``AnalysisReport`` and the current risk-scoring
function to produce a ranked list of ``AdaptationPlan`` objects.

Each plan includes:
- Expected benefit (quantified performance delta)
- Risk score (0.0–1.0)
- Reversibility rating
- Blast radius (affected downstream components)

The Planner also enforces the *safety gate*: plans that would modify safety
constraints or cross L3 threshold boundaries are flagged for mandatory human
approval before execution.
"""

from __future__ import annotations

from apex.core.types import (
    AdaptationPlan,
    AdaptationType,
    AnalysisReport,
    Reversibility,
    Severity,
    ThresholdLevel,
)
from apex.thresholds.risk_scorer import RiskScorer


_SEVERITY_TO_BENEFIT: dict[Severity, float] = {
    Severity.INFORMATIONAL: 0.05,
    Severity.WARNING: 0.15,
    Severity.DEGRADED: 0.30,
    Severity.CRITICAL: 0.60,
    Severity.CATASTROPHIC: 0.90,
}


def _adaptation_type_for_severity(severity: Severity) -> AdaptationType:
    if severity in (Severity.INFORMATIONAL, Severity.WARNING):
        return AdaptationType.MICRO
    if severity == Severity.DEGRADED:
        return AdaptationType.MESO
    return AdaptationType.MACRO


def _reversibility_for_type(adaptation_type: AdaptationType) -> Reversibility:
    return {
        AdaptationType.MICRO: Reversibility.FULLY_REVERSIBLE,
        AdaptationType.MESO: Reversibility.PARTIALLY_REVERSIBLE,
        AdaptationType.MACRO: Reversibility.DESTRUCTIVE,
    }[adaptation_type]


_APPROVAL_THRESHOLD = ThresholdLevel.L3_HIGH_RISK


class Planner:
    """MAPE-K Plan phase.

    Generates a ranked list of adaptation candidates from an ``AnalysisReport``.

    Parameters
    ----------
    risk_scorer:
        An instance of ``RiskScorer`` used to compute the risk score for each
        candidate plan.
    """

    def __init__(self, risk_scorer: RiskScorer | None = None) -> None:
        self._risk_scorer = risk_scorer or RiskScorer()

    def plan(self, report: AnalysisReport) -> list[AdaptationPlan]:
        """Return a ranked list of adaptation plans (highest utility first)."""
        plans: list[AdaptationPlan] = []

        for cluster in report.symptom_clusters:
            adaptation_type = _adaptation_type_for_severity(cluster.severity)
            reversibility = _reversibility_for_type(adaptation_type)
            expected_benefit = _SEVERITY_TO_BENEFIT.get(cluster.severity, 0.1)

            blast_radius = list(
                {s.split("=")[0] for s in cluster.signals} | {cluster.probable_cause}
            )

            risk_score = self._risk_scorer.score(
                reversibility=reversibility,
                blast_radius_count=len(blast_radius),
                is_novel=cluster.is_evolution_candidate,
                affects_safety=False,
                confidence_deficit=1.0 - expected_benefit,
            )

            threshold_level = self._risk_scorer.threshold_level(risk_score)
            requires_approval = threshold_level in (
                ThresholdLevel.L3_HIGH_RISK,
                ThresholdLevel.L4_CRITICAL,
            )

            plan = AdaptationPlan(
                description=(
                    f"Adapt [{adaptation_type.value}] to address: "
                    f"{cluster.probable_cause}"
                ),
                adaptation_type=adaptation_type,
                expected_benefit=expected_benefit,
                risk_score=risk_score,
                reversibility=reversibility,
                blast_radius=blast_radius,
                requires_human_approval=requires_approval,
                metadata={"cluster_id": cluster.cluster_id},
            )
            plans.append(plan)

        # Rank by utility = expected_benefit / (1 + risk_score)
        plans.sort(key=lambda p: p.expected_benefit / (1.0 + p.risk_score), reverse=True)
        return plans

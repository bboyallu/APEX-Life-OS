"""Threshold calibration review (§Phase 5) — continuous hardening of the
autonomic threshold taxonomy.

The ``ThresholdCalibrator`` reviews the adaptation history stored in the
``KnowledgeBase`` and compares *predicted* risk (the threshold level derived
from each plan's risk score) against *actual* outcomes (committed vs. rolled
back vs. blocked).  It produces a ``CalibrationReport`` with per-level
statistics and human-reviewable recommendations.

Recommendations are advisory only: threshold boundaries are safety-relevant
configuration, so the calibrator never mutates the ``RiskScorer`` itself.
Applying a recommendation is an L3+ act requiring human approval.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import ExecutionStatus, ThresholdLevel
from apex.thresholds.risk_scorer import RiskScorer


class LevelCalibrationStats(BaseModel):
    """Outcome statistics for a single threshold level."""

    level: ThresholdLevel
    total: int = 0
    committed: int = 0
    rolled_back: int = 0
    blocked: int = 0

    @property
    def rollback_rate(self) -> float:
        """Fraction of executed (non-blocked) plans that were rolled back."""
        executed = self.committed + self.rolled_back
        if executed == 0:
            return 0.0
        return self.rolled_back / executed


class CalibrationRecommendation(BaseModel):
    """A single advisory recommendation from a calibration review."""

    level: ThresholdLevel
    kind: str  # "tighten" | "relax" | "insufficient_data"
    message: str


class CalibrationReport(BaseModel):
    """Structured output of a threshold calibration review."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sample_size: int = 0
    stats: list[LevelCalibrationStats] = Field(default_factory=list)
    recommendations: list[CalibrationRecommendation] = Field(default_factory=list)

    def stats_for(self, level: ThresholdLevel) -> LevelCalibrationStats | None:
        for s in self.stats:
            if s.level == level:
                return s
        return None


# Levels on which the system may act without synchronous human approval.
_AUTONOMOUS_LEVELS = (
    ThresholdLevel.L0_ROUTINE,
    ThresholdLevel.L1_NOTABLE,
    ThresholdLevel.L2_SIGNIFICANT,
)

_GATED_LEVELS = (ThresholdLevel.L3_HIGH_RISK, ThresholdLevel.L4_CRITICAL)


class ThresholdCalibrator:
    """Reviews adaptation outcomes to validate threshold boundaries.

    Parameters
    ----------
    knowledge_base:
        Shared KB holding the adaptation history.
    risk_scorer:
        Scorer whose ``threshold_level`` mapping is under review.
    min_samples:
        Minimum executions at a level before a recommendation is made.
    max_autonomous_rollback_rate:
        If an autonomous level (L0–L2) rolls back more often than this,
        the calibrator recommends tightening (lowering) its upper boundary.
    max_gated_rollback_rate:
        If a human-gated level (L3–L4) shows a rollback rate at or below
        this with sufficient samples, the calibrator notes that its lower
        boundary may be relaxed after human review.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        risk_scorer: RiskScorer | None = None,
        *,
        min_samples: int = 5,
        max_autonomous_rollback_rate: float = 0.10,
        max_gated_rollback_rate: float = 0.02,
    ) -> None:
        self._kb = knowledge_base
        self._scorer = risk_scorer or RiskScorer()
        self._min_samples = min_samples
        self._max_autonomous_rollback_rate = max_autonomous_rollback_rate
        self._max_gated_rollback_rate = max_gated_rollback_rate

    def review(self) -> CalibrationReport:
        """Run a calibration review over the full adaptation history."""
        stats: dict[ThresholdLevel, LevelCalibrationStats] = {
            level: LevelCalibrationStats(level=level) for level in ThresholdLevel
        }

        history = self._kb.get_adaptation_history()
        for entry in history:
            level = self._scorer.threshold_level(entry.plan.risk_score)
            s = stats[level]
            s.total += 1
            if entry.result.status == ExecutionStatus.ROLLED_BACK:
                s.rolled_back += 1
            elif entry.result.status == ExecutionStatus.BLOCKED:
                s.blocked += 1
            elif entry.result.status == ExecutionStatus.COMMITTED:
                s.committed += 1

        recommendations = self._make_recommendations(stats)
        return CalibrationReport(
            sample_size=len(history),
            stats=list(stats.values()),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_recommendations(
        self, stats: dict[ThresholdLevel, LevelCalibrationStats]
    ) -> list[CalibrationRecommendation]:
        recommendations: list[CalibrationRecommendation] = []

        for level in _AUTONOMOUS_LEVELS:
            s = stats[level]
            executed = s.committed + s.rolled_back
            if executed < self._min_samples:
                continue
            if s.rollback_rate > self._max_autonomous_rollback_rate:
                recommendations.append(
                    CalibrationRecommendation(
                        level=level,
                        kind="tighten",
                        message=(
                            f"{level.value} rollback rate {s.rollback_rate:.0%} exceeds "
                            f"tolerance {self._max_autonomous_rollback_rate:.0%} over "
                            f"{executed} executions — consider lowering the {level.value} "
                            "upper boundary (requires human approval)."
                        ),
                    )
                )

        for level in _GATED_LEVELS:
            s = stats[level]
            executed = s.committed + s.rolled_back
            if executed < self._min_samples:
                continue
            if s.rollback_rate <= self._max_gated_rollback_rate:
                recommendations.append(
                    CalibrationRecommendation(
                        level=level,
                        kind="relax",
                        message=(
                            f"{level.value} rollback rate {s.rollback_rate:.0%} is within "
                            f"tolerance {self._max_gated_rollback_rate:.0%} over "
                            f"{executed} approved executions — the {level.value} lower "
                            "boundary may be relaxed after human review."
                        ),
                    )
                )

        return recommendations

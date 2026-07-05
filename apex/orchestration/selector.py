"""Decision Orchestration — path selection (§4.2).

The ``PathSelector`` maps a ``DecisionContext`` to the appropriate
``DecisionPath`` using the algorithm defined in §4.2::

    function selectPath(context):
        if context.urgency == CRITICAL and context.risk < LOW_THRESHOLD:
            return REFLEXIVE
        if context.novelty > NOVELTY_THRESHOLD:
            return COLLABORATIVE
        if context.affects_safety_constraints or context.risk > HIGH_THRESHOLD:
            return ADVERSARIAL
        if context.confidence > CONFIDENCE_THRESHOLD:
            return DELIBERATIVE
        return COLLABORATIVE  // default to caution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from apex.core.types import DecisionPath


class Urgency(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DecisionContext:
    """Context fed into the path-selection algorithm."""

    urgency: Urgency = Urgency.NORMAL
    risk: float = 0.0                     # 0.0–1.0
    novelty: float = 0.0                  # 0.0–1.0
    confidence: float = 1.0              # 0.0–1.0
    affects_safety_constraints: bool = False
    metadata: dict = field(default_factory=dict)


# Threshold constants from §4.2
_LOW_THRESHOLD = 0.15
_HIGH_THRESHOLD = 0.60
_NOVELTY_THRESHOLD = 0.70
_CONFIDENCE_THRESHOLD = 0.80


class PathSelector:
    """Selects the appropriate orchestration path for a given context.

    Parameters
    ----------
    low_threshold, high_threshold, novelty_threshold, confidence_threshold:
        Override the default threshold values.
    """

    def __init__(
        self,
        low_threshold: float = _LOW_THRESHOLD,
        high_threshold: float = _HIGH_THRESHOLD,
        novelty_threshold: float = _NOVELTY_THRESHOLD,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.novelty_threshold = novelty_threshold
        self.confidence_threshold = confidence_threshold

    def select(self, context: DecisionContext) -> DecisionPath:
        """Return the ``DecisionPath`` for the given context."""
        if (
            context.urgency == Urgency.CRITICAL
            and context.risk < self.low_threshold
        ):
            return DecisionPath.REFLEXIVE

        if context.novelty > self.novelty_threshold:
            return DecisionPath.COLLABORATIVE

        if (
            context.affects_safety_constraints
            or context.risk > self.high_threshold
        ):
            return DecisionPath.ADVERSARIAL

        if context.confidence > self.confidence_threshold:
            return DecisionPath.DELIBERATIVE

        return DecisionPath.COLLABORATIVE

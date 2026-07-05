"""apex.core — Core data types and the Knowledge Base."""

from apex.core.knowledge_base import KnowledgeBase
from apex.core.types import (
    AdaptationPlan,
    AdaptationType,
    AlertPayload,
    AnalysisReport,
    ConstraintType,
    DecisionPath,
    DecisionRecord,
    ExecutionResult,
    ExecutionStatus,
    Reversibility,
    Severity,
    SymptomCluster,
    ThresholdLevel,
    VerificationResult,
)

__all__ = [
    "KnowledgeBase",
    "AdaptationPlan",
    "AdaptationType",
    "AlertPayload",
    "AnalysisReport",
    "ConstraintType",
    "DecisionPath",
    "DecisionRecord",
    "ExecutionResult",
    "ExecutionStatus",
    "Reversibility",
    "Severity",
    "SymptomCluster",
    "ThresholdLevel",
    "VerificationResult",
]

"""Core data types for the APEX self-evolving AI system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity levels produced by the Analyze phase."""

    INFORMATIONAL = "informational"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class ThresholdLevel(str, Enum):
    """Autonomic threshold levels that govern human-oversight requirements."""

    L0_ROUTINE = "L0"       # 0.00–0.15 — fully autonomous
    L1_NOTABLE = "L1"       # 0.15–0.35 — autonomous + logged
    L2_SIGNIFICANT = "L2"   # 0.35–0.60 — async human notification
    L3_HIGH_RISK = "L3"     # 0.60–0.80 — synchronous human approval
    L4_CRITICAL = "L4"      # 0.80–1.00 — immediate alert + hard block


class Reversibility(str, Enum):
    FULLY_REVERSIBLE = "fully_reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    DESTRUCTIVE = "destructive"


class AdaptationType(str, Enum):
    """Granularity of a proposed adaptation."""

    MICRO = "micro"    # prompt tuning, threshold recalibration
    MESO = "meso"      # submodule replacement, tool rewiring
    MACRO = "macro"    # architecture changes, model fine-tuning


class DecisionPath(str, Enum):
    """Orchestration paths from the path registry (§4.1)."""

    REFLEXIVE = "reflexive"         # ~ms, low rigor, no human loop
    DELIBERATIVE = "deliberative"   # ~seconds, medium rigor, optional human
    COLLABORATIVE = "collaborative" # ~minutes, high rigor, human required
    ADVERSARIAL = "adversarial"     # ~minutes, maximum rigor, mandatory human


class ConstraintType(str, Enum):
    HARD = "hard"     # immutable — safety / ethics / legal
    SOFT = "soft"     # mutable via Plan
    LEARNED = "learned"  # inferred from history, validated before use


class VerificationResult(str, Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNDECIDABLE = "undecidable"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    SHADOW = "shadow"
    CANARY = "canary"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Core payloads
# ---------------------------------------------------------------------------


class SymptomCluster(BaseModel):
    """A group of correlated signals pointing at a common root cause."""

    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signals: list[str]
    probable_cause: str
    severity: Severity
    is_evolution_candidate: bool = False


class AnalysisReport(BaseModel):
    """Structured output of the Analyze phase (§2.2)."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symptom_clusters: list[SymptomCluster] = Field(default_factory=list)
    overall_severity: Severity = Severity.INFORMATIONAL
    proposed_evolution_targets: list[str] = Field(default_factory=list)


class AdaptationPlan(BaseModel):
    """A candidate adaptation strategy produced by the Plan phase (§2.3)."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str
    adaptation_type: AdaptationType
    expected_benefit: float = Field(ge=0.0, description="Quantified performance delta")
    risk_score: float = Field(ge=0.0, le=1.0)
    reversibility: Reversibility = Reversibility.FULLY_REVERSIBLE
    blast_radius: list[str] = Field(default_factory=list, description="Affected components")
    requires_human_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    """Audit record emitted for every significant decision (§3.4)."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision_path: DecisionPath
    summary: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    verification_result: VerificationResult
    proof_trace: str | None = None
    rules_evaluated: list[str] = Field(default_factory=list)
    justification: str = ""
    plan_id: str | None = None


class AlertPayload(BaseModel):
    """Structured alert payload sent to human operators (§6.3)."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: ThresholdLevel
    evolution_summary: str
    risk_score: float = Field(ge=0.0, le=1.0)
    affected_components: list[str] = Field(default_factory=list)
    reversibility: Reversibility
    proof_trace_url: str | None = None
    approve_url: str | None = None
    deny_url: str | None = None
    auto_deny_at: datetime | None = None


class ExecutionResult(BaseModel):
    """Outcome of the Execute phase for a single adaptation plan."""

    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str
    status: ExecutionStatus
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""  # cryptographic signature placeholder
    rollback_available: bool = True
    notes: str = ""

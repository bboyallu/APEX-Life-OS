# APEX Life OS

A **self-evolving AI system** reference implementation built on the APEX blueprint.  
APEX continuously monitors its own performance, reasons about its limitations, proposes structural or behavioural changes, and enacts those changes — all while remaining bounded by safety constraints and human oversight.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               Human Oversight Interface             │
│        (Dashboard · Alert Channels · Veto API)      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           Decision Orchestration Layer              │
│    (Runtime path selection · Policy arbitration)    │
└──────┬────────────────┬────────────────┬────────────┘
       │                │                │
┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼───────┐
│  MAPE-K     │  │ Neuro-      │  │ Autonomic   │
│  Control    │  │ Symbolic    │  │ Threshold   │
│  Loop       │  │ Layer       │  │ Engine      │
└──────┬──────┘  └──────┬──────┘  └─────┬───────┘
       │                │                │
┌──────▼────────────────▼────────────────▼───────────┐
│                  Knowledge Base (K)                 │
│   (System state · History · Rules · World model)   │
└─────────────────────────────────────────────────────┘
```

### Packages

| Package | Description |
|---|---|
| `apex.core` | Data types (`AdaptationPlan`, `AnalysisReport`, …) and the `KnowledgeBase` |
| `apex.mape` | MAPE-K loop: `Monitor`, `Analyzer`, `Planner`, `Executor`, `MAPELoop` |
| `apex.neuro_symbolic` | `NeuralSubsystem`, `SymbolicSubsystem`, `VerificationPipeline` |
| `apex.orchestration` | `PathSelector`, `DecisionOrchestrator`, policy arbitration |
| `apex.thresholds` | `RiskScorer`, `AutonomicThresholdEngine` with dead-man switch |
| `apex.alerts` | `AlertChannels` (voice/push/SMS/email), `AlertSystem`, approval API |
| `apex.governance` | `SafetyConstraintRegistry` (immutable core), `AuditLedger` |
| `apex.system` | `ApexSystem` — top-level façade wiring everything together |

---

## Quick Start

```python
from apex import ApexSystem
from apex.mape.analyzer import SignalRule
from apex.core.types import Severity

# Create the system (wire in your alert handlers for production)
system = ApexSystem(
    push_handler=lambda payload: send_push(payload),
    email_handler=lambda payload: send_email(payload),
    voice_handler=lambda payload: make_call(payload),   # L4 only
    on_freeze=lambda: page_on_call_engineer(),
)

# Register a signal rule
system.add_signal_rule(
    SignalRule("error_rate", upper_threshold=0.05, severity=Severity.WARNING)
)

# Publish telemetry
system.publish_metric("api_gateway", "error_rate", 0.08)

# Run one adaptation cycle (Monitor → Analyze → Plan → Execute)
report = system.run_cycle()
print(report.overall_severity)   # Severity.WARNING

# Signal that the oversight interface is reachable (resets dead-man switch)
system.heartbeat()

# Verify the audit chain integrity
valid, message = system.verify_audit_chain()
print(valid, message)  # True  "Chain valid. N entries verified."
```

---

## MAPE-K Loop

### Monitor
Collects telemetry via a `TelemetryBus`. `AnomalyDetector` instances watch rolling windows over individual metrics and fire callbacks when upper/lower thresholds are exceeded.

### Analyze
Correlates raw signal events with registered `SignalRule` objects and anomaly alerts, scores severity (`informational → catastrophic`), and emits a structured `AnalysisReport`.

### Plan
Generates `AdaptationPlan` candidates ranked by utility (`expected_benefit / (1 + risk_score)`). Plans crossing L3/L4 thresholds are flagged for mandatory human approval.

### Execute
Implements a staged rollout:
1. **Shadow mode** — new behaviour runs in parallel without affecting live output.
2. **Canary** — gradual traffic shifting with live rollback triggers.
3. **Atomic commit** — all-or-nothing state transition.

Automatic rollback fires if post-execution metrics worsen beyond the configured delta.

---

## Neuro-Symbolic Layer

```
Neural Subsystem          Symbolic Subsystem
──────────────────        ──────────────────────────
LLM / model            →  First-order logic rules
Pattern recognizer     →  Constraint registry
Confidence estimator      ↓
                       Verifier (check against rules)
                          ↓
                       Justified Decision + Proof trace
```

- **Verified** → decision proceeds with proof trace attached.
- **Refuted** → neural subsystem receives counterexample and regenerates (up to `max_retries`).
- **Undecidable** → escalated to human oversight.

Every decision emits a `DecisionRecord` with a natural-language summary, proof trace, confidence score, and rules evaluated.

---

## Autonomic Threshold Engine

| Level | Risk Score | Action |
|---|---|---|
| L0 — Routine | 0.00–0.15 | Fully autonomous |
| L1 — Notable | 0.15–0.35 | Autonomous + logged |
| L2 — Significant | 0.35–0.60 | Async human notification |
| L3 — High-Risk | 0.60–0.80 | Synchronous human approval required |
| L4 — Critical | 0.80–1.00 | Immediate alert + hard block |

**Dead-man switch**: If the oversight interface is unreachable for longer than the configured timeout (default: 15 minutes), all L2+ evolutions are automatically suspended.

---

## Alert System

| Channel | Used For |
|---|---|
| Voice call | L4 Critical |
| Push / In-App | L3 High-Risk |
| SMS | Fallback for L3–L4 if push fails |
| Email | All L2+ events (always sent) |

### Approval timeouts

| Event | Timeout | Default |
|---|---|---|
| L2 notification | 4 hours | Auto-approve if no veto |
| L3 approval | 30 minutes | Auto-deny |
| L4 critical | 10 minutes | Auto-deny + system freeze |

---

## Governance & Safety

- **Immutable safety core** (§7.1): Five hard constraints (`no-self-replication`, `no-constraint-override`, `minimal-footprint`, `human-veto`, `audit-integrity`) loaded at startup; cannot be modified by any autonomous process.
- **Cryptographic audit ledger**: Blockchain-style SHA-256 chained append-only log. Every metric publication, adaptation cycle, and alert is recorded. `verify_audit_chain()` detects any tampering.
- **Human veto rights**: Immediate halt, retroactive rollback (72-hour window), component freeze, full audit access.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=apex --cov-report=term-missing
```

### Project layout

```
apex/
  __init__.py          # ApexSystem façade
  system.py            # Top-level wiring
  core/                # Types + KnowledgeBase
  mape/                # Monitor, Analyzer, Planner, Executor, MAPELoop
  neuro_symbolic/      # Neural, Symbolic, VerificationPipeline
  orchestration/       # PathSelector, DecisionOrchestrator
  thresholds/          # RiskScorer, AutonomicThresholdEngine
  alerts/              # AlertChannels, AlertSystem
  governance/          # SafetyConstraintRegistry, AuditLedger
tests/
  test_knowledge_base.py
  test_mape.py
  test_neuro_symbolic.py
  test_orchestration.py
  test_thresholds.py
  test_alerts.py
  test_governance.py
  test_system.py
```

---

## Implementation Roadmap

- [x] **Phase 1** — Foundation: Knowledge Base, Monitor, basic Analyze, audit ledger, alert channels
- [x] **Phase 2** — Adaptation loop: Plan, Execute, shadow/canary rollout, Autonomic Threshold Engine
- [x] **Phase 3** — Neuro-Symbolic: SymbolicSubsystem, VerificationPipeline, DecisionRecord, Collaborative/Adversarial paths
- [x] **Phase 4** — Full oversight: Push, voice, SMS alert system; human approval API; dead-man switch; auto-deny timeouts
- [ ] **Phase 5** — Continuous hardening: Third-party audit, threshold calibration review, causal model expansion with deployment history

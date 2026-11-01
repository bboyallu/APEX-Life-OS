# APEX Life OS

A **governed personal AI agent** — chat with it, give it tools, let it remember — where every action is risk-scored, human-gated when it matters, and written to a tamper-evident audit ledger. Built on the APEX self-evolving-AI blueprint (MAPE-K loop, neuro-symbolic verification, autonomic thresholds).

## Get started in 60 seconds

```bash
pip install .            # Python 3.11+
apex chat                # works immediately, no API key needed (offline mode)
```

Unlock full natural-language reasoning with any backend:

```bash
export ANTHROPIC_API_KEY=...        # Claude
export OPENAI_API_KEY=...           # OpenAI
apex --provider ollama chat         # local models via Ollama (no key)
# OPENAI_BASE_URL=... works with LM Studio, vLLM, OpenRouter, etc.
```

```bash
apex ask "what is 18% of 2450?"                 # one-shot
echo "summarise README.md" | apex ask -         # stdin
apex ask --json "add task call mum"             # machine-readable
apex memory list | search | add | forget        # long-term memory
apex tasks                                      # your to-do list
apex audit --verify                             # prove the log wasn't tampered with
apex doctor                                     # diagnose configuration
```

Or with Docker: `docker build -t apex . && docker run -it -v apex-data:/data apex`.

### What the agent can do

| Tool | Risk level | Notes |
|---|---|---|
| `calculate`, `current_time`, `system_status`, `recall`, `list_tasks`, `list_dir`, `read_file` | L0 | read-only, autonomous |
| `remember`, `add_task`, `complete_task` | L0–L1 | logged |
| `forget` | L1 | partially reversible |
| `write_file` | L2 | allowed, flagged for notification, audit-logged |
| `run_command` (`--allow-shell`) | L3 | **asks you every time**; dangerous commands (`sudo`, `rm -rf /`, `curl \| sh`…) are blocked outright |

File tools are sandboxed to `--workspace` (default: current directory), including symlink escapes. State lives in `~/.apex` (`$APEX_HOME`): `MEMORY.md`, `USER.md`, `tasks.json`, `audit.jsonl`, `sessions/`.

### Safety model

Each tool call becomes an `AdaptationPlan`, is scored by `RiskScorer`, and evaluated by the `AutonomicThresholdEngine`: L0–L1 run autonomously, L2 runs with notification, **L3 needs your approval** (denied when non-interactive; `--yes` pre-approves L3), **L4 is always blocked**. The dead-man switch suspends L3 when no human is present. Tool output is treated as data — the system prompt tells the model never to follow instructions found inside files or memories.

### Embedding in Python

```python
from apex.agent import Agent, create_provider

agent = Agent.create(create_provider("auto"), approver=lambda req: input(f"allow {req.tool}? ") == "y")
print(agent.run("Remember that my sister's birthday is 3 March").text)
```

Add your own tool with `agent.tools.register(Tool(name, description, json_schema, handler, reversibility=..., blast_radius=(...)))` — it is governed automatically.

---

## The APEX framework underneath

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
| `apex.memory` | Pluggable long-term memory: `MemoryProvider`, built-in `MEMORY.md`/`USER.md` store, provider registry |
| `apex.governance` | `SafetyConstraintRegistry` (immutable core), `AuditLedger` |
| `apex.agent` | The user-facing agent: LLM providers, governed tools, `Agent` loop, workspace |
| `apex.cli` | `apex` command-line interface |
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

## Memory Providers

APEX supports pluggable long-term memory backends. The default is the zero-dependency **built-in** provider, which persists memories to human-readable markdown files.

| Provider | Connection |
|---|---|
| `byterover` | requires API key |
| `hindsight` | API key / local |
| `holographic` | local |
| `honcho` | API key / local |
| `mem0` | API key / local |
| `openviking` | API key / local |
| `retaindb` | API key / local |
| `supermemory` | requires API key |
| `builtin` *(default)* | `MEMORY.md` / `USER.md` |

Only `builtin` ships with an implementation; external providers are catalogued and enabled by registering a factory:

```python
from apex.memory import default_registry, MemoryScope

registry = default_registry()
registry.register_factory("mem0", make_mem0_provider)   # your adapter
provider = registry.create("mem0")
```

Memory operations on `ApexSystem` are audit-logged:

```python
entry = system.remember("build", "Use pytest", citation="pyproject.toml")
system.search_memories("pytest")                        # -> [entry]
system.recall_memories(MemoryScope.USER)                # user-scoped prefs
system.forget_memory(entry.entry_id)                    # True
```

- **Repository scope** → `MEMORY.md` — facts about the system, visible to all agents.
- **User scope** → `USER.md` — preferences of the current human operator.

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
  memory/              # MemoryProvider, BuiltinMemoryProvider, registry
  governance/          # SafetyConstraintRegistry, AuditLedger
tests/
  test_knowledge_base.py
  test_mape.py
  test_neuro_symbolic.py
  test_orchestration.py
  test_thresholds.py
  test_alerts.py
  test_memory.py
  test_governance.py
  test_system.py
  test_agent.py
```

---

## Implementation Roadmap

- [x] **Phase 1** — Foundation: Knowledge Base, Monitor, basic Analyze, audit ledger, alert channels
- [x] **Phase 2** — Adaptation loop: Plan, Execute, shadow/canary rollout, Autonomic Threshold Engine
- [x] **Phase 3** — Neuro-Symbolic: SymbolicSubsystem, VerificationPipeline, DecisionRecord, Collaborative/Adversarial paths
- [x] **Phase 4** — Full oversight: Push, voice, SMS alert system; human approval API; dead-man switch; auto-deny timeouts
- [ ] **Phase 5** — Continuous hardening: Third-party audit, threshold calibration review, causal model expansion with deployment history

"""Tests for the KnowledgeBridge — the knowledge-to-evolution loop."""

from __future__ import annotations

from apex.core.types import Severity
from apex.knowledge.bridge import EVOLUTION_TOPIC, KnowledgeBridge, KnowledgeSignal
from apex.knowledge.vault import KnowledgeVault
from apex.system import ApexSystem


def _make_vault(tmp_path, raw_content: str, name: str = "2025-01-gateway-notes.md"):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / name).write_text(raw_content, encoding="utf-8")
    vault = KnowledgeVault(root=tmp_path)
    vault.process_raw()
    return vault


class TestSignalExtraction:
    def test_extracts_signal_with_severity(self, tmp_path):
        vault = _make_vault(
            tmp_path,
            "topic: gateway\n\n"
            "signal: api_gateway :: retry storm amplifying latency [degraded]\n",
        )
        bridge = KnowledgeBridge(vault=vault)
        signals = bridge.extract_signals()
        assert len(signals) == 1
        assert signals[0].target == "api_gateway"
        assert signals[0].description == "retry storm amplifying latency"
        assert signals[0].severity == Severity.DEGRADED
        assert signals[0].source == "wiki/gateway.md"

    def test_default_and_invalid_severity_fall_back_to_warning(self, tmp_path):
        vault = _make_vault(
            tmp_path,
            "topic: gateway\n\n"
            "signal: svc-a :: needs a cache layer\n"
            "signal: svc-b :: something odd [bogus]\n",
        )
        signals = KnowledgeBridge(vault=vault).extract_signals()
        assert [s.severity for s in signals] == [Severity.WARNING, Severity.WARNING]

    def test_extraction_is_deduplicated_and_persistent(self, tmp_path):
        vault = _make_vault(
            tmp_path,
            "topic: gateway\n\nsignal: api_gateway :: reduce retries\n",
        )
        bridge = KnowledgeBridge(vault=vault)
        assert len(bridge.extract_signals()) == 1
        assert bridge.extract_signals() == []
        # A fresh bridge instance shares the persisted state.
        assert KnowledgeBridge(vault=vault).extract_signals() == []

    def test_non_signal_content_yields_nothing(self, tmp_path):
        vault = _make_vault(tmp_path, "topic: notes\n\nJust ordinary notes.\n")
        assert KnowledgeBridge(vault=vault).extract_signals() == []


class TestRecordCycle:
    def test_record_writes_raw_note_without_new_signals(self, tmp_path):
        vault = _make_vault(
            tmp_path,
            "topic: gateway\n\nsignal: api_gateway :: reduce retries [critical]\n",
        )
        bridge = KnowledgeBridge(vault=vault)
        signals = bridge.extract_signals()

        from apex.core.types import AnalysisReport

        report = AnalysisReport(proposed_evolution_targets=["api_gateway"])
        path = bridge.record_cycle(cycle=1, report=report, signals=signals)

        assert path.name == f"{EVOLUTION_TOPIC}.md"
        text = path.read_text(encoding="utf-8")
        assert "Cycle 1" in text
        assert "api_gateway" in text
        # The log itself must never contain signal directives.
        vault.process_raw()
        assert bridge.extract_signals() == []

    def test_signal_digest_is_stable(self):
        a = KnowledgeSignal(target="x", description="y")
        b = KnowledgeSignal(target="x", description="y")
        assert a.digest == b.digest


class TestKnowledgeInformedCycle:
    def test_full_loop(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "2025-02-observations.md").write_text(
            "topic: gateway\n\n"
            "The gateway is struggling under load.\n"
            "signal: api_gateway :: connection pool exhausted under load [degraded]\n",
            encoding="utf-8",
        )
        system = ApexSystem(knowledge_root=str(tmp_path))

        report = system.run_knowledge_informed_cycle()

        # Knowledge drove the evolution proposal.
        assert "api_gateway" in report.proposed_evolution_targets
        assert any(
            "Knowledge-derived signal" in c.probable_cause
            for c in report.symptom_clusters
        )

        # Outcome was recorded back into raw/ and folded into the wiki.
        assert (raw / f"{EVOLUTION_TOPIC}.md").exists()
        assert (tmp_path / "wiki" / f"{EVOLUTION_TOPIC}.md").exists()

        # Every step is on the shared, verifiable audit ledger.
        events = [e.event_type for e in system.audit_ledger.read()]
        assert "knowledge_signals_extracted" in events
        assert "mape_cycle_complete" in events
        assert "evolution_recorded_to_knowledge" in events
        ok, _ = system.verify_audit_chain()
        assert ok

    def test_second_cycle_does_not_refire_same_signal(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "note.md").write_text(
            "topic: gateway\n\nsignal: api_gateway :: reduce retries\n",
            encoding="utf-8",
        )
        system = ApexSystem(knowledge_root=str(tmp_path))
        first = system.run_knowledge_informed_cycle()
        assert "api_gateway" in first.proposed_evolution_targets

        second = system.run_knowledge_informed_cycle()
        assert "api_gateway" not in second.proposed_evolution_targets

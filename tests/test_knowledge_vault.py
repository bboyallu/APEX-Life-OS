"""Tests for apex.knowledge — the built-in raw/wiki/outputs knowledge base."""

from __future__ import annotations

from pathlib import Path

import pytest

from apex.knowledge.vault import KnowledgeVault
from apex.system import ApexSystem


@pytest.fixture()
def vault(tmp_path: Path) -> KnowledgeVault:
    (tmp_path / "raw").mkdir()
    return KnowledgeVault(root=tmp_path)


def _write_raw(root: Path, name: str, content: str) -> Path:
    path = root / "raw" / name
    path.write_text(content, encoding="utf-8")
    return path


class TestIngestion:
    def test_ingests_new_files_into_wiki(self, vault: KnowledgeVault, tmp_path: Path):
        _write_raw(tmp_path, "stoicism-notes.txt", "Marcus Aurelius on dichotomy of control.")
        report = vault.process_raw()

        assert report.ingested == ["raw/stoicism-notes.txt"]
        article = tmp_path / "wiki" / "stoicism-notes.md"
        assert article.exists()
        text = article.read_text(encoding="utf-8")
        assert "Marcus Aurelius" in text
        assert "Source: raw/stoicism-notes.txt" in text

    def test_topic_line_overrides_filename(self, vault: KnowledgeVault, tmp_path: Path):
        _write_raw(tmp_path, "random-dump.txt", "Topic: Deep Work\nFocus blocks matter.")
        vault.process_raw()
        assert (tmp_path / "wiki" / "deep-work.md").exists()

    def test_idempotent_reprocessing(self, vault: KnowledgeVault, tmp_path: Path):
        _write_raw(tmp_path, "a.txt", "alpha content here")
        first = vault.process_raw()
        second = vault.process_raw()

        assert first.ingested == ["raw/a.txt"]
        assert second.ingested == []
        assert second.skipped == ["raw/a.txt"]

    def test_changed_file_updates_instead_of_duplicating(
        self, vault: KnowledgeVault, tmp_path: Path
    ):
        raw = _write_raw(tmp_path, "b.txt", "original body text")
        vault.process_raw()
        raw.write_text("revised body text", encoding="utf-8")
        report = vault.process_raw()

        assert report.updated == ["raw/b.txt"]
        text = (tmp_path / "wiki" / "b.md").read_text(encoding="utf-8")
        assert "revised body text" in text
        assert "original body text" not in text

    def test_raw_files_are_never_modified(self, vault: KnowledgeVault, tmp_path: Path):
        raw = _write_raw(tmp_path, "c.txt", "immutable input")
        before = raw.read_bytes()
        vault.process_raw()
        assert raw.read_bytes() == before

    def test_raw_readme_is_ignored(self, vault: KnowledgeVault, tmp_path: Path):
        _write_raw(tmp_path, "README.md", "This is the folder readme.")
        report = vault.process_raw()
        assert report.ingested == []

    def test_multiple_files_same_topic_merge_into_one_article(
        self, vault: KnowledgeVault, tmp_path: Path
    ):
        _write_raw(tmp_path, "one.txt", "Topic: Fitness\nZone 2 cardio.")
        _write_raw(tmp_path, "two.txt", "Topic: Fitness\nProgressive overload.")
        report = vault.process_raw()

        assert report.articles == ["fitness"]
        text = (tmp_path / "wiki" / "fitness.md").read_text(encoding="utf-8")
        assert "Zone 2 cardio" in text
        assert "Progressive overload" in text


class TestIndexAndCrossReferences:
    def test_index_lists_all_articles(self, vault: KnowledgeVault, tmp_path: Path):
        _write_raw(tmp_path, "alpha.txt", "first note body")
        _write_raw(tmp_path, "beta.txt", "second note body")
        vault.process_raw()

        index = (tmp_path / "wiki" / "index.md").read_text(encoding="utf-8")
        assert "[Alpha](./alpha.md)" in index
        assert "[Beta](./beta.md)" in index

    def test_related_articles_are_cross_linked(self, vault: KnowledgeVault, tmp_path: Path):
        shared = "productivity systems compounding habits leverage"
        _write_raw(tmp_path, "habits.txt", f"Notes about {shared} and identity.")
        _write_raw(tmp_path, "leverage.txt", f"Thoughts on {shared} and capital.")
        _write_raw(tmp_path, "cooking.txt", "Sourdough hydration ratios only.")
        vault.process_raw()

        habits = (tmp_path / "wiki" / "habits.md").read_text(encoding="utf-8")
        assert "(./leverage.md)" in habits
        assert "cooking" not in habits

    def test_article_for_deleted_raw_source_is_removed(
        self, vault: KnowledgeVault, tmp_path: Path
    ):
        raw = _write_raw(tmp_path, "temp.txt", "temporary note")
        vault.process_raw()
        assert (tmp_path / "wiki" / "temp.md").exists()
        raw.unlink()
        vault.process_raw()
        assert not (tmp_path / "wiki" / "temp.md").exists()


class TestReports:
    def test_report_written_to_outputs_with_date_prefix(
        self, vault: KnowledgeVault, tmp_path: Path
    ):
        _write_raw(tmp_path, "notes.txt", "Kubernetes autoscaling saves money.")
        vault.process_raw()
        path = vault.generate_report("kubernetes autoscaling")

        assert path.parent == tmp_path / "outputs"
        assert path.name[:4].isdigit()
        text = path.read_text(encoding="utf-8")
        assert "Kubernetes autoscaling" in text
        assert "Source:" in text

    def test_outputs_are_append_only(self, vault: KnowledgeVault, tmp_path: Path):
        first = vault.generate_report("same query")
        second = vault.generate_report("same query")
        assert first != second
        assert first.exists() and second.exists()

    def test_no_match_report_says_so(self, vault: KnowledgeVault):
        path = vault.generate_report("zzz-nonexistent-zzz")
        assert "No matching knowledge" in path.read_text(encoding="utf-8")


class TestApexSystemIntegration:
    def test_system_processes_knowledge_and_audits(self, tmp_path: Path):
        (tmp_path / "raw").mkdir()
        _write_raw(tmp_path, "note.txt", "APEX integration note.")
        system = ApexSystem(knowledge_root=str(tmp_path))

        report = system.process_knowledge()
        assert report.ingested == ["raw/note.txt"]
        assert (tmp_path / "wiki" / "note.md").exists()

        events = [e.event_type for e in system.audit_ledger.read()]
        assert "knowledge_processed" in events
        ok, _ = system.verify_audit_chain()
        assert ok

    def test_system_generates_knowledge_report_and_audits(self, tmp_path: Path):
        (tmp_path / "raw").mkdir()
        _write_raw(tmp_path, "note.txt", "Latency budgets are important.")
        system = ApexSystem(knowledge_root=str(tmp_path))
        system.process_knowledge()

        output = system.generate_knowledge_report("latency budgets")
        assert Path(output).exists()
        events = [e.event_type for e in system.audit_ledger.read()]
        assert "knowledge_report_generated" in events

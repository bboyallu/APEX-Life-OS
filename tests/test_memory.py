"""Tests for the memory subsystem (providers, registry, ApexSystem wiring)."""

from __future__ import annotations

import pytest

from apex.memory import (
    BuiltinMemoryProvider,
    MemoryEntry,
    MemoryProviderInfo,
    MemoryScope,
    default_registry,
)
from apex.system import ApexSystem


# ---------------------------------------------------------------------------
# BuiltinMemoryProvider
# ---------------------------------------------------------------------------


class TestBuiltinMemoryProvider:
    def test_remember_creates_memory_md(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        provider.remember(MemoryEntry(subject="build", fact="Use pytest"))
        assert (tmp_path / "MEMORY.md").exists()
        assert "Use pytest" in (tmp_path / "MEMORY.md").read_text()

    def test_user_scope_goes_to_user_md(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        provider.remember(
            MemoryEntry(scope=MemoryScope.USER, subject="style", fact="Be concise")
        )
        assert (tmp_path / "USER.md").exists()
        assert not (tmp_path / "MEMORY.md").exists()

    def test_recall_round_trip(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        stored = provider.remember(
            MemoryEntry(subject="build", fact="Use pytest", citation="pyproject.toml")
        )
        entries = provider.recall()
        assert len(entries) == 1
        e = entries[0]
        assert e.entry_id == stored.entry_id
        assert e.subject == "build"
        assert e.fact == "Use pytest"
        assert e.citation == "pyproject.toml"
        assert e.scope == MemoryScope.REPOSITORY

    def test_recall_filters_by_scope(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        provider.remember(MemoryEntry(subject="a", fact="repo fact"))
        provider.remember(
            MemoryEntry(scope=MemoryScope.USER, subject="b", fact="user pref")
        )
        assert len(provider.recall()) == 2
        assert len(provider.recall(MemoryScope.USER)) == 1
        assert provider.recall(MemoryScope.USER)[0].fact == "user pref"

    def test_search_is_case_insensitive(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        provider.remember(MemoryEntry(subject="testing", fact="Run Pytest with -q"))
        provider.remember(MemoryEntry(subject="lint", fact="No linter configured"))
        hits = provider.search("PYTEST")
        assert len(hits) == 1
        assert hits[0].subject == "testing"

    def test_forget_removes_entry(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        stored = provider.remember(MemoryEntry(subject="a", fact="temp"))
        provider.remember(MemoryEntry(subject="b", fact="keep"))
        assert provider.forget(stored.entry_id) is True
        remaining = provider.recall()
        assert len(remaining) == 1
        assert remaining[0].fact == "keep"

    def test_forget_unknown_id_returns_false(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        assert provider.forget("nonexistent") is False

    def test_recall_empty_when_no_files(self, tmp_path):
        provider = BuiltinMemoryProvider(root=tmp_path)
        assert provider.recall() == []

    def test_persistence_across_instances(self, tmp_path):
        BuiltinMemoryProvider(root=tmp_path).remember(
            MemoryEntry(subject="durable", fact="survives restart")
        )
        entries = BuiltinMemoryProvider(root=tmp_path).recall()
        assert len(entries) == 1
        assert entries[0].fact == "survives restart"


# ---------------------------------------------------------------------------
# MemoryProviderRegistry
# ---------------------------------------------------------------------------


class TestMemoryProviderRegistry:
    def test_catalogue_matches_setup_screen(self):
        registry = default_registry()
        names = {info.name for info in registry.list_providers()}
        assert names == {
            "byterover",
            "hindsight",
            "holographic",
            "honcho",
            "mem0",
            "openviking",
            "retaindb",
            "supermemory",
            "builtin",
        }

    def test_builtin_is_default_and_available(self):
        registry = default_registry()
        defaults = [i for i in registry.list_providers() if i.is_default]
        assert len(defaults) == 1
        assert defaults[0].name == "builtin"
        assert registry.is_available("builtin")

    def test_create_builtin(self, tmp_path):
        provider = default_registry().create("builtin", root=tmp_path)
        assert isinstance(provider, BuiltinMemoryProvider)

    def test_external_provider_without_factory_raises(self):
        with pytest.raises(NotImplementedError):
            default_registry().create("mem0")

    def test_unknown_provider_raises_keyerror(self):
        with pytest.raises(KeyError):
            default_registry().create("nope")

    def test_register_factory_for_catalogue_provider(self, tmp_path):
        registry = default_registry()
        registry.register_factory("mem0", lambda: BuiltinMemoryProvider(root=tmp_path))
        assert registry.is_available("mem0")
        assert isinstance(registry.create("mem0"), BuiltinMemoryProvider)

    def test_register_new_provider_requires_info(self, tmp_path):
        registry = default_registry()
        with pytest.raises(KeyError):
            registry.register_factory("custom", BuiltinMemoryProvider)
        registry.register_factory(
            "custom",
            lambda: BuiltinMemoryProvider(root=tmp_path),
            info=MemoryProviderInfo(
                name="custom",
                description="local",
                requires_api_key=False,
                supports_local=True,
            ),
        )
        assert registry.get_info("custom") is not None
        assert registry.is_available("custom")


# ---------------------------------------------------------------------------
# ApexSystem wiring
# ---------------------------------------------------------------------------


class TestApexSystemMemory:
    def _system(self, tmp_path) -> ApexSystem:
        return ApexSystem(memory_provider=BuiltinMemoryProvider(root=tmp_path))

    def test_default_provider_is_builtin(self):
        assert ApexSystem().memory.name == "builtin"

    def test_remember_and_recall(self, tmp_path):
        system = self._system(tmp_path)
        entry = system.remember("build", "Use pytest", citation="pyproject.toml")
        recalled = system.recall_memories()
        assert [e.entry_id for e in recalled] == [entry.entry_id]

    def test_remember_is_audited(self, tmp_path):
        system = self._system(tmp_path)
        system.remember("build", "Use pytest")
        events = [e.event_type for e in system.audit_ledger.read()]
        assert "memory_stored" in events

    def test_forget_is_audited(self, tmp_path):
        system = self._system(tmp_path)
        entry = system.remember("build", "Use pytest")
        assert system.forget_memory(entry.entry_id) is True
        events = [e.event_type for e in system.audit_ledger.read()]
        assert "memory_forgotten" in events

    def test_forget_unknown_not_audited(self, tmp_path):
        system = self._system(tmp_path)
        assert system.forget_memory("missing") is False
        events = [e.event_type for e in system.audit_ledger.read()]
        assert "memory_forgotten" not in events

    def test_search_memories(self, tmp_path):
        system = self._system(tmp_path)
        system.remember("build", "Use pytest")
        system.remember("style", "Be concise", scope=MemoryScope.USER)
        assert len(system.search_memories("pytest")) == 1
        assert len(system.search_memories("concise", MemoryScope.USER)) == 1

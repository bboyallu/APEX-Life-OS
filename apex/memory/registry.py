"""Memory provider registry — the catalogue behind "Memory provider setup".

Mirrors the interactive setup screen::

    ( ) byterover    — requires API key
    ( ) hindsight    — API key / local
    ( ) holographic  — local
    ( ) honcho       — API key / local
    ( ) mem0         — API key / local
    ( ) openviking   — API key / local
    ( ) retaindb     — API key / local
    ( ) supermemory  — requires API key
    (•) Built-in only — MEMORY.md / USER.md (default)

Only the built-in provider ships with an implementation; the others are
listed with their connection requirements and may be enabled by registering
a factory (``registry.register_factory("mem0", make_mem0_provider)``).
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from apex.memory.builtin import BuiltinMemoryProvider
from apex.memory.provider import MemoryProvider


class MemoryProviderInfo(BaseModel):
    """Catalogue metadata for a memory provider."""

    name: str
    description: str
    requires_api_key: bool
    supports_local: bool
    is_default: bool = False


_CATALOGUE: list[MemoryProviderInfo] = [
    MemoryProviderInfo(name="byterover", description="requires API key", requires_api_key=True, supports_local=False),
    MemoryProviderInfo(name="hindsight", description="API key / local", requires_api_key=True, supports_local=True),
    MemoryProviderInfo(name="holographic", description="local", requires_api_key=False, supports_local=True),
    MemoryProviderInfo(name="honcho", description="API key / local", requires_api_key=True, supports_local=True),
    MemoryProviderInfo(name="mem0", description="API key / local", requires_api_key=True, supports_local=True),
    MemoryProviderInfo(name="openviking", description="API key / local", requires_api_key=True, supports_local=True),
    MemoryProviderInfo(name="retaindb", description="API key / local", requires_api_key=True, supports_local=True),
    MemoryProviderInfo(name="supermemory", description="requires API key", requires_api_key=True, supports_local=False),
    MemoryProviderInfo(name="builtin", description="MEMORY.md / USER.md (default)", requires_api_key=False, supports_local=True, is_default=True),
]


class MemoryProviderRegistry:
    """Registry of available memory providers and their factories."""

    def __init__(self) -> None:
        self._catalogue: dict[str, MemoryProviderInfo] = {
            info.name: info for info in _CATALOGUE
        }
        self._factories: dict[str, Callable[..., MemoryProvider]] = {
            "builtin": BuiltinMemoryProvider,
        }

    def list_providers(self) -> list[MemoryProviderInfo]:
        return list(self._catalogue.values())

    def get_info(self, name: str) -> MemoryProviderInfo | None:
        return self._catalogue.get(name)

    def is_available(self, name: str) -> bool:
        """Return ``True`` if the provider has an installed factory."""
        return name in self._factories

    def register_factory(
        self,
        name: str,
        factory: Callable[..., MemoryProvider],
        info: MemoryProviderInfo | None = None,
    ) -> None:
        """Install (or override) a provider factory.

        ``info`` may be provided to add a provider not in the catalogue.
        """
        if info is not None:
            self._catalogue[name] = info
        elif name not in self._catalogue:
            raise KeyError(
                f"Unknown provider {name!r}; pass a MemoryProviderInfo to add it."
            )
        self._factories[name] = factory

    def create(self, name: str = "builtin", **kwargs) -> MemoryProvider:
        """Instantiate a provider by name."""
        if name not in self._catalogue:
            raise KeyError(f"Unknown memory provider: {name!r}")
        factory = self._factories.get(name)
        if factory is None:
            info = self._catalogue[name]
            raise NotImplementedError(
                f"Provider {name!r} ({info.description}) has no installed factory. "
                f"Register one with register_factory({name!r}, ...)."
            )
        return factory(**kwargs)


def default_registry() -> MemoryProviderRegistry:
    """Return a fresh registry with the standard catalogue."""
    return MemoryProviderRegistry()

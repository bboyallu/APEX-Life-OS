"""Memory subsystem — pluggable long-term memory providers.

The default provider is the built-in markdown store (``MEMORY.md`` /
``USER.md``).  External providers (mem0, supermemory, …) can be registered
via :mod:`apex.memory.registry`.
"""

from apex.memory.provider import MemoryEntry, MemoryProvider, MemoryScope
from apex.memory.builtin import BuiltinMemoryProvider
from apex.memory.registry import (
    MemoryProviderInfo,
    MemoryProviderRegistry,
    default_registry,
)
from apex.memory.vaults import (
    Episode,
    MemoryVaultStore,
    VaultKeyError,
    VaultSnapshot,
    make_vault_key,
    render_memory_context,
)

__all__ = [
    "MemoryEntry",
    "MemoryProvider",
    "MemoryScope",
    "BuiltinMemoryProvider",
    "MemoryProviderInfo",
    "MemoryProviderRegistry",
    "default_registry",
    "Episode",
    "MemoryVaultStore",
    "VaultKeyError",
    "VaultSnapshot",
    "make_vault_key",
    "render_memory_context",
]

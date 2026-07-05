"""Memory provider interface — the contract all memory backends implement.

A memory provider gives the system durable, queryable long-term memory that
survives individual adaptation cycles.  Two scopes are supported:

* ``REPOSITORY`` — facts about the system / codebase, visible to all agents.
* ``USER`` — preferences of the current human operator.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    REPOSITORY = "repository"
    USER = "user"


class MemoryEntry(BaseModel):
    """A single stored memory."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scope: MemoryScope = MemoryScope.REPOSITORY
    subject: str
    fact: str
    citation: str = ""


class MemoryProvider(ABC):
    """Abstract base class for memory backends."""

    #: Machine-readable provider identifier (e.g. ``"builtin"``, ``"mem0"``).
    name: str = "abstract"

    @abstractmethod
    def remember(self, entry: MemoryEntry) -> MemoryEntry:
        """Persist a memory entry and return it."""

    @abstractmethod
    def recall(self, scope: MemoryScope | None = None) -> list[MemoryEntry]:
        """Return all stored entries, optionally filtered by scope."""

    @abstractmethod
    def search(self, query: str, scope: MemoryScope | None = None) -> list[MemoryEntry]:
        """Return entries whose subject or fact contains ``query`` (case-insensitive)."""

    @abstractmethod
    def forget(self, entry_id: str) -> bool:
        """Delete an entry by id.  Return ``True`` if it existed."""

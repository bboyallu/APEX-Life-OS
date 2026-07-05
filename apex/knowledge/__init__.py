"""apex.knowledge — built-in personal knowledge base (raw/ → wiki/ → outputs/).

Implements the schema defined in ``KNOWLEDGE_BASE.md`` at the repository
root: raw material dropped into ``raw/`` is compiled into cross-referenced
wiki articles under ``wiki/``, and on-demand reports are written to
``outputs/``.
"""

from apex.knowledge.bridge import KnowledgeBridge, KnowledgeSignal
from apex.knowledge.vault import IngestReport, KnowledgeVault, WikiArticle

__all__ = [
    "IngestReport",
    "KnowledgeBridge",
    "KnowledgeSignal",
    "KnowledgeVault",
    "WikiArticle",
]

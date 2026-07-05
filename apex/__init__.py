"""APEX Life OS — self-evolving AI system.

Implements the architecture described in the APEX Self-Evolving AI System
Blueprint, including:

- MAPE-K closed-control adaptation loop
- Neuro-symbolic reasoning and verification
- Decision orchestration with path selection
- Autonomic threshold engine and risk scoring
- High-priority outbound alert system
- Governance: immutable safety core and audit ledger
"""

from apex.core import KnowledgeBase
from apex.system import ApexSystem

__version__ = "0.1.0"

__all__ = [
    "ApexSystem",
    "KnowledgeBase",
    "__version__",
]

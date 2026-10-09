"""Step 3 — Type-Aware Answering: route each typed sub-query to its expert solver.

Solvers: Math (structured computation), Logic (bidirectional consistency), Fact
(multi-source weighted RAG), General (direct generation). `route` dispatches by type.
"""

from .general_solver import solve_general
from .logic_solver import solve_logic
from .math_solver import solve_math
from .rag_solver import (
    CachedRetriever,
    Evidence,
    Retriever,
    SerperRetriever,
    SnapshotCache,
    format_evidence,
    solve_rag,
)
from .router import route

__all__ = [
    "solve_math",
    "solve_logic",
    "solve_general",
    "solve_rag",
    "route",
    "Retriever",
    "SerperRetriever",
    "CachedRetriever",
    "SnapshotCache",
    "Evidence",
    "format_evidence",
]

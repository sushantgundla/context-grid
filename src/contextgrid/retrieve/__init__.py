"""Retrieval strategies, and the registry of them.

The index is *where* the vectors live. The strategy is *how* they are used. Keeping the two
apart is what turns "should I use agentic retrieval?" from a rewrite into a cell in a grid.
"""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.retrieve.base import (
    RetrievalStrategy,
    RetrievalTrace,
    Searcher,
    fuse,
)
from contextgrid.retrieve.strategies import (
    DecomposedRetrieval,
    RetrievalError,
    SimpleRetrieval,
    WidenedRetrieval,
)

RETRIEVERS: Registry[RetrievalStrategy] = Registry(family="retrieval")

RETRIEVERS.register(
    "simple", doc="One search per query. The arm every other strategy has to beat."
)(SimpleRetrieval)
RETRIEVERS.register(
    "widened", shorthand="factor", doc="Search deeper than asked, then cut back. No model calls."
)(WidenedRetrieval)
RETRIEVERS.register(
    "decomposed",
    shorthand="max_parts",
    doc="Split a multi-part question and search each. Mechanical, so it costs nothing.",
)(DecomposedRetrieval)


def get_retriever(spec: str | RetrievalStrategy | None) -> RetrievalStrategy:
    """Resolve a strategy from a spec, or pass one through. `None` means plain search."""
    if spec is None:
        return SimpleRetrieval()
    return RETRIEVERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "RETRIEVERS",
    "DecomposedRetrieval",
    "RetrievalError",
    "RetrievalStrategy",
    "RetrievalTrace",
    "Searcher",
    "SimpleRetrieval",
    "WidenedRetrieval",
    "fuse",
    "get_retriever",
]

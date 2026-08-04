"""Chunkers, and the registry of them.

Chunkers all cut up the same text, which is what makes comparing them fair without any
re-annotation of ground truth: gold is stored as character spans and resolved to whichever
chunks each strategy happened to produce.
"""

from __future__ import annotations

from contextgrid.chunk.base import ChunkBuilder, ChunkerError, chunk_id
from contextgrid.chunk.fixed import FixedTokenChunker
from contextgrid.chunk.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from contextgrid.chunk.semantic import (
    SemanticChunker,
    profile_summary,
    similarity_profile,
)
from contextgrid.chunk.sentence import SentenceWindowChunker, sentence_ranges
from contextgrid.chunk.structural import StructuralChunker
from contextgrid.core.protocols import Chunker
from contextgrid.core.registry import Registry

CHUNKERS: Registry[Chunker] = Registry(family="chunker")

CHUNKERS.register("fixed", shorthand="size", doc="Fixed-size token windows with overlap.")(
    FixedTokenChunker
)
CHUNKERS.register(
    "recursive", shorthand="size", doc="Split on the largest separator that fits. The default."
)(RecursiveChunker)
CHUNKERS.register("sentence", shorthand="window", doc="A sliding window of whole sentences.")(
    SentenceWindowChunker
)
CHUNKERS.register(
    "structural", shorthand="max_size", doc="One chunk per section, bounded by size."
)(StructuralChunker)
CHUNKERS.register(
    "semantic", shorthand="percentile", doc="Cut where consecutive sentences change topic."
)(SemanticChunker)


def get_chunker(spec: str | Chunker) -> Chunker:
    """Resolve a chunker from a spec like `recursive:512,overlap=64`, or pass one through."""
    return CHUNKERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "CHUNKERS",
    "DEFAULT_SEPARATORS",
    "ChunkBuilder",
    "ChunkerError",
    "FixedTokenChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
    "StructuralChunker",
    "chunk_id",
    "get_chunker",
    "profile_summary",
    "sentence_ranges",
    "similarity_profile",
]

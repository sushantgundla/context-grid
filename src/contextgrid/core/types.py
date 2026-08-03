"""Re-exports of the core value objects.

The types live in three focused modules -- `span`, `documents`, `evalset` -- because they
change for different reasons and each is easier to hold in your head alone. This module is
the flat surface for anything that just wants a type without caring where it lives.
"""

from __future__ import annotations

from contextgrid.core.documents import (
    Block,
    BlockKind,
    Chunk,
    ChunkSet,
    Document,
    MediaType,
    ParsedDocument,
    RetrievedChunk,
    SourceFile,
    chunks_of,
    spans_of,
)
from contextgrid.core.evalset import (
    EvalItem,
    EvalSet,
    GoldAnchor,
    GoldSpan,
    Qrels,
    QuestionType,
    RelevanceLabel,
    Run,
)
from contextgrid.core.span import (
    Span,
    coverage_fraction,
    covered_length,
    intersection_length,
    merge_spans,
    total_length,
)

__all__ = [
    "Block",
    "BlockKind",
    "Chunk",
    "ChunkSet",
    "Document",
    "EvalItem",
    "EvalSet",
    "GoldAnchor",
    "GoldSpan",
    "MediaType",
    "ParsedDocument",
    "Qrels",
    "QuestionType",
    "RelevanceLabel",
    "RetrievedChunk",
    "Run",
    "SourceFile",
    "Span",
    "chunks_of",
    "coverage_fraction",
    "covered_length",
    "intersection_length",
    "merge_spans",
    "spans_of",
    "total_length",
]

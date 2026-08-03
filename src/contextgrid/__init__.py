"""context-grid: a lab for grounding pipelines.

Sweep parser x chunker x embedder x index x reranker on your own documents, and get back
ranked, reproducible results scored on quality, latency and cost.

Everything rests on one property: a piece of text always knows which characters of which
source document it came from. That is what makes comparing two chunkers -- or two parsers --
a valid thing to do at all.
"""

from __future__ import annotations

from contextgrid.core.errors import (
    ContextGridError,
    DocumentError,
    EvalSetError,
    MissingExtraError,
    ResolutionError,
    SpanError,
)
from contextgrid.core.types import (
    Chunk,
    Document,
    EvalItem,
    EvalSet,
    GoldSpan,
    RelevanceLabel,
    RetrievedChunk,
    Span,
    coverage_fraction,
    covered_length,
    intersection_length,
    merge_spans,
    total_length,
)
from contextgrid.core.warnings import GridWarning, Severity, WarningCode, WarningLog
from contextgrid.score.resolve import (
    GoldResolution,
    Resolution,
    ResolutionPolicy,
    SpanResolver,
    character_f1,
    character_precision,
    character_recall,
    gold_coverage_by_chunk,
    retrieved_character_count,
)

__version__ = "0.0.1"

__all__ = [
    "Chunk",
    "ContextGridError",
    "Document",
    "DocumentError",
    "EvalItem",
    "EvalSet",
    "EvalSetError",
    "GoldResolution",
    "GoldSpan",
    "GridWarning",
    "MissingExtraError",
    "RelevanceLabel",
    "Resolution",
    "ResolutionError",
    "ResolutionPolicy",
    "RetrievedChunk",
    "Severity",
    "Span",
    "SpanError",
    "SpanResolver",
    "WarningCode",
    "WarningLog",
    "__version__",
    "character_f1",
    "character_precision",
    "character_recall",
    "coverage_fraction",
    "covered_length",
    "gold_coverage_by_chunk",
    "intersection_length",
    "merge_spans",
    "retrieved_character_count",
    "total_length",
]

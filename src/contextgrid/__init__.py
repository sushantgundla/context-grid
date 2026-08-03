"""context-grid: a lab for grounding pipelines.

Sweep parser x chunker x embedder x index x reranker on your own documents, and get back
ranked, reproducible results scored on quality, latency and cost.

Everything rests on one property: a piece of text always knows which characters of which
source document it came from. That is what makes comparing two chunkers -- or two parsers --
a valid thing to do at all.
"""

from __future__ import annotations

from contextgrid.chunk import CHUNKERS, ChunkerError, get_chunker
from contextgrid.core.errors import (
    ContextGridError,
    DocumentError,
    EvalSetError,
    MissingExtraError,
    ResolutionError,
    SpanError,
)
from contextgrid.core.protocols import Chunker, Parser, Tokenizer
from contextgrid.core.registry import Registry, UnknownPluginError
from contextgrid.core.types import (
    Block,
    BlockKind,
    Chunk,
    ChunkSet,
    Document,
    EvalItem,
    EvalSet,
    GoldAnchor,
    GoldSpan,
    MediaType,
    ParsedDocument,
    QuestionType,
    RelevanceLabel,
    RetrievedChunk,
    SourceFile,
    Span,
    coverage_fraction,
    covered_length,
    intersection_length,
    merge_spans,
    total_length,
)
from contextgrid.core.warnings import GridWarning, Severity, WarningCode, WarningLog
from contextgrid.corpus import (
    Corpus,
    CorpusError,
    CorpusFingerprint,
    fingerprint,
    fingerprint_sources,
)
from contextgrid.parse import PARSERS, get_parser
from contextgrid.score.anchor import (
    AnchorMatch,
    AnchorResolver,
    MatchStrategy,
    collapse_whitespace,
)
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
from contextgrid.tokens import TOKENIZERS, get_tokenizer

__version__ = "0.0.3"

__all__ = [
    "CHUNKERS",
    "PARSERS",
    "TOKENIZERS",
    "AnchorMatch",
    "AnchorResolver",
    "Block",
    "BlockKind",
    "Chunk",
    "ChunkSet",
    "Chunker",
    "ChunkerError",
    "ContextGridError",
    "Corpus",
    "CorpusError",
    "CorpusFingerprint",
    "Document",
    "DocumentError",
    "EvalItem",
    "EvalSet",
    "EvalSetError",
    "GoldAnchor",
    "GoldResolution",
    "GoldSpan",
    "GridWarning",
    "MatchStrategy",
    "MediaType",
    "MissingExtraError",
    "ParsedDocument",
    "Parser",
    "QuestionType",
    "Registry",
    "RelevanceLabel",
    "Resolution",
    "ResolutionError",
    "ResolutionPolicy",
    "RetrievedChunk",
    "Severity",
    "SourceFile",
    "Span",
    "SpanError",
    "SpanResolver",
    "Tokenizer",
    "UnknownPluginError",
    "WarningCode",
    "WarningLog",
    "__version__",
    "character_f1",
    "character_precision",
    "character_recall",
    "collapse_whitespace",
    "coverage_fraction",
    "covered_length",
    "fingerprint",
    "fingerprint_sources",
    "get_chunker",
    "get_parser",
    "get_tokenizer",
    "gold_coverage_by_chunk",
    "intersection_length",
    "merge_spans",
    "retrieved_character_count",
    "total_length",
]

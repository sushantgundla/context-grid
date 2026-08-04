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

# Library chunkers. Registered lazily so `pip install context-grid` stays one dependency and
# `import contextgrid` stays cheap; asking for one without the extra raises MissingExtraError
# naming what to install.
#
# The five above are ours and are offset-exact by construction. These are what people actually
# deploy. Having all three sources on one axis -- ours, chonkie's, LangChain's -- is the
# comparison this package exists to make, and the reason none of them was reimplemented.
_CHONKIE = (
    ("chonkie:token", "ChonkieTokenChunker", "size", "Fixed token windows, chonkie's."),
    (
        "chonkie:recursive",
        "ChonkieRecursiveChunker",
        "size",
        "Chonkie's recursive splitter. The head-to-head against ours.",
    ),
    ("chonkie:sentence", "ChonkieSentenceChunker", "size", "Whole sentences, chonkie's."),
    (
        "chonkie:code",
        "ChonkieCodeChunker",
        "size",
        "Splits on the syntax tree. Nothing hand-written comes close.",
    ),
)

for _name, _attr, _shorthand, _doc in _CHONKIE:
    CHUNKERS.register_lazy(
        _name,
        module="contextgrid.chunk.chonkie",
        attr=_attr,
        extra="chunk",
        package="chonkie",
        shorthand=_shorthand,
        doc=_doc,
    )

_LANGCHAIN = (
    (
        "langchain:recursive",
        "LangChainRecursiveChunker",
        "What most deployed systems are actually running.",
    ),
    ("langchain:character", "LangChainCharacterChunker", "One separator only. The naive baseline."),
    ("langchain:markdown", "LangChainMarkdownChunker", "Recursive, Markdown boundaries first."),
)

for _name, _attr, _doc in _LANGCHAIN:
    CHUNKERS.register_lazy(
        _name,
        module="contextgrid.chunk.langchain",
        attr=_attr,
        extra="chunk",
        package="langchain-text-splitters",
        shorthand="size",
        doc=_doc,
    )


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

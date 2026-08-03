"""Source files, parsed documents, blocks and chunks.

The pipeline shape these types encode:

    SourceFile  --parser-->  ParsedDocument  --chunker-->  Chunk
    (raw bytes)              (text + blocks)               (retrievable unit)

The important subtlety is that **the parser produces the text**. For a PDF there is no
canonical text until something extracts it, and two parsers extract different text -- that
is the whole reason parser choice is worth measuring. So character offsets are always
offsets into one particular parse, and `ParsedDocument.text_hash` is what stops two parses
being mixed up with each other.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contextgrid.core.errors import DocumentError
from contextgrid.core.span import Span
from contextgrid.core.warnings import WarningLog

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------


class MediaType(str, Enum):
    """The input formats the pipeline knows about."""

    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    UNKNOWN = "application/octet-stream"

    @classmethod
    def from_suffix(cls, suffix: str) -> MediaType:
        """Guess from a file extension. Unknown extensions are not an error here --
        a parser declines them explicitly, which produces a better message."""
        return _SUFFIX_TO_MEDIA_TYPE.get(suffix.lower().lstrip("."), cls.UNKNOWN)


_SUFFIX_TO_MEDIA_TYPE = {
    "txt": MediaType.TEXT,
    "text": MediaType.TEXT,
    "md": MediaType.MARKDOWN,
    "markdown": MediaType.MARKDOWN,
    "mdx": MediaType.MARKDOWN,
    "html": MediaType.HTML,
    "htm": MediaType.HTML,
    "pdf": MediaType.PDF,
    "docx": MediaType.DOCX,
    "pptx": MediaType.PPTX,
    "xlsx": MediaType.XLSX,
}


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One input file, before anything has been extracted from it.

    `raw` holds the bytes when they have been read. For born-digital text formats a parser
    will decode them; for a PDF it will do considerably more.
    """

    id: str
    media_type: MediaType = MediaType.UNKNOWN
    path: str | None = None
    raw: bytes | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int | None:
        return None if self.raw is None else len(self.raw)

    def content_hash(self) -> str:
        """SHA-256 of the raw bytes. The cache key every downstream stage hangs off."""
        if self.raw is None:
            raise DocumentError(
                f"source file {self.id!r} has no bytes loaded, so it cannot be hashed"
            )
        return hashlib.sha256(self.raw).hexdigest()

    def text(self, encoding: str = "utf-8") -> str:
        """Decode the bytes as text. Only meaningful for text formats."""
        if self.raw is None:
            raise DocumentError(f"source file {self.id!r} has no bytes loaded")
        return self.raw.decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """A body of text that character offsets refer to.

    Produced by a parser from a `SourceFile`. Two parsers over the same file produce two
    documents with the same `id` and different text, which is exactly the thing under test
    -- so anything holding spans must also know which parse they belong to.
    """

    id: str
    text: str
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)

    def span(self) -> Span:
        """A span covering the whole document."""
        return Span(self.id, 0, len(self.text))

    def slice(self, span: Span) -> str:
        """The text a span refers to.

        Raises rather than silently returning a short string when the span belongs to a
        different document or runs past the end -- a truncated slice would be scored as if
        it were real evidence.
        """
        if span.doc_id != self.id:
            raise DocumentError(
                f"span belongs to document {span.doc_id!r}, cannot slice document {self.id!r}"
            )
        if span.end > len(self.text):
            raise DocumentError(
                f"span {span.start}-{span.end} runs past the end of document "
                f"{self.id!r} (length {len(self.text)})"
            )
        return self.text[span.start : span.end]

    def contains_span(self, span: Span) -> bool:
        return span.doc_id == self.id and span.end <= len(self.text)

    def text_hash(self) -> str:
        """SHA-256 of the text. Identifies which parse a set of offsets belongs to."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class BlockKind(str, Enum):
    """What a parser thinks a region of the document is.

    Structural chunkers use this, and so does the diagnosis of why a parser lost a table.
    """

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    TABLE_ROW = "table_row"
    LIST = "list"
    LIST_ITEM = "list_item"
    CODE = "code"
    QUOTE = "quote"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Block:
    """A structural region of a parsed document, with its position in the text."""

    span: Span
    text: str
    kind: BlockKind = BlockKind.PARAGRAPH
    page: int | None = None
    level: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.span.doc_id

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING


# ---------------------------------------------------------------------------
# Parsed documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """One parser's reading of one source file.

    `offsets_exact` is the honesty flag for the parse. It is true when every block's text is
    literally `document.text[block.span]`. A parser that reflows columns, re-orders reading
    sequence or rewrites table cells may not be able to promise that, and saying so is better
    than letting a user draw a confident conclusion from an approximate comparison.
    """

    document: Document
    blocks: tuple[Block, ...] = ()
    parser: str = "unknown"
    parser_version: str = "0"
    offsets_exact: bool = True
    page_count: int | None = None
    duration_ms: float | None = None
    warnings: WarningLog = field(default_factory=WarningLog)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def text(self) -> str:
        return self.document.text

    def text_hash(self) -> str:
        """Identifies this parse. Chunks carrying offsets are only valid against this hash."""
        return self.document.text_hash()

    def verify_blocks(self) -> list[Block]:
        """Blocks whose text does not match the document at their own span.

        Empty for any parser claiming exact offsets. The conformance suite asserts that.
        """
        return [
            block
            for block in self.blocks
            if not self.document.contains_span(block.span)
            or self.document.slice(block.span) != block.text
        ]

    def blocks_of(self, *kinds: BlockKind) -> tuple[Block, ...]:
        wanted = set(kinds)
        return tuple(block for block in self.blocks if block.kind in wanted)

    def block_at(self, position: int) -> Block | None:
        """The block containing a character position, if any."""
        for block in self.blocks:
            if block.span.start <= position < block.span.end:
                return block
        return None

    def page_at(self, position: int) -> int | None:
        block = self.block_at(position)
        return None if block is None else block.page

    def heading_path_at(self, position: int) -> tuple[str, ...]:
        """The chain of headings above a position, outermost first.

        Attached to chunks as metadata, where it is one of the cheapest retrieval gains
        available -- a chunk that knows it sits under "Termination > Notice" is far easier
        to match than a bare paragraph.
        """
        path: list[tuple[int, str]] = []
        for block in self.blocks:
            if block.span.start > position:
                break
            if not block.is_heading:
                continue
            level = block.level if block.level is not None else 1
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, block.text.strip()))
        return tuple(text for _, text in path)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text, and where it came from.

    `offsets_exact` is the honesty flag. Most chunkers slice the document, so their text is
    literally `document.text[span]` and the flag is true. Some do not: contextual retrieval
    prepends an LLM-written summary, proposition extraction rewrites sentences into atomic
    facts. Those chunks still carry the span they derive from, but their text is not a slice
    of it, and scoring built on them is approximate. Saying so is better than hiding it.
    """

    id: str
    span: Span
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    offsets_exact: bool = True

    @property
    def doc_id(self) -> str:
        return self.span.doc_id

    @property
    def char_start(self) -> int:
        return self.span.start

    @property
    def char_end(self) -> int:
        return self.span.end

    @property
    def char_length(self) -> int:
        return self.span.length

    def token_count(self, tokenizer: str) -> int | None:
        """Length in tokens under a named tokenizer, if it has been measured.

        Chunk size is meaningless without naming the tokenizer -- "512 with 50 overlap"
        describes different text under cl100k_base than under a BERT wordpiece vocabulary.
        Sizes are therefore stored per tokenizer and never as a single number.
        """
        return self.token_counts.get(tokenizer)

    def matches_source(self, document: Document) -> bool:
        """True when this chunk's text is exactly the document text its span points at.

        The invariant every offset-exact chunker must satisfy, and what the conformance
        suite checks.
        """
        if not document.contains_span(self.span):
            return False
        return document.slice(self.span) == self.text


@dataclass(frozen=True, slots=True)
class ChunkSet:
    """Every chunk one chunker produced from one parse, kept together with its provenance.

    The `text_hash` is what makes mixing parses impossible: chunks carrying offsets into
    PyMuPDF's extraction are meaningless against Docling's, and this is the field that
    catches it.
    """

    chunks: tuple[Chunk, ...]
    source_id: str
    parser: str
    chunker: str
    text_hash: str
    tokenizer: str | None = None
    offsets_exact: bool = True
    warnings: WarningLog = field(default_factory=WarningLog)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    def belongs_to(self, parsed: ParsedDocument) -> bool:
        """Whether these chunks were cut from this exact parse."""
        return self.text_hash == parsed.text_hash()

    def require_parse(self, parsed: ParsedDocument) -> None:
        """Raise unless these chunks came from this parse.

        Called before scoring. Comparing chunks against the wrong parse produces numbers
        that look entirely reasonable and are meaningless.
        """
        if not self.belongs_to(parsed):
            raise DocumentError(
                f"chunk set from parser {self.parser!r} does not match the given parse of "
                f"{parsed.id!r} (parser {parsed.parser!r}). Character offsets are only valid "
                "against the parse that produced them."
            )

    @property
    def total_characters(self) -> int:
        return sum(chunk.char_length for chunk in self.chunks)


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk a configuration returned for a query, with its position and score."""

    chunk: Chunk
    score: float
    rank: int
    rank_before_rerank: int | None = None

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def moved(self) -> int | None:
        """Positions gained by reranking. Positive means it moved up."""
        if self.rank_before_rerank is None:
            return None
        return self.rank_before_rerank - self.rank


def chunks_of(retrieved: Sequence[RetrievedChunk]) -> list[Chunk]:
    return [r.chunk for r in retrieved]


def spans_of(chunks: Iterable[Chunk]) -> list[Span]:
    return [c.span for c in chunks]

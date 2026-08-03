"""Shared machinery for chunkers.

Every chunker's job reduces to the same thing: decide character ranges, then hand them here
to become chunks. Keeping construction in one place means the invariants -- exact offsets,
per-tokenizer sizes, heading provenance, stable ids -- are guaranteed once rather than
re-implemented correctly in nine places and incorrectly in a tenth.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.errors import ContextGridError
from contextgrid.core.protocols import Tokenizer
from contextgrid.core.span import Span
from contextgrid.tokens import get_tokenizer


class ChunkerError(ContextGridError, ValueError):
    """A chunker was configured in a way that cannot produce sensible chunks."""


def chunk_id(doc_id: str, start: int, end: int) -> str:
    """A stable, readable identifier.

    Derived from position rather than a counter, so the same chunker over the same text
    produces the same ids every run -- which is what lets cached results be reused and
    lets two runs be diffed at all.
    """
    return f"{doc_id}:{start}-{end}"


class ChunkBuilder:
    """Turns character ranges into chunks for one parsed document.

    Heading lookup is precomputed. Walking the block list for every chunk would be
    quadratic, which is invisible on a test fixture and unacceptable on a real corpus.
    """

    def __init__(
        self,
        parsed: ParsedDocument,
        tokenizers: Sequence[Tokenizer] | None = None,
    ) -> None:
        self.parsed = parsed
        self.tokenizers = list(tokenizers) if tokenizers else [get_tokenizer(None)]
        self._heading_positions: list[int] = []
        self._heading_paths: list[tuple[str, ...]] = []
        self._page_positions: list[int] = []
        self._pages: list[int] = []
        self._precompute()

    def _precompute(self) -> None:
        path: list[tuple[int, str]] = []
        for block in self.parsed.blocks:
            if block.is_heading:
                level = block.level if block.level is not None else 1
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, block.text.strip()))
                self._heading_positions.append(block.span.start)
                self._heading_paths.append(tuple(text for _, text in path))
            if block.page is not None:
                self._page_positions.append(block.span.start)
                self._pages.append(block.page)

    def heading_path_at(self, position: int) -> tuple[str, ...]:
        index = bisect.bisect_right(self._heading_positions, position) - 1
        return self._heading_paths[index] if index >= 0 else ()

    def page_at(self, position: int) -> int | None:
        index = bisect.bisect_right(self._page_positions, position) - 1
        return self._pages[index] if index >= 0 else None

    def build(
        self,
        start: int,
        end: int,
        *,
        index: int | None = None,
        extra_meta: dict[str, object] | None = None,
    ) -> Chunk:
        text = self.parsed.text[start:end]
        meta: dict[str, object] = {
            "heading_path": self.heading_path_at(start),
            "page": self.page_at(start),
            "parser": self.parsed.parser,
        }
        if index is not None:
            meta["index"] = index
        if extra_meta:
            meta.update(extra_meta)

        return Chunk(
            id=chunk_id(self.parsed.id, start, end),
            span=Span(self.parsed.id, start, end),
            text=text,
            meta=meta,
            token_counts={tok.name: tok.count(text) for tok in self.tokenizers},
            # A chunk is only as exact as the parse it was cut from.
            offsets_exact=self.parsed.offsets_exact,
        )

    def build_all(self, ranges: Sequence[tuple[int, int]]) -> list[Chunk]:
        return [
            self.build(start, end, index=index)
            for index, (start, end) in enumerate(ranges)
            if end > start
        ]


# ---------------------------------------------------------------------------
# token-range helpers
# ---------------------------------------------------------------------------


def trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a range past surrounding whitespace, without ever inverting it."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def token_starts(spans: Sequence[tuple[int, int]]) -> list[int]:
    return [start for start, _ in spans]


def expand_back_by_tokens(
    spans: Sequence[tuple[int, int]],
    starts: Sequence[int],
    position: int,
    tokens: int,
) -> int:
    """Move a chunk's start backwards by a number of tokens, for overlap.

    Working in tokens rather than characters keeps the overlap the same size as the chunk
    parameter it is expressed in -- an overlap of 50 characters and an overlap of 50 tokens
    are very different amounts of context, and conflating them is a common way these
    comparisons drift.
    """
    if tokens <= 0 or not spans:
        return position
    index = bisect.bisect_left(starts, position)
    return spans[max(0, index - tokens)][0]


def validate_size_and_overlap(size: int, overlap: int) -> None:
    if size <= 0:
        raise ChunkerError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ChunkerError(f"overlap must be >= 0, got {overlap}")
    if overlap >= size:
        raise ChunkerError(
            f"overlap ({overlap}) must be smaller than size ({size}); "
            "an overlap at or above the chunk size never advances through the document"
        )

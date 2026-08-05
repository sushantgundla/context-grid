"""Recursive character chunking.

The de-facto default in every RAG framework, and the arm most real systems are actually
running. It splits on the largest natural boundary that fits -- paragraphs, then lines, then
sentences, then words -- and only falls back to cutting mid-word when nothing else works.

Implemented over character ranges rather than by concatenating strings, so every chunk is a
literal slice of the document and the offsets are exact by construction rather than by
careful bookkeeping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from contextgrid.chunk.base import (
    ChunkBuilder,
    expand_back_by_tokens,
    token_starts,
    trim_range,
    validate_size_and_overlap,
)
from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.protocols import Tokenizer
from contextgrid.tokens import get_tokenizer

#: Largest natural boundary first. The empty string is the last resort: cut anywhere.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    """Split on the largest separator that produces pieces within `size` tokens.

    `separators` is tried in order. Each piece too large for the current separator is split
    again with the next one down, then adjacent pieces are packed back together greedily up
    to `size` -- which is what stops a document of one-line paragraphs becoming a document
    of one-line chunks.
    """

    size: int = 512
    #: `None` means an eighth of the size, which is 64 at the default 512 -- exactly the fixed
    #: default this replaced. Fixed, it made `recursive:64` an error: an inherited 64 collides
    #: with a size of 64, and refusing a perfectly reasonable chunk size because of a default
    #: the user never named is a bad axis value. An overlap they *do* name is still checked.
    overlap: int | None = None
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "recursive"
    version: ClassVar[str] = "1"

    _tokenizer: Tokenizer = field(init=False, repr=False, compare=False)
    #: `overlap` with the default resolved against `size`. Always an int.
    _overlap: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self) -> None:
        resolved = self.size // 8 if self.overlap is None else self.overlap
        # Both, deliberately. `overlap` is what a manifest records, and `None` there would say
        # nothing about what actually ran; `_overlap` is the same number typed as an int, so
        # the arithmetic below needs no narrowing at every use.
        object.__setattr__(self, "overlap", resolved)
        object.__setattr__(self, "_overlap", resolved)
        validate_size_and_overlap(self.size, self._overlap)
        object.__setattr__(self, "_tokenizer", get_tokenizer(self.tokenizer))

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        if not text.strip():
            return []

        pieces = self._split(text, 0, len(text), self.separators)
        packed = self._pack(text, pieces)
        ranges = self._apply_overlap(text, packed)

        builder = ChunkBuilder(parsed, [self._tokenizer])
        return builder.build_all(ranges)

    # -- splitting -----------------------------------------------------------

    def _split(
        self, text: str, start: int, end: int, separators: tuple[str, ...]
    ) -> list[tuple[int, int]]:
        """Break one range into pieces that each fit, recursing down the separator list."""
        if start >= end:
            return []
        if self._fits(text, start, end):
            return [(start, end)]
        if not separators:
            return self._split_by_tokens(text, start, end)

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return self._split_by_tokens(text, start, end)

        parts = _split_on(text, start, end, separator)
        if len(parts) == 1:
            # This separator does not appear here; try a finer one.
            return self._split(text, start, end, rest)

        pieces: list[tuple[int, int]] = []
        for piece_start, piece_end in parts:
            pieces.extend(self._split(text, piece_start, piece_end, rest))
        return pieces

    def _split_by_tokens(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Last resort: cut at token boundaries, ignoring meaning entirely."""
        spans = self._tokenizer.token_spans(text[start:end])
        if not spans:
            return [(start, end)]
        ranges: list[tuple[int, int]] = []
        for cursor in range(0, len(spans), self.size):
            window = spans[cursor : cursor + self.size]
            ranges.append((start + window[0][0], start + window[-1][1]))
        return ranges

    def _fits(self, text: str, start: int, end: int) -> bool:
        return self._tokenizer.count(text[start:end]) <= self.size

    # -- packing -------------------------------------------------------------

    def _pack(self, text: str, pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge adjacent pieces greedily while they still fit.

        Without this, a document of short paragraphs produces a chunk per paragraph and the
        `size` parameter does nothing at all -- a failure mode that looks like a working
        chunker right up until you plot recall against chunk size and get a flat line.
        """
        packed: list[tuple[int, int]] = []
        for start, end in pieces:
            if not packed:
                packed.append((start, end))
                continue
            open_start, _ = packed[-1]
            if self._fits(text, open_start, end):
                packed[-1] = (open_start, end)
            else:
                packed.append((start, end))
        return [trim_range(text, start, end) for start, end in packed]

    # -- overlap -------------------------------------------------------------

    def _apply_overlap(self, text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if self._overlap <= 0 or len(ranges) < 2:
            return ranges
        spans = self._tokenizer.token_spans(text)
        starts = token_starts(spans)
        expanded = [ranges[0]]
        for start, end in ranges[1:]:
            expanded.append((expand_back_by_tokens(spans, starts, start, self._overlap), end))
        return expanded


def _split_on(text: str, start: int, end: int, separator: str) -> list[tuple[int, int]]:
    """Split a range on a separator, keeping the separator with the piece before it.

    Keeping it attached means the pieces tile the range exactly, with no characters lost
    between them -- which is what allows a chunk to be a literal slice.
    """
    pattern = re.escape(separator)
    parts: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(pattern, text[start:end]):
        boundary = start + match.end()
        if boundary > cursor:
            parts.append((cursor, boundary))
            cursor = boundary
    if cursor < end:
        parts.append((cursor, end))
    return parts or [(start, end)]

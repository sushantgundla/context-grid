"""Structural, heading-aware chunking.

Cuts on the document's own structure: a chunk is a section, from one heading to the next.
Sections too large to be one chunk are split further with the recursive chunker; sections too
small are merged with their neighbours.

This is the arm that usually wins on documentation and contracts, and it is the one that
depends on the parser having found the headings at all -- which makes it the sharpest
demonstration of why parser choice belongs on the grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar

from contextgrid.chunk.base import ChunkBuilder, ChunkerError, trim_range
from contextgrid.chunk.recursive import RecursiveChunker
from contextgrid.core.documents import BlockKind, Chunk, Document, ParsedDocument
from contextgrid.core.protocols import Tokenizer
from contextgrid.tokens import get_tokenizer

Splitter = Callable[[int, int], list[tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class StructuralChunker:
    """One chunk per section, bounded by `max_size` and `min_size` tokens.

    `keep_heading_path` prepends the chain of headings above a chunk to its text -- so a
    paragraph under "Termination > Notice period" carries those words into its embedding.
    It reliably helps retrieval, and it makes the chunk text no longer a literal slice of the
    document, so chunks produced that way declare `offsets_exact=False`.
    """

    max_size: int = 512
    #: `None` means an eighth of `max_size`, which is 64 at the default 512 -- exactly the
    #: fixed default this replaced. Fixed, it made `structural:64` an error: the shorthand sets
    #: `max_size` and the inherited 64 then collided with it. Refusing a reasonable size
    #: because of a default the user never named is a bad axis value.
    min_size: int | None = None
    #: `min_size` with the default resolved against `max_size`. Always an int.
    _min_size: int = field(init=False, repr=False, compare=False, default=0)
    keep_heading_path: bool = False
    split_tables: bool = False
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "structural"
    version: ClassVar[str] = "1"

    _tokenizer: Tokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ChunkerError(f"max_size must be positive, got {self.max_size}")
        resolved = self.max_size // 8 if self.min_size is None else self.min_size
        # Both, deliberately: `min_size` is what a manifest records and `None` there would say
        # nothing about what ran; `_min_size` is the same number typed as an int.
        object.__setattr__(self, "min_size", resolved)
        object.__setattr__(self, "_min_size", resolved)
        if self._min_size < 0:
            raise ChunkerError(f"min_size must be >= 0, got {self._min_size}")
        if self._min_size >= self.max_size:
            raise ChunkerError(
                f"min_size ({self._min_size}) must be below max_size ({self.max_size})"
            )
        object.__setattr__(self, "_tokenizer", get_tokenizer(self.tokenizer))

    # -- the protocol --------------------------------------------------------

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        if not text.strip():
            return []

        sections = self._sections(parsed)
        if not sections:
            # Nothing structural to work with, so fall back rather than return nothing. A
            # parser that found no headings should score badly, not produce an empty index.
            return self._recursive().chunk(parsed)

        builder = ChunkBuilder(parsed, [self._tokenizer])
        split = self._splitter(parsed)
        protected = self._protected_ranges(parsed)

        chunks: list[Chunk] = []
        for start, end in self._merge_small(text, sections):
            for piece_start, piece_end in self._cut(text, start, end, split, protected):
                chunks.append(self._build(builder, piece_start, piece_end, len(chunks)))
        return chunks

    # -- section discovery ---------------------------------------------------

    def _sections(self, parsed: ParsedDocument) -> list[tuple[int, int]]:
        """Ranges running from each heading to the next one."""
        headings = [block for block in parsed.blocks if block.is_heading]
        if not headings:
            return []

        text = parsed.text
        bounds: list[tuple[int, int]] = []

        preamble = trim_range(text, 0, headings[0].span.start)
        if preamble[1] > preamble[0]:
            bounds.append(preamble)

        for index, heading in enumerate(headings):
            end = headings[index + 1].span.start if index + 1 < len(headings) else len(text)
            trimmed = trim_range(text, heading.span.start, end)
            if trimmed[1] > trimmed[0]:
                bounds.append(trimmed)
        return bounds

    def _merge_small(self, text: str, sections: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Fold sections below `min_size` into the section before them.

        A document of many short headings otherwise produces a chunk per heading, each too
        small to carry any context -- the classic failure of naive structural chunking.
        """
        if self._min_size <= 0:
            return sections
        merged: list[tuple[int, int]] = []
        for start, end in sections:
            if not merged:
                merged.append((start, end))
                continue
            previous_start, _ = merged[-1]
            too_small = self._count(text, start, end) < self._min_size
            if too_small and self._count(text, previous_start, end) <= self.max_size:
                merged[-1] = (previous_start, end)
            else:
                merged.append((start, end))
        return merged

    # -- cutting oversized sections ------------------------------------------

    def _cut(
        self,
        text: str,
        start: int,
        end: int,
        split: Splitter,
        protected: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if self._count(text, start, end) <= self.max_size:
            return [(start, end)]

        inside = [(s, e) for s, e in protected if s >= start and e <= end]
        if not inside:
            return split(start, end)

        # Cutting a table in half is one of the most damaging things a chunker does and one
        # of the hardest to spot from a leaderboard. Tables come out whole unless asked
        # otherwise, even when that means a chunk over max_size.
        ranges: list[tuple[int, int]] = []
        cursor = start
        for table_start, table_end in sorted(inside):
            if table_start > cursor:
                ranges.extend(split(cursor, table_start))
            ranges.append((table_start, table_end))
            cursor = table_end
        if cursor < end:
            ranges.extend(split(cursor, end))
        return [(s, e) for s, e in ranges if e > s]

    def _protected_ranges(self, parsed: ParsedDocument) -> list[tuple[int, int]]:
        if self.split_tables:
            return []
        return [
            (block.span.start, block.span.end)
            for block in parsed.blocks
            if block.kind in {BlockKind.TABLE, BlockKind.TABLE_ROW}
        ]

    def _splitter(self, parsed: ParsedDocument) -> Splitter:
        """A function splitting one range of this document with the recursive chunker."""
        inner = self._recursive()

        def split(start: int, end: int) -> list[tuple[int, int]]:
            section = ParsedDocument(
                document=Document(id=parsed.id, text=parsed.text[start:end]),
                parser=parsed.parser,
                parser_version=parsed.parser_version,
                offsets_exact=parsed.offsets_exact,
            )
            return [
                (start + chunk.char_start, start + chunk.char_end) for chunk in inner.chunk(section)
            ]

        return split

    def _recursive(self) -> RecursiveChunker:
        return RecursiveChunker(size=self.max_size, overlap=0, tokenizer=self._tokenizer)

    # -- chunk construction --------------------------------------------------

    def _build(self, builder: ChunkBuilder, start: int, end: int, index: int) -> Chunk:
        chunk = builder.build(start, end, index=index)
        if not self.keep_heading_path:
            return chunk

        path = builder.heading_path_at(start)
        if not path:
            return chunk

        prefix = " > ".join(path)
        # Prepending text means the chunk is no longer a slice of the document, so it can no
        # longer honestly claim exact offsets. The span still points at where it came from.
        return replace(
            chunk,
            text=f"{prefix}\n\n{chunk.text}",
            offsets_exact=False,
            meta={**chunk.meta, "heading_prefix": prefix},
        )

    def _count(self, text: str, start: int, end: int) -> int:
        return self._tokenizer.count(text[start:end])

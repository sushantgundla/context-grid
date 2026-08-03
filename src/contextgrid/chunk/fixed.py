"""Fixed-size token chunking.

The baseline everyone claims to have tuned. Cuts every `size` tokens with `overlap` tokens
carried backwards from the previous chunk, ignoring sentence, paragraph and table
boundaries entirely.

Worth having precisely because it is naive: it is the arm every cleverer chunker has to beat,
and on a surprising number of corpora it does not lose by much.
"""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class FixedTokenChunker:
    """Cut every `size` tokens, carrying `overlap` tokens of context backwards.

    Sizes are measured with `tokenizer`, whose name is recorded on every chunk. Two runs
    using "512" under different tokenizers are not comparable, and this is what makes that
    visible rather than silent.
    """

    size: int = 512
    overlap: int = 64
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "fixed"
    version: ClassVar[str] = "1"

    _tokenizer: Tokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_size_and_overlap(self.size, self.overlap)
        object.__setattr__(self, "_tokenizer", get_tokenizer(self.tokenizer))

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        spans = self._tokenizer.token_spans(text)
        if not spans:
            return []

        starts = token_starts(spans)
        builder = ChunkBuilder(parsed, [self._tokenizer])
        step = self.size - self.overlap

        ranges: list[tuple[int, int]] = []
        for cursor in range(0, len(spans), step):
            window = spans[cursor : cursor + self.size]
            if not window:
                break
            start = expand_back_by_tokens(spans, starts, window[0][0], self.overlap)
            end = window[-1][1]
            ranges.append(trim_range(text, start, end))
            if cursor + self.size >= len(spans):
                break

        return builder.build_all(_deduplicate(ranges))


def _deduplicate(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop repeated and fully-contained ranges.

    Overlap on a very short document can make the last window identical to the one before
    it. Emitting both would double-count that text in every character-level metric.
    """
    kept: list[tuple[int, int]] = []
    for start, end in ranges:
        if kept and start >= kept[-1][0] and end <= kept[-1][1]:
            continue
        kept.append((start, end))
    return kept

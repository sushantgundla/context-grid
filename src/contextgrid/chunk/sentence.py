"""Sentence-window chunking.

Chunks are a sliding window of whole sentences. Cheap, boundary-respecting, and it beats
more sophisticated strategies often enough to be worth keeping as a serious arm rather than
a baseline.

Sentence detection is a regex, not a linguistic model. It handles common abbreviations and
decimals and will still be wrong sometimes -- on "Fig. 3 shows" it usually holds, on
"e.g. the party of the first part" it sometimes does not. The failure is visible in the
chunk boundaries rather than hidden, which is the right kind of wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.chunk.base import ChunkBuilder, ChunkerError, trim_range
from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.protocols import Tokenizer
from contextgrid.tokens import get_tokenizer

#: Abbreviations that are never ordinary words, so case does not matter.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "inc",
        "ltd",
        "corp",
        "approx",
        "cf",
        "al",
        "eg",
        "ie",
    }
)

#: Abbreviations that collide with common words, so only the capitalised form counts.
#: "No. 5" is a number; "she said no." is a sentence, and treating the second as an
#: abbreviation swallows the sentence boundary and merges two chunks into one.
_CAPITALISED_ABBREVIATIONS = frozenset({"No", "Fig", "St", "Co", "Art", "Sec", "Ch", "Vol"})

_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")  # noqa: RUF001
_TRAILING_TOKEN = re.compile(r"([A-Za-z]+|\d+)[.!?\"'”’)\]]*$")  # noqa: RUF001


@dataclass(frozen=True, slots=True)
class SentenceWindowChunker:
    """A sliding window of `window` sentences, advancing `stride` sentences at a time.

    `stride` smaller than `window` produces overlapping chunks, which is usually what you
    want: a fact stated across a sentence boundary survives in at least one window.
    """

    window: int = 3
    stride: int = 1
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "sentence"
    version: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ChunkerError(f"window must be positive, got {self.window}")
        if self.stride <= 0:
            raise ChunkerError(
                f"stride must be positive, got {self.stride}; "
                "a stride of zero never advances through the document"
            )

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        sentences = sentence_ranges(text)
        if not sentences:
            return []

        builder = ChunkBuilder(parsed, [get_tokenizer(self.tokenizer)])
        ranges: list[tuple[int, int]] = []
        for cursor in range(0, len(sentences), self.stride):
            group = sentences[cursor : cursor + self.window]
            if not group:
                break
            ranges.append(trim_range(text, group[0][0], group[-1][1]))
            if cursor + self.window >= len(sentences):
                break

        return builder.build_all(_drop_contained(ranges))


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of sentences, with surrounding whitespace trimmed."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if _is_abbreviation(text, match.start()):
            continue
        trimmed = _trim(text, cursor, end)
        if trimmed is not None:
            ranges.append(trimmed)
        cursor = end
    trimmed = _trim(text, cursor, len(text))
    if trimmed is not None:
        ranges.append(trimmed)
    return ranges


def _is_abbreviation(text: str, position: int) -> bool:
    """Whether the full stop at `position` is part of an abbreviation or a decimal."""
    if position + 1 < len(text) and text[position + 1].isdigit():
        return True  # a decimal point, as in "3.5"
    match = _TRAILING_TOKEN.search(text[:position])
    if match is None:
        return False
    word = match.group(1)
    if word.isdigit():
        return True  # a numbered list item or a section number
    return word.lower() in _ABBREVIATIONS or word in _CAPITALISED_ABBREVIATIONS


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    start, end = trim_range(text, start, end)
    return None if start >= end else (start, end)


def _drop_contained(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove windows fully inside the previous one, which the tail of a document produces."""
    kept: list[tuple[int, int]] = []
    for start, end in ranges:
        if kept and start >= kept[-1][0] and end <= kept[-1][1]:
            continue
        kept.append((start, end))
    return kept

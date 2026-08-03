"""Proof that the conformance suites can fail.

A suite that passes everything might be checking nothing. Each test here builds a plugin
with one specific, realistic bug and asserts the corresponding invariant catches it.

The bugs are not invented. Every one of them is a mistake that is easy to make, produces
plausible-looking output, and would corrupt results without failing anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import pytest

from contextgrid.core.documents import (
    BlockKind,
    Chunk,
    Document,
    MediaType,
    ParsedDocument,
    SourceFile,
)
from contextgrid.core.span import Span
from contextgrid.parse import MarkdownParser
from tests.support import CONTRACT, source

PARSER = MarkdownParser()
CONTRACT_SOURCE = source("contract", CONTRACT)
PARSED = PARSER.parse(CONTRACT_SOURCE)


def blocks_are_literal_slices(parsed: ParsedDocument) -> bool:
    return parsed.verify_blocks() == []


def chunks_are_literal_slices(chunks: list[Chunk], document: Document) -> bool:
    return all(chunk.matches_source(document) for chunk in chunks if chunk.offsets_exact)


def covers_all_content(spans: list[Span], text: str) -> bool:
    covered: set[int] = set()
    for span in spans:
        covered.update(range(span.start, span.end))
    return all(index in covered for index, char in enumerate(text) if not char.isspace())


# ---------------------------------------------------------------------------
# parser bugs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OffByOneParser:
    """A parser whose block spans are shifted by one character.

    The classic. Comes from treating a range as inclusive at both ends, produces text that
    looks almost right, and shifts every gold-span resolution by one character.
    """

    name: ClassVar[str] = "off-by-one"
    version: ClassVar[str] = "0"

    def supports(self, media_type: MediaType) -> bool:
        return True

    def parse(self, src: SourceFile) -> ParsedDocument:
        good = PARSER.parse(src)
        shifted = tuple(
            replace(block, span=Span(block.span.doc_id, block.span.start, block.span.end - 1))
            for block in good.blocks
            if block.span.length > 1
        )
        return replace(good, blocks=shifted, parser=self.name)


@dataclass(frozen=True, slots=True)
class NormalisingParser:
    """A parser that tidies whitespace but keeps the original offsets.

    Very common in real wrappers: the library returns cleaned-up text, the wrapper records
    where it thinks the text came from, and the two drift apart.
    """

    name: ClassVar[str] = "normalising"
    version: ClassVar[str] = "0"

    def supports(self, media_type: MediaType) -> bool:
        return True

    def parse(self, src: SourceFile) -> ParsedDocument:
        good = PARSER.parse(src)
        cleaned = tuple(replace(block, text=" ".join(block.text.split())) for block in good.blocks)
        return replace(good, blocks=cleaned, parser=self.name)


@dataclass(frozen=True, slots=True)
class TableLosingParser:
    """A parser that drops tables entirely.

    Exactly what a fast PDF extractor does to a financial report, and the reason the parser
    belongs on the grid. The content simply vanishes, and nothing downstream can tell.
    """

    name: ClassVar[str] = "table-losing"
    version: ClassVar[str] = "0"

    def supports(self, media_type: MediaType) -> bool:
        return True

    def parse(self, src: SourceFile) -> ParsedDocument:
        good = PARSER.parse(src)
        kept = tuple(block for block in good.blocks if block.kind is not BlockKind.TABLE)
        return replace(good, blocks=kept, parser=self.name)


def test_offset_invariant_catches_an_off_by_one() -> None:
    assert blocks_are_literal_slices(PARSED)
    assert not blocks_are_literal_slices(OffByOneParser().parse(CONTRACT_SOURCE))


def test_offset_invariant_catches_silent_normalisation() -> None:
    parsed = NormalisingParser().parse(CONTRACT_SOURCE)
    assert not blocks_are_literal_slices(parsed)


def test_coverage_invariant_catches_a_lost_table() -> None:
    good = PARSER.parse(CONTRACT_SOURCE)
    assert covers_all_content([b.span for b in good.blocks], good.text)

    lossy = TableLosingParser().parse(CONTRACT_SOURCE)
    assert not covers_all_content([b.span for b in lossy.blocks], lossy.text)


def test_a_broken_parser_is_still_shaped_like_a_parser() -> None:
    """The bugs above are behavioural, not structural. Type checks alone would miss all of
    them, which is the argument for behavioural conformance suites."""
    from contextgrid.core.protocols import Parser

    assert isinstance(OffByOneParser(), Parser)
    assert isinstance(TableLosingParser(), Parser)


# ---------------------------------------------------------------------------
# chunker bugs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GappyChunker:
    """A chunker that skips the text between chunks.

    Comes from advancing the cursor by `size` while emitting only `size - overlap`
    characters. The chunks look fine individually.
    """

    name: ClassVar[str] = "gappy"
    version: ClassVar[str] = "0"

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        chunks: list[Chunk] = []
        for start in range(0, len(text), 200):
            end = min(start + 120, len(text))
            if end <= start:
                continue
            chunks.append(
                Chunk(
                    id=f"{parsed.id}:{start}-{end}",
                    span=Span(parsed.id, start, end),
                    text=text[start:end],
                    token_counts={"regex": 1},
                )
            )
        return chunks


@dataclass(frozen=True, slots=True)
class LyingChunker:
    """A chunker that rewrites text while claiming exact offsets.

    What a contextual-retrieval chunker does if it forgets to set the flag: prepend an
    LLM-written summary, keep saying the text is a literal slice.
    """

    name: ClassVar[str] = "lying"
    version: ClassVar[str] = "0"

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        return [
            Chunk(
                id=f"{parsed.id}:0-{len(text)}",
                span=Span(parsed.id, 0, len(text)),
                text=f"Context: this is a contract.\n\n{text}",
                token_counts={"regex": 1},
                offsets_exact=True,
            )
        ]


@dataclass(frozen=True, slots=True)
class CollidingChunker:
    """A chunker whose ids repeat.

    Two chunks with one id collapse into a single entry in every qrel and every run,
    silently discarding one of them.
    """

    name: ClassVar[str] = "colliding"
    version: ClassVar[str] = "0"

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        half = len(text) // 2
        return [
            Chunk(
                id=f"{parsed.id}:chunk",
                span=Span(parsed.id, 0, half),
                text=text[:half],
                token_counts={"regex": 1},
            ),
            Chunk(
                id=f"{parsed.id}:chunk",
                span=Span(parsed.id, half, len(text)),
                text=text[half:],
                token_counts={"regex": 1},
            ),
        ]


def test_coverage_invariant_catches_a_gappy_chunker() -> None:
    chunks = GappyChunker().chunk(PARSED)
    assert chunks
    assert not covers_all_content([c.span for c in chunks], PARSED.text)


def test_offset_invariant_catches_a_chunker_that_lies_about_rewriting() -> None:
    chunks = LyingChunker().chunk(PARSED)
    assert not chunks_are_literal_slices(chunks, PARSED.document)


def test_the_same_chunker_passes_once_it_tells_the_truth() -> None:
    """Rewriting text is allowed. Claiming it did not is the bug."""
    honest = [replace(c, offsets_exact=False) for c in LyingChunker().chunk(PARSED)]
    assert chunks_are_literal_slices(honest, PARSED.document)


def test_uniqueness_invariant_catches_colliding_ids() -> None:
    ids = [chunk.id for chunk in CollidingChunker().chunk(PARSED)]
    assert len(ids) != len(set(ids))


@pytest.mark.parametrize(
    "broken", [GappyChunker(), LyingChunker(), CollidingChunker()], ids=lambda c: c.name
)
def test_broken_chunkers_still_satisfy_the_type(broken: object) -> None:
    from contextgrid.core.protocols import Chunker

    assert isinstance(broken, Chunker)

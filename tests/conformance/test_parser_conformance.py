"""The contract every parser must satisfy.

There will eventually be a dozen parsers here, several of them wrapping large third-party
libraries whose behaviour changes between releases. A parser that quietly loses character
offsets would corrupt every number derived from it without failing anything, so the
invariants are asserted centrally and every implementation is run through them.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from contextgrid.core.documents import MediaType, ParsedDocument, SourceFile
from contextgrid.core.errors import DocumentError
from contextgrid.core.protocols import Parser
from contextgrid.core.warnings import WarningCode
from tests.support import ALL_PARSERS, ALL_SOURCES, source

PARSER_IDS = [p.name for p in ALL_PARSERS]
SOURCE_IDS = [s.id for s in ALL_SOURCES]


@pytest.fixture(params=ALL_PARSERS, ids=PARSER_IDS)
def parser(request: pytest.FixtureRequest) -> Parser:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=ALL_SOURCES, ids=SOURCE_IDS)
def sample(request: pytest.FixtureRequest, parser: Parser) -> SourceFile:
    source_file: SourceFile = request.param
    if not parser.supports(source_file.media_type):
        pytest.skip(f"{parser.name} does not read {source_file.media_type.value}")
    return source_file


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_satisfies_the_protocol(parser: Parser) -> None:
    assert isinstance(parser, Parser)


def test_has_a_name_and_a_version(parser: Parser) -> None:
    assert parser.name
    assert parser.version


def test_declares_what_it_supports(parser: Parser) -> None:
    assert parser.supports(MediaType.TEXT) or parser.supports(MediaType.PDF)


# ---------------------------------------------------------------------------
# the offset invariant
# ---------------------------------------------------------------------------


def test_every_block_is_a_literal_slice_of_the_document(parser: Parser, sample: SourceFile) -> None:
    """The invariant the whole package rests on.

    If `document.text[block.span]` is not `block.text`, then gold spans resolved against
    this parse point at the wrong characters, and every downstream metric is wrong in a way
    no leaderboard would show.
    """
    parsed = parser.parse(sample)
    if not parsed.offsets_exact:
        pytest.skip(f"{parser.name} declares approximate offsets")
    assert parsed.verify_blocks() == []


def test_blocks_stay_inside_the_document(parser: Parser, sample: SourceFile) -> None:
    parsed = parser.parse(sample)
    for block in parsed.blocks:
        assert parsed.document.contains_span(block.span)
        assert block.span.doc_id == sample.id


def test_blocks_are_in_reading_order(parser: Parser, sample: SourceFile) -> None:
    parsed = parser.parse(sample)
    starts = [block.span.start for block in parsed.blocks]
    assert starts == sorted(starts)


def test_blocks_do_not_overlap(parser: Parser, sample: SourceFile) -> None:
    """Overlapping blocks would double-count text in every structural measure."""
    parsed = parser.parse(sample)
    for earlier, later in pairwise(parsed.blocks):
        assert earlier.span.end <= later.span.start


def test_no_block_is_empty(parser: Parser, sample: SourceFile) -> None:
    parsed = parser.parse(sample)
    assert all(block.text.strip() for block in parsed.blocks)


def test_blocks_cover_every_non_whitespace_character(parser: Parser, sample: SourceFile) -> None:
    """A parser may drop whitespace between blocks. It may not drop content.

    Text that appears in no block is text no chunker will ever see and no retriever can
    ever return -- an invisible hole in the corpus.
    """
    parsed = parser.parse(sample)
    covered = set()
    for block in parsed.blocks:
        covered.update(range(block.span.start, block.span.end))
    missing = {
        index
        for index, char in enumerate(parsed.text)
        if not char.isspace() and index not in covered
    }
    assert not missing, f"{parser.name} dropped {len(missing)} characters"


# ---------------------------------------------------------------------------
# determinism and provenance
# ---------------------------------------------------------------------------


def test_parsing_twice_gives_the_same_result(parser: Parser, sample: SourceFile) -> None:
    """Non-determinism would make cached results and run comparison meaningless."""
    first = parser.parse(sample)
    second = parser.parse(sample)
    assert first.text == second.text
    assert first.blocks == second.blocks
    assert first.text_hash() == second.text_hash()


def test_records_which_parser_produced_it(parser: Parser, sample: SourceFile) -> None:
    parsed = parser.parse(sample)
    assert parsed.parser == parser.name
    assert parsed.parser_version == parser.version


def test_document_keeps_the_source_id(parser: Parser, sample: SourceFile) -> None:
    assert parser.parse(sample).document.id == sample.id


def test_text_hash_differs_when_the_text_differs(parser: Parser) -> None:
    """The guard that stops chunks from one parse being scored against another."""
    if parser.supports(MediaType.PDF):
        from tests.pdf_fixtures import contract_pdf, prose_pdf
        from tests.support import pdf

        one = parser.parse(pdf("a", contract_pdf()))
        two = parser.parse(pdf("a", prose_pdf()))
    else:
        one = parser.parse(source("a", "The notice period is thirty days."))
        two = parser.parse(source("a", "The notice period is sixty days."))
    assert one.text_hash() != two.text_hash()


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


def empty_for(parser: Parser) -> SourceFile:
    """A document with no retrievable text, in a format this parser accepts."""
    if parser.supports(MediaType.PDF):
        from tests.pdf_fixtures import scanned_pdf
        from tests.support import pdf

        return pdf("scanned-pdf", scanned_pdf())
    return source("empty", "")


def test_empty_document_parses_without_raising(parser: Parser) -> None:
    parsed = parser.parse(empty_for(parser))
    assert parsed.blocks == ()
    assert parsed.text == ""


def test_empty_document_says_so(parser: Parser) -> None:
    """A page with no text layer is a scan. Saying nothing would let it score zero silently."""
    parsed = parser.parse(empty_for(parser))
    assert parsed.warnings.of_code(WarningCode.EMPTY_TEXT_LAYER)


def test_whitespace_only_document_produces_no_blocks(parser: Parser) -> None:
    if parser.supports(MediaType.PDF):
        pytest.skip("a PDF has no whitespace-only equivalent")
    assert parser.parse(source("blank", "  \n\n \t ")).blocks == ()


def test_unread_source_file_is_a_clear_error(parser: Parser) -> None:
    """Better than an AttributeError on None three frames down."""
    unread = SourceFile(id="never-read", media_type=MediaType.TEXT, path="x.txt")
    with pytest.raises(DocumentError, match="no bytes loaded"):
        parser.parse(unread)


def test_returns_a_parsed_document(parser: Parser, sample: SourceFile) -> None:
    assert isinstance(parser.parse(sample), ParsedDocument)

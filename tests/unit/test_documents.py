"""Unit tests for source files, parses and chunk sets.

The theme is the one subtlety that makes the parser axis possible: two parsers over the same
file produce two different texts, so a set of character offsets is only meaningful against
the one parse that produced it.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import (
    Block,
    BlockKind,
    Chunk,
    ChunkSet,
    Document,
    MediaType,
    ParsedDocument,
    SourceFile,
)
from contextgrid.core.errors import DocumentError
from contextgrid.core.span import Span

TEXT = "# Heading\n\nThe notice period is thirty days.\n"


def parsed(text: str = TEXT, parser: str = "markdown") -> ParsedDocument:
    return ParsedDocument(document=Document(id="d", text=text), parser=parser)


# ---------------------------------------------------------------------------
# MediaType
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("pdf", MediaType.PDF),
        (".pdf", MediaType.PDF),
        ("PDF", MediaType.PDF),
        ("md", MediaType.MARKDOWN),
        ("markdown", MediaType.MARKDOWN),
        ("txt", MediaType.TEXT),
        ("docx", MediaType.DOCX),
        ("xyz", MediaType.UNKNOWN),
    ],
)
def test_media_type_from_suffix(suffix: str, expected: MediaType) -> None:
    assert MediaType.from_suffix(suffix) is expected


# ---------------------------------------------------------------------------
# SourceFile
# ---------------------------------------------------------------------------


def test_source_file_reports_its_size_and_hash() -> None:
    source = SourceFile(id="a", media_type=MediaType.TEXT, raw=b"hello")
    assert source.size_bytes == 5
    assert len(source.content_hash()) == 64


def test_identical_bytes_hash_identically() -> None:
    """The cache key every downstream stage hangs off."""
    a = SourceFile(id="a", raw=b"same")
    b = SourceFile(id="b", raw=b"same")
    assert a.content_hash() == b.content_hash()


def test_an_unread_file_has_no_size_and_cannot_be_hashed() -> None:
    source = SourceFile(id="a", path="a.txt")
    assert source.size_bytes is None
    with pytest.raises(DocumentError, match="cannot be hashed"):
        source.content_hash()


def test_decoding_an_unread_file_is_a_clear_error() -> None:
    with pytest.raises(DocumentError, match="no bytes loaded"):
        SourceFile(id="a").text()


def test_undecodable_bytes_do_not_crash() -> None:
    """A corpus with one badly encoded file should not take the whole run down."""
    assert SourceFile(id="a", raw=b"\xff\xfe caf\xe9").text()


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------


def test_parse_exposes_its_text_and_id() -> None:
    document = parsed()
    assert document.id == "d"
    assert document.text == TEXT


def test_two_parsers_of_the_same_file_hash_differently() -> None:
    """The guard that makes the parser axis safe: offsets from one parse are meaningless
    against another, and the hash is what catches the mistake."""
    pymupdf_like = parsed("Notice period: thirty days", parser="a")
    docling_like = parsed("Notice period:  thirty  days", parser="b")
    assert pymupdf_like.text_hash() != docling_like.text_hash()


def test_verify_blocks_finds_a_block_that_lies_about_its_position() -> None:
    good = Block(span=Span("d", 0, 9), text=TEXT[0:9])
    bad = Block(span=Span("d", 0, 9), text="something else")
    assert (
        ParsedDocument(document=Document(id="d", text=TEXT), blocks=(good,)).verify_blocks() == []
    )
    assert ParsedDocument(document=Document(id="d", text=TEXT), blocks=(bad,)).verify_blocks() == [
        bad
    ]


def test_verify_blocks_finds_a_block_past_the_end() -> None:
    over = Block(span=Span("d", 0, len(TEXT) + 10), text=TEXT)
    assert ParsedDocument(document=Document(id="d", text=TEXT), blocks=(over,)).verify_blocks()


def test_page_lookup_uses_the_last_block_at_or_before_the_position() -> None:
    document = ParsedDocument(
        document=Document(id="d", text="a" * 100),
        blocks=(
            Block(span=Span("d", 0, 40), text="a" * 40, page=1),
            Block(span=Span("d", 40, 100), text="a" * 60, page=2),
        ),
    )
    assert document.page_at(10) == 1
    assert document.page_at(50) == 2


def test_page_is_unknown_before_any_block() -> None:
    document = ParsedDocument(
        document=Document(id="d", text="a" * 100),
        blocks=(Block(span=Span("d", 10, 40), text="a" * 30, page=1),),
    )
    assert document.page_at(5) is None


def test_block_at_returns_none_outside_every_block() -> None:
    assert parsed().block_at(5) is None


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_chunk_reports_its_character_length() -> None:
    chunk = Chunk(id="c", span=Span("d", 10, 25), text="x" * 15)
    assert chunk.char_length == 15


# ---------------------------------------------------------------------------
# ChunkSet
# ---------------------------------------------------------------------------


def make_set(text_hash: str) -> ChunkSet:
    return ChunkSet(
        chunks=(Chunk(id="c0", span=Span("d", 0, 9), text=TEXT[0:9]),),
        source_id="d",
        parser="markdown",
        chunker="recursive",
        text_hash=text_hash,
    )


def test_chunk_set_is_sized_and_iterable() -> None:
    chunk_set = make_set(parsed().text_hash())
    assert len(chunk_set) == 1
    assert [c.id for c in chunk_set] == ["c0"]
    assert chunk_set.total_characters == 9


def test_a_chunk_set_belongs_to_the_parse_that_produced_it() -> None:
    document = parsed()
    assert make_set(document.text_hash()).belongs_to(document)


def test_scoring_chunks_against_the_wrong_parse_is_refused() -> None:
    """Offsets into PyMuPDF's extraction are meaningless against Docling's. The numbers
    would look entirely reasonable and mean nothing."""
    other = parsed("completely different text", parser="other")
    with pytest.raises(DocumentError, match="only valid against the parse"):
        make_set("a-hash-from-a-different-parse").require_parse(other)


def test_require_parse_passes_for_the_right_parse() -> None:
    document = parsed()
    make_set(document.text_hash()).require_parse(document)  # does not raise


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


def test_block_knows_its_document_and_whether_it_is_a_heading() -> None:
    heading = Block(span=Span("d", 0, 9), text="# Heading", kind=BlockKind.HEADING, level=1)
    body = Block(span=Span("d", 11, 20), text="body text")
    assert heading.doc_id == "d"
    assert heading.is_heading
    assert not body.is_heading

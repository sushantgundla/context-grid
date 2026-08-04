"""Layout parsers: docling, marker and pymupdf4llm.

Most of the logic worth testing is not in the engines -- it is in turning the Markdown they
emit into blocks with exact offsets, keeping tables whole and attributing pages. That part
needs no engine at all and is tested directly.

`pymupdf4llm` is fast enough to run for real on every suite. `docling` and `marker` load vision
models and take minutes on a first run, so they are behind `CG_SLOW_PARSERS=1`. They have been
run for real: docling parses the contract fixture to 4 blocks with exact offsets, finding the
table `pymupdf` flattens into prose.
"""

from __future__ import annotations

import os

import pytest

from contextgrid.core.documents import BlockKind, MediaType, SourceFile
from contextgrid.parse import PARSERS, get_parser
from contextgrid.parse.layout import (
    PyMuPDF4LLMParser,
    _blocks_from_markdown,
)
from contextgrid.parse.pymupdf import PyMuPDFParser
from tests.pdf_fixtures import contract_pdf, prose_pdf

slow = pytest.mark.skipif(
    os.environ.get("CG_SLOW_PARSERS") != "1",
    reason="loads vision models and takes minutes; set CG_SLOW_PARSERS=1 to run",
)


# ---------------------------------------------------------------------------
# Markdown into blocks -- no engine needed
# ---------------------------------------------------------------------------


def test_a_table_stays_in_one_block() -> None:
    """A table split across blocks is a table no chunker can be told to keep together, and on
    a financial document that decides whether the answer is retrievable at all."""
    blocks = _blocks_from_markdown(
        "Fees are below.\n\n"
        "| Service | Fee |\n|---|---|\n| Standard | 1200 |\n| Premium | 3400 |\n\n"
        "Payable within 30 days.\n"
    )
    tables = [text for text, kind, _ in blocks if kind is BlockKind.TABLE]
    assert len(tables) == 1
    assert "Standard" in tables[0]
    assert "Premium" in tables[0]


def test_headings_come_out_as_headings() -> None:
    blocks = _blocks_from_markdown("# Title\n\nSome prose.\n\n## Section\n\nMore prose.\n")
    headings = [text for text, kind, _ in blocks if kind is BlockKind.HEADING]
    assert headings == ["Title", "Section"]


def test_the_hash_marks_are_not_part_of_the_heading_text() -> None:
    """They would end up inside the chunk, inside the embedding, and inside a gold quote that
    then fails to resolve."""
    blocks = _blocks_from_markdown("### Termination for cause\n")
    assert blocks[0][0] == "Termination for cause"


def test_paragraphs_are_split_on_blank_lines() -> None:
    blocks = _blocks_from_markdown("First para\nstill first.\n\nSecond para.\n")
    assert [text for text, _, _ in blocks] == ["First para\nstill first.", "Second para."]


def test_a_table_between_two_paragraphs_keeps_its_place() -> None:
    """Reading order is what parsers most often get wrong, and getting it wrong destroys
    retrieval on any document where the answer depends on what a sentence sat next to."""
    blocks = _blocks_from_markdown("Before.\n\n| a | b |\n|---|---|\n\nAfter.\n")
    assert [kind for _, kind, _ in blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.PARAGRAPH,
    ]


def test_page_markers_set_the_page_and_are_dropped() -> None:
    """The comment is scaffolding. Left in the text it lands inside a chunk and then inside an
    embedding."""
    blocks = _blocks_from_markdown(
        "<!-- page: 1 -->\nFirst page.\n\n<!-- page: 2 -->\nSecond page.\n"
    )
    assert [(text, page) for text, _, page in blocks] == [("First page.", 1), ("Second page.", 2)]
    assert not any("page:" in text for text, _, _ in blocks)


def test_empty_markdown_produces_no_blocks() -> None:
    assert _blocks_from_markdown("") == []
    assert _blocks_from_markdown("\n\n   \n") == []


def test_a_table_at_the_very_end_is_not_lost() -> None:
    """An off-by-one in the flush would silently drop the last table in every document."""
    blocks = _blocks_from_markdown("Intro.\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert blocks[-1][1] is BlockKind.TABLE


# ---------------------------------------------------------------------------
# pymupdf4llm -- fast enough to run for real
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contract() -> SourceFile:
    return SourceFile(id="contract", raw=contract_pdf(), media_type=MediaType.PDF)


def test_every_block_is_a_literal_slice_of_the_text(contract: SourceFile) -> None:
    """What changes between parsers is *what the text is*, not whether offsets into it can be
    trusted. Gold spans resolve against this parse, so a block that is not a slice moves every
    piece of evidence in the document."""
    parsed = PyMuPDF4LLMParser().parse(contract)
    for block in parsed.blocks:
        assert parsed.text[block.span.start : block.span.end] == block.text


def test_it_finds_the_table_the_plain_extractor_flattens(contract: SourceFile) -> None:
    """The whole reason this arm is on the axis. Same extraction engine, different output: one
    returns a table, the other returns a soup of digits in reading order."""
    markdown = PyMuPDF4LLMParser().parse(contract)
    plain = PyMuPDFParser().parse(contract)

    assert any(block.kind is BlockKind.TABLE for block in markdown.blocks)
    assert not any(block.kind is BlockKind.TABLE for block in plain.blocks)


def test_headings_are_marked_as_headings(contract: SourceFile) -> None:
    parsed = PyMuPDF4LLMParser().parse(contract)
    assert any(block.is_heading for block in parsed.blocks)


def test_it_records_which_engine_produced_the_text(contract: SourceFile) -> None:
    """Two parsers over the same PDF produce different text, so a parse that does not say which
    one it came from cannot be reproduced or compared."""
    parsed = PyMuPDF4LLMParser().parse(contract)
    assert parsed.parser == "pymupdf4llm"
    assert parsed.meta["output_format"] == "markdown"


def test_pages_are_attributed(contract: SourceFile) -> None:
    parsed = PyMuPDF4LLMParser().parse(contract)
    assert parsed.page_count >= 1
    assert any(block.page is not None for block in parsed.blocks)


def test_parsing_twice_gives_the_same_text(contract: SourceFile) -> None:
    """Caching and diffing both rest on this."""
    parser = PyMuPDF4LLMParser()
    assert parser.parse(contract).text == parser.parse(contract).text


def test_prose_parses_without_inventing_tables() -> None:
    source = SourceFile(id="prose", raw=prose_pdf(), media_type=MediaType.PDF)
    parsed = PyMuPDF4LLMParser().parse(source)
    assert parsed.text.strip()
    assert not any(block.kind is BlockKind.TABLE for block in parsed.blocks)


def test_a_document_parses_the_same_whatever_was_parsed_before_it(
    contract: SourceFile,
) -> None:
    """The bug that made process isolation non-negotiable.

    pymupdf4llm's output depends on which documents went through the same interpreter before
    it -- state persists in MuPDF's C layer, below Python, so reloading the module and emptying
    MuPDF's store both fail to clear it. Without isolation this prose PDF parses to 1182
    characters alone and 919 mangled ones ("notce perod s trty") after the contract.

    A corpus that parses differently depending on file order cannot be reproduced, and the
    parse is what every offset, every chunk and every score downstream rests on.
    """
    prose = SourceFile(id="prose", raw=prose_pdf(), media_type=MediaType.PDF)
    parser = PyMuPDF4LLMParser()

    alone = parser.parse(prose)
    parser.parse(contract)
    after = parser.parse(prose)

    assert alone.text == after.text


def test_isolation_is_on_by_default() -> None:
    """Off, the failure it re-enables is silent."""
    assert PyMuPDF4LLMParser().isolate
    assert (
        PyMuPDF4LLMParser()
        .parse(SourceFile(id="prose", raw=prose_pdf(), media_type=MediaType.PDF))
        .meta["isolated"]
    )


def test_a_broken_pdf_says_which_document_it_was() -> None:
    """The worker runs in another process, so its traceback has to be carried back with the
    document's name attached or a failed sweep says nothing useful."""
    from contextgrid.core.errors import DocumentError

    source = SourceFile(id="not-a-pdf", raw=b"this is not a PDF", media_type=MediaType.PDF)
    with pytest.raises(DocumentError, match="not-a-pdf"):
        PyMuPDF4LLMParser().parse(source)


def test_source_bytes_that_were_never_read_is_a_clear_error() -> None:
    source = SourceFile(id="unread", raw=None, media_type=MediaType.PDF)
    from contextgrid.core.errors import DocumentError

    with pytest.raises(DocumentError, match="no bytes loaded"):
        PyMuPDF4LLMParser().parse(source)


def test_it_only_claims_the_formats_it_can_read() -> None:
    parser = PyMuPDF4LLMParser()
    assert parser.supports(MediaType.PDF)
    assert not parser.supports(MediaType.DOCX)


def test_docling_claims_more_than_pdf() -> None:
    """A corpus of DOCX has no other parser on this axis, which is most of why docling earns
    its weight."""
    from contextgrid.parse.layout import DoclingParser

    assert DoclingParser().supports(MediaType.DOCX)
    assert DoclingParser().supports(MediaType.HTML)


# ---------------------------------------------------------------------------
# reachable from a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["docling", "marker", "pymupdf4llm"])
def test_every_parser_is_registered(name: str) -> None:
    assert name in PARSERS
    assert PARSERS.describe()[name]


@pytest.mark.parametrize(
    "spec",
    ["pymupdf4llm", "pymupdf4llm:table_strategy=lines", "docling:ocr=true", "marker:en"],
)
def test_every_parser_is_reachable_from_one_config_line(spec: str) -> None:
    assert get_parser(spec).name in {"pymupdf4llm", "docling", "marker"}


def test_the_parser_axis_can_now_sweep_three_real_engines() -> None:
    from contextgrid.grid import matrix

    configs = matrix(parser=["pymupdf", "pymupdf4llm", "docling", "marker"]).expand("factorial")
    assert len({config.parser for config in configs}) == 4


# ---------------------------------------------------------------------------
# the heavy ones
# ---------------------------------------------------------------------------


@slow
def test_docling_parses_with_exact_offsets(contract: SourceFile) -> None:
    from contextgrid.parse.layout import DoclingParser

    parsed = DoclingParser().parse(contract)
    assert parsed.text.strip()
    for block in parsed.blocks:
        assert parsed.text[block.span.start : block.span.end] == block.text


@slow
def test_docling_finds_the_table(contract: SourceFile) -> None:
    from contextgrid.parse.layout import DoclingParser

    parsed = DoclingParser().parse(contract)
    assert any(block.kind is BlockKind.TABLE for block in parsed.blocks)


@slow
def test_marker_parses_with_exact_offsets(contract: SourceFile) -> None:
    from contextgrid.parse.layout import MarkerParser

    parsed = MarkerParser().parse(contract)
    assert parsed.text.strip()
    for block in parsed.blocks:
        assert parsed.text[block.span.start : block.span.end] == block.text

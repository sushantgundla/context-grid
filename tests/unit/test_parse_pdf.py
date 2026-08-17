"""Unit tests for the PDF parsers, and for what they disagree about.

The conformance suite already proves both keep their offsets honest. What matters here is
the disagreement: two parsers reading the same bytes produce different text, different
structure and different retrievable evidence. That difference is the axis nothing else in
the field measures, so it needs to be pinned down rather than assumed.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import BlockKind, MediaType, SourceFile
from contextgrid.core.errors import DocumentError
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import WarningCode
from contextgrid.parse.builder import TextAssembler, infer_heading_levels, round_size
from contextgrid.parse.pdfplumber import PDFPlumberParser
from contextgrid.parse.pymupdf import PyMuPDFParser
from contextgrid.score.anchor import AnchorResolver
from tests.pdf_fixtures import contract_pdf, mixed_pdf, prose_pdf, scanned_pdf
from tests.support import pdf

CONTRACT = pdf("contract-pdf", contract_pdf())
PROSE = pdf("prose-pdf", prose_pdf())
SCANNED = pdf("scanned-pdf", scanned_pdf())
MIXED = pdf("mixed-pdf", mixed_pdf())

PARSERS = [PyMuPDFParser(), PDFPlumberParser()]
IDS = [p.name for p in PARSERS]


@pytest.fixture(params=PARSERS, ids=IDS)
def parser(request: pytest.FixtureRequest) -> PyMuPDFParser | PDFPlumberParser:
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# what both parsers must do
# ---------------------------------------------------------------------------


def test_both_read_the_prose(parser: PyMuPDFParser) -> None:
    parsed = parser.parse(CONTRACT)
    assert "thirty days written notice" in " ".join(parsed.text.split())


def test_both_keep_offsets_exact_on_a_pdf(parser: PyMuPDFParser) -> None:
    """A PDF has no text until the parser makes some, which is exactly where offsets
    usually drift. The assembler makes it structurally impossible."""
    parsed = parser.parse(CONTRACT)
    assert parsed.offsets_exact
    assert parsed.verify_blocks() == []


def test_both_record_the_page_count_and_how_long_they_took(parser: PyMuPDFParser) -> None:
    parsed = parser.parse(PROSE)
    assert parsed.page_count == 2
    assert parsed.duration_ms is not None
    assert parsed.duration_ms > 0


def test_both_tag_blocks_with_their_page(parser: PyMuPDFParser) -> None:
    parsed = parser.parse(PROSE)
    assert {block.page for block in parsed.blocks} == {1, 2}


def test_both_decline_formats_they_cannot_read(parser: PyMuPDFParser) -> None:
    assert parser.supports(MediaType.PDF)
    assert not parser.supports(MediaType.MARKDOWN)


def test_both_report_a_page_with_no_text_layer(parser: PyMuPDFParser) -> None:
    """A scan scores zero whatever the retriever does. Saying so beats a silent zero."""
    warnings = parser.parse(SCANNED).warnings.of_code(WarningCode.EMPTY_TEXT_LAYER)
    assert warnings
    assert "OCR" in warnings[0].message


def test_both_report_a_blank_page_inside_a_readable_document(parser: PyMuPDFParser) -> None:
    parsed = parser.parse(MIXED)
    assert "Schedule A" in parsed.text
    assert parsed.warnings.of_code(WarningCode.EMPTY_TEXT_LAYER)


def test_both_refuse_an_unread_file(parser: PyMuPDFParser) -> None:
    with pytest.raises(DocumentError, match="no bytes loaded"):
        parser.parse(SourceFile(id="x", media_type=MediaType.PDF))


# ---------------------------------------------------------------------------
# where they differ -- the reason the parser is an axis
# ---------------------------------------------------------------------------


def test_the_two_parsers_produce_different_text_from_the_same_bytes() -> None:
    """The premise of the whole project, stated as a test."""
    one = PyMuPDFParser().parse(CONTRACT)
    two = PDFPlumberParser().parse(CONTRACT)
    assert one.text != two.text
    assert one.text_hash() != two.text_hash()


def test_pymupdf_shatters_a_table_into_loose_cells() -> None:
    """It has no idea what a table is. Every cell becomes its own block, and the
    relationship between a row label and its value exists only as page geometry."""
    parsed = PyMuPDFParser().parse(CONTRACT)
    assert not parsed.blocks_of(BlockKind.TABLE)
    assert any(block.text == "3400" for block in parsed.blocks)


def test_pdfplumber_keeps_the_table_together() -> None:
    parsed = PDFPlumberParser().parse(CONTRACT)
    tables = parsed.blocks_of(BlockKind.TABLE)
    assert len(tables) == 1
    assert "Premium" in tables[0].text
    assert "3400" in tables[0].text


def test_pymupdf_infers_headings_from_font_size() -> None:
    """A PDF has no headings, only larger text. Guessing is what makes structural chunking
    possible at all, and two parsers guessing differently is a real effect on retrieval."""
    headings = [b for b in PyMuPDFParser().parse(CONTRACT).blocks if b.is_heading]
    assert [b.text for b in headings] == ["Master Services Agreement", "2. Termination"]
    assert [b.level for b in headings] == [1, 2]


def test_pdfplumber_finds_no_headings_so_structural_chunking_has_nothing_to_use() -> None:
    assert not PDFPlumberParser().parse(CONTRACT).blocks_of(BlockKind.HEADING)


def test_heading_detection_can_be_switched_off() -> None:
    parsed = PyMuPDFParser(detect_headings=False).parse(CONTRACT)
    assert not parsed.blocks_of(BlockKind.HEADING)


def test_a_document_set_entirely_in_one_size_has_no_headings() -> None:
    """Nothing is larger than the body, so nothing is promoted. The alternative -- calling
    the first line a heading -- would invent structure that is not there."""
    assert not PyMuPDFParser().parse(PROSE).blocks_of(BlockKind.HEADING)


# ---------------------------------------------------------------------------
# the end-to-end point: one eval set, two parses
# ---------------------------------------------------------------------------


def evalset() -> EvalSet:
    return EvalSet(
        id="es",
        items=(
            EvalItem(
                id="q_prose",
                question="How much notice is required?",
                anchors=(GoldAnchor(source_id="contract-pdf", quote="thirty days written notice"),),
            ),
            EvalItem(
                id="q_table",
                question="What is the Premium monthly fee?",
                anchors=(GoldAnchor(source_id="contract-pdf", quote="Premium 3400 500"),),
            ),
        ),
    )


@pytest.mark.parametrize("parser", PARSERS, ids=IDS)
def test_prose_evidence_survives_both_parsers(parser: PyMuPDFParser) -> None:
    parsed = parser.parse(CONTRACT)
    resolved, _ = AnchorResolver().resolve(evalset(), {"contract-pdf": parsed})
    prose = resolved.get("q_prose")
    assert prose is not None
    assert prose.is_answerable
    # And it points at real characters in that parser's own text.
    found = " ".join(parsed.document.slice(prose.gold[0].span).split())
    assert found == "thirty days written notice"


def test_table_evidence_depends_on_how_the_parser_rendered_the_table() -> None:
    """The measurement, not a bug.

    Ground truth quoting a table row as a person would read it -- "Premium 3400 500" --
    matches whichever parse laid the row out that way. pdfplumber's markdown pipes give an
    embedder a better signal and stop that quote matching, which is a genuine trade-off
    rather than a formatting preference, so it is a parameter.
    """
    plain = PDFPlumberParser(table_format="plain").parse(CONTRACT)
    piped = PDFPlumberParser(table_format="pipe").parse(CONTRACT)

    resolved_plain, _ = AnchorResolver().resolve(evalset(), {"contract-pdf": plain})
    resolved_piped, _ = AnchorResolver().resolve(evalset(), {"contract-pdf": piped})

    # `is_resolved`: whether *this* parse located the quote. Both items are answerable --
    # they carry the same anchor -- and only one parse can find it.
    assert resolved_plain.get("q_table").is_resolved  # type: ignore[union-attr]
    assert not resolved_piped.get("q_table").is_resolved  # type: ignore[union-attr]


def test_evidence_a_parser_cannot_produce_is_reported_at_the_set_level() -> None:
    piped = PDFPlumberParser(table_format="pipe").parse(CONTRACT)
    _, log = AnchorResolver().resolve(evalset(), {"contract-pdf": piped})

    summary = log.of_code(WarningCode.ANCHOR_NOT_FOUND)[-1]
    assert "could not locate 1 of 2" in summary.message
    assert summary.detail["lost"] == 1


def test_an_unknown_table_format_says_what_is_allowed() -> None:
    with pytest.raises(DocumentError, match="Choose one of"):
        PDFPlumberParser(table_format="latex").parse(CONTRACT)


def test_tables_can_be_left_alone_entirely() -> None:
    parsed = PDFPlumberParser(extract_tables=False).parse(CONTRACT)
    assert not parsed.blocks_of(BlockKind.TABLE)
    assert "Premium" in parsed.text  # the cells are still there, just as loose lines


# ---------------------------------------------------------------------------
# margin stripping
# ---------------------------------------------------------------------------


def test_a_margin_ratio_drops_text_at_the_top_and_bottom_of_the_page() -> None:
    """Repeated page furniture is one of the quietest ways to poison dense retrieval."""
    full = PyMuPDFParser().parse(PROSE)
    trimmed = PyMuPDFParser(margin_ratio=0.2).parse(PROSE)
    assert len(trimmed.blocks) < len(full.blocks)


# ---------------------------------------------------------------------------
# the assembler
# ---------------------------------------------------------------------------


def test_the_assembler_keeps_text_and_spans_in_step() -> None:
    assembler = TextAssembler("d")
    assembler.add("First block.")
    assembler.add("Second block.", kind=BlockKind.HEADING, page=2, level=1)
    parsed = assembler.build(parser="test", version="1")

    assert parsed.text == "First block.\n\nSecond block."
    assert parsed.verify_blocks() == []
    assert parsed.blocks[1].page == 2
    assert parsed.blocks[1].level == 1


def test_the_assembler_drops_blank_blocks() -> None:
    assembler = TextAssembler("d")
    assert assembler.add("   \n ") is None
    assert len(assembler) == 0


def test_the_assembler_trims_each_block() -> None:
    """Leading whitespace inside a block would make the span disagree with the text."""
    assembler = TextAssembler("d")
    assembler.add("  padded  ")
    parsed = assembler.build(parser="test", version="1")
    assert parsed.text == "padded"
    assert parsed.verify_blocks() == []


def test_the_separator_belongs_to_no_block() -> None:
    """It is layout the parser invented, not content it found."""
    assembler = TextAssembler("d", separator="\n")
    assembler.add("one")
    assembler.add("two")
    parsed = assembler.build(parser="test", version="1")
    assert parsed.text == "one\ntwo"
    assert parsed.block_at(3) is None


def test_an_empty_assembler_builds_an_empty_document() -> None:
    parsed = TextAssembler("d").build(parser="test", version="1")
    assert parsed.text == ""
    assert parsed.blocks == ()


# ---------------------------------------------------------------------------
# heading inference
# ---------------------------------------------------------------------------


def test_heading_levels_rank_sizes_above_the_body() -> None:
    levels = infer_heading_levels([(18.0, 25), (14.0, 14), (11.0, 300)])
    assert levels == {18.0: 1, 14.0: 2}


def test_body_size_is_decided_by_characters_not_by_line_count() -> None:
    """The bug this weighting exists to prevent.

    A bordered table contributes a dozen two-word cells set smaller than the body. By line
    count they outvote the prose, the inferred body size drops to the cell size, and the
    actual body text gets promoted to a heading.

    A 9pt table under 11pt prose isolates it: `min_ratio` cannot save this one, because 11
    really is meaningfully larger than 9. Only counting characters gets it right.
    """
    lines = [(18.0, 25), (11.0, 60), (11.0, 55), (11.0, 50), *[(9.0, 6)] * 12]

    by_characters = infer_heading_levels(lines)
    assert 11.0 not in by_characters  # body text stays body text
    assert by_characters == {18.0: 1}

    by_lines = infer_heading_levels([(size, 1) for size, _ in lines])
    assert 11.0 in by_lines  # the naive version promotes the prose to a heading


def test_slightly_larger_text_is_emphasis_not_structure() -> None:
    assert infer_heading_levels([(11.5, 20), (11.0, 300)]) == {}


def test_no_sizes_gives_no_headings() -> None:
    assert infer_heading_levels([]) == {}


def test_sizes_are_quantised_so_near_identical_ones_agree() -> None:
    assert round_size(11.04) == round_size(11.0) == 11.0
    assert round_size(11.4) == 11.5

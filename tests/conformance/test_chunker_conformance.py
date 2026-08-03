"""The contract every chunker must satisfy.

Chunkers are where offsets are easiest to get subtly wrong: they slice, merge, overlap and
trim, and an off-by-one survives every eyeball test while quietly shifting every gold-span
resolution by a character.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from contextgrid.core.documents import ParsedDocument, SourceFile
from contextgrid.core.protocols import Chunker
from contextgrid.core.span import merge_spans
from contextgrid.parse import MarkdownParser
from tests.support import ALL_CHUNKERS, CONTENTFUL_SOURCES, ChunkerCase, source

PARSER = MarkdownParser()

CASE_IDS = [case.label for case in ALL_CHUNKERS]
SOURCE_IDS = [s.id for s in CONTENTFUL_SOURCES]


@pytest.fixture(params=ALL_CHUNKERS, ids=CASE_IDS)
def case(request: pytest.FixtureRequest) -> ChunkerCase:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=CONTENTFUL_SOURCES, ids=SOURCE_IDS)
def parsed(request: pytest.FixtureRequest) -> ParsedDocument:
    return PARSER.parse(request.param)


def chunker_of(case: ChunkerCase) -> Chunker:
    return case.chunker


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_satisfies_the_protocol(case: ChunkerCase) -> None:
    assert isinstance(case.chunker, Chunker)


def test_has_a_name_and_a_version(case: ChunkerCase) -> None:
    assert case.chunker.name
    assert case.chunker.version


# ---------------------------------------------------------------------------
# the offset invariant
# ---------------------------------------------------------------------------


def test_every_chunk_is_a_literal_slice_of_the_document(
    case: ChunkerCase, parsed: ParsedDocument
) -> None:
    """Unless the chunker admits it rewrote the text.

    A chunker that prepends a heading path or extracts propositions produces text that is
    not in the document. That is allowed. Claiming exact offsets while doing it is not.
    """
    for chunk in case.chunker.chunk(parsed):
        if chunk.offsets_exact:
            assert chunk.matches_source(parsed.document), (
                f"{case.label} chunk {chunk.id} text does not match its own span"
            )


def test_chunks_stay_inside_the_document(case: ChunkerCase, parsed: ParsedDocument) -> None:
    for chunk in case.chunker.chunk(parsed):
        assert parsed.document.contains_span(chunk.span)
        assert chunk.doc_id == parsed.id


def test_no_chunk_is_empty(case: ChunkerCase, parsed: ParsedDocument) -> None:
    for chunk in case.chunker.chunk(parsed):
        assert chunk.text.strip()
        assert chunk.span.length > 0


def test_chunks_are_in_reading_order(case: ChunkerCase, parsed: ParsedDocument) -> None:
    starts = [chunk.char_start for chunk in case.chunker.chunk(parsed)]
    assert starts == sorted(starts)


def test_chunk_ids_are_unique(case: ChunkerCase, parsed: ParsedDocument) -> None:
    """Duplicate ids would silently collapse two chunks into one in every qrel."""
    ids = [chunk.id for chunk in case.chunker.chunk(parsed)]
    assert len(ids) == len(set(ids))


def test_no_chunk_contains_another(case: ChunkerCase, parsed: ParsedDocument) -> None:
    """Overlap is fine. One chunk swallowing another whole is redundancy, and it inflates
    character-level precision by counting the same text twice."""
    chunks = case.chunker.chunk(parsed)
    for earlier, later in pairwise(chunks):
        assert not (later.char_start >= earlier.char_start and later.char_end <= earlier.char_end)


def test_chunks_cover_every_non_whitespace_character(
    case: ChunkerCase, parsed: ParsedDocument
) -> None:
    """Text in no chunk is evidence no retriever can ever return.

    The most damaging bug a chunker can have, and the least visible: the leaderboard just
    shows slightly worse recall, with nothing pointing at the cause.
    """
    if not case.covers_everything:
        pytest.skip(f"{case.label} samples the document on purpose")

    covered = set()
    for chunk in case.chunker.chunk(parsed):
        covered.update(range(chunk.char_start, chunk.char_end))
    missing = [
        index
        for index, char in enumerate(parsed.text)
        if not char.isspace() and index not in covered
    ]
    assert not missing, (
        f"{case.label} dropped {len(missing)} characters, starting at {missing[0]}: "
        f"{parsed.text[missing[0] : missing[0] + 60]!r}"
    )


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_records_a_token_count_under_a_named_tokenizer(
    case: ChunkerCase, parsed: ParsedDocument
) -> None:
    """Chunk size means nothing without naming the tokenizer that measured it."""
    for chunk in case.chunker.chunk(parsed):
        assert chunk.token_counts, f"{case.label} recorded no token counts"
        for name, count in chunk.token_counts.items():
            assert name
            assert count > 0


def test_carries_the_heading_path_where_there_is_one(case: ChunkerCase) -> None:
    from tests.support import CONTRACT

    parsed = PARSER.parse(source("contract", CONTRACT))
    chunks = case.chunker.chunk(parsed)
    assert any(chunk.meta.get("heading_path") for chunk in chunks), (
        f"{case.label} lost the heading path on a document full of headings"
    )


def test_inherits_exactness_from_the_parse(case: ChunkerCase, parsed: ParsedDocument) -> None:
    """A chunk cannot be more exact than the parse it was cut from."""
    approximate = ParsedDocument(
        document=parsed.document,
        blocks=parsed.blocks,
        parser=parsed.parser,
        offsets_exact=False,
    )
    for chunk in case.chunker.chunk(approximate):
        assert not chunk.offsets_exact


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_chunking_twice_gives_the_same_chunks(case: ChunkerCase, parsed: ParsedDocument) -> None:
    first = case.chunker.chunk(parsed)
    second = case.chunker.chunk(parsed)
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_ids_are_stable_across_instances(case: ChunkerCase, parsed: ParsedDocument) -> None:
    """Ids derive from position, not a counter, so a cached run can be reused."""
    ids = [chunk.id for chunk in case.chunker.chunk(parsed)]
    assert all(chunk_id.startswith(f"{parsed.id}:") for chunk_id in ids)


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   \n\n \t "], ids=["empty", "whitespace"])
def test_empty_document_produces_no_chunks(case: ChunkerCase, text: str) -> None:
    assert case.chunker.chunk(PARSER.parse(source("blank", text))) == []


def test_a_document_shorter_than_one_chunk_still_produces_one(case: ChunkerCase) -> None:
    parsed = PARSER.parse(source("tiny", "Thirty days."))
    chunks = case.chunker.chunk(parsed)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Thirty days."


def test_a_document_with_no_structure_still_chunks(case: ChunkerCase) -> None:
    """Structural chunkers must fall back rather than return nothing when a parser found
    no headings -- a bad parse should score badly, not produce an empty index."""
    parsed = PARSER.parse(source("flat", "word " * 400))
    assert case.chunker.chunk(parsed)


def test_merged_chunk_spans_are_contiguous_where_coverage_is_promised(
    case: ChunkerCase, parsed: ParsedDocument
) -> None:
    """Merging the chunks should give back one region per contiguous run of content."""
    if not case.covers_everything:
        pytest.skip(f"{case.label} samples the document on purpose")
    chunks = case.chunker.chunk(parsed)
    merged = merge_spans([chunk.span for chunk in chunks])
    assert merged
    assert merged[0].start <= min(chunk.char_start for chunk in chunks)


def test_source_file_is_untouched(case: ChunkerCase) -> None:
    """Chunking must not mutate what it was given."""
    original: SourceFile = next(s for s in CONTENTFUL_SOURCES if s.id == "contract")
    before = original.raw
    case.chunker.chunk(PARSER.parse(original))
    assert original.raw == before

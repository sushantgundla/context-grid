"""Unit tests for the individual chunkers.

The conformance suite proves they all keep their offsets honest. These pin down what makes
each one different, and the configuration mistakes each one should refuse.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from contextgrid.chunk import (
    CHUNKERS,
    ChunkerError,
    FixedTokenChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    StructuralChunker,
    get_chunker,
    sentence_ranges,
)
from contextgrid.core.documents import BlockKind
from contextgrid.parse import MarkdownParser
from contextgrid.tokens import CharacterTokenizer, RegexTokenizer
from tests.support import CONTRACT, source

MD = MarkdownParser()


def parse(text: str):  # type: ignore[no-untyped-def]
    return MD.parse(source("d", text))


CONTRACT_PARSE = MD.parse(source("contract", CONTRACT))
WORDS = parse(" ".join(f"word{i}" for i in range(200)))


# ---------------------------------------------------------------------------
# configuration is validated up front
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunker", [FixedTokenChunker, RecursiveChunker])
def test_overlap_at_or_above_size_is_refused(chunker: type) -> None:
    """It would never advance through the document -- an infinite loop or a single chunk,
    depending on the implementation. Better to refuse the configuration."""
    with pytest.raises(ChunkerError, match="never advances"):
        chunker(size=100, overlap=100)


@pytest.mark.parametrize("chunker", [FixedTokenChunker, RecursiveChunker])
def test_non_positive_size_is_refused(chunker: type) -> None:
    with pytest.raises(ChunkerError, match="must be positive"):
        chunker(size=0)


@pytest.mark.parametrize("chunker", [FixedTokenChunker, RecursiveChunker])
def test_negative_overlap_is_refused(chunker: type) -> None:
    with pytest.raises(ChunkerError, match="overlap must be >= 0"):
        chunker(size=100, overlap=-1)


def test_sentence_stride_must_advance() -> None:
    with pytest.raises(ChunkerError, match="never advances"):
        SentenceWindowChunker(window=3, stride=0)


def test_structural_min_must_be_below_max() -> None:
    with pytest.raises(ChunkerError, match="must be below max_size"):
        StructuralChunker(max_size=64, min_size=64)


# ---------------------------------------------------------------------------
# fixed-size
# ---------------------------------------------------------------------------


def test_fixed_respects_the_size_it_was_given() -> None:
    chunks = FixedTokenChunker(size=20, overlap=0).chunk(WORDS)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.token_counts["regex"] == 20


def test_fixed_overlap_repeats_text_between_neighbours() -> None:
    chunks = FixedTokenChunker(size=20, overlap=5).chunk(WORDS)
    assert chunks[1].char_start < chunks[0].char_end


def test_fixed_without_overlap_does_not_repeat() -> None:
    chunks = FixedTokenChunker(size=20, overlap=0).chunk(WORDS)
    for earlier, later in pairwise(chunks):
        assert later.char_start >= earlier.char_end


def test_fixed_measures_in_the_tokenizer_it_was_given() -> None:
    """The same "size" produces very different chunks under different tokenizers, which is
    exactly why the tokenizer is recorded on every chunk."""
    by_word = FixedTokenChunker(size=20, overlap=0, tokenizer=RegexTokenizer()).chunk(WORDS)
    by_char = FixedTokenChunker(size=20, overlap=0, tokenizer=CharacterTokenizer()).chunk(WORDS)
    assert len(by_char) > len(by_word)
    assert "regex" in by_word[0].token_counts
    assert "character" in by_char[0].token_counts


def test_fixed_on_a_document_smaller_than_one_chunk() -> None:
    chunks = FixedTokenChunker(size=1000, overlap=100).chunk(parse("Short document."))
    assert len(chunks) == 1
    assert chunks[0].text == "Short document."


# ---------------------------------------------------------------------------
# recursive
# ---------------------------------------------------------------------------


def test_recursive_prefers_paragraph_boundaries() -> None:
    document = parse("First paragraph here.\n\nSecond paragraph here.\n\nThird one here.")
    chunks = RecursiveChunker(size=4, overlap=0).chunk(document)
    assert any(chunk.text.strip().endswith(".") for chunk in chunks)


def test_recursive_packs_short_paragraphs_together() -> None:
    """Without packing, a document of one-line paragraphs yields a chunk per line and the
    size parameter does nothing -- a flat line on every chunk-size plot."""
    document = parse("\n\n".join(f"Line {i}." for i in range(20)))
    packed = RecursiveChunker(size=100, overlap=0).chunk(document)
    unpacked = RecursiveChunker(size=5, overlap=0).chunk(document)
    assert len(packed) < len(unpacked)


def test_recursive_falls_back_to_cutting_mid_text_when_nothing_else_fits() -> None:
    document = parse("word " * 200)
    chunks = RecursiveChunker(size=10, overlap=0).chunk(document)
    assert len(chunks) > 10


def test_recursive_never_exceeds_size_by_much() -> None:
    chunks = RecursiveChunker(size=30, overlap=0).chunk(CONTRACT_PARSE)
    for chunk in chunks:
        assert chunk.token_counts["regex"] <= 30


def test_recursive_overlap_reaches_backwards() -> None:
    plain = RecursiveChunker(size=30, overlap=0).chunk(CONTRACT_PARSE)
    overlapped = RecursiveChunker(size=30, overlap=10).chunk(CONTRACT_PARSE)
    assert overlapped[1].char_start < plain[1].char_start


def test_custom_separators() -> None:
    document = parse("a;b;c;d;e;f;g;h")
    chunks = RecursiveChunker(size=2, overlap=0, separators=(";", "")).chunk(document)
    assert len(chunks) > 1


# ---------------------------------------------------------------------------
# sentence windows
# ---------------------------------------------------------------------------


def test_sentence_ranges_split_on_terminators() -> None:
    text = "One. Two! Three? Four."
    assert [text[s:e] for s, e in sentence_ranges(text)] == [
        "One.",
        "Two!",
        "Three?",
        "Four.",
    ]


def test_abbreviations_do_not_end_a_sentence() -> None:
    text = "Dr. Smith approved it. The board agreed."
    assert len(sentence_ranges(text)) == 2


def test_decimals_do_not_end_a_sentence() -> None:
    text = "It grew by 3.5 percent last year. That is the figure."
    assert len(sentence_ranges(text)) == 2


def test_numbered_clauses_do_not_end_a_sentence() -> None:
    text = "See clause 2. It covers termination."
    assert len(sentence_ranges(text)) == 1


def test_a_quoted_sentence_end_is_still_a_sentence_end() -> None:
    text = 'She said "no." The meeting ended.'
    assert len(sentence_ranges(text)) == 2


def test_ambiguous_abbreviations_only_count_when_capitalised() -> None:
    """ "No." is a number and "no." is a word. Treating every "no." as an abbreviation
    swallows a real sentence boundary and merges two chunks into one."""
    assert len(sentence_ranges("See No. 5 for details. It applies.")) == 2
    assert len(sentence_ranges("The answer was no. The vote failed.")) == 2


def test_sentence_window_groups_whole_sentences() -> None:
    document = parse("One. Two. Three. Four. Five.")
    chunks = SentenceWindowChunker(window=2, stride=2).chunk(document)
    assert chunks[0].text == "One. Two."


def test_sentence_stride_below_window_overlaps() -> None:
    document = parse("One. Two. Three. Four.")
    chunks = SentenceWindowChunker(window=2, stride=1).chunk(document)
    assert chunks[0].text == "One. Two."
    assert chunks[1].text == "Two. Three."


def test_sentence_stride_above_window_samples() -> None:
    """A legitimate configuration that deliberately leaves text out, which is why the
    coverage conformance check has an opt-out rather than being unconditional."""
    document = parse("One. Two. Three. Four. Five. Six.")
    chunks = SentenceWindowChunker(window=1, stride=3).chunk(document)
    assert [c.text for c in chunks] == ["One.", "Four."]


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------


def test_structural_cuts_on_headings() -> None:
    chunks = StructuralChunker(max_size=1000, min_size=0).chunk(CONTRACT_PARSE)
    assert len(chunks) == 6
    assert chunks[1].text.startswith("## 1. Term")


def test_structural_merges_sections_below_min_size() -> None:
    many = parse("\n\n".join(f"## Section {i}\n\nShort." for i in range(10)))
    merged = StructuralChunker(max_size=500, min_size=20).chunk(many)
    unmerged = StructuralChunker(max_size=500, min_size=0).chunk(many)
    assert len(merged) < len(unmerged)


def test_structural_splits_sections_above_max_size() -> None:
    big = parse("## Big section\n\n" + "word " * 300)
    chunks = StructuralChunker(max_size=50, min_size=0).chunk(big)
    assert len(chunks) > 1


def test_structural_keeps_a_table_whole_by_default() -> None:
    """Cutting a table in half is one of the most damaging things a chunker does, and one
    of the hardest to spot from a leaderboard."""
    chunks = StructuralChunker(max_size=20, min_size=0).chunk(CONTRACT_PARSE)
    table = CONTRACT_PARSE.blocks_of(BlockKind.TABLE)[0]
    assert any(chunk.span.contains(table.span) for chunk in chunks)


def test_structural_splits_tables_when_asked() -> None:
    whole = StructuralChunker(max_size=20, min_size=0, split_tables=False).chunk(CONTRACT_PARSE)
    split = StructuralChunker(max_size=20, min_size=0, split_tables=True).chunk(CONTRACT_PARSE)
    assert len(split) > len(whole)


def test_structural_falls_back_when_the_parser_found_no_headings() -> None:
    """A parser that loses the structure should score badly, not produce an empty index."""
    flat = parse("word " * 300)
    assert not [b for b in flat.blocks if b.is_heading]
    assert StructuralChunker(max_size=50, min_size=0).chunk(flat)


def test_keep_heading_path_prepends_and_admits_it_is_no_longer_a_slice() -> None:
    chunks = StructuralChunker(max_size=1000, min_size=0, keep_heading_path=True).chunk(
        CONTRACT_PARSE
    )
    prefixed = [c for c in chunks if c.meta.get("heading_prefix")]
    assert prefixed
    for chunk in prefixed:
        assert chunk.text.startswith(chunk.meta["heading_prefix"])
        assert not chunk.offsets_exact  # it rewrote the text, so it says so


# ---------------------------------------------------------------------------
# resolution from spec strings
# ---------------------------------------------------------------------------


def test_registry_knows_every_chunker() -> None:
    assert CHUNKERS.names() == ["fixed", "recursive", "sentence", "structural"]


def test_spec_strings_build_configured_chunkers() -> None:
    chunker = get_chunker("recursive:256,overlap=32")
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.size == 256
    assert chunker.overlap == 32


def test_get_chunker_passes_an_instance_through() -> None:
    instance = FixedTokenChunker()
    assert get_chunker(instance) is instance


def test_shorthand_differs_per_chunker() -> None:
    assert CHUNKERS.parse_spec("fixed:512") == ("fixed", {"size": 512})
    assert CHUNKERS.parse_spec("sentence:4") == ("sentence", {"window": 4})
    assert CHUNKERS.parse_spec("structural:800") == ("structural", {"max_size": 800})

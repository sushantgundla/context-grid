"""Unit tests for the text and Markdown parsers.

The conformance suite proves the invariants hold. These pin down the behaviour that makes
the Markdown parser worth having: finding the structure that structural chunking depends on.
"""

from __future__ import annotations

from contextgrid.core.documents import BlockKind, MediaType
from contextgrid.parse import MarkdownParser, TextParser
from tests.support import CONTRACT, source

MD = MarkdownParser()
TXT = TextParser()


def kinds(text: str) -> list[BlockKind]:
    return [block.kind for block in MD.parse(source("d", text)).blocks]


def texts(text: str) -> list[str]:
    return [block.text for block in MD.parse(source("d", text)).blocks]


# ---------------------------------------------------------------------------
# plain text
# ---------------------------------------------------------------------------


def test_text_parser_splits_on_blank_lines() -> None:
    parsed = TXT.parse(source("d", "First para.\n\nSecond para.\n\n\nThird."))
    assert [b.text for b in parsed.blocks] == ["First para.", "Second para.", "Third."]


def test_text_parser_treats_single_newlines_as_one_paragraph() -> None:
    parsed = TXT.parse(source("d", "One line\nstill the same paragraph."))
    assert len(parsed.blocks) == 1


def test_text_parser_calls_everything_a_paragraph() -> None:
    parsed = TXT.parse(source("d", "# Not a heading here\n\nJust text."))
    assert {b.kind for b in parsed.blocks} == {BlockKind.PARAGRAPH}


# ---------------------------------------------------------------------------
# markdown structure
# ---------------------------------------------------------------------------


def test_atx_headings_are_found_with_their_level() -> None:
    parsed = MD.parse(source("d", "# One\n\n## Two\n\n### Three\n"))
    headings = [(b.text, b.level) for b in parsed.blocks if b.is_heading]
    assert headings == [("# One", 1), ("## Two", 2), ("### Three", 3)]


def test_setext_headings_are_found() -> None:
    parsed = MD.parse(source("d", "Title\n=====\n\nBody text.\n"))
    assert parsed.blocks[0].kind is BlockKind.HEADING
    assert parsed.blocks[0].level == 1


def test_a_dashed_rule_after_a_paragraph_is_a_heading_not_a_list() -> None:
    parsed = MD.parse(source("d", "Subtitle\n--------\n\nBody.\n"))
    assert parsed.blocks[0].kind is BlockKind.HEADING
    assert parsed.blocks[0].level == 2


def test_lists_tables_quotes_and_code_are_distinguished() -> None:
    document = (
        "- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n> quoted\n\n```python\nx = 1\n```\n"
    )
    assert set(kinds(document)) == {
        BlockKind.LIST_ITEM,
        BlockKind.TABLE,
        BlockKind.QUOTE,
        BlockKind.CODE,
    }


def test_a_hash_inside_a_code_fence_is_not_a_heading() -> None:
    """The reason parsing is a stateful line walk rather than a regex over the whole text."""
    document = "```\n# not a heading\n```\n"
    assert kinds(document) == [BlockKind.CODE]


def test_an_unterminated_code_fence_runs_to_the_end() -> None:
    parsed = MD.parse(source("d", "text\n\n```python\nx = 1\n"))
    assert parsed.blocks[-1].kind is BlockKind.CODE
    assert parsed.blocks[-1].span.end == len(parsed.text)


def test_consecutive_list_items_group_into_one_block() -> None:
    assert texts("- one\n- two\n- three\n") == ["- one\n- two\n- three"]


def test_a_list_after_a_paragraph_is_a_separate_block() -> None:
    assert kinds("Intro text.\n- one\n- two\n") == [BlockKind.PARAGRAPH, BlockKind.LIST_ITEM]


def test_numbered_lists_are_recognised() -> None:
    assert kinds("1. one\n2. two\n") == [BlockKind.LIST_ITEM]


# ---------------------------------------------------------------------------
# heading paths
# ---------------------------------------------------------------------------


def test_heading_path_walks_up_the_levels() -> None:
    parsed = MD.parse(source("contract", CONTRACT))
    position = parsed.text.index("Either party may terminate")
    assert parsed.heading_path_at(position) == (
        "# Master Services Agreement",
        "## 2. Termination",
        "### 2.1 Notice period",
    )


def test_heading_path_pops_back_out_at_a_shallower_heading() -> None:
    parsed = MD.parse(source("contract", CONTRACT))
    position = parsed.text.index("Fees are payable")
    assert parsed.heading_path_at(position) == (
        "# Master Services Agreement",
        "## 3. Fees",
    )


def test_heading_path_is_empty_before_the_first_heading() -> None:
    parsed = MD.parse(source("d", "Preamble text.\n\n# Later heading\n"))
    assert parsed.heading_path_at(0) == ()


# ---------------------------------------------------------------------------
# block lookup
# ---------------------------------------------------------------------------


def test_block_at_a_position() -> None:
    parsed = MD.parse(source("d", "# Heading\n\nSome body text.\n"))
    block = parsed.block_at(parsed.text.index("body"))
    assert block is not None
    assert block.kind is BlockKind.PARAGRAPH


def test_block_at_a_gap_is_none() -> None:
    parsed = MD.parse(source("d", "One.\n\nTwo.\n"))
    assert parsed.block_at(parsed.text.index("\n\n")) is None


def test_blocks_of_a_kind() -> None:
    parsed = MD.parse(source("contract", CONTRACT))
    assert len(parsed.blocks_of(BlockKind.TABLE)) == 1
    assert len(parsed.blocks_of(BlockKind.HEADING)) == 6


# ---------------------------------------------------------------------------
# media types
# ---------------------------------------------------------------------------


def test_both_parsers_accept_text_and_markdown() -> None:
    for parser in (TXT, MD):
        assert parser.supports(MediaType.TEXT)
        assert parser.supports(MediaType.MARKDOWN)


def test_neither_claims_to_read_pdfs() -> None:
    for parser in (TXT, MD):
        assert not parser.supports(MediaType.PDF)

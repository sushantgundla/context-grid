"""`summary()` prints the tables share even when it is zero.

`if self.table_ratio:` is falsey at exactly 0.0, so the field vanished for any parser that
found no tables. `docs/guide/cli.md` says the line reports what share of the text is tables,
and a field that disappears reads as "this parser does not report tables" when it actually
means "this parser found none" -- opposite conclusions for somebody choosing a parser.

On the same corpus `markdown` finds 17% and `text` finds 0%. That gap is the parser axis
showing itself, and it is only visible if the 0% is printed.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.corpus import Corpus, fingerprint
from contextgrid.parse import MarkdownParser, TextParser

TABLE_DOC = """# Pricing

Some words before the table.

| Plan | Price |
| --- | --- |
| Free | 0 |
| Team | 20 |

And some words after it.
"""


def _summary(text: str, parser: object) -> str:
    corpus = Corpus.from_texts({"pricing.md": text}, media_type=MediaType.MARKDOWN)
    parses = {source.id: parser.parse(source) for source in corpus}  # type: ignore[attr-defined]
    return fingerprint(corpus, parses).summary()


def test_markdown_reports_a_non_zero_tables_share() -> None:
    summary = _summary(TABLE_DOC, MarkdownParser())

    assert "via markdown" in summary
    assert "tables" in summary
    assert "0% tables" not in summary


def test_text_parser_reports_zero_rather_than_dropping_the_field() -> None:
    """The whole point: `text` finds no tables, and must say so out loud."""
    summary = _summary(TABLE_DOC, TextParser())

    assert "via text" in summary
    assert summary.endswith("0% tables")


def test_a_corpus_with_no_tables_at_all_still_reports_zero() -> None:
    summary = _summary("# Notes\n\nJust prose, no table anywhere.\n", MarkdownParser())

    assert summary.endswith("0% tables")


def test_both_parsers_differ_on_the_same_corpus() -> None:
    """The parser axis, visible in one line each."""
    assert _summary(TABLE_DOC, MarkdownParser()) != _summary(TABLE_DOC, TextParser())


def test_unparsed_fingerprint_still_omits_the_parsed_fields() -> None:
    """No parse means no character count and no tables share -- that guard is correct."""
    summary = fingerprint(Corpus.from_texts({"a.md": TABLE_DOC})).summary()

    assert "tables" not in summary
    assert "chars via" not in summary
    assert summary == f"1 files, {len(TABLE_DOC.encode()):,} bytes"


def test_summary_is_still_one_line() -> None:
    assert "\n" not in _summary(TABLE_DOC, TextParser())


@pytest.mark.parametrize("parser", [MarkdownParser(), TextParser()])
def test_every_parsed_summary_has_all_four_fields(parser: object) -> None:
    """Nothing splits this string positionally, but the field count should stay stable."""
    assert len(_summary(TABLE_DOC, parser).split(", ")) == 4

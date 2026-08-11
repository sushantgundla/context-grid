"""Unit tests for loading and profiling a corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus, CorpusError, fingerprint, fingerprint_sources
from contextgrid.corpus.fingerprint import require_parsed_text
from contextgrid.parse import MarkdownParser
from contextgrid.pipeline import Config, build
from tests.pdf_fixtures import prose_pdf
from tests.support import API_DOCS, CONTRACT

MD = MarkdownParser()


def _from_dir_error(path: Path) -> str:
    with pytest.raises(CorpusError) as error:
        Corpus.from_dir(path)
    return str(error.value)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "contract.md").write_text(CONTRACT)
    (tmp_path / "notes.txt").write_text("Some plain notes.")
    nested = tmp_path / "guides"
    nested.mkdir()
    (nested / "api.md").write_text(API_DOCS)
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "config.md").write_text("not corpus content")
    return tmp_path


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_from_dir_reads_matching_files_recursively(corpus_dir: Path) -> None:
    corpus = Corpus.from_dir(corpus_dir)
    assert set(corpus.ids) == {"contract.md", "notes.txt", "guides/api.md"}


def test_from_dir_skips_unmatched_extensions_and_dot_directories(corpus_dir: Path) -> None:
    """A .git directory full of Markdown is not a corpus, however the glob was written."""
    corpus = Corpus.from_dir(corpus_dir)
    assert not any("ignore" in i or ".git" in i for i in corpus.ids)


def test_from_dir_can_stay_shallow(corpus_dir: Path) -> None:
    corpus = Corpus.from_dir(corpus_dir, recursive=False)
    assert "guides/api.md" not in corpus.ids


def test_from_dir_honours_a_file_cap(corpus_dir: Path) -> None:
    assert len(Corpus.from_dir(corpus_dir, max_files=2)) == 2


def test_from_dir_detects_media_types(corpus_dir: Path) -> None:
    corpus = Corpus.from_dir(corpus_dir)
    assert corpus.require("contract.md").media_type is MediaType.MARKDOWN
    assert corpus.require("notes.txt").media_type is MediaType.TEXT


def test_ids_are_relative_so_they_stay_readable(corpus_dir: Path) -> None:
    """A leaderboard row saying `guides/api.md` beats one saying /Users/…/tmp/xyz/api.md."""
    assert "guides/api.md" in Corpus.from_dir(corpus_dir).ids


def test_a_directory_with_nothing_matching_says_what_to_do(tmp_path: Path) -> None:
    (tmp_path / "data.parquet").write_bytes(b"x")
    with pytest.raises(CorpusError, match="rename the files"):
        Corpus.from_dir(tmp_path)


def test_nothing_matching_names_the_extensions_that_are_actually_there(tmp_path: Path) -> None:
    (tmp_path / "data.parquet").write_bytes(b"x")
    (tmp_path / "notes.rst").write_text("x")
    message = _from_dir_error(tmp_path)
    assert ".parquet" in message
    assert ".rst" in message


def test_an_empty_directory_says_it_is_empty_rather_than_listing_extensions(
    tmp_path: Path,
) -> None:
    assert "no files at all" in _from_dir_error(tmp_path)


def test_nothing_matching_does_not_send_a_config_user_after_a_key_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """`patterns` is a `from_dir` argument, not a config key. The old message hid that.

    A config file is the primary interface, so "Pass `patterns`" left a CLI user with
    nowhere to go: adding `patterns:` to the file is rejected as an unknown key.
    """
    message = _from_dir_error(tmp_path)
    assert "no `patterns:` config key" in message
    assert "Corpus.from_dir" in message
    assert "Python-API only" in message


def test_a_missing_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="not a directory"):
        Corpus.from_dir(tmp_path / "nope")


def test_from_files_reads_an_explicit_list(corpus_dir: Path) -> None:
    corpus = Corpus.from_files([corpus_dir / "contract.md", corpus_dir / "notes.txt"])
    assert corpus.ids == ("contract.md", "notes.txt")


def test_from_files_rejects_a_directory(corpus_dir: Path) -> None:
    with pytest.raises(CorpusError, match="not a file"):
        Corpus.from_files([corpus_dir])


def test_from_texts_is_the_ten_second_path() -> None:
    corpus = Corpus.from_texts({"a": "First document.", "b": "Second document."})
    assert len(corpus) == 2
    assert corpus.require("a").text() == "First document."


def test_duplicate_ids_are_refused() -> None:
    from contextgrid.core.documents import SourceFile

    with pytest.raises(CorpusError, match="duplicate source id"):
        Corpus(files=(SourceFile(id="a", raw=b"x"), SourceFile(id="a", raw=b"y")))


# ---------------------------------------------------------------------------
# access
# ---------------------------------------------------------------------------


def test_require_lists_what_is_available() -> None:
    corpus = Corpus.from_texts({"a": "x", "b": "y"})
    with pytest.raises(CorpusError, match="Available: a, b"):
        corpus.require("c")


def test_get_returns_none_rather_than_raising() -> None:
    assert Corpus.from_texts({"a": "x"}).get("b") is None


def test_of_type_filters() -> None:
    corpus = Corpus.from_texts({"a": "x"}, media_type=MediaType.MARKDOWN)
    assert corpus.of_type(MediaType.MARKDOWN)
    assert not corpus.of_type(MediaType.PDF)


def test_corpus_is_sized_and_iterable() -> None:
    corpus = Corpus.from_texts({"a": "xx", "b": "yyy"})
    assert len(corpus) == 2
    assert [s.id for s in corpus] == ["a", "b"]
    assert corpus.total_bytes == 5


# ---------------------------------------------------------------------------
# content hash
# ---------------------------------------------------------------------------


def test_the_same_documents_hash_the_same_whatever_order_they_arrive_in() -> None:
    """Part of the run manifest: two runs over the same corpus must agree."""
    one = Corpus.from_texts({"a": "first", "b": "second"})
    two = Corpus.from_texts({"b": "second", "a": "first"})
    assert one.content_hash() == two.content_hash()


def test_changing_one_document_changes_the_corpus_hash() -> None:
    one = Corpus.from_texts({"a": "first", "b": "second"})
    two = Corpus.from_texts({"a": "first", "b": "second edited"})
    assert one.content_hash() != two.content_hash()


# ---------------------------------------------------------------------------
# fingerprint from bytes alone
# ---------------------------------------------------------------------------


def test_source_fingerprint_counts_files_and_types(corpus_dir: Path) -> None:
    print_ = fingerprint_sources(Corpus.from_dir(corpus_dir))
    assert print_.file_count == 3
    assert print_.media_types["text/markdown"] == 2
    assert print_.total_bytes > 0
    assert not print_.is_parsed


def test_byte_identical_files_are_grouped() -> None:
    corpus = Corpus.from_texts({"a": "same text", "b": "same text", "c": "different"})
    print_ = fingerprint_sources(corpus)
    assert print_.duplicate_groups == (("a", "b"),)
    assert print_.duplicate_file_count == 1
    assert print_.duplicate_rate == pytest.approx(1 / 3)


def test_duplicates_produce_a_hint_before_anything_is_parsed() -> None:
    corpus = Corpus.from_texts({"a": "same", "b": "same"})
    hints = fingerprint_sources(corpus).hints()
    assert any("byte-identical" in hint for hint in hints)


# ---------------------------------------------------------------------------
# fingerprint with a parse
# ---------------------------------------------------------------------------


def profile(texts: dict[str, str]):  # type: ignore[no-untyped-def]
    corpus = Corpus.from_texts(texts, media_type=MediaType.MARKDOWN)
    parses = {s.id: MD.parse(s) for s in corpus}
    return fingerprint(corpus, parses)


def test_a_parse_adds_content_statistics() -> None:
    print_ = profile({"contract.md": CONTRACT})
    assert print_.is_parsed
    assert print_.parser == "markdown"
    assert print_.total_characters == len(CONTRACT)
    assert print_.heading_count == 6
    assert print_.table_characters > 0


def test_a_table_heavy_corpus_says_the_parser_will_dominate() -> None:
    """The hint that turns a blank matrix builder into a guided decision."""
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(60))
    hints = profile({"data.md": f"# Numbers\n\n{table}\n"}).hints()
    assert any("tables" in hint and "Parser choice" in hint for hint in hints)


def test_a_code_heavy_corpus_suggests_a_code_aware_embedder() -> None:
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(200)) + "\n```"
    hints = profile({"snippets.md": f"# Code\n\n{code}\n"}).hints()
    assert any("code-aware embedder" in hint for hint in hints)


def test_a_corpus_with_no_headings_says_structural_chunking_has_nothing_to_do() -> None:
    hints = profile({"flat.md": "Just prose. " * 200}).hints()
    assert any("No headings" in hint for hint in hints)


def test_a_heading_heavy_corpus_recommends_structural_chunking() -> None:
    document = "\n\n".join(f"## Section {i}\n\nSome body text here." for i in range(12))
    hints = profile({"doc.md": document}).hints()
    assert any("Structural chunking usually wins" in hint for hint in hints)


def test_short_documents_warn_that_large_chunk_sizes_cannot_differentiate() -> None:
    hints = profile({f"note{i}.md": "# T\n\nShort." for i in range(3)}).hints()
    assert any("cannot differentiate" in hint for hint in hints)


def test_long_documents_suggest_parent_document_retrieval() -> None:
    hints = profile({"big.md": "# T\n\n" + ("word " * 20_000)}).hints()
    assert any("Parent-document retrieval" in hint for hint in hints)


def test_empty_documents_are_reported_as_a_parser_or_ocr_problem() -> None:
    hints = profile({"good.md": "# T\n\nText.", "scanned.md": "   "}).hints()
    assert any("came back empty" in hint and "OCR" in hint for hint in hints)


def test_block_kinds_are_counted() -> None:
    kinds = profile({"contract.md": CONTRACT}).block_kinds
    assert kinds["heading"] == 6
    assert kinds["table"] == 1


def test_length_statistics() -> None:
    print_ = profile({"a.md": "x" * 100, "b.md": "y" * 300})
    assert print_.mean_length == 200
    assert print_.median_length == 200
    assert sorted(print_.document_lengths) == [100, 300]


def test_summary_is_one_line() -> None:
    summary = profile({"contract.md": CONTRACT}).summary()
    assert "\n" not in summary
    assert "markdown" in summary


def test_fingerprint_without_parses_falls_back_to_bytes_only() -> None:
    corpus = Corpus.from_texts({"a": "text"})
    assert not fingerprint(corpus).is_parsed
    assert not fingerprint(corpus, {}).is_parsed
    assert not fingerprint(corpus, []).is_parsed


# ---------------------------------------------------------------------------
# a parser that read nothing
# ---------------------------------------------------------------------------


def test_a_parser_that_read_nothing_is_named_instead_of_the_embedder() -> None:
    """The parser is the fault. `TfidfEmbedder` was only the first thing to trip over it.

    A parser that does not match the file types declines every file, the corpus arrives at
    the embedder empty, and the user was told to call `prepare()` -- about a step they never
    took and a component they never chose.
    """
    corpus = Corpus.from_texts({"r.md": "Refunds within 30 days."}, media_type=MediaType.MARKDOWN)
    with pytest.raises(CorpusError) as error:
        require_parsed_text(corpus, {}, parser="pymupdf")

    message = str(error.value)
    assert "pymupdf" in message  # the thing that failed
    assert "r.md" in message  # a file it could not read
    assert "text/markdown" in message  # why the two do not match
    assert "prepare()" not in message


def test_documents_that_parse_to_nothing_count_as_read_nothing() -> None:
    """Declining every file and returning blanks for every file are the same failure."""
    corpus = Corpus.from_texts({"a.md": "text", "b.md": "more"}, media_type=MediaType.MARKDOWN)
    unreadable = Corpus.from_texts({"a.md": "  ", "b.md": ""})
    blank = {source.id: MD.parse(source) for source in unreadable}
    with pytest.raises(CorpusError, match="read no text"):
        require_parsed_text(corpus, blank, parser="pymupdf")


def test_one_readable_document_is_enough_to_carry_on() -> None:
    """A corpus of scans with one text page is a warning, not a dead end."""
    corpus = Corpus.from_texts({"a.md": "text", "b.md": "  "}, media_type=MediaType.MARKDOWN)
    require_parsed_text(corpus, {s.id: MD.parse(s) for s in corpus}, parser="markdown")


def test_a_partly_skipped_corpus_is_not_an_error() -> None:
    """Ten PDFs and one Markdown file read by a PDF parser is a working sweep, not a stop.

    Only the total wipeout is fatal. Erroring on a partial skip would break every corpus
    that happens to carry a stray file the chosen parser does not handle.
    """
    texts = {f"report{i}.pdf": f"Page {i} of the report." for i in range(10)}
    corpus = Corpus.from_texts({**texts, "notes.md": "# Notes"})
    parsed = {source.id: MD.parse(source) for source in corpus if source.id.endswith(".pdf")}
    require_parsed_text(corpus, parsed, parser="pymupdf")


def test_an_empty_corpus_is_somebody_elses_error() -> None:
    require_parsed_text(Corpus(files=()), {}, parser="markdown")


def test_the_check_accepts_a_sequence_of_parses_like_fingerprint_does() -> None:
    corpus = Corpus.from_texts({"a.md": "text"}, media_type=MediaType.MARKDOWN)
    require_parsed_text(corpus, [MD.parse(s) for s in corpus], parser="markdown")


def test_more_than_three_unreadable_files_are_summarised() -> None:
    corpus = Corpus.from_texts({f"f{i}.md": "text" for i in range(5)})
    with pytest.raises(CorpusError, match="and 2 more"):
        require_parsed_text(corpus, {}, parser="pymupdf")


def test_building_a_pipeline_on_an_unreadable_corpus_blames_the_parser() -> None:
    """The repro: `contextgrid profile ./documents --parser pymupdf` on a Markdown corpus.

    It used to reach `TfidfEmbedder`, which had been prepared on nothing and said so. The
    parse is where it went wrong, and now that is where it stops.
    """
    corpus = Corpus.from_texts({"r.md": "Refunds within 30 days."}, media_type=MediaType.MARKDOWN)
    with pytest.raises(CorpusError, match="'pymupdf' parser read no text"):
        build(Config(parser="pymupdf"), corpus)


def test_building_a_pipeline_on_a_partly_readable_corpus_still_runs_and_still_warns() -> None:
    """The regression most likely to bite: a stray file the parser skips must not stop a run."""
    corpus = Corpus(
        files=(
            SourceFile(id="report.pdf", media_type=MediaType.PDF, raw=prose_pdf()),
            SourceFile(id="notes.md", media_type=MediaType.MARKDOWN, raw=b"# Notes\n\nSome text."),
        )
    )
    built = build(Config(parser="pymupdf"), corpus)

    assert built.chunks
    skipped = [w for w in built.warnings if w.code is WarningCode.PARSER_FALLBACK]
    assert [w.subject for w in skipped] == ["notes.md"]


def test_an_unparsed_fingerprint_gives_no_content_hints() -> None:
    """It has not read anything yet, so it should not pretend to have an opinion."""
    corpus = Corpus.from_texts({"a": "text", "b": "other"})
    assert fingerprint_sources(corpus).hints() == []

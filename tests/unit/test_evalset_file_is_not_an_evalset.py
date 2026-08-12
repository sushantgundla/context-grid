"""Pointing `evalset:` at something that is not an eval set.

A directory, a spreadsheet, a PDF, a file in the wrong encoding. Every one of them used to
arrive as a raw OS or Python string with an `error:` prefix in front of it --
`[Errno 21] Is a directory: /path/to/whatever`, or `'utf-8' codec can't decode byte 0xca in
position 0`. Both true, and neither says what an eval set is or what should have been named.

The guard lives in `read_jsonl` and `read_csv` rather than in `read_evalset`, because there
are two places that choose a reader: `read_evalset`, and `config/loader.py`. Putting it in
the readers is the only way `check`, `run` and `evalset` give one answer about one file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.core.errors import EvalSetError
from contextgrid.evalset.io import read_csv, read_evalset

QUESTIONS = [{"id": "q1", "question": "What is the notice period?"}]
COLUMNS = "id,question\nq1,What is the notice period?\n"

#: The first bytes of a real PNG. Any non-UTF-8 bytes would do; these are recognisably not text.
NOT_TEXT = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xca\xfe"


# ---------------------------------------------------------------------------
# a directory
# ---------------------------------------------------------------------------


def test_a_directory_says_it_is_a_directory(tmp_path: Path) -> None:
    folder = tmp_path / "questions"
    folder.mkdir()

    with pytest.raises(EvalSetError) as caught:
        read_evalset(folder)

    message = str(caught.value)
    assert "is a directory" in message
    assert ".jsonl or .csv" in message
    assert "Errno" not in message


def test_a_directory_named_like_a_csv_says_the_same(tmp_path: Path) -> None:
    """`read_evalset` sends anything ending `.csv` to a different reader. Both readers open
    their own file, so both need the guard."""
    folder = tmp_path / "questions.csv"
    folder.mkdir()

    for read in (read_evalset, read_csv):
        with pytest.raises(EvalSetError, match="is a directory"):
            read(folder)


def test_the_directory_message_names_the_path(tmp_path: Path) -> None:
    folder = tmp_path / "questions"
    folder.mkdir()

    with pytest.raises(EvalSetError) as caught:
        read_evalset(folder)

    assert str(folder) in str(caught.value)


# ---------------------------------------------------------------------------
# a file that is not text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["questions.jsonl", "questions.csv", "questions"])
def test_a_binary_file_says_it_is_not_text(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(NOT_TEXT)

    with pytest.raises(EvalSetError) as caught:
        read_evalset(path)

    message = str(caught.value)
    assert "is not text" in message
    assert "An eval set is a JSONL or CSV file of questions" in message
    assert "codec can't decode" not in message


def test_a_file_in_the_wrong_encoding_says_where_it_broke(tmp_path: Path) -> None:
    """Latin-1 questions from an old export. Real, and worth a byte offset to find."""
    path = tmp_path / "questions.jsonl"
    path.write_bytes("caf\xe9 au lait\n".encode("latin-1"))

    with pytest.raises(EvalSetError, match="is not text") as caught:
        read_evalset(path)

    assert "at byte 3" in str(caught.value)


@pytest.mark.parametrize(
    ("name", "hint"),
    [
        ("questions.xlsx", "Excel workbook"),
        ("questions.xls", "Excel workbook"),
        ("questions.ods", "spreadsheet"),
        ("questions.pdf", "a PDF"),
        ("questions.docx", "Word document"),
        ("questions.parquet", "Parquet"),
        ("questions.zip", "zip archive"),
        ("questions.db", "database file"),
    ],
)
def test_a_binary_format_we_recognise_is_named(tmp_path: Path, name: str, hint: str) -> None:
    """The spreadsheet the questions were written in is the easy mistake to make, and
    "invalid continuation byte" is a terrible way to be told about it."""
    path = tmp_path / name
    path.write_bytes(NOT_TEXT)

    with pytest.raises(EvalSetError) as caught:
        read_evalset(path)

    assert hint in str(caught.value)


def test_an_unrecognised_binary_file_still_says_what_was_wanted(tmp_path: Path) -> None:
    """No hint to give, so it gives the one thing it does know."""
    path = tmp_path / "questions.bin"
    path.write_bytes(NOT_TEXT)

    with pytest.raises(EvalSetError) as caught:
        read_evalset(path)

    assert "This looks like" not in str(caught.value)
    assert "An eval set is a JSONL or CSV file of questions" in str(caught.value)


# ---------------------------------------------------------------------------
# a text file that is not either format
# ---------------------------------------------------------------------------


def test_a_config_pointed_at_itself_says_the_extension_is_wrong(tmp_path: Path) -> None:
    """A YAML file parses as text and fails on line 1. The JSON error alone leaves somebody
    hunting for a typo in a file that was never the right kind of file."""
    path = tmp_path / "config.yaml"
    path.write_text("evalset: ./questions.jsonl\n", encoding="utf-8")

    with pytest.raises(EvalSetError) as caught:
        read_evalset(path)

    message = str(caught.value)
    assert "`.yaml` file is not a format an eval set is read from" in message
    assert ".jsonl" in message
    assert ".csv" in message


@pytest.mark.parametrize("name", ["questions.jsonl", "questions.json", "questions"])
def test_a_normal_extension_gets_no_lecture_about_extensions(tmp_path: Path, name: str) -> None:
    """The file is the right kind and the line is broken. Do not muddy that with format
    advice -- `.json` and no extension at all are both perfectly ordinary here."""
    path = tmp_path / name
    path.write_text("{oops\n", encoding="utf-8")

    with pytest.raises(EvalSetError, match="is not valid JSON") as caught:
        read_evalset(path)

    assert "not a format an eval set is read from" not in str(caught.value)


def test_a_line_that_is_not_an_object_says_so(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("5\n", encoding="utf-8")

    with pytest.raises(EvalSetError, match="one question per line") as caught:
        read_evalset(path)

    assert "questions.jsonl:1 is a number" in str(caught.value)


def test_a_broken_header_line_says_so(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"_evalset": 7}\n', encoding="utf-8")

    with pytest.raises(EvalSetError, match="`_evalset` is a number"):
        read_evalset(path)


# ---------------------------------------------------------------------------
# and the files that are eval sets still read
# ---------------------------------------------------------------------------


def test_a_real_jsonl_file_still_reads(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in QUESTIONS) + "\n", encoding="utf-8")

    assert len(read_evalset(path)) == 1
    assert read_evalset(path).items[0].question == "What is the notice period?"


def test_a_real_csv_file_still_reads(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    path.write_text(COLUMNS, encoding="utf-8")

    assert len(read_evalset(path)) == 1


def test_a_csv_with_a_byte_order_mark_still_reads(tmp_path: Path) -> None:
    """Excel writes one. Reading the whole file at once must not lose the `utf-8-sig`
    handling that strips it."""
    path = tmp_path / "questions.csv"
    path.write_bytes(b"\xef\xbb\xbf" + COLUMNS.encode("utf-8"))

    loaded = read_evalset(path)

    assert len(loaded) == 1
    assert loaded.items[0].id == "q1"


def test_a_csv_with_a_newline_inside_a_field_still_reads(tmp_path: Path) -> None:
    """The quoted newline is why `csv` wants `newline=""`. Reading into a string first must
    keep that, or a two-line question becomes two questions."""
    path = tmp_path / "questions.csv"
    path.write_text('id,question\nq1,"Two lines\nin one field?"\n', encoding="utf-8")

    loaded = read_evalset(path)

    assert len(loaded) == 1
    assert loaded.items[0].question == "Two lines\nin one field?"


def test_a_missing_file_still_says_it_is_missing(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="no eval set at"):
        read_evalset(tmp_path / "absent.jsonl")

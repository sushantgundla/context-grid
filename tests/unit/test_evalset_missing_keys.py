"""A hand-written eval set with the wrong key in it.

`error: 'source_id'` was the whole message. Four characters in quotes, from a bare `KeyError`
that escaped `GoldAnchor.from_dict` and got printed by the CLI's top-level handler. It named
no file, no line, no item, and did not say what was expected -- and it arrived at the one
moment a new user is most likely to be hand-writing JSONL for the first time.

The mistake it punishes is a reasonable one. `read_csv` accepts `doc_id`, `file`, `document`
and three more as column names for the same field, because a subject-matter expert handing
over a spreadsheet should not have to reformat it. JSONL takes none of them. So the reader
who followed the CSV page and then wrote a JSONL file by hand gets `'source_id'` and no clue
that the two formats differ.

These tests pin the three things the message has to carry: which key is missing, where the
line is, and -- when the key that is present is one of the CSV aliases -- that the alias is
real but belongs to the other format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.core.errors import EvalSetError
from contextgrid.core.evalset import EvalItem, GoldAnchor
from contextgrid.evalset.io import read_jsonl


def write(path: Path, *records: dict[str, object]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


# ---------------------------------------------------------------------------
# the anchor
# ---------------------------------------------------------------------------


def test_an_anchor_without_source_id_says_so() -> None:
    with pytest.raises(EvalSetError) as caught:
        GoldAnchor.from_dict({"quote": "Refunds are issued within 30 days."})

    message = str(caught.value)
    assert "source_id" in message
    assert message != "'source_id'"  # the old bare KeyError, and nothing else


def test_an_anchor_using_the_csv_alias_is_told_it_is_a_csv_alias() -> None:
    """`doc_id` is a real column name -- in the other format. Say that, don't just refuse."""
    with pytest.raises(EvalSetError) as caught:
        GoldAnchor.from_dict({"doc_id": "refunds.md", "quote": "Refunds are issued."})

    message = str(caught.value)
    assert "doc_id" in message
    assert "source_id" in message
    assert "CSV" in message


def test_an_anchor_without_a_quote_says_so() -> None:
    with pytest.raises(EvalSetError) as caught:
        GoldAnchor.from_dict({"source_id": "refunds.md"})

    assert "quote" in str(caught.value)


def test_the_message_lists_the_keys_the_anchor_did_have() -> None:
    """So the reader can see their own typo rather than guess at it."""
    with pytest.raises(EvalSetError) as caught:
        GoldAnchor.from_dict({"sourceid": "refunds.md", "quote": "Refunds are issued."})

    assert "sourceid" in str(caught.value)


# ---------------------------------------------------------------------------
# the item
# ---------------------------------------------------------------------------


def test_an_item_without_a_question_says_so() -> None:
    with pytest.raises(EvalSetError) as caught:
        EvalItem.from_dict({"id": "q1"})

    assert "question" in str(caught.value)


def test_an_item_using_a_csv_alias_for_the_question_is_told() -> None:
    with pytest.raises(EvalSetError) as caught:
        EvalItem.from_dict({"id": "q1", "q": "How long do refunds take?"})

    message = str(caught.value)
    assert "question" in message
    assert "CSV" in message


# ---------------------------------------------------------------------------
# through the reader, which is where a user actually meets it
# ---------------------------------------------------------------------------


def test_the_file_and_line_are_named(tmp_path: Path) -> None:
    """A one-line message about a 200-line file is not much of a message."""
    path = write(
        tmp_path / "questions.jsonl",
        {"id": "q1", "question": "Fine.", "anchors": [{"source_id": "a.md", "quote": "Yes."}]},
        {"id": "q2", "question": "Broken.", "anchors": [{"doc_id": "b.md", "quote": "No."}]},
    )

    with pytest.raises(EvalSetError) as caught:
        read_jsonl(path)

    message = str(caught.value)
    assert "questions.jsonl:2" in message
    assert "source_id" in message


def test_the_item_id_is_named_when_the_line_has_one(tmp_path: Path) -> None:
    path = write(
        tmp_path / "questions.jsonl",
        {"id": "nw13", "question": "Broken.", "anchors": [{"doc_id": "b.md", "quote": "No."}]},
    )

    with pytest.raises(EvalSetError) as caught:
        read_jsonl(path)

    assert "nw13" in str(caught.value)


def test_a_good_file_still_reads(tmp_path: Path) -> None:
    """The guard must not cost the happy path anything."""
    path = write(
        tmp_path / "questions.jsonl",
        {
            "id": "q1",
            "question": "How long?",
            "anchors": [{"source_id": "a.md", "quote": "30 days."}],
        },
    )

    evalset = read_jsonl(path)

    assert len(evalset.items) == 1
    assert evalset.items[0].anchors[0].source_id == "a.md"

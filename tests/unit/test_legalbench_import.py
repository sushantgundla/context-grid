"""Reading a LegalBench-RAG benchmark file, including the ones somebody typed by hand.

Two rules are being pinned here, and they pull in opposite directions on purpose.

An *incomplete* annotation is dropped quietly -- a snippet with no `file_path`, or a `span`
with fewer than two entries. That is documented, it is what a partly annotated benchmark
really looks like, and erroring on it would make a large public dataset unusable over a
handful of rows.

A *misshapen* field raises, naming the file, the test, the snippet and what was expected.
`"span": "117,202"` used to import as characters 1 to 1 -- a real span, in the right file,
pointing at the wrong two characters, scored against for the rest of the run. Silence there
buys nothing and costs the whole result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contextgrid.core.errors import EvalSetError
from contextgrid.core.span import Span
from contextgrid.evalset.io import read_legalbench_rag

SNIPPET = {"file_path": "contract.txt", "span": [10, 42]}


def write(tmp_path: Path, payload: Any, *, name: str = "lb.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def one_test(**fields: Any) -> dict[str, Any]:
    """A single well-formed test, with whatever the caller wants broken about it."""
    return {"query": "What is the notice period?", "snippets": [dict(SNIPPET)], **fields}


# ---------------------------------------------------------------------------
# the two accepted top-level shapes
# ---------------------------------------------------------------------------


def test_a_bare_array_of_tests_is_accepted(tmp_path: Path) -> None:
    """Documented in both cli.md and evalsets.md, and it used to crash with
    `'list' object has no attribute 'get'`. Several published dumps of the benchmark are
    shaped this way."""
    loaded = read_legalbench_rag(write(tmp_path, [one_test(id="q1")]))

    assert len(loaded) == 1
    assert loaded.items[0].id == "q1"
    assert loaded.items[0].gold[0].span == Span("contract.txt", 10, 42)


def test_an_object_with_a_tests_array_is_accepted(tmp_path: Path) -> None:
    loaded = read_legalbench_rag(write(tmp_path, {"tests": [one_test(id="q1")]}))

    assert len(loaded) == 1
    assert loaded.items[0].gold[0].span == Span("contract.txt", 10, 42)


def test_both_shapes_load_identically(tmp_path: Path) -> None:
    """The bare array is not a lesser form -- it produces the same eval set."""
    tests = [one_test(id="q1"), one_test(id="q2", query="How long is the cure period?")]

    bare = read_legalbench_rag(write(tmp_path, tests, name="bare.json"))
    wrapped = read_legalbench_rag(write(tmp_path, {"tests": tests}, name="wrapped.json"))

    assert [item.to_dict() for item in bare] == [item.to_dict() for item in wrapped]


def test_an_empty_benchmark_loads_as_an_empty_set(tmp_path: Path) -> None:
    assert len(read_legalbench_rag(write(tmp_path, []))) == 0
    assert len(read_legalbench_rag(write(tmp_path, {"tests": []}))) == 0


# ---------------------------------------------------------------------------
# incomplete annotations are dropped, as documented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("what", "snippet"),
    [
        ("no file_path", {"span": [10, 42]}),
        ("no span", {"file_path": "contract.txt"}),
        ("neither", {}),
        ("null file_path", {"file_path": None, "span": [10, 42]}),
        ("null span", {"file_path": "contract.txt", "span": None}),
        ("blank file_path", {"file_path": "   ", "span": [10, 42]}),
        ("one-entry span", {"file_path": "contract.txt", "span": [10]}),
        ("empty span", {"file_path": "contract.txt", "span": []}),
    ],
)
def test_an_incomplete_snippet_is_dropped_not_rejected(
    tmp_path: Path, what: str, snippet: dict[str, Any]
) -> None:
    """A partly annotated benchmark still loads. The test survives without that gold, which
    is what the docs promise and what the real dataset needs."""
    loaded = read_legalbench_rag(write(tmp_path, [one_test(snippets=[snippet])]))

    assert len(loaded) == 1, what
    assert loaded.items[0].gold == (), what


def test_a_test_with_no_query_is_skipped(tmp_path: Path) -> None:
    payload = [one_test(id="a"), one_test(id="b", query="  "), one_test(id="c")]

    assert [item.id for item in read_legalbench_rag(write(tmp_path, payload))] == ["a", "c"]


def test_a_test_with_no_snippets_still_loads(tmp_path: Path) -> None:
    """It can never be scored, but it is a question somebody wrote, and reporting it as
    gold-less is more use than dropping it."""
    for payload in ([{"query": "q"}], [{"query": "q", "snippets": None}]):
        loaded = read_legalbench_rag(write(tmp_path, payload))
        assert len(loaded) == 1
        assert loaded.items[0].gold == ()


def test_ids_are_the_position_in_the_file(tmp_path: Path) -> None:
    """Deliberately not consecutive. A skipped test leaves its number unused, so `lb3`
    always means "the fourth test in the file" -- both to somebody reading an error and to
    the same benchmark loaded again after that empty query is filled in."""
    payload = [{"query": "a"}, {"query": ""}, {"query": "c"}, {"query": "d"}]

    assert [item.id for item in read_legalbench_rag(write(tmp_path, payload))] == [
        "lb0",
        "lb2",
        "lb3",
    ]


def test_a_given_id_wins_over_the_default(tmp_path: Path) -> None:
    payload = [one_test(id="contract-nda-01"), one_test(), one_test(id=7)]

    assert [item.id for item in read_legalbench_rag(write(tmp_path, payload))] == [
        "contract-nda-01",
        "lb1",
        "7",
    ]


def test_the_answer_is_carried_through(tmp_path: Path) -> None:
    loaded = read_legalbench_rag(write(tmp_path, [one_test(answer="Thirty days.")]))

    assert loaded.items[0].answer == "Thirty days."
    assert read_legalbench_rag(write(tmp_path, [one_test()])).items[0].answer is None


def test_offsets_written_as_strings_are_read_as_numbers(tmp_path: Path) -> None:
    """CSV round-trips and hand-edited files quote their numbers. The offsets are
    unambiguous, so read them rather than refuse them."""
    snippet = {"file_path": "contract.txt", "span": ["10", "42"]}

    loaded = read_legalbench_rag(write(tmp_path, [one_test(snippets=[snippet])]))

    assert loaded.items[0].gold[0].span == Span("contract.txt", 10, 42)


def test_the_imported_set_says_its_gold_is_span_level(tmp_path: Path) -> None:
    loaded = read_legalbench_rag(write(tmp_path, [one_test()]))

    assert loaded.meta["granularity"] == "span"
    assert loaded.source == "legalbench-rag"


# ---------------------------------------------------------------------------
# misshapen fields raise a message naming the file
# ---------------------------------------------------------------------------


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="no LegalBench-RAG file at"):
        read_legalbench_rag(tmp_path / "absent.json")


def test_a_file_that_is_not_json_says_so(tmp_path: Path) -> None:
    """Not a raw `JSONDecodeError`. The CLI prints whatever comes out under `error:`."""
    path = tmp_path / "lb.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(EvalSetError, match="is not valid JSON"):
        read_legalbench_rag(path)


@pytest.mark.parametrize(
    ("what", "payload", "expected"),
    [
        ("a string", "hello", "is a string"),
        ("a number", 7, "is a number"),
        ("null", None, "is null"),
        ("true", True, "is a true/false value"),
        ("an object with no tests key", {"questions": []}, "has no `tests` key"),
        ("an empty object", {}, "Keys found: none"),
        ("tests as an object", {"tests": {"a": 1}}, "`tests` is an object"),
        ("tests as a string", {"tests": "q1"}, "`tests` is a string"),
        ("tests as a number", {"tests": 3}, "`tests` is a number"),
        ("a test that is a string", {"tests": ["q1"]}, "test 0 is a string"),
        ("a test that is a number", {"tests": [5]}, "test 0 is a number"),
        ("a test that is null", {"tests": [None]}, "test 0 is null"),
        ("a test that is an array", {"tests": [[1, 2]]}, "test 0 is an array"),
        ("query as an object", [{"query": {"text": "q"}}], "`query` is an object"),
        ("query as a number", [{"query": 5}], "`query` is a number"),
        ("id as an object", [{"id": {"a": 1}, "query": "q"}], "`id` is an object"),
        ("answer as an object", [{"query": "q", "answer": {"a": 1}}], "`answer` is an object"),
        ("snippets as a string", [{"query": "q", "snippets": "a.md"}], "`snippets` is a string"),
        ("snippets as an object", [{"query": "q", "snippets": {}}], "`snippets` is an object"),
        ("snippets as a number", [{"query": "q", "snippets": 2}], "`snippets` is a number"),
        ("a snippet that is a string", [{"query": "q", "snippets": ["a"]}], "snippet 0 is a str"),
        ("a snippet that is an array", [{"query": "q", "snippets": [[1]]}], "snippet 0 is an arr"),
        ("a snippet that is null", [{"query": "q", "snippets": [None]}], "snippet 0 is null"),
    ],
)
def test_a_misshapen_file_names_the_file_and_the_problem(
    tmp_path: Path, what: str, payload: Any, expected: str
) -> None:
    """Every one of these used to be an `AttributeError` wearing an `error:` prefix, or a
    silent empty import. The message has to name the file, because the person reading it is
    looking at a directory of benchmark files."""
    path = write(tmp_path, payload)

    with pytest.raises(EvalSetError, match=expected) as caught:
        read_legalbench_rag(path)

    assert str(path) in str(caught.value), what


@pytest.mark.parametrize(
    ("what", "span", "expected"),
    [
        ("a string", "117,202", "`span` is a string"),
        ("a number", 9, "`span` is a number"),
        ("an object", {"start": 1, "end": 2}, "`span` is an object"),
        ("words", ["start", "end"], "whole numbers"),
        ("nulls", [None, None], "whole numbers"),
        ("nested arrays", [[1], [2]], "whole numbers"),
        ("true and false", [True, False], "whole numbers"),
        ("backwards", [90, 10], "must be >= start"),
        ("negative", [-5, 10], "must be >= 0"),
    ],
)
def test_a_misshapen_span_names_the_snippet(
    tmp_path: Path, what: str, span: Any, expected: str
) -> None:
    """`"span": "117,202"` is the one that matters: it used to import as characters 1 to 1,
    a real span pointing at the wrong text, and score against it silently."""
    payload = [one_test(snippets=[{"file_path": "contract.txt", "span": span}])]

    with pytest.raises(EvalSetError, match=expected) as caught:
        read_legalbench_rag(write(tmp_path, payload))

    assert "snippet 0" in str(caught.value), what


@pytest.mark.parametrize("file_path", [3, ["contract.txt"], {"name": "contract.txt"}])
def test_a_file_path_that_is_not_a_string_is_rejected(tmp_path: Path, file_path: Any) -> None:
    """`str()` on a list produces `"['contract.txt']"`, which is a document id nothing will
    ever match, and the run reports it as a missing document instead of a broken file."""
    payload = [one_test(snippets=[{"file_path": file_path, "span": [10, 42]}])]

    with pytest.raises(EvalSetError, match="`file_path` is"):
        read_legalbench_rag(write(tmp_path, payload))


def test_the_message_names_the_test_that_is_wrong(tmp_path: Path) -> None:
    """A benchmark has thousands of tests. "Something is wrong somewhere" is not a message
    anybody can act on."""
    payload = [one_test(), one_test(), one_test(snippets=[{"file_path": "c.txt", "span": "bad"}])]

    with pytest.raises(EvalSetError, match="test 2, snippet 0"):
        read_legalbench_rag(write(tmp_path, payload))

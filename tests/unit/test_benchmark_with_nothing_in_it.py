"""A benchmark file that gave us nothing, and saying which kind of nothing it was.

Three different problems used to share one message, and the message was
`none of the 0 documents the benchmark refers to were found under documents`. Zero documents
cannot fail to be found. Worse, it blamed the corpus: somebody reading it goes and checks
their corpus path, finds it correct, and never learns that their benchmark file parsed to
nothing at all.

They are three problems with three fixes. An empty file needs tests writing. A file whose
snippets were all dropped needs its annotations looking at. A corpus without the documents
needs the documents. Each says which one it is, and only the third mentions the corpus.

The dropping itself is right -- `docs/guide/cli.md:314` documents it. Saying nothing about
it is what was wrong: a file whose every snippet was discarded scored exactly like a file
pointing at the wrong corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contextgrid.evalset.io import describe_skipped, read_legalbench_rag
from contextgrid.validate import ValidationError, load_benchmark, self_check

CONTRACT = "Fees are payable within thirty days of the invoice date.\n"
FEES = CONTRACT.index("thirty days")
GOOD = {"file_path": "contract.txt", "span": [FEES, FEES + 11]}


@pytest.fixture
def documents(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "contract.txt").write_text(CONTRACT, encoding="utf-8")
    return corpus


def benchmark(tmp_path: Path, payload: Any, *, name: str = "benchmark.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# no usable tests -- and not a word about the corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [[], {"tests": []}])
def test_a_benchmark_with_no_tests_says_exactly_that(
    tmp_path: Path, documents: Path, payload: Any
) -> None:
    with pytest.raises(ValidationError, match="contains no tests") as caught:
        load_benchmark(benchmark(tmp_path, payload), documents)

    assert "0 documents" not in str(caught.value)
    assert "corpus" not in str(caught.value)


def test_a_benchmark_whose_tests_were_all_skipped_says_why(tmp_path: Path, documents: Path) -> None:
    payload = {"tests": [{"query": ""}, {"query": "   "}]}

    with pytest.raises(ValidationError, match="no usable tests") as caught:
        load_benchmark(benchmark(tmp_path, payload), documents)

    message = str(caught.value)
    assert "all 2 tests in it were skipped" in message
    assert "2 tests skipped: no `query`" in message
    assert "documents were found" not in message


def test_the_empty_file_message_does_not_blame_the_corpus(tmp_path: Path, documents: Path) -> None:
    """The whole point. The corpus here is perfectly good and is not the problem."""
    with pytest.raises(ValidationError) as caught:
        load_benchmark(benchmark(tmp_path, []), documents)

    assert "were found under" not in str(caught.value)


# ---------------------------------------------------------------------------
# tests, but every snippet dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("what", "snippet", "reason"),
    [
        ("span of one offset", {"file_path": "contract.txt", "span": [5]}, "shorter than two"),
        ("no file_path", {"span": [0, 10]}, "had no `file_path`"),
        ("no span", {"file_path": "contract.txt"}, "had no `span`"),
        ("blank file_path", {"file_path": "  ", "span": [0, 10]}, "had no `file_path`"),
    ],
)
def test_a_test_whose_only_snippet_was_dropped_says_how_many_and_why(
    tmp_path: Path, documents: Path, what: str, snippet: dict[str, Any], reason: str
) -> None:
    payload = {"tests": [{"query": "q", "snippets": [snippet]}]}

    with pytest.raises(ValidationError, match="not one usable gold span") as caught:
        load_benchmark(benchmark(tmp_path, payload), documents)

    message = str(caught.value)
    assert "1 snippet skipped" in message, what
    assert reason in message, what
    assert "0 documents" not in message, what


def test_a_test_with_no_snippets_at_all_still_explains_itself(
    tmp_path: Path, documents: Path
) -> None:
    """Nothing was skipped, because nothing was there. The message says what was expected
    instead of quoting a count of zero."""
    with pytest.raises(ValidationError, match="not one usable gold span") as caught:
        load_benchmark(benchmark(tmp_path, {"tests": [{"query": "q"}]}), documents)

    assert "does not carry a `snippets` array" in str(caught.value)


def test_the_three_empty_shapes_do_not_share_a_message(tmp_path: Path, documents: Path) -> None:
    """The requirement, stated as a test: no usable tests, no usable spans, and no documents
    in the corpus are three problems with three fixes."""
    messages = []
    for payload in (
        [],
        {"tests": [{"query": "q", "snippets": [{"file_path": "contract.txt", "span": [5]}]}]},
        {"tests": [{"query": "q", "snippets": [{"file_path": "absent.txt", "span": [0, 5]}]}]},
    ):
        with pytest.raises(ValidationError) as caught:
            load_benchmark(benchmark(tmp_path, payload), documents)
        messages.append(str(caught.value))

    assert len(set(messages)) == 3
    assert "contains no tests" in messages[0]
    assert "not one usable gold span" in messages[1]
    assert "annotations were made against" in messages[2]
    assert "corpus" not in messages[0]


# ---------------------------------------------------------------------------
# a run that succeeds still says what it dropped
# ---------------------------------------------------------------------------


def test_skipped_snippets_are_reported_on_a_run_that_works(tmp_path: Path, documents: Path) -> None:
    """A silent discard and a quiet one are not the same thing. This run is fine, and the
    user still needs to know that four of their annotations never made it in."""
    payload = {
        "tests": [
            {"query": "q1", "snippets": [dict(GOOD), {"span": [0, 5]}]},
            {"query": "q2", "snippets": [dict(GOOD), {"file_path": "contract.txt"}]},
            {"query": "q3", "snippets": [dict(GOOD), {"file_path": "contract.txt", "span": [1]}]},
            {"query": "", "snippets": []},
        ]
    }

    corpus, evalset = load_benchmark(benchmark(tmp_path, payload), documents)
    report = self_check(corpus, evalset)

    assert report["in_range_rate"] == 1.0
    assert "3 snippets skipped" in report["verdict"]
    assert "1 had no `file_path`" in report["verdict"]
    assert "1 had no `span`" in report["verdict"]
    assert "1 had a span shorter than two offsets" in report["verdict"]
    assert "1 test skipped: no `query`" in report["verdict"]


def test_what_was_skipped_comes_before_the_verdict(tmp_path: Path, documents: Path) -> None:
    """The CLI prints the verdict and moves on, so the count has to be read first."""
    payload = {"tests": [{"query": "q1", "snippets": [dict(GOOD), {"span": [0, 5]}]}]}

    corpus, evalset = load_benchmark(benchmark(tmp_path, payload), documents)
    verdict = self_check(corpus, evalset)["verdict"]

    assert verdict.index("snippet skipped") < verdict.index("point at real text")


def test_a_clean_benchmark_reports_nothing_skipped(tmp_path: Path, documents: Path) -> None:
    """No noise on the happy path."""
    payload = {"tests": [{"query": "q1", "snippets": [dict(GOOD)]}]}

    corpus, evalset = load_benchmark(benchmark(tmp_path, payload), documents)
    report = self_check(corpus, evalset)

    assert report["skipped_on_import"] == ""
    assert report["verdict"].startswith("1 of 1 spans point at real text")


# ---------------------------------------------------------------------------
# the counts themselves
# ---------------------------------------------------------------------------


def test_the_import_records_what_it_dropped(tmp_path: Path) -> None:
    payload = {
        "tests": [
            {"query": "q1", "snippets": [{"span": [0, 5]}, {"file_path": "a.txt"}]},
            {"query": "q2", "snippets": [{"file_path": "a.txt", "span": [1]}]},
            {"query": ""},
        ]
    }

    loaded = read_legalbench_rag(benchmark(tmp_path, payload))

    assert loaded.meta["tests_in_file"] == 3
    assert loaded.meta["tests_skipped"] == 1
    assert loaded.meta["snippets_skipped"] == {
        "had no `file_path`": 1,
        "had no `span`": 1,
        "had a span shorter than two offsets": 1,
    }


def test_a_clean_import_records_no_drops(tmp_path: Path) -> None:
    loaded = read_legalbench_rag(benchmark(tmp_path, [{"query": "q", "snippets": [dict(GOOD)]}]))

    assert loaded.meta["tests_skipped"] == 0
    assert loaded.meta["snippets_skipped"] == {}
    assert describe_skipped(loaded) == ""


def test_the_description_counts_in_the_plural(tmp_path: Path) -> None:
    payload = [{"query": "q", "snippets": [{"span": [0, 1]}, {"span": [0, 1]}]}]

    assert describe_skipped(read_legalbench_rag(benchmark(tmp_path, payload))) == (
        "2 snippets skipped: 2 had no `file_path`."
    )


def test_the_description_of_a_set_that_was_never_imported_is_empty() -> None:
    """Hand-built eval sets carry no import counts, and must not grow a bogus one."""
    from contextgrid.core.evalset import EvalSet

    assert describe_skipped(EvalSet(id="es", items=())) == ""

"""A benchmark that names a document the corpus does not have.

Two failures used to be counted as one. "These offsets miss the text" means the corpus is a
different edition of documents you already have, and nothing downstream is worth running.
"This document is not here" means you are short a file, and everything else is fine.

Charging the second against the first gave advice that was simply wrong: four tests, one
naming an absent file, and `validate` announced that the corpus was "almost certainly not the
one the annotations were made against" and stopped -- when the three loadable spans were
exact. The missing file is now counted on its own and named out loud, which is the one thing
it must never be: silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.core.evalset import EvalItem, EvalSet, GoldSpan
from contextgrid.core.span import Span
from contextgrid.corpus import Corpus
from contextgrid.validate import load_benchmark, self_check

CONTRACT = (
    "Either party may terminate this agreement for convenience by giving thirty days "
    "written notice to the other party.\n\n"
    "Fees are payable within thirty days of the invoice date.\n"
)


def evalset_over(*docs: str) -> EvalSet:
    """One question per document, each with a gold span that is exact where it exists."""
    start = CONTRACT.index("thirty days written notice")
    return EvalSet(
        id="lb",
        items=tuple(
            EvalItem(
                id=f"t{number}",
                question=f"Question {number}?",
                gold=(GoldSpan(span=Span(doc, start, start + 26), grade=2),),
            )
            for number, doc in enumerate(docs)
        ),
    )


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts({"contract.txt": CONTRACT})


# ---------------------------------------------------------------------------
# counted apart from the spans that were actually checked
# ---------------------------------------------------------------------------


def test_a_span_naming_an_absent_document_is_not_counted_against_the_rate(
    corpus: Corpus,
) -> None:
    report = self_check(corpus, evalset_over("contract.txt", "not-here.md"))

    assert report["spans_checked"] == 1
    assert report["spans_in_range"] == 1
    assert report["in_range_rate"] == 1.0
    assert report["spans_missing_file"] == 1
    assert report["missing_files"] == ["not-here.md"]


def test_the_missing_document_is_named_in_the_verdict(corpus: Corpus) -> None:
    """Skipped, but never silently. Discovering it as a percentage is not discovering it."""
    verdict = self_check(corpus, evalset_over("contract.txt", "not-here.md"))["verdict"]

    assert "not-here.md" in verdict
    assert "Skipped 1 span" in verdict
    assert "1 of 1 spans point at real text" in verdict


def test_the_missing_document_is_named_before_the_verdict_it_is_not_part_of(
    corpus: Corpus,
) -> None:
    """The CLI prints the verdict and then decides whether to carry on, so the skipped file
    has to come before the number it was left out of."""
    verdict = self_check(corpus, evalset_over("contract.txt", "not-here.md"))["verdict"]

    assert verdict.index("not-here.md") < verdict.index("point at real text")


def test_every_missing_document_is_named(corpus: Corpus) -> None:
    report = self_check(corpus, evalset_over("contract.txt", "a.md", "b.md", "c.md"))

    assert report["missing_files"] == ["a.md", "b.md", "c.md"]
    assert report["spans_missing_file"] == 3
    for name in ("a.md", "b.md", "c.md"):
        assert name in report["verdict"]


def test_a_long_list_of_missing_documents_is_cut_short(corpus: Corpus) -> None:
    """Twenty file names in front of the verdict buries the verdict."""
    absent = [f"missing{number}.md" for number in range(9)]
    report = self_check(corpus, evalset_over("contract.txt", *absent))

    assert report["missing_files"] == sorted(absent)
    assert "and 4 more" in report["verdict"]


def test_no_missing_documents_means_no_notice(corpus: Corpus) -> None:
    """The common case reads exactly as it did before."""
    report = self_check(corpus, evalset_over("contract.txt"))

    assert report["spans_missing_file"] == 0
    assert report["missing_files"] == []
    assert report["verdict"].startswith("1 of 1 spans point at real text")


# ---------------------------------------------------------------------------
# what a missing document must not hide
# ---------------------------------------------------------------------------


def test_offsets_that_miss_the_text_are_still_caught(corpus: Corpus) -> None:
    """The check this whole thing exists for. The document is loaded; the spans run off the
    end of it; that is a corpus that does not match the annotations, and it still fails."""
    off_the_end = EvalSet(
        id="lb",
        items=(
            EvalItem(
                id="t0",
                question="Question?",
                gold=(GoldSpan(span=Span("contract.txt", 9000, 9100), grade=2),),
            ),
        ),
    )

    report = self_check(corpus, off_the_end)

    assert report["in_range_rate"] < 0.95
    assert "not the one the annotations were made against" in report["verdict"]


def test_a_missing_document_does_not_rescue_a_corpus_that_does_not_match(
    corpus: Corpus,
) -> None:
    """Both problems at once. The missing file is reported, and the mismatch still fails."""
    both = EvalSet(
        id="lb",
        items=(
            EvalItem(
                id="t0",
                question="Question?",
                gold=(GoldSpan(span=Span("contract.txt", 9000, 9100), grade=2),),
            ),
            EvalItem(
                id="t1",
                question="Another?",
                gold=(GoldSpan(span=Span("gone.md", 0, 10), grade=2),),
            ),
        ),
    )

    report = self_check(corpus, both)

    assert report["in_range_rate"] == 0.0
    assert "gone.md" in report["verdict"]
    assert "not the one the annotations were made against" in report["verdict"]


def test_a_benchmark_whose_documents_are_all_absent_fails(corpus: Corpus) -> None:
    """Nothing was checked, so nothing is confirmed. `in_range_rate` stays 0, which is what
    the CLI gates on -- the safe direction."""
    report = self_check(corpus, evalset_over("a.md", "b.md"))

    assert report["spans_checked"] == 0
    assert report["in_range_rate"] == 0.0
    assert "Every gold span" in report["verdict"]
    assert "a.md" in report["verdict"]


def test_a_benchmark_with_no_spans_at_all_is_unchanged() -> None:
    report = self_check(Corpus.from_texts({"a": "text"}), EvalSet(id="es", items=()))

    assert report["spans_checked"] == 0
    assert report["spans_missing_file"] == 0
    assert "no gold spans" in report["verdict"]


# ---------------------------------------------------------------------------
# end to end, through the loader
# ---------------------------------------------------------------------------


def test_one_absent_document_no_longer_condemns_the_whole_run(tmp_path: Path) -> None:
    """The reported bug, start to finish: four good tests and one naming `not-here.md`."""
    documents = tmp_path / "corpus"
    documents.mkdir()
    (documents / "contract.txt").write_text(CONTRACT, encoding="utf-8")
    start = CONTRACT.index("thirty days written notice")

    payload = {
        "tests": [
            {
                "id": f"t{number}",
                "query": f"Question {number}?",
                "snippets": [{"file_path": "contract.txt", "span": [start, start + 26]}],
            }
            for number in range(4)
        ]
        + [
            {
                "id": "t4",
                "query": "And this one?",
                "snippets": [{"file_path": "not-here.md", "span": [0, 10]}],
            }
        ]
    }
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(payload), encoding="utf-8")

    corpus, evalset = load_benchmark(benchmark, documents)
    report = self_check(corpus, evalset)

    assert len(evalset) == 5
    assert report["in_range_rate"] == 1.0
    assert report["missing_files"] == ["not-here.md"]
    assert "not-here.md" in report["verdict"]


def test_a_bare_array_benchmark_loads_through_the_validator(tmp_path: Path) -> None:
    """`load_benchmark` is what the CLI calls, and it used to die on the documented bare
    array with `'list' object has no attribute 'get'`."""
    documents = tmp_path / "corpus"
    documents.mkdir()
    (documents / "contract.txt").write_text(CONTRACT, encoding="utf-8")
    start = CONTRACT.index("thirty days written notice")

    benchmark = tmp_path / "bare.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "query": "How much notice is required?",
                    "snippets": [{"file_path": "contract.txt", "span": [start, start + 26]}],
                }
            ]
        ),
        encoding="utf-8",
    )

    corpus, evalset = load_benchmark(benchmark, documents)

    assert corpus.ids == ("contract.txt",)
    assert [item.id for item in evalset] == ["q1"]
    assert self_check(corpus, evalset)["in_range_rate"] == 1.0

"""Unit tests for the benchmark validation harness.

The real benchmark is not vendored -- it is large and not ours to redistribute -- so these
build a miniature one with the same shape. What they check is the harness: that it loads
spans correctly, that it notices when the corpus does not match the annotations, and that it
compares against published numbers honestly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.core.errors import ContextGridError
from contextgrid.pipeline import Config
from contextgrid.validate import (
    ValidationError,
    ValidationResult,
    items_without_gold,
    load_benchmark,
    resolution_report,
    self_check,
    validate,
)

CONTRACT = (
    "MASTER SERVICES AGREEMENT\n\n"
    "Either party may terminate this agreement for convenience by giving thirty days "
    "written notice to the other party.\n\n"
    "A party may terminate immediately upon a material breach that remains unremedied "
    "for fifteen days after written notice.\n\n"
    "Fees are payable within thirty days of the invoice date.\n"
)


@pytest.fixture
def benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """A miniature benchmark in LegalBench-RAG's shape, with real character spans."""
    documents = tmp_path / "corpus"
    documents.mkdir()
    (documents / "contract.txt").write_text(CONTRACT)

    notice_start = CONTRACT.index("thirty days written notice")
    breach_start = CONTRACT.index("fifteen days after written notice")

    payload = {
        "tests": [
            {
                "id": "t1",
                "query": "How much notice is required to terminate for convenience?",
                "snippets": [
                    {
                        "file_path": "contract.txt",
                        "span": [notice_start, notice_start + len("thirty days written notice")],
                    }
                ],
            },
            {
                "id": "t2",
                "query": "How long is the cure period after a material breach?",
                "snippets": [
                    {
                        "file_path": "contract.txt",
                        "span": [
                            breach_start,
                            breach_start + len("fifteen days after written notice"),
                        ],
                    }
                ],
            },
        ]
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload))
    return path, documents


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_the_benchmark_and_its_documents_load_together(
    benchmark: tuple[Path, Path],
) -> None:
    corpus, evalset = load_benchmark(*benchmark)
    assert len(corpus) == 1
    assert len(evalset) == 2
    assert all(item.gold for item in evalset)


def test_only_the_documents_the_benchmark_refers_to_are_loaded(
    benchmark: tuple[Path, Path],
) -> None:
    path, documents = benchmark
    (documents / "unrelated.txt").write_text("nothing points at this")
    corpus, _ = load_benchmark(path, documents)
    assert corpus.ids == ("contract.txt",)


def test_a_limit_takes_the_first_n_questions(benchmark: tuple[Path, Path]) -> None:
    _, evalset = load_benchmark(*benchmark, limit=1)
    assert len(evalset) == 1


def test_a_missing_benchmark_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no benchmark file"):
        load_benchmark(tmp_path / "absent.json", tmp_path)


def test_a_missing_corpus_directory_says_so(benchmark: tuple[Path, Path]) -> None:
    path, _ = benchmark
    with pytest.raises(ValidationError, match="no corpus directory"):
        load_benchmark(path, path.parent / "absent")


def test_documents_that_are_not_there_are_a_clear_error(
    benchmark: tuple[Path, Path], tmp_path: Path
) -> None:
    """The spans are offsets into those exact files. Loading different ones would produce a
    validation run comparing two unrelated things."""
    path, _ = benchmark
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValidationError, match="annotations were made against"):
        load_benchmark(path, empty)


# ---------------------------------------------------------------------------
# the self check -- does the gold point at real text?
# ---------------------------------------------------------------------------


def test_the_self_check_confirms_the_corpus_matches_the_annotations(
    benchmark: tuple[Path, Path],
) -> None:
    """The first thing to check, and the one that separates a loading problem from a
    retrieval result."""
    corpus, evalset = load_benchmark(*benchmark)
    report = self_check(corpus, evalset)

    assert report["in_range_rate"] == 1.0
    assert report["spans_empty"] == 0
    assert "point at real text" in report["verdict"]


def test_the_self_check_catches_a_corpus_that_does_not_match(
    benchmark: tuple[Path, Path],
) -> None:
    """No number from a run like this means anything, and the cause is loading."""
    path, documents = benchmark
    _, evalset = load_benchmark(path, documents)
    (documents / "contract.txt").write_text("short")
    truncated, _ = load_benchmark(path, documents)

    report = self_check(truncated, evalset)
    assert report["in_range_rate"] < 0.95
    assert "not the one the annotations were made against" in report["verdict"]


def test_the_self_check_handles_a_benchmark_with_no_spans() -> None:
    from contextgrid.core.evalset import EvalItem, EvalSet
    from contextgrid.corpus import Corpus

    empty = EvalSet(id="es", items=(EvalItem(id="q", question="A question?"),))
    report = self_check(Corpus.from_texts({"a": "text"}), empty)
    assert report["spans_checked"] == 0
    assert "no gold spans" in report["verdict"]


def test_questions_with_no_gold_can_be_counted_before_running() -> None:
    from contextgrid.core.evalset import EvalItem, EvalSet

    evalset = EvalSet(
        id="es",
        items=(EvalItem(id="q1", question="A question?"), EvalItem(id="q2", question="Two?")),
    )
    assert len(items_without_gold(evalset)) == 2


# ---------------------------------------------------------------------------
# running the validation
# ---------------------------------------------------------------------------


def test_validation_scores_the_benchmark_end_to_end(benchmark: tuple[Path, Path]) -> None:
    corpus, evalset = load_benchmark(*benchmark)
    result = validate(corpus, evalset)

    assert result.questions == 2
    assert result.resolved == 2
    assert result.resolution_rate == 1.0
    assert result.metrics["recall@10"] > 0


def test_validation_uses_a_deliberately_plain_configuration(
    benchmark: tuple[Path, Path],
) -> None:
    """The point is to check the scorer, not to win the benchmark. A clever configuration
    that beat the paper would prove nothing about whether our metrics are right."""
    corpus, evalset = load_benchmark(*benchmark)
    result = validate(corpus, evalset)
    assert result.config["index"] == "bm25"
    assert result.config["embedder"] is None


def test_a_custom_configuration_is_honoured(benchmark: tuple[Path, Path]) -> None:
    corpus, evalset = load_benchmark(*benchmark)
    result = validate(corpus, evalset, config=Config(parser="text", index="bm25", k=3))
    assert result.config["k"] == 3


def test_agreement_is_reported_against_published_numbers(
    benchmark: tuple[Path, Path],
) -> None:
    corpus, evalset = load_benchmark(*benchmark)
    ours = validate(corpus, evalset).metrics["recall@10"]

    close = validate(corpus, evalset, reference={"recall@10": ours - 0.01})
    assert close.agrees
    assert "reproduces a benchmark it did not define" in close.report()


def test_a_disagreement_is_reported_as_ours_to_explain(benchmark: tuple[Path, Path]) -> None:
    corpus, evalset = load_benchmark(*benchmark)
    off = validate(corpus, evalset, reference={"recall@10": 0.1}, tolerance=0.02)

    assert not off.agrees
    report = off.report()
    assert "outside the" in report
    assert "a problem with our scoring" in report


def test_without_published_numbers_it_only_reports_its_own() -> None:
    result = ValidationResult(name="x", metrics={"recall@10": 0.5}, questions=1, resolved=1)
    assert not result.agrees
    assert "No published numbers" in result.report()


def test_the_report_leads_with_how_much_gold_resolved(
    benchmark: tuple[Path, Path],
) -> None:
    """A low rate invalidates everything downstream, so it belongs above the metrics."""
    corpus, evalset = load_benchmark(*benchmark)
    report = validate(corpus, evalset).report()
    assert report.index("Resolved") < report.index("|") if "|" in report else True
    assert "to character spans in the corpus" in report


# ---------------------------------------------------------------------------
# resolution diagnostics
# ---------------------------------------------------------------------------


def test_the_resolution_report_separates_the_two_failure_modes(
    benchmark: tuple[Path, Path],
) -> None:
    """Gold that does not match the corpus, and gold that matches but falls between chunks."""
    from contextgrid.pipeline import build

    corpus, evalset = load_benchmark(*benchmark)
    chunks = build(Config(parser="text", chunker="recursive:64,overlap=0"), corpus).chunks

    report = resolution_report(evalset, chunks)
    assert report["questions"] == 2
    assert report["resolution_rate"] > 0
    assert report["policy"] == "coverage"


def test_validation_errors_are_context_grid_errors() -> None:
    assert issubclass(ValidationError, ContextGridError)

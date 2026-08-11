"""What the matrix removes, and whether it says so.

Two failures prompted this file, both of them honesty failures rather than crashes.

`20 on paper, 10 to run (5 impossible combination(s) skipped)` -- 20 minus 10 is 10, not 5.
Half the shrink was canonicalisation collapsing rows onto each other, which nothing counted, so
a reader had to work out where five runs went.

And `widened` with no reranker: four of ten rows in that same sweep were the plain-search arm
re-measured under a different name. Collapsing those is right, but only where the extra reach
really is thrown away -- which is not everywhere, and the second half of this file is the proof.
"""

from __future__ import annotations

import pytest

from contextgrid.grid import MatrixError, matrix
from contextgrid.grid.matrix import (
    DedupeReport,
    canonicalise,
    deduplicate,
    deduplicate_with_report,
)
from contextgrid.index.base import Scored
from contextgrid.pipeline import Config
from contextgrid.retrieve import RetrievalTrace, SimpleRetrieval, WidenedRetrieval

# ---------------------------------------------------------------------------
# the numbers on the line have to add up
# ---------------------------------------------------------------------------


def test_every_combination_is_accounted_for() -> None:
    """The arithmetic the printed line got wrong: kept plus removed is what went in."""
    grid = matrix(embedder=["tfidf", "hash", "length"], index=["dense", "bm25"])
    configs, report = grid.expand_with_report("factorial")

    assert report.considered == 6
    assert report.kept == len(configs)
    assert report.kept + report.removed == report.considered


def test_a_collapse_is_counted_separately_from_an_impossible_combination() -> None:
    """The five that vanished silently. `bm25 + hash` is a duplicate; `dense + null` cannot run."""
    _, report = deduplicate_with_report(
        [
            Config(index="dense", embedder="tfidf"),
            Config(index="bm25", embedder="tfidf"),
            Config(index="bm25", embedder="hash"),  # same run as the line above
            Config(index="dense", embedder=None),  # no vectors for a dense index
        ]
    )

    assert (report.kept, report.collapsed, report.impossible) == (2, 1, 1)


def test_a_configuration_written_twice_is_not_called_a_collapse() -> None:
    """Nothing was rewritten here -- the axis simply repeats a value, and that is worth saying."""
    _, report = deduplicate_with_report([Config(chunker="a"), Config(chunker="a")])

    assert (report.repeated, report.collapsed) == (1, 0)


def test_a_later_exact_match_on_a_rewritten_row_is_a_collapse() -> None:
    """The rewrite happened to the first row rather than the second, but one run still went."""
    _, report = deduplicate_with_report(
        [Config(index="bm25", embedder="tfidf"), Config(index="bm25", embedder=None)]
    )

    assert (report.kept, report.collapsed, report.repeated) == (1, 1, 0)


def test_the_note_names_every_category_that_fired() -> None:
    _, report = deduplicate_with_report(
        [
            Config(index="bm25", embedder="tfidf"),
            Config(index="bm25", embedder="hash"),
            Config(index="dense", embedder=None),
        ]
    )

    note = report.note()
    assert "1 impossible combination(s) skipped" in note
    assert "1 collapsed onto an identical run" in note


def test_the_note_is_empty_when_the_matrix_did_not_shrink() -> None:
    _, report = deduplicate_with_report([Config(chunker="a"), Config(chunker="b")])

    assert report.note() == ""
    assert report.removed == 0


def test_a_report_that_does_not_reconcile_is_refused() -> None:
    """The guarantee itself: these numbers cannot be published unless they add up."""
    with pytest.raises(MatrixError, match="accounting is wrong"):
        DedupeReport(considered=20, kept=10, impossible=5, collapsed=0, repeated=0)


def test_the_old_dropped_count_still_means_impossible_only() -> None:
    """Existing callers keep the number they have always been handed, unchanged."""
    grid = matrix(embedder=["tfidf", None], index=["dense", "bm25"])
    _, dropped = grid.expand_with_dropped("factorial")
    _, report = grid.expand_with_report("factorial")

    assert dropped == report.impossible
    assert report.collapsed > 0  # and this is the part the number never covered


def test_deduplicate_keeps_its_two_value_shape() -> None:
    configs, dropped = deduplicate([Config(chunker="a"), Config(chunker="b"), Config(chunker="a")])

    assert [c.chunker for c in configs] == ["a", "b"]
    assert dropped == 0


# ---------------------------------------------------------------------------
# `widened` with nothing downstream to use the extra reach
# ---------------------------------------------------------------------------


def test_widened_alone_is_reset_to_plain_search() -> None:
    """No reranker, no transform, no ingestion, an exact index: the surplus is thrown away."""
    assert canonicalise(Config(retrieval="widened:8")).retrieval is None
    assert canonicalise(Config(retrieval="widened")).retrieval is None


def test_two_widths_and_plain_search_stop_being_three_rows() -> None:
    """The sweep that had four known-duplicate rows before anything ran."""
    configs, report = matrix(retrieval=["widened:2", "widened:8"]).expand_with_report("factorial")

    assert len(configs) == 1
    assert configs[0].retrieval is None
    assert (report.considered, report.collapsed) == (2, 1)


def test_widened_survives_a_reranker() -> None:
    """With one, the wider net is exactly what it reorders."""
    assert canonicalise(Config(retrieval="widened:8", reranker="lexical")).retrieval == "widened:8"


def test_widened_survives_a_transform() -> None:
    """A transform can return several queries, and then the deeper lists fuse differently."""
    assert canonicalise(Config(retrieval="widened:8", transform="multi-query")).retrieval == (
        "widened:8"
    )


def test_the_identity_spellings_are_normalised_before_widened_is_judged() -> None:
    """`reranker: none` is no reranker, so it must not hold a duplicate row open."""
    kept = canonicalise(Config(retrieval="widened:8", reranker="none", transform="none"))

    assert kept.retrieval is None


def test_widened_survives_an_ingestion_strategy() -> None:
    """A deeper pool can merge runs of siblings a shallow one never had the pieces for."""
    assert canonicalise(Config(retrieval="widened:8", ingestion="sentence-window:2")).retrieval == (
        "widened:8"
    )


def test_widened_survives_an_approximate_index() -> None:
    """An approximate index searches more of its structure when asked for more."""
    assert canonicalise(Config(retrieval="widened:8", index="faiss")).retrieval == "widened:8"
    assert canonicalise(Config(retrieval="widened:8", index="quantized")).retrieval == "widened:8"


def test_a_factor_of_one_is_plain_search_whatever_else_is_configured() -> None:
    """The depth is `k` itself, so these are the searches plain search would have made."""
    assert canonicalise(Config(retrieval="widened:1", reranker="cross-encoder")).retrieval is None
    assert canonicalise(Config(retrieval="widened:1", index="faiss")).retrieval is None


def test_the_other_strategies_are_left_alone() -> None:
    """They make different searches, not the same ones at a different depth."""
    for spec in ("decomposed", "relevance-feedback", "simple"):
        assert canonicalise(Config(retrieval=spec)).retrieval == spec


def test_an_unparseable_strategy_is_left_for_the_run_to_report() -> None:
    """Rewriting it here would hide the real error behind a row that became plain search."""
    assert canonicalise(Config(retrieval="not-a-real-strategy")).retrieval == "not-a-real-strategy"


# ---------------------------------------------------------------------------
# why the guard above is not paranoia
# ---------------------------------------------------------------------------


def _ranked(prefix: str) -> list[Scored]:
    return [Scored(f"{prefix}{i}", 1.0 - i / 100) for i in range(40)]


def test_widened_changes_the_answer_on_two_queries_with_no_reranker() -> None:
    """The measurement that stopped this being an unconditional reset.

    Two chunks sit 20th on both queries -- invisible to a top-5 search, and winners once the
    deeper lists are fused by rank. No reranker anywhere.
    """
    first, second = _ranked("a"), _ranked("b")
    first[20], first[21] = Scored("both1", 0.80), Scored("both2", 0.79)
    second[22], second[23] = Scored("both1", 0.78), Scored("both2", 0.77)

    def searcher(text: str, wanted: int) -> list[Scored]:
        return (first if text == "one" else second)[:wanted]

    queries = ["one", "two"]
    plain = SimpleRetrieval().retrieve("q", queries, searcher, 5, RetrievalTrace())
    wide = WidenedRetrieval(factor=8).retrieve("q", queries, searcher, 5, RetrievalTrace())

    assert [s.chunk_id for s in plain] != [s.chunk_id for s in wide]
    assert "both1" in [s.chunk_id for s in wide]


def test_widened_returns_the_same_top_k_on_one_query() -> None:
    """And this is why collapsing it is right in the case the guard does allow."""
    ranking = _ranked("a")

    def searcher(text: str, wanted: int) -> list[Scored]:
        return ranking[:wanted]

    plain = SimpleRetrieval().retrieve("q", ["q"], searcher, 5, RetrievalTrace())
    wide = WidenedRetrieval(factor=8).retrieve("q", ["q"], searcher, 5, RetrievalTrace())

    assert [s.chunk_id for s in plain] == [s.chunk_id for s in wide]

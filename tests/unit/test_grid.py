"""Unit tests for the matrix, the runner, results and costing."""

from __future__ import annotations

import pytest

from contextgrid.cache import CacheStats, MemoryCache
from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus
from contextgrid.cost.model import CostModel, Pricing
from contextgrid.grid import Matrix, MatrixError, Runner, SweepMode, estimate_cost, matrix
from contextgrid.grid.matrix import canonicalise, deduplicate
from contextgrid.pipeline import Config, Timings
from tests.support import API_DOCS, CONTRACT

QUESTIONS = [
    ("q1", "How much notice is needed to terminate for convenience?", "contract.md", "thirty days"),
    ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
    ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
    ("q4", "What happens on a material breach?", "contract.md", "fifteen days of written notice"),
    ("q5", "What does GET /widgets return?", "api.md", "Returns 404 when the id is unknown"),
]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )


@pytest.fixture
def evalset() -> EvalSet:
    return EvalSet(
        id="es",
        items=tuple(
            EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
            for i, q, s, t in QUESTIONS
        ),
    )


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------


def test_a_single_value_does_not_need_wrapping_in_a_list() -> None:
    assert matrix(chunker="recursive:512").chunker == ("recursive:512",)


def test_the_shape_is_the_multiplication_written_out() -> None:
    shape = matrix(chunker=["a", "b"], index=["dense", "bm25", "hybrid"]).shape()
    assert shape.startswith("1 \u00d7 2 \u00d7 1 \u00d7 3")
    assert shape.endswith("= 6")


def test_the_identity_reranker_is_the_same_run_as_no_reranker() -> None:
    """ "none" is the identity, so `reranker=["none", "lexical"]` is two arms, not three --
    and sweeping candidate depth under it would credit the depth axis with differences it
    did not cause."""
    configs = matrix(reranker=["none", "lexical"], candidates=[5, 20, 50]).expand("factorial")
    assert len(configs) == 4  # one for no reranking, three depths for the lexical one
    assert sum(1 for c in configs if c.reranker is None) == 1


def test_candidate_depth_is_a_real_axis_when_something_reranks() -> None:
    """The parameter most reranking advice omits, and where most of the effect lives."""
    configs = matrix(reranker="lexical", candidates=[10, 100]).expand("factorial")
    assert sorted(c.candidates for c in configs) == [10, 100]


def test_an_empty_axis_is_refused() -> None:
    with pytest.raises(MatrixError, match="is empty"):
        Matrix(chunker=())


def test_k_must_be_at_least_one() -> None:
    with pytest.raises(MatrixError, match="k must be at least 1"):
        Matrix(k=0)


def test_only_axes_with_more_than_one_value_can_teach_you_anything() -> None:
    assert matrix(chunker=["a", "b"], index="dense").varying_axes == ("chunker",)


def test_factorial_covers_every_combination() -> None:
    assert len(matrix(chunker=["a", "b"], embedder=["tfidf", "hash"]).expand("factorial")) == 4


def test_ofat_is_linear_rather_than_exponential() -> None:
    """Four axes with three values each is 81 configurations factorial and 9 as OFAT."""
    wide = matrix(
        parser=["a", "b", "c"],
        chunker=["a", "b", "c"],
        embedder=["tfidf", "hash", "length"],
        index=["dense", "hybrid", "bm25"],
    )
    assert len(wide.expand("factorial")) > len(wide.expand("ofat"))
    assert len(wide.expand("ofat")) <= 1 + 4 * 2


def test_ofat_starts_from_the_baseline() -> None:
    configs = matrix(chunker=["a", "b"]).expand("ofat")
    assert configs[0] == matrix(chunker=["a", "b"]).baseline()


def test_stage_configs_vary_one_axis_and_fix_the_rest() -> None:
    grid = matrix(chunker=["a", "b", "c"])
    staged = grid.stage_configs("chunker", Config(chunker="a", index="dense"))
    assert [c.chunker for c in staged] == ["a", "b", "c"]
    assert {c.index for c in staged} == {"dense"}


def test_an_unknown_axis_is_refused() -> None:
    with pytest.raises(MatrixError, match="unknown axis"):
        matrix().stage_configs("temperature", Config())


# ---------------------------------------------------------------------------
# removing configurations that are not actually different
# ---------------------------------------------------------------------------


def test_an_index_that_ignores_vectors_drops_the_embedder() -> None:
    """`bm25 + tfidf` and `bm25 + hash` are the same run under two names."""
    assert canonicalise(Config(index="bm25", embedder="tfidf")).embedder is None
    assert canonicalise(Config(index="dense", embedder="tfidf")).embedder == "tfidf"


def test_the_sparse_arm_of_a_sweep_collapses() -> None:
    grid = matrix(embedder=["tfidf", "hash", "length"], index=["dense", "bm25"])
    configs = grid.expand("factorial")
    assert len(configs) == 4  # three dense arms, one bm25 -- not six
    assert sum(1 for c in configs if c.index == "bm25") == 1


def test_deduplication_keeps_the_original_order() -> None:
    configs = deduplicate([Config(chunker="a"), Config(chunker="b"), Config(chunker="a")])
    assert [c.chunker for c in configs] == ["a", "b"]


def test_an_index_that_cannot_be_built_is_left_for_the_run_to_report() -> None:
    """Silently dropping it here would hide the real error behind a missing row."""
    assert canonicalise(Config(index="not-a-real-index")).index == "not-a-real-index"


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def test_one_configuration_end_to_end(corpus: Corpus, evalset: EvalSet) -> None:
    result = Runner(corpus=corpus).run_one(
        Config(chunker="sentence:1", embedder="tfidf", index="dense", k=3), evalset
    )
    assert result.scored_queries == len(QUESTIONS)
    assert 0.0 <= result.metric("recall@5") <= 1.0
    assert result.chunk_count > 0
    assert result.timings.build_ms >= 0


def test_a_sweep_reuses_work_across_configurations(corpus: Corpus, evalset: EvalSet) -> None:
    """The claim that makes a grid affordable, checked rather than believed."""
    stats = CacheStats()
    runner = Runner(corpus=corpus, cache=MemoryCache(), stats=stats)
    grid = matrix(chunker="sentence:1", index=["dense", "hybrid"], embedder="tfidf")
    runner.run(grid, evalset, mode="factorial")

    # Same parser and chunker on both arms, so each source is parsed and chunked once.
    assert stats.by_stage["parse"][0] > 0
    assert stats.by_stage["chunk"][0] > 0


def test_sweeping_indexes_never_re_embeds(corpus: Corpus, evalset: EvalSet) -> None:
    stats = CacheStats()
    runner = Runner(corpus=corpus, cache=MemoryCache(), stats=stats)
    runner.run(
        matrix(chunker="sentence:1", embedder="tfidf", index=["dense", "hybrid"]),
        evalset,
        mode="factorial",
    )
    hits, misses = stats.by_stage["embed"]
    assert misses == 1  # embedded once
    assert hits >= 1  # and reused after that


def test_a_budget_stops_the_sweep_and_says_so(corpus: Corpus, evalset: EvalSet) -> None:
    grid = matrix(
        chunker=[
            "sentence:1",
            "fixed:20,overlap=0",
            "recursive:40,overlap=0",
            "structural:60,min_size=8",
        ]
    )
    results = Runner(corpus=corpus).run(grid, evalset, mode="factorial", budget_seconds=0.0)
    assert len(results) < 4
    assert results.warnings.of_code(WarningCode.BUDGET_REACHED)


def test_progress_is_reported(corpus: Corpus, evalset: EvalSet) -> None:
    seen: list[int] = []
    Runner(corpus=corpus).run(
        matrix(chunker=["sentence:1", "fixed:20,overlap=0"]),
        evalset,
        mode="factorial",
        on_progress=lambda index, total, config: seen.append(index),
    )
    assert seen == [1, 2]


def test_staged_runs_fewer_configurations_than_factorial(corpus: Corpus, evalset: EvalSet) -> None:
    grid = matrix(
        chunker=["sentence:1", "fixed:20,overlap=0", "recursive:40,overlap=0"],
        embedder=["tfidf", "hash:64"],
        index=["dense", "hybrid"],
    )
    staged = Runner(corpus=corpus).run(grid, evalset, mode="staged")
    assert len(staged) < len(grid.expand("factorial"))
    assert "final" in staged.meta


def test_staged_admits_it_cannot_see_interactions(corpus: Corpus, evalset: EvalSet) -> None:
    """The honest caveat on the mode most people actually want."""
    grid = matrix(chunker=["sentence:1", "fixed:20,overlap=0"], index=["dense", "hybrid"])
    results = Runner(corpus=corpus).run(grid, evalset, mode="staged")
    caveats = results.warnings.of_code(WarningCode.NON_DETERMINISTIC_STAGE)
    assert caveats
    assert "may not be the best configuration" in caveats[0].message


def test_a_configuration_that_resolves_no_evidence_is_marked_unsound(
    corpus: Corpus,
) -> None:
    impossible = EvalSet(
        id="es",
        items=(
            EvalItem(
                id="q1",
                question="q",
                anchors=(GoldAnchor(source_id="contract.md", quote="text that is not there"),),
            ),
        ),
    )
    result = Runner(corpus=corpus).run_one(Config(chunker="sentence:1"), impossible)
    assert not result.is_sound
    assert result.warnings.of_code(WarningCode.GOLD_SPAN_UNREACHABLE)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def test_the_leaderboard_always_carries_latency_and_cost(corpus: Corpus, evalset: EvalSet) -> None:
    """Not optional columns. Omitting them invites the mistake this tool exists to prevent."""
    results = Runner(corpus=corpus).run(
        matrix(chunker=["sentence:1", "fixed:20,overlap=0"]), evalset, mode="factorial"
    )
    row = results.leaderboard("recall@5")[0]
    assert "p95_ms" in row
    assert "cost_per_1k" in row


def test_the_axis_effect_averages_over_everything_that_used_a_value(
    corpus: Corpus, evalset: EvalSet
) -> None:
    results = Runner(corpus=corpus).run(
        matrix(chunker=["sentence:1", "fixed:20,overlap=0"], index=["dense", "hybrid"]),
        evalset,
        mode="factorial",
    )
    effect = results.axis_effect("chunker", "recall@5")
    assert set(effect) == {"sentence:1", "fixed:20,overlap=0"}


def test_the_pareto_frontier_is_never_empty(corpus: Corpus, evalset: EvalSet) -> None:
    results = Runner(corpus=corpus).run(
        matrix(chunker=["sentence:1", "fixed:20,overlap=0"]), evalset, mode="factorial"
    )
    assert results.pareto("recall@5", "p95_ms")


def test_comparing_two_configurations_shows_where_they_disagreed(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """Two configs with the same mean can succeed on completely different questions."""
    results = Runner(corpus=corpus, headline="recall@3").run(
        matrix(chunker=["sentence:1", "fixed:12,overlap=0"]), evalset, mode="factorial"
    )
    labels = [run.label for run in results]
    comparison = results.compare(labels[0], labels[1], "recall@3")
    assert comparison["queries_compared"] > 0
    assert comparison["left_wins"] + comparison["right_wins"] == comparison["queries_disagreed"]


def test_comparing_an_unknown_configuration_says_which_one(
    corpus: Corpus, evalset: EvalSet
) -> None:
    results = Runner(corpus=corpus).run(matrix(), evalset)
    with pytest.raises(KeyError, match="nonsense"):
        results.compare(results.runs[0].label, "nonsense")


def test_the_summary_is_plain_english(corpus: Corpus, evalset: EvalSet) -> None:
    results = Runner(corpus=corpus).run(
        matrix(chunker=["sentence:1", "fixed:20,overlap=0"]), evalset, mode="factorial"
    )
    summary = results.summary("recall@5")
    assert "scored best on recall@5" in summary
    # It runs a real paired test rather than hedging, and on five questions the honest
    # answer is almost always that the two cannot be told apart.
    assert "not distinguishable" in summary or "beats" in summary


def test_an_empty_result_set_says_so() -> None:
    from contextgrid.report.results import Results

    assert Results().summary() == "No configurations were run."


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


def test_a_local_model_costs_time_rather_than_tokens() -> None:
    model = CostModel(machine_usd_per_hour=0.10)
    cost = model.estimate(
        embedder="tfidf", index_tokens=100_000, query_tokens_per_query=12, compute_seconds=36
    )
    assert not cost.metered
    assert cost.query_usd_per_1k == 0.0
    assert cost.index_usd == pytest.approx(0.001)


def test_a_hosted_model_costs_per_token_at_index_and_at_query_time() -> None:
    cost = CostModel().estimate(
        embedder="text-embedding-3-small", index_tokens=1_000_000, query_tokens_per_query=10
    )
    assert cost.metered
    assert cost.index_usd == pytest.approx(0.02)
    assert cost.query_usd_per_1k == pytest.approx(0.0002)


def test_a_spec_string_still_finds_the_price() -> None:
    assert CostModel().pricing_for("text-embedding-3-small:256").embed_per_million == 0.02


def test_an_unpriced_model_is_costed_at_zero_and_flagged() -> None:
    """Understating a cost silently is worse than refusing to guess."""
    model = CostModel()
    model.pricing_for("some-new-model")
    assert model.warnings


def test_total_cost_combines_the_one_off_and_the_recurring() -> None:
    cost = CostModel().estimate(
        embedder="text-embedding-3-small", index_tokens=1_000_000, query_tokens_per_query=10
    )
    assert cost.total_at(10_000) == pytest.approx(0.02 + 0.002)


def test_a_custom_price_table_is_honoured() -> None:
    model = CostModel(prices={"mine": Pricing(embed_per_million=1.0)})
    assert model.estimate(
        embedder="mine", index_tokens=1_000_000, query_tokens_per_query=0
    ).index_usd == pytest.approx(1.0)


def test_estimating_a_sweep_before_running_it(corpus: Corpus) -> None:
    estimate = estimate_cost(
        matrix(chunker=["a", "b"], index=["dense", "bm25"]), corpus, mode="factorial"
    )
    assert estimate["configurations"] > 0
    assert "\u00d7" in estimate["shape"]
    assert estimate["estimated_usd"] >= 0


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


def test_percentiles_over_no_queries_are_zero_rather_than_an_error() -> None:
    assert Timings().percentile(0.95) == 0.0


def test_percentiles_pick_the_right_position() -> None:
    timings = Timings(query_ms=[1.0, 2.0, 3.0, 4.0, 100.0])
    assert timings.percentile(0.5) == 3.0
    assert timings.percentile(0.95) == 100.0


def test_build_time_is_the_sum_of_the_stages() -> None:
    timings = Timings(parse_ms=1.0, chunk_ms=2.0, embed_ms=3.0, index_ms=4.0)
    assert timings.build_ms == 10.0
    assert timings.as_dict()["build_ms"] == 10.0


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def test_a_config_label_reads_like_a_leaderboard_row() -> None:
    assert Config("markdown", "recursive:512", "tfidf", "dense").label == (
        "markdown · recursive:512 · tfidf · dense"
    )


def test_a_config_without_an_embedder_says_so_by_omission() -> None:
    assert Config("markdown", "recursive:512", None, "bm25").label == (
        "markdown · recursive:512 · bm25"
    )


def test_configs_are_hashable_so_they_can_be_deduplicated() -> None:
    assert len({Config(), Config(), Config(chunker="other")}) == 2


def test_sweep_modes_are_named() -> None:
    assert {m.value for m in SweepMode} == {"factorial", "ofat", "staged"}

"""Cross-checking our metrics against ranx.

The core of this package installs with numpy and nothing else, so the metrics are implemented
here rather than delegated to a library that would drag in a JIT compiler. That is only
defensible if they are right.

`ranx` is the peer-reviewed reference implementation (ECIR 2022). These tests generate random
judgements and random runs -- including the awkward shapes: empty results, graded relevance,
ties, more relevant chunks than the cut-off -- and assert the two agree to floating-point
tolerance. "Our numbers match ranx on a thousand random cases" is a stronger claim than "we
used ranx", and anyone can re-run it.
"""

from __future__ import annotations

import random

import pytest

from contextgrid.score.metrics import evaluate

ranx = pytest.importorskip("ranx", reason="ranx is a dev dependency")

KS = (1, 3, 5, 10)
METRIC_NAMES = ("recall", "precision", "hit_rate", "mrr", "map", "ndcg")


def make_case(
    seed: int,
    *,
    queries: int = 12,
    pool: int = 40,
    max_relevant: int = 6,
    max_grade: int = 3,
    max_returned: int = 15,
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    """Random judgements and a random ranked run over a shared pool of chunks."""
    rng = random.Random(seed)
    chunks = [f"c{i}" for i in range(pool)]

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, list[str]] = {}

    for index in range(queries):
        query_id = f"q{index}"
        relevant = rng.sample(chunks, rng.randint(1, max_relevant))
        qrels[query_id] = {c: rng.randint(1, max_grade) for c in relevant}

        returned = rng.sample(chunks, rng.randint(0, max_returned))
        run[query_id] = returned

    return qrels, run


def as_ranx(qrels: dict[str, dict[str, int]], run: dict[str, list[str]]) -> tuple:  # type: ignore[type-arg]
    """ranx wants scores rather than positions, so give descending scores by rank."""
    scored = {
        query_id: {chunk_id: float(len(ranked) - i) for i, chunk_id in enumerate(ranked)}
        for query_id, ranked in run.items()
    }
    # ranx rejects a run with no results at all for a query, so give it a placeholder that
    # cannot be relevant. Our implementation handles the empty list directly.
    for results in scored.values():
        if not results:
            results["__none__"] = 0.0
    return ranx.Qrels(qrels), ranx.Run(scored)


@pytest.mark.parametrize("seed", range(25))
def test_every_metric_matches_ranx(seed: int) -> None:
    qrels, run = make_case(seed)
    ours = evaluate(qrels, run, ks=KS, metrics=METRIC_NAMES)

    reference_qrels, reference_run = as_ranx(qrels, run)
    wanted = [f"{name}@{k}" for name in METRIC_NAMES for k in KS]
    theirs = ranx.evaluate(reference_qrels, reference_run, wanted)

    for metric in wanted:
        assert ours[metric] == pytest.approx(theirs[metric], abs=1e-9), (
            f"{metric} disagrees on seed {seed}: ours {ours[metric]}, ranx {theirs[metric]}"
        )


@pytest.mark.parametrize("seed", range(10))
def test_binary_relevance_also_matches(seed: int) -> None:
    """Graded gold is the interesting case, but plenty of imported eval sets are binary."""
    qrels, run = make_case(seed, max_grade=1)
    ours = evaluate(qrels, run, ks=KS, metrics=METRIC_NAMES)

    reference_qrels, reference_run = as_ranx(qrels, run)
    wanted = [f"{name}@{k}" for name in METRIC_NAMES for k in KS]
    theirs = ranx.evaluate(reference_qrels, reference_run, wanted)

    for metric in wanted:
        assert ours[metric] == pytest.approx(theirs[metric], abs=1e-9)


def test_more_relevant_chunks_than_the_cut_off_matches() -> None:
    """The case where MAP conventions usually diverge between implementations."""
    qrels, run = make_case(99, max_relevant=12, max_returned=20)
    ours = evaluate(qrels, run, ks=(3, 5), metrics=("map", "recall", "ndcg"))

    reference_qrels, reference_run = as_ranx(qrels, run)
    wanted = ["map@3", "map@5", "recall@3", "recall@5", "ndcg@3", "ndcg@5"]
    theirs = ranx.evaluate(reference_qrels, reference_run, wanted)

    for metric in wanted:
        assert ours[metric] == pytest.approx(theirs[metric], abs=1e-9)


def test_a_run_that_returns_nothing_matches() -> None:
    qrels = {"q0": {"c1": 2}, "q1": {"c2": 1}}
    run: dict[str, list[str]] = {"q0": [], "q1": ["c2"]}
    ours = evaluate(qrels, run, ks=(5,), metrics=METRIC_NAMES)

    reference_qrels, reference_run = as_ranx(qrels, run)
    wanted = [f"{name}@5" for name in METRIC_NAMES]
    theirs = ranx.evaluate(reference_qrels, reference_run, wanted)

    for metric in wanted:
        assert ours[metric] == pytest.approx(theirs[metric], abs=1e-9)


# ---------------------------------------------------------------------------
# the corners ranx does not reach
# ---------------------------------------------------------------------------
#
# ranx compares the metrics on well-formed inputs, which is where a formula goes wrong. These
# cover the branches it never exercises: a question with no relevant chunk at all, a nonsense
# cut-off, an unknown metric name, and the two rank functions -- which had no test whatsoever
# and are exported from the package root.


def test_a_question_with_no_relevant_chunk_scores_zero_not_a_crash() -> None:
    """An eval set carrying a question whose evidence never resolved is normal, not
    exceptional. Dividing by an empty relevant set would take down the whole sweep."""
    from contextgrid.score.metrics import (
        average_precision,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    empty: dict[str, int] = {}
    for score in (
        recall_at_k(empty, ["a", "b"], 5),
        precision_at_k(empty, ["a", "b"], 5),
        average_precision(empty, ["a", "b"], 5),
        ndcg_at_k(empty, ["a", "b"], 5),
    ):
        assert score == 0.0


def test_a_cutoff_of_zero_scores_zero_rather_than_dividing_by_it() -> None:
    from contextgrid.score.metrics import precision_at_k

    assert precision_at_k({"a": 1}, ["a"], 0) == 0.0


def test_an_unknown_metric_name_lists_the_real_ones() -> None:
    """`evaluate` and `per_query` are the two public entry points, and a typo in a headline
    should not reach a leaderboard as a silent absence."""
    from contextgrid.score.metrics import evaluate, per_query

    with pytest.raises(ValueError, match="Available: hit_rate"):
        evaluate({"q": {"a": 1}}, {"q": ["a"]}, metrics=["recall", "f1"])

    with pytest.raises(ValueError, match="unknown metric 'f1'"):
        per_query({"q": {"a": 1}}, {"q": ["a"]}, "f1", 5)


def test_the_rank_of_the_first_relevant_result_is_one_based() -> None:
    """Position 1 is the top of the list. Zero-based here would make "mean rank 0.0" mean
    perfect, which reads as failure to every person who sees it."""
    from contextgrid.score.metrics import rank_of_first_relevant

    assert rank_of_first_relevant({"b": 1}, ["a", "b", "c"], 5) == 2
    assert rank_of_first_relevant({"a": 1}, ["a", "b"], 5) == 1


def test_nothing_relevant_in_the_list_has_no_rank_rather_than_a_large_one() -> None:
    """`None`, not a sentinel like 999. A sentinel averaged into a mean rank silently invents
    a number nobody measured."""
    from contextgrid.score.metrics import rank_of_first_relevant

    assert rank_of_first_relevant({"z": 1}, ["a", "b"], 5) is None


def test_a_relevant_result_past_the_cutoff_does_not_count() -> None:
    from contextgrid.score.metrics import rank_of_first_relevant

    assert rank_of_first_relevant({"c": 1}, ["a", "b", "c"], 2) is None
    assert rank_of_first_relevant({"c": 1}, ["a", "b", "c"], 3) == 3


def test_the_mean_rank_comes_back_with_how_many_queries_it_missed() -> None:
    """Returned together on purpose. A mean rank of 2.0 means something very different computed
    over three of four queries than over four of four, and the average alone hides it."""
    from contextgrid.score.metrics import mean_rank_of_first_relevant

    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"z": 1}}
    run = {"q1": ["a"], "q2": ["x", "b"], "q3": ["x", "y"]}

    mean, missed = mean_rank_of_first_relevant(qrels, run, k=5)
    assert mean == pytest.approx(1.5)  # ranks 1 and 2
    assert missed == 1  # q3 never found anything


def test_a_question_with_no_gold_is_not_counted_as_missed() -> None:
    """It was never answerable, so counting it as a miss would blame the retriever for the
    eval set."""
    from contextgrid.score.metrics import mean_rank_of_first_relevant

    qrels = {"q1": {"a": 1}, "q2": {}}
    mean, missed = mean_rank_of_first_relevant(qrels, {"q1": ["a"], "q2": []}, k=5)

    assert mean == 1.0
    assert missed == 0


def test_finding_nothing_at_all_gives_no_mean_rather_than_zero() -> None:
    """Zero would read as "found everything first", the exact opposite of the truth."""
    from contextgrid.score.metrics import mean_rank_of_first_relevant

    mean, missed = mean_rank_of_first_relevant({"q1": {"a": 1}}, {"q1": ["x"]}, k=5)
    assert mean is None
    assert missed == 1


def test_a_query_missing_from_the_run_is_a_miss_not_a_crash() -> None:
    """A configuration that failed on one question still has to be scorable on the rest."""
    from contextgrid.score.metrics import mean_rank_of_first_relevant

    mean, missed = mean_rank_of_first_relevant({"q1": {"a": 1}, "q2": {"b": 1}}, {"q1": ["a"]}, k=5)
    assert mean == 1.0
    assert missed == 1


def test_available_metrics_matches_what_evaluate_accepts() -> None:
    """The list is what error messages promise and what the config validator checks against."""
    from contextgrid.score.metrics import available_metrics, evaluate

    names = available_metrics()
    assert names == tuple(sorted(names))
    scores = evaluate({"q": {"a": 1}}, {"q": ["a"]}, metrics=list(names), ks=[1])
    for name in names:
        assert f"{name}@1" in scores

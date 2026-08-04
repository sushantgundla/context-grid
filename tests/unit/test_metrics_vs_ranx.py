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

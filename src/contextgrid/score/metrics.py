"""Retrieval metrics.

Implemented here rather than delegated, for one reason: the core of this package installs
with numpy and nothing else, and a metrics library that drags in a JIT compiler would make
`pip install context-grid` a minute-long event.

That is only defensible if the implementations are right, so they are checked against `ranx`
-- the peer-reviewed reference -- on randomly generated qrels and runs, in CI. "Our numbers
agree exactly with ranx on ten thousand random cases" is a stronger claim than "we used
ranx", and it is one anybody can re-run.

Every metric takes graded relevance judgements (`{chunk_id: grade}`) and a ranked list of
chunk ids. Anything with a grade above zero is relevant; nDCG uses the grades themselves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from contextgrid.core.evalset import Qrels

#: The cut-offs reported by default. Small values show precision, large ones show whether the
#: evidence is present at all -- and for RAG the second question is usually the real one.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)


# ---------------------------------------------------------------------------
# per-query metrics
# ---------------------------------------------------------------------------


def recall_at_k(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top k.

    The headline metric for retrieval-augmented generation. A generator does not need the
    evidence ranked first, it needs the evidence *present* -- so recall at the k you actually
    put in the prompt is the number that predicts whether the answer can be right at all.
    """
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    if not relevant:
        return 0.0
    found = sum(1 for chunk_id in ranked[:k] if chunk_id in relevant)
    return found / len(relevant)


def precision_at_k(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant. Divided by k, not by how many were returned.

    Dividing by the number returned would let a configuration that finds only three chunks
    score 1.0 for precision@10, which flatters exactly the configurations that are failing.
    """
    if k <= 0:
        return 0.0
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    return sum(1 for chunk_id in ranked[:k] if chunk_id in relevant) / k


def hit_rate_at_k(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """1.0 when anything relevant is in the top k. The most intuitive number here."""
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    return 1.0 if any(chunk_id in relevant for chunk_id in ranked[:k]) else 0.0


def reciprocal_rank(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """1 / the position of the first relevant result, or 0 if there is none in the top k."""
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def rank_of_first_relevant(
    judgements: Mapping[str, int], ranked: Sequence[str], k: int | None = None
) -> int | None:
    """Where the first relevant result actually landed, or None if it did not.

    More useful than the reciprocal when debugging: "it was at rank 14" says something a
    score of 0.071 does not.
    """
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    limit = len(ranked) if k is None else k
    for position, chunk_id in enumerate(ranked[:limit], start=1):
        if chunk_id in relevant:
            return position
    return None


def average_precision(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """Precision measured at each relevant hit, averaged over all relevant chunks.

    **The denominator is the total number of relevant chunks, not `min(relevant, k)`.** Both
    conventions exist and they disagree by a lot. This one is trec_eval's, which `ranx`
    follows and which the IR literature means by MAP@k, so it is the one used here.

    The consequence is worth stating plainly: a query with 20 relevant chunks cannot score
    above 0.25 at k=5, however perfect the ranking. That is a property of the metric rather
    than a fact about the retriever, and it is why MAP is reported alongside recall rather
    than instead of it.
    """
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    if not relevant:
        return 0.0

    hits = 0
    total = 0.0
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            hits += 1
            total += hits / position

    return total / len(relevant)


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain: each gain divided by log2 of its position plus one."""
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """nDCG with graded relevance, against the best possible ordering of this query's gold.

    Grades are what make this worth computing. With binary gold it degenerates into a noisier
    hit rate, which is why the eval set carries "fully answers" and "partially relevant" as
    separate grades rather than collapsing them.
    """
    gains = [float(judgements.get(chunk_id, 0)) for chunk_id in ranked[:k]]
    ideal = sorted((float(grade) for grade in judgements.values()), reverse=True)[:k]

    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

#: name -> (function, whether it takes a k)
_METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": reciprocal_rank,
    "map": average_precision,
    "ndcg": ndcg_at_k,
}


def evaluate(
    qrels: Qrels,
    run: Mapping[str, Sequence[str]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    metrics: Sequence[str] = tuple(_METRICS),
) -> dict[str, float]:
    """Score a whole run, averaged over queries.

    Only queries present in `qrels` are scored. A query with no relevant chunks cannot be
    got right or wrong, and including it as a zero would drag the mean down for a reason
    that has nothing to do with the retriever.

    A query in the qrels that the run never answered scores zero, because that is a real
    failure rather than a missing measurement.
    """
    unknown = set(metrics) - set(_METRICS)
    if unknown:
        raise ValueError(
            f"unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(_METRICS))}"
        )

    scored = [query_id for query_id in qrels if qrels[query_id]]
    if not scored:
        return {}

    results: dict[str, float] = {}
    for metric in metrics:
        function = _METRICS[metric]
        for k in ks:
            total = sum(function(qrels[qid], run.get(qid, ()), k) for qid in scored)
            results[f"{metric}@{k}"] = total / len(scored)
    return results


def per_query(
    qrels: Qrels,
    run: Mapping[str, Sequence[str]],
    metric: str,
    k: int,
) -> dict[str, float]:
    """One metric for every query separately.

    The input to paired significance testing, and to finding the questions where two
    configurations actually disagree.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}. Available: {', '.join(sorted(_METRICS))}")
    function = _METRICS[metric]
    return {
        query_id: function(judgements, run.get(query_id, ()), k)
        for query_id, judgements in qrels.items()
        if judgements
    }


def mean_rank_of_first_relevant(
    qrels: Qrels, run: Mapping[str, Sequence[str]], *, k: int | None = None
) -> tuple[float | None, int]:
    """Average position of the first relevant result, and how many queries had none.

    Returned together on purpose. A mean rank of 2.1 means something very different when it
    was computed over 90 of 100 queries than over 40 of them, and reporting the average
    alone hides the difference.
    """
    ranks = [
        rank
        for query_id, judgements in qrels.items()
        if judgements
        for rank in [rank_of_first_relevant(judgements, run.get(query_id, ()), k)]
        if rank is not None
    ]
    answered = sum(1 for judgements in qrels.values() if judgements)
    if not ranks:
        return None, answered
    return sum(ranks) / len(ranks), answered - len(ranks)


def available_metrics() -> tuple[str, ...]:
    return tuple(sorted(_METRICS))

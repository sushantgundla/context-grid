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

**A chunk id counts once, however many times the ranking repeats it.** Retrieval is a set
question wearing an ordered coat: recall asks which of the relevant chunks reached the
context, not how many slots they filled. Counting a repeat as a second retrieval let
`recall@3` return 1.5 -- a number the scale does not have -- and made a retriever score
better for the bug of returning the same chunk three times. `evaluate()` refuses such a run
outright; the functions below are correct on one anyway, so a caller who does their own
de-duplication is not left guessing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.core.evalset import Qrels
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.score.base import METRICS

#: The cut-offs reported by default. Small values show precision, large ones show whether the
#: evidence is present at all -- and for RAG the second question is usually the real one.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)

#: The default `metrics=` for `evaluate()` -- these six, regardless of what else has been
#: registered into `METRICS` by the time it runs. A plugin somebody installed becoming part
#: of every run's default columns just because it exists would be its own kind of surprise;
#: `run.metrics` (see `config/schema.py`) is how it gets opted in.
BUILTIN_METRIC_NAMES: tuple[str, ...] = (
    "recall",
    "precision",
    "hit_rate",
    "mrr",
    "map",
    "ndcg",
)


# ---------------------------------------------------------------------------
# per-query metrics
# ---------------------------------------------------------------------------


def recall_at_k(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top k.

    The headline metric for retrieval-augmented generation. A generator does not need the
    evidence ranked first, it needs the evidence *present* -- so recall at the k you actually
    put in the prompt is the number that predicts whether the answer can be right at all.

    Distinct chunks, counted against the window as it actually is. A run of `c1, c1, c1` at
    k=3 found one of its two relevant chunks and wasted two slots doing it: recall is 0.5.
    The repeats are not promoted out of the way to let a fourth result in, because the top
    three is what a prompt would have been built from.
    """
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    if not relevant:
        return 0.0
    found = len(relevant.intersection(ranked[:k]))
    return found / len(relevant)


def precision_at_k(judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant. Divided by k, not by how many were returned.

    Dividing by the number returned would let a configuration that finds only three chunks
    score 1.0 for precision@10, which flatters exactly the configurations that are failing.

    The numerator counts distinct chunks for the same reason. A repeat of something already
    in the context is wasted space, so `c1, c1, x` is one useful result in three -- exactly
    what `c1, y, x` is -- rather than two.
    """
    if k <= 0:
        return 0.0
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    return len(relevant.intersection(ranked[:k])) / k


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

    A chunk contributes at its first appearance and never again. Without that, a ranking of
    `c1, c1, c1` scored three hits against two relevant chunks and came out at 1.5.
    """
    relevant = {chunk_id for chunk_id, grade in judgements.items() if grade > 0}
    if not relevant:
        return 0.0

    hits = 0
    total = 0.0
    counted: set[str] = set()
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant and chunk_id not in counted:
            counted.add(chunk_id)
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

    A repeated chunk gains nothing the second time. Its information is already in the
    context, and paying gain for it again is what let a ranking out-score its own ideal.
    """
    seen: set[str] = set()
    gains: list[float] = []
    for chunk_id in ranked[:k]:
        gains.append(0.0 if chunk_id in seen else float(judgements.get(chunk_id, 0)))
        seen.add(chunk_id)
    ideal = sorted((float(grade) for grade in judgements.values()), reverse=True)[:k]

    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0


# ---------------------------------------------------------------------------
# the six built-ins, as `Metric` plugins
# ---------------------------------------------------------------------------
#
# The functions above are the real public API -- `recall_at_k` and friends are what tests and
# callers use directly, and that does not change. These are just enough wrapping to let the
# `METRICS` registry resolve the same six by name, the way it resolves any other plugin.
# `score/__init__.py` registers them, the way `chunk/__init__.py` registers `FixedTokenChunker`
# and the rest into `CHUNKERS`.


@dataclass(frozen=True, slots=True)
class RecallMetric:
    """Recall@k as a registered plugin. See `recall_at_k` for the definition."""

    name: ClassVar[str] = "recall"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return recall_at_k(judgements, ranked, k)


@dataclass(frozen=True, slots=True)
class PrecisionMetric:
    """Precision@k as a registered plugin. See `precision_at_k` for the definition."""

    name: ClassVar[str] = "precision"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return precision_at_k(judgements, ranked, k)


@dataclass(frozen=True, slots=True)
class HitRateMetric:
    """Hit rate@k as a registered plugin. See `hit_rate_at_k` for the definition."""

    name: ClassVar[str] = "hit_rate"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return hit_rate_at_k(judgements, ranked, k)


@dataclass(frozen=True, slots=True)
class MRRMetric:
    """Reciprocal rank as a registered plugin. See `reciprocal_rank` for the definition."""

    name: ClassVar[str] = "mrr"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return reciprocal_rank(judgements, ranked, k)


@dataclass(frozen=True, slots=True)
class MAPMetric:
    """Mean average precision as a registered plugin. See `average_precision` for the
    definition, and the note on its denominator convention."""

    name: ClassVar[str] = "map"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return average_precision(judgements, ranked, k)


@dataclass(frozen=True, slots=True)
class NDCGMetric:
    """Graded nDCG as a registered plugin. See `ndcg_at_k` for the definition."""

    name: ClassVar[str] = "ndcg"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return ndcg_at_k(judgements, ranked, k)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _is_scorable(judgements: Mapping[str, int]) -> bool:
    """True when this question has at least one chunk judged relevant.

    The one rule for "can this question be got right or wrong", used by everything that
    averages over questions. `{}` and `{"c1": 0}` are two spellings of "nothing relevant
    here" and both fail it.

    They used to disagree. `{}` is falsy so it was skipped; `{"c1": 0}` is a non-empty dict
    so it was scored, always at zero, and halved the mean of a two-question set. A resolver
    emitting a grade-0 row for every question it could not resolve would then have halved
    every number in a sweep, for a reason having nothing to do with the retriever.
    """
    return any(grade > 0 for grade in judgements.values())


def _check_ks(ks: Sequence[int]) -> None:
    """Reject cut-offs below 1, rather than handing them to a slice that accepts anything.

    `ranked[:-1]` is valid Python and means "all but the last", so `ks=[-1]` used to come
    back as a plausible number for a cut-off nobody could describe.
    """
    bad = sorted({k for k in ks if k < 1})
    if bad:
        raise ValueError(
            f"cut-off k must be at least 1, got {', '.join(str(k) for k in bad)}. "
            "A k of 0 or below is not a smaller top-k, it is a slice that means something else."
        )


def _check_no_duplicates(run: Mapping[str, Sequence[str]], scored: Sequence[str]) -> None:
    """Refuse a ranking that returns the same chunk id twice.

    Rejected rather than quietly de-duplicated, and rejected rather than clamped. A retriever
    handing back the same chunk twice is broken, and the whole point of a metric is to say so
    -- a number repaired on the way out would put a plausible score on a leaderboard and
    leave the bug in place. Only the questions that will actually be averaged are checked; a
    ranking nothing scores is nobody's business.
    """
    for query_id in scored:
        ranked = run.get(query_id, ())
        seen: set[str] = set()
        repeated: set[str] = set()
        for chunk_id in ranked:
            if chunk_id in seen:
                repeated.add(chunk_id)
            seen.add(chunk_id)
        if repeated:
            raise ValueError(
                f"the run for query {query_id!r} returns the same chunk id more than once: "
                f"{', '.join(repr(c) for c in sorted(repeated))}. A ranking is a list of distinct "
                "chunks, and scoring one with repeats in it would report a retriever's bug as "
                "a score. De-duplicate the ranking, keeping each chunk at its best position."
            )


def evaluate(
    qrels: Qrels,
    run: Mapping[str, Sequence[str]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    metrics: Sequence[str] = BUILTIN_METRIC_NAMES,
    warnings: WarningLog | None = None,
) -> dict[str, float]:
    """Score a whole run, averaged over queries.

    Only queries present in `qrels` **with at least one chunk judged relevant** are scored. A
    query with no relevant chunks cannot be got right or wrong, and including it as a zero
    would drag the mean down for a reason that has nothing to do with the retriever. `{}` and
    `{"c1": 0}` both mean that and are both excluded -- see `_is_scorable`.

    A query in the qrels that the run never answered scores zero, because that is a real
    failure rather than a missing measurement.

    Two shapes are rejected rather than scored. A `k` below 1 is not a cut-off, and a ranking
    that returns the same chunk id twice is a broken retriever rather than a result.

    `metrics` is resolved through `METRICS` -- the six built-ins by default, or any mix of
    those and registered custom metrics. **A metric that raises is left out of the result
    entirely, not scored as zero**: `RunResult.has(name)` exists precisely so a metric nobody
    successfully computed reads as absent rather than as a measured 0.0, which is what let a
    configuration with perfect recall score 0/100 on `composite()` the one time this was
    gotten wrong. Pass `warnings` (a `WarningLog`) to have the failure recorded rather than
    silently dropped -- `Runner.run_one` does.
    """
    unknown = set(metrics) - set(METRICS.names())
    if unknown:
        raise ValueError(
            f"unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(METRICS.names())}"
        )

    _check_ks(ks)

    scored = [query_id for query_id, judgements in qrels.items() if _is_scorable(judgements)]
    if not scored:
        return {}

    _check_no_duplicates(run, scored)

    results: dict[str, float] = {}
    for metric in metrics:
        instance = METRICS.create(metric)
        try:
            per_k = {
                k: sum(instance.evaluate(qrels[qid], run.get(qid, ()), k) for qid in scored)
                / len(scored)
                for k in ks
            }
        except Exception as error:  # a custom metric's bug must not take the whole run down
            if warnings is not None:
                warnings.add(
                    WarningCode.METRIC_FAILED,
                    f"the {metric!r} metric raised {error!r} and was left out of this run's "
                    "results rather than silently scoring zero",
                    severity=Severity.CAUTION,
                    stage="score",
                    subject=metric,
                )
            continue
        results.update({f"{metric}@{k}": value for k, value in per_k.items()})
    return results


def per_query(
    qrels: Qrels,
    run: Mapping[str, Sequence[str]],
    metric: str,
    k: int,
) -> dict[str, float]:
    """One metric for every query separately.

    The input to paired significance testing, and to finding the questions where two
    configurations actually disagree. `metric` is resolved through `METRICS`, so a
    registered custom metric works here exactly like a built-in one.

    Which questions appear, and which inputs are refused, match `evaluate()` exactly: a
    question with nothing judged relevant is left out, and a k below 1 or a ranking with a
    repeated chunk id raises.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}. Available: {', '.join(METRICS.names())}")
    _check_ks([k])
    scored = [query_id for query_id, judgements in qrels.items() if _is_scorable(judgements)]
    _check_no_duplicates(run, scored)

    instance = METRICS.create(metric)
    return {
        query_id: instance.evaluate(qrels[query_id], run.get(query_id, ()), k)
        for query_id in scored
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
        if _is_scorable(judgements)
        for rank in [rank_of_first_relevant(judgements, run.get(query_id, ()), k)]
        if rank is not None
    ]
    answered = sum(1 for judgements in qrels.values() if _is_scorable(judgements))
    if not ranks:
        return None, answered
    return sum(ranks) / len(ranks), answered - len(ranks)


def available_metrics() -> tuple[str, ...]:
    return tuple(METRICS.names())

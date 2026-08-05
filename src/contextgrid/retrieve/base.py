"""Retrieval strategies: how the index is used, as distinct from what the index is.

The index and the strategy are two different decisions, and conflating them is why "should I
use agentic RAG?" is unanswerable in practice. `dense`, `bm25`, `faiss:hnsw` and `pgvector` are
*stores*: where the vectors live and how a single search is executed. A retrieval strategy is
what sits on top -- how many searches happen, who decides what to search for, and whether the
answer to one search changes the next.

Keeping them apart means the interesting question becomes a cell in a grid rather than a
rewrite: *does agentic retrieval beat plain search on my pgvector index, and is it worth the
model calls?* Nobody can currently answer that about their own corpus, because measuring it
means building both and no tool builds both.

The protocol is deliberately tiny. A strategy is handed a `Searcher` -- a function that runs
one search against whatever index the configuration chose -- and returns ranked chunk ids. It
never sees the index type, so every strategy works with every store, and adding a store does
not touch any strategy.

**Every strategy that costs model calls reports them.** A strategy that quietly makes four
calls per query looks identical on a recall chart to one that makes none, and the entire point
of this package is that those two things are not the same.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from contextgrid.index.base import Scored

#: Runs one search. Given the query text and how many results to return, hands back a ranked
#: list. The strategy neither knows nor cares whether that was BM25, HNSW or Postgres.
Searcher = Callable[[str, int], Sequence[Scored]]


@dataclass(slots=True)
class RetrievalTrace:
    """What a strategy actually did, for the columns a recall number cannot carry.

    Two strategies with the same recall and wildly different `model_calls` are a decision, not
    a tie. Without this the leaderboard would present them as equivalent.
    """

    searches: int = 0
    model_calls: int = 0
    queries: list[str] = field(default_factory=list)
    #: Free-form, for strategies with something specific to say -- how many rounds an iterative
    #: search took before it stopped finding anything new, for instance.
    notes: dict[str, object] = field(default_factory=dict)

    def record_search(self, query: str) -> None:
        self.searches += 1
        self.queries.append(query)

    def record_model_call(self, count: int = 1) -> None:
        self.model_calls += count

    def merge(self, other: RetrievalTrace) -> None:
        self.searches += other.searches
        self.model_calls += other.model_calls
        self.queries.extend(other.queries)
        self.notes.update(other.notes)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Turns a question into ranked chunks, using whatever index it was given."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def uses_model(self) -> bool:
        """True when this strategy calls a language model per query.

        Read by the runner, which refuses to start a sweep containing one of these without a
        spending limit -- a strategy that decides its own number of calls has no upper bound
        anybody can eyeball.
        """
        ...

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
    ) -> list[Scored]:
        """Rank chunks for one question.

        `query` is the question as asked. `queries` is what the transform axis made of it --
        usually one string, sometimes several. A strategy is free to ignore `queries` and work
        from `query` alone, and an agentic one will.
        """
        ...


def fuse(results: Sequence[Sequence[Scored]], k: int) -> list[Scored]:
    """Combine several ranked lists into one, by rank.

    Reciprocal rank fusion rather than score averaging, and that is not a detail. A cosine
    similarity from one query and a cosine similarity from another are not on the same scale,
    however similar the numbers look -- averaging them lets whichever query happened to produce
    larger magnitudes dominate a result it did not earn. Ranks have no such problem.
    """
    from contextgrid.index.base import top_k
    from contextgrid.index.hybrid import reciprocal_rank_fusion

    if len(results) == 1:
        return list(results[0])[:k]
    return top_k(reciprocal_rank_fusion(results), k)

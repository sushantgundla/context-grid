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

A strategy that wants to *read* what it found -- relevance feedback, pseudo-relevance
expansion, anything that decides its next move from the text of a hit rather than just its id
-- is handed a second, equally narrow thing: `Lookup`. It resolves a chunk id to the `Chunk`
behind it and nothing else. Neither `Searcher` nor `Lookup` can reach the index itself, an
embedder, or another chunk's text -- a strategy can search, and it can read what came back, and
that is the entire surface.

**Every strategy that costs model calls reports them.** A strategy that quietly makes four
calls per query looks identical on a recall chart to one that makes none, and the entire point
of this package is that those two things are not the same.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from contextgrid.core.documents import Chunk
from contextgrid.evalset.llm import LLMError
from contextgrid.index.base import Scored


def needs_model_error(name: str) -> LLMError:
    """The one message a model-backed strategy raises when it was handed no model.

    Built in one place so `get_retriever` and the strategy itself cannot drift apart, and so
    the list of alternatives is read off the registry rather than typed out twice.

    It names `run.model` rather than `get_retriever`, because a config file is how almost
    everybody reaches this axis and an error naming a Python function they never call is an
    error they cannot act on.
    """
    from contextgrid.retrieve import model_free_retrievers

    return LLMError(
        f"the {name!r} retrieval strategy needs a model. Set `run.model` in your config, or "
        f"use one of the model-free strategies: {', '.join(model_free_retrievers())}"
    )

#: Runs one search. Given the query text and how many results to return, hands back a ranked
#: list. The strategy neither knows nor cares whether that was BM25, HNSW or Postgres.
Searcher = Callable[[str, int], Sequence[Scored]]

#: Reads one already-retrieved chunk by id. Backed by `BuiltPipeline.chunk_by_id()`, so it
#: returns exactly what a strategy would expect a `Searcher` hit to mean: the *retrievable*
#: chunk for a plain id, or the wider *presentation* passage for an id a sibling-merge produced
#: -- whichever one the hit actually stands for. Returns `None` for an id it does not
#: recognise, which should not normally happen for an id a `Searcher` call just returned, but a
#: strategy must not assume it never will.
#:
#: A strategy can only ever look up a chunk it already has the id for -- there is no way to
#: enumerate or browse through `Lookup`, which is what keeps it from being the index in
#: disguise.
Lookup = Callable[[str], "Chunk | None"]


def _no_lookup(chunk_id: str) -> Chunk | None:
    """The default `lookup`: every call site written before this parameter existed.

    Defaulting to "nothing found" rather than making `lookup` required means the three
    strategies that have no use for chunk text, and every test that calls `.retrieve(...)`
    directly, keep compiling and behaving exactly as they did. Only a strategy that actually
    reads text -- and the pipeline, which always passes a real one -- needs to know this
    parameter is there at all.
    """
    del chunk_id
    return None


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

        Read by the runner, which warns before starting a sweep containing one of these with
        no spending limit -- a strategy that decides its own number of calls has no upper bound
        anybody can eyeball in advance. A warning rather than a refusal, because it is the
        user's money and they may well mean it; `budget_usd` is the ceiling that actually
        stops it.

        Also read by `get_retriever`, which hands such a strategy the configured model rather
        than letting it build its own, and **refuses outright** when there is no model to hand
        over. A strategy that builds its own ignores `run.model` and spends money nothing can
        meter, so the configuration is costed at zero and `budget_usd` cannot stop it.
        """
        ...

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
        lookup: Lookup = _no_lookup,
    ) -> list[Scored]:
        """Rank chunks for one question.

        `query` is the question as asked. `queries` is what the transform axis made of it --
        usually one string, sometimes several. A strategy is free to ignore `queries` and work
        from `query` alone, and an agentic one will.

        `lookup` reads the text behind a `chunk_id` a `searcher` call returned -- for a
        strategy that decides its next search from what the last one found (relevance
        feedback, pseudo-relevance expansion) rather than from ids and scores alone. Most
        strategies have no use for it and can ignore the parameter entirely.
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

"""The retrieval strategies that need no language model.

`simple` is the baseline every other strategy has to beat, and it is not a placeholder: most
of the clever ones do not clear their own cost, and that cannot be shown without the plain arm
on the same chart with the same latency and dollars beside it.

The rest add structure rather than intelligence -- more searches, wider nets, results fed back
into the next query. They cost latency and no model calls, which puts them in a genuinely
different bracket from anything agentic and makes them the honest middle of the axis.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.core.errors import ContextGridError
from contextgrid.index.base import Scored
from contextgrid.retrieve.base import RetrievalTrace, Searcher, fuse


class RetrievalError(ContextGridError, ValueError):
    """A retrieval strategy was configured in a way that cannot work."""


@dataclass(frozen=True, slots=True)
class SimpleRetrieval:
    """One search per query, fused if the transform produced several.

    Exactly what this package did before retrieval became an axis, extracted unchanged. It is
    the arm every other strategy is measured against, and on a great many corpora it wins --
    which is itself worth publishing, because the field's default advice assumes otherwise.
    """

    name: ClassVar[str] = "simple"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
    ) -> list[Scored]:
        del query
        results = []
        for text in queries:
            trace.record_search(text)
            results.append(searcher(text, k))
        return fuse(results, k)


@dataclass(frozen=True, slots=True)
class WidenedRetrieval:
    """Search deeper than asked for, then cut back.

    Free recall, sometimes. Asking the index for `k * factor` and returning the top `k` changes
    nothing on its own -- but it changes a great deal once a reranker is in the pipeline, and it
    is the cheapest way to find out whether the retriever's ordering or its *reach* is what is
    limiting a configuration.

    Costs a little index time and no model calls, which makes it the first thing to try before
    reaching for anything that bills per query.
    """

    factor: int = 4

    name: ClassVar[str] = "widened"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise RetrievalError(f"widened factor must be at least 1, got {self.factor}")

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
    ) -> list[Scored]:
        del query
        depth = k * self.factor
        results = []
        for text in queries:
            trace.record_search(text)
            results.append(searcher(text, depth))
        trace.notes["depth"] = depth
        return fuse(results, k)


#: Words too common to be worth a search of their own.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "your",
    }
)

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class DecomposedRetrieval:
    """Split the question into parts and search each, then fuse.

    A question like "what is the refund window and does it cover digital goods?" has two
    answers, usually in two different chunks. One search ranks whichever half the embedding
    happened to favour and the other half is simply lost.

    Splitting is deliberately mechanical -- on conjunctions and clause punctuation, with a
    minimum length so fragments do not become searches of their own. A model would split it
    better and would cost a call per query; this arm exists to show how much of that gain is
    available for nothing, which is the comparison the `agentic` arm needs to be judged against.
    """

    #: Meaningful words -- stopwords removed -- a fragment needs before it earns a search of
    #: its own. Two, because "notice period" and "digital goods" are exactly the fragments this
    #: is for, and a floor of three drops both.
    min_words: int = 2
    max_parts: int = 4

    name: ClassVar[str] = "decomposed"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    SPLITTERS: ClassVar[tuple[str, ...]] = (
        r"\band\b",
        r"\bor\b",
        r"\balso\b",
        r"[;?]",
        r",\s+(?=and|or|what|how|when|where|why|who|does|is|are)",
    )

    def __post_init__(self) -> None:
        if self.max_parts < 1:
            raise RetrievalError(f"max_parts must be at least 1, got {self.max_parts}")

    def parts(self, query: str) -> list[str]:
        pieces = re.split("|".join(self.SPLITTERS), query, flags=re.IGNORECASE)

        # The whole question always leads. Decomposition is meant to add recall, not replace
        # the search that was already working.
        kept = [query]
        seen = {_signature(query)}
        for piece in pieces:
            fragment = piece.strip(" ,;?.")
            signature = _signature(fragment)
            # A trailing "?" splits a one-part question into itself, and searching the same
            # words twice costs a round trip to buy nothing.
            if signature in seen or len(_meaningful_words(fragment)) < self.min_words:
                continue
            seen.add(signature)
            kept.append(fragment)
            if len(kept) == self.max_parts:
                break
        return kept

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
    ) -> list[Scored]:
        del queries  # decomposition works from the question as asked
        results = []
        for text in self.parts(query):
            trace.record_search(text)
            results.append(searcher(text, k))
        trace.notes["parts"] = len(results)
        return fuse(results, k)


def _signature(text: str) -> tuple[str, ...]:
    """What makes two fragments the same search: the words that carry meaning, in order."""
    return tuple(_meaningful_words(text))


def _meaningful_words(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]

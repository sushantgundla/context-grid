"""Rerankers, and the parameter that actually matters about them.

Half the advice in this field is "use a reranker", almost none of it says how deep a candidate
list to feed one, and that second question is where most of the effect lives. A reranker over
the top 10 can only reorder what the retriever already found; over the top 100 it can rescue
evidence the retriever ranked 47th. The cost scales with the depth and the benefit does not,
so somewhere on that curve is the right answer for a given corpus.

The candidate-depth curve is one of the most useful things this tool can produce and nobody
publishes it. It falls out of putting `candidates` on the grid as an ordinary axis.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from contextgrid.core.documents import Chunk
from contextgrid.index.base import Scored, top_k

_WORD = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Reranker(Protocol):
    """Reorders a candidate list using the query and the passage together.

    The difference from a retriever is that a reranker sees both at once, so it can judge
    whether a passage answers *this* question rather than whether it is nearby in a vector
    space. That is why it helps, and why it costs more per candidate.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def rerank(self, query: str, candidates: Sequence[Chunk], k: int) -> list[Scored]: ...


@dataclass(frozen=True, slots=True)
class NoReranker:
    """Keeps the retriever's order. The arm every reranker has to beat.

    Not a placeholder. Half of "use a reranker" advice is untested, and the honest comparison
    needs the baseline on the same chart with the same latency and cost columns.
    """

    name: ClassVar[str] = "none"
    version: ClassVar[str] = "1"

    def rerank(self, query: str, candidates: Sequence[Chunk], k: int) -> list[Scored]:
        del query
        # Descending scores preserve the incoming order through any later sort.
        return [
            Scored(chunk.id, float(len(candidates) - position))
            for position, chunk in enumerate(candidates[:k])
        ]


@dataclass(frozen=True, slots=True)
class LexicalOverlapReranker:
    """Scores a passage by how much of the query it actually contains.

    A cross-encoder without the encoder: what fraction of the query's words appear in the
    passage, divided by length so a long passage cannot win by containing everything.

    Weak, free, and genuinely useful as the floor. A neural reranker that costs real money and
    beats this by 0.01 has told you something important about whether to deploy it.
    """

    length_penalty: float = 0.25

    name: ClassVar[str] = "lexical"
    version: ClassVar[str] = "1"

    def rerank(self, query: str, candidates: Sequence[Chunk], k: int) -> list[Scored]:
        terms = {word.lower() for word in _WORD.findall(query)}
        if not terms:
            return NoReranker().rerank(query, candidates, k)

        scores: dict[str, float] = {}
        for chunk in candidates:
            words = [word.lower() for word in _WORD.findall(chunk.text)]
            if not words:
                scores[chunk.id] = 0.0
                continue
            matched = sum(1 for word in words if word in terms)
            coverage = len({word for word in words if word in terms}) / len(terms)
            density = matched / (len(words) ** self.length_penalty)
            scores[chunk.id] = coverage + 0.1 * density

        return top_k(scores, k)


@dataclass(frozen=True, slots=True)
class MMRReranker:
    """Maximal marginal relevance: relevance minus similarity to what is already chosen.

    The fix for a top-5 that is five near-copies of the same paragraph. Overlapping chunks
    make a leaderboard look fine -- the evidence really is retrieved, five times -- while the
    generator sees one fact spread across the whole context window.

    `diversity` at 0 keeps the original order; at 1 it picks the most different passage
    available regardless of relevance.
    """

    diversity: float = 0.3

    name: ClassVar[str] = "mmr"
    version: ClassVar[str] = "1"

    def rerank(self, query: str, candidates: Sequence[Chunk], k: int) -> list[Scored]:
        del query
        remaining = list(candidates)
        chosen: list[Chunk] = []
        scores: dict[str, float] = {}

        # Relevance is the incoming rank, since MMR reorders rather than rescoring.
        relevance = {
            chunk.id: 1.0 - position / max(len(candidates), 1)
            for position, chunk in enumerate(candidates)
        }

        while remaining and len(chosen) < k:
            best = max(
                remaining,
                key=lambda chunk: (
                    (1 - self.diversity) * relevance[chunk.id]
                    - self.diversity * _max_similarity(chunk, chosen)
                ),
            )
            scores[best.id] = float(k - len(chosen))
            chosen.append(best)
            remaining.remove(best)

        return [Scored(chunk.id, scores[chunk.id]) for chunk in chosen]


def _max_similarity(chunk: Chunk, chosen: Sequence[Chunk]) -> float:
    """How much this passage repeats something already selected, by word overlap."""
    if not chosen:
        return 0.0
    words = {word.lower() for word in _WORD.findall(chunk.text)}
    if not words:
        return 0.0

    best = 0.0
    for other in chosen:
        other_words = {word.lower() for word in _WORD.findall(other.text)}
        if not other_words:
            continue
        overlap = len(words & other_words) / len(words | other_words)
        best = max(best, overlap)
    return best

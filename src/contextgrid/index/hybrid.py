"""Hybrid search: combining a dense and a sparse index.

Two fusion methods, because they behave differently and the difference matters.

**Reciprocal rank fusion** ignores scores entirely and uses only positions. That makes it
robust -- there is no need to make a cosine similarity and a BM25 score comparable, which
they are not -- and it means a result the dense side ranked first with a score of 0.31
counts exactly as much as one it ranked first with 0.98.

**Weighted score fusion** keeps the magnitudes, after normalising each side to a common
range. It can express "the dense side is usually right, so lean on it", which RRF cannot,
and it is more sensitive to one side producing an outlier.

Which wins is corpus-dependent, which is the entire premise of this tool, so both are here
and neither is the default by assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.embed.base import Vectors
from contextgrid.index.base import Index, Scored, top_k


class FusionError(ContextGridError, ValueError):
    """A hybrid index was configured in a way that cannot combine its two sides."""


@dataclass(slots=True)
class HybridIndex:
    """Search a dense and a sparse index, then fuse the two result lists.

    `candidates` is how deep each side is read before fusing. Too shallow and a result the
    dense side ranked 30th can never be rescued by the sparse side agreeing with it, which
    is exactly the case hybrid search exists to catch.
    """

    dense: Index
    sparse: Index
    fusion: str = "rrf"
    #: RRF's smoothing constant. 60 is the value from the original paper and is rarely tuned;
    #: it controls how quickly the contribution of lower ranks decays.
    rrf_k: int = 60
    #: Weight on the dense side for weighted fusion. 0.5 is an even split.
    alpha: float = 0.5
    candidates: int = 100

    name: ClassVar[str] = "hybrid"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = True
    is_exact: ClassVar[bool] = True

    _size: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fusion not in {"rrf", "weighted"}:
            raise FusionError(f"unknown fusion {self.fusion!r}. Choose 'rrf' or 'weighted'.")
        if not 0.0 <= self.alpha <= 1.0:
            raise FusionError(f"alpha must be between 0 and 1, got {self.alpha}")
        if self.candidates < 1:
            raise FusionError(f"candidates must be at least 1, got {self.candidates}")

    def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None:
        self.dense.build(chunks, vectors)
        self.sparse.build(chunks, vectors)
        self._size = len(chunks)

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        depth = max(self.candidates, k)
        dense_hits = self.dense.search(text, vector, depth)
        sparse_hits = self.sparse.search(text, vector, depth)

        if self.fusion == "rrf":
            scores = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self.rrf_k)
        else:
            scores = weighted_fusion(dense_hits, sparse_hits, alpha=self.alpha)
        return top_k(scores, k)

    def size_bytes(self) -> int:
        return self.dense.size_bytes() + self.sparse.size_bytes()

    def __len__(self) -> int:
        return self._size


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Scored]], *, k: int = 60
) -> dict[str, float]:
    """Fuse ranked lists by position: each contributes `1 / (k + rank)`.

    Scores are never compared across lists, only ranks -- which is the point. A cosine
    similarity of 0.31 and a BM25 score of 14.2 are not on the same scale and no amount of
    normalisation makes them mean the same thing.
    """
    if k < 1:
        raise FusionError(f"rrf_k must be at least 1, got {k}")

    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, scored in enumerate(results, start=1):
            scores[scored.chunk_id] = scores.get(scored.chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_fusion(
    dense: Sequence[Scored], sparse: Sequence[Scored], *, alpha: float = 0.5
) -> dict[str, float]:
    """Fuse by score, after min-max normalising each side to [0, 1].

    A document missing from one side scores zero there rather than being dropped, so a
    result both sides half-liked can still beat one that only appeared in a single list.
    """
    dense_scores = _min_max(dense)
    sparse_scores = _min_max(sparse)

    scores: dict[str, float] = {}
    for chunk_id in set(dense_scores) | set(sparse_scores):
        scores[chunk_id] = alpha * dense_scores.get(chunk_id, 0.0) + (1 - alpha) * (
            sparse_scores.get(chunk_id, 0.0)
        )
    return scores


def _min_max(results: Sequence[Scored]) -> dict[str, float]:
    """Scale one result list to [0, 1]. A list where every score is equal maps to 1.0."""
    if not results:
        return {}
    values = [scored.score for scored in results]
    lowest, highest = min(values), max(values)
    if highest == lowest:
        return {scored.chunk_id: 1.0 for scored in results}
    span = highest - lowest
    return {scored.chunk_id: (scored.score - lowest) / span for scored in results}

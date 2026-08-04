"""Exact dense search.

Brute force, which for a corpus this tool is meant for is both fast enough and the right
default. It is also the reference every approximate index has to be judged against: an ANN
configuration that returns 92% of what exact search would have found is a perfectly good
trade, but only if somebody measured the 8%.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.embed.base import Vectors, normalise
from contextgrid.index.base import Scored, top_k


class IndexBuildError(ContextGridError, ValueError):
    """An index was built or queried inconsistently."""


@dataclass(slots=True)
class ExactDenseIndex:
    """Cosine or dot-product search over every vector, with no approximation.

    `metric="cosine"` normalises both sides, so it is unaffected by vector magnitude.
    `metric="dot"` is not, which is correct for models trained that way and wrong for models
    that were not -- another quiet way a comparison becomes unfair.
    """

    metric: str = "cosine"

    name: ClassVar[str] = "dense"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = True
    is_exact: ClassVar[bool] = True

    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.metric not in {"cosine", "dot"}:
            raise IndexBuildError(f"unknown metric {self.metric!r}. Choose 'cosine' or 'dot'.")

    def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None:
        if vectors is None:
            raise IndexBuildError(
                f"the {self.name!r} index needs vectors. Give it an embedder, or use a "
                "sparse index that works on text alone."
            )
        if len(chunks) != vectors.shape[0]:
            raise IndexBuildError(
                f"got {len(chunks)} chunks and {vectors.shape[0]} vectors. "
                "One of them is out of step with the other, and every score would be wrong."
            )

        self._ids = [chunk.id for chunk in chunks]
        matrix = np.asarray(vectors, dtype=np.float32)
        self._matrix = normalise(matrix) if self.metric == "cosine" else matrix

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        del text
        if vector is None:
            raise IndexBuildError(f"the {self.name!r} index needs a query vector")
        if not self._ids:
            return []

        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._matrix.shape[1]:
            raise IndexBuildError(
                f"query has {query.shape[1]} dimensions but the index was built with "
                f"{self._matrix.shape[1]}. The query and the documents were embedded by "
                "different models."
            )
        if self.metric == "cosine":
            query = normalise(query)

        scores = (self._matrix @ query.T).ravel()
        return top_k(dict(zip(self._ids, (float(s) for s in scores), strict=True)), k)

    def size_bytes(self) -> int:
        return int(self._matrix.nbytes)

    def __len__(self) -> int:
        return len(self._ids)

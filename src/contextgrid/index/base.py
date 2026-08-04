"""The search interface, and what every index has to report about itself."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contextgrid.core.documents import Chunk
from contextgrid.embed.base import Vectors


@dataclass(frozen=True, slots=True)
class Scored:
    """One result: which chunk, and how well it did."""

    chunk_id: str
    score: float


@runtime_checkable
class Index(Protocol):
    """Holds chunks and finds the ones most like a query.

    `search` takes both the query text and its vector, because sparse and dense indexes need
    different things and a hybrid needs both. An index ignores whichever it does not use.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def needs_vectors(self) -> bool:
        """False for indexes that work on text alone, so the embedder can be skipped."""
        ...

    @property
    def is_exact(self) -> bool:
        """False for approximate search.

        An approximate index must be compared against its exact twin before its numbers mean
        anything. Tuning `efSearch` without knowing what recall it cost is guessing, and it
        is guessing in the direction that looks good.
        """
        ...

    def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None: ...

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]: ...

    def size_bytes(self) -> int:
        """Roughly how much memory this index occupies.

        One-time cost people forget entirely when comparing a 1536-dimension model against
        a 384-dimension one.
        """
        ...


def top_k(scores: dict[str, float], k: int) -> list[Scored]:
    """The k highest-scoring entries, ties broken by chunk id.

    Deterministic tie-breaking matters more than it looks: two chunks with identical scores
    would otherwise swap places between runs, and a leaderboard that moves when nothing
    changed destroys trust in every other number on it.
    """
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Scored(chunk_id, score) for chunk_id, score in ordered[:k]]

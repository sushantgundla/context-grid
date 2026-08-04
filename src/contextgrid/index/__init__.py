"""Indexes, and the registry of them."""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.index.base import Index, Scored, top_k
from contextgrid.index.dense import ExactDenseIndex, IndexBuildError
from contextgrid.index.hybrid import (
    FusionError,
    HybridIndex,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from contextgrid.index.sparse import BM25Index

INDEXES: Registry[Index] = Registry(family="index")

INDEXES.register("dense", shorthand="metric", doc="Exact dense search. The reference arm.")(
    ExactDenseIndex
)
INDEXES.register("bm25", shorthand="k1", doc="Okapi BM25. No model, no vectors.")(BM25Index)


def _hybrid(
    fusion: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
    candidates: int = 100,
    metric: str = "cosine",
    k1: float = 1.5,
    b: float = 0.75,
) -> Index:
    """Dense and sparse together, fused by rank or by normalised score."""
    return HybridIndex(
        dense=ExactDenseIndex(metric=metric),
        sparse=BM25Index(k1=k1, b=b),
        fusion=fusion,
        rrf_k=rrf_k,
        alpha=alpha,
        candidates=candidates,
    )


INDEXES.register("hybrid", shorthand="fusion", doc="Dense plus BM25, fused by rank or score.")(
    _hybrid
)


def get_index(spec: str | Index) -> Index:
    """Resolve an index from a spec like `hybrid:weighted,alpha=0.7`, or pass one through."""
    return INDEXES.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "INDEXES",
    "BM25Index",
    "ExactDenseIndex",
    "FusionError",
    "HybridIndex",
    "Index",
    "IndexBuildError",
    "Scored",
    "get_index",
    "reciprocal_rank_fusion",
    "top_k",
    "weighted_fusion",
]

"""Indexes, and the registry of them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextgrid.core.registry import Registry
from contextgrid.index.base import Index, Scored, top_k
from contextgrid.index.dense import ExactDenseIndex, IndexBuildError
from contextgrid.index.hybrid import (
    FusionError,
    HybridIndex,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from contextgrid.index.quantize import (
    BinaryCodec,
    CompressionReport,
    ProductCodec,
    Quantization,
    QuantizationError,
    QuantizedDenseIndex,
    ScalarCodec,
    recall_against_exact,
)
from contextgrid.index.sparse import BM25Index

if TYPE_CHECKING:  # Imported for the public names only; both need optional dependencies.
    from contextgrid.index.ann import FaissIndex, USearchIndex
    from contextgrid.index.pgvector import PgVectorIndex

INDEXES: Registry[Index] = Registry(family="index")

INDEXES.register("dense", shorthand="metric", doc="Exact dense search. The reference arm.")(
    ExactDenseIndex
)
INDEXES.register("bm25", shorthand="k1", doc="Okapi BM25. No model, no vectors.")(BM25Index)
INDEXES.register(
    "quantized",
    shorthand="scheme",
    doc="Compressed dense search: scalar, product or binary, with optional rescoring.",
)(QuantizedDenseIndex)


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

# Approximate search. Until these landed the chart this package most wanted to draw -- what
# did approximation actually cost you? -- had nothing to plot.
INDEXES.register_lazy(
    "faiss",
    module="contextgrid.index.ann",
    attr="FaissIndex",
    extra="index",
    # The distribution, not the import name -- `pip install faiss` fetches something else.
    package="faiss-cpu",
    shorthand="kind",
    doc="faiss: flat, hnsw, ivf or ivfpq. Four index types on one axis.",
)
INDEXES.register_lazy(
    "usearch",
    module="contextgrid.index.ann",
    attr="USearchIndex",
    extra="index",
    package="usearch",
    shorthand="dtype",
    # `f32/f16/i8`, matching `USearchIndex.DTYPES`. It said `f32/f16/i8/b1`, which is the one
    # dtype this index deliberately does not have: usearch's binary mode wants bit-packed
    # input and a Hamming metric, and `usearch:dtype=b1` has always raised. So the single
    # sentence `contextgrid plugins` prints about this index was pointing at the one setting
    # guaranteed to fail. `quantized:binary` is the arm that does binary properly.
    doc="usearch HNSW, with f32/f16/i8 storage. A second opinion on the same idea.",
)
INDEXES.register_lazy(
    "pgvector",
    module="contextgrid.index.pgvector",
    attr="PgVectorIndex",
    extra="pgvector",
    package="psycopg",
    shorthand="kind",
    doc="Postgres with pgvector: exact, hnsw or ivfflat. What people actually deploy on.",
)


def get_index(spec: str | Index) -> Index:
    """Resolve an index from a spec like `hybrid:weighted,alpha=0.7`, or pass one through."""
    return INDEXES.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "INDEXES",
    "BM25Index",
    "BinaryCodec",
    "CompressionReport",
    "ExactDenseIndex",
    "FaissIndex",
    "FusionError",
    "HybridIndex",
    "Index",
    "IndexBuildError",
    "PgVectorIndex",
    "ProductCodec",
    "Quantization",
    "QuantizationError",
    "QuantizedDenseIndex",
    "ScalarCodec",
    "Scored",
    "USearchIndex",
    "get_index",
    "recall_against_exact",
    "reciprocal_rank_fusion",
    "top_k",
    "weighted_fusion",
]

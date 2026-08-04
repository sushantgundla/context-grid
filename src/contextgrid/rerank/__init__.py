"""Rerankers, and the registry of them."""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.rerank.base import (
    LexicalOverlapReranker,
    MMRReranker,
    NoReranker,
    Reranker,
)

RERANKERS: Registry[Reranker] = Registry(family="reranker")

RERANKERS.register("none", doc="Keep the retriever's order. The arm every reranker must beat.")(
    NoReranker
)
RERANKERS.register(
    "lexical", shorthand="length_penalty", doc="Query-term coverage. Free, and the floor."
)(LexicalOverlapReranker)
RERANKERS.register(
    "mmr", shorthand="diversity", doc="Maximal marginal relevance. Fixes a top-5 of near-copies."
)(MMRReranker)

# Cross-encoders arrive with the extra that provides them.
for _name, _attr, _doc in [
    ("bge-reranker-base", "BgeRerankerBase", "BAAI bge-reranker-base cross-encoder."),
    ("bge-reranker-v2-m3", "BgeRerankerV2M3", "BAAI bge-reranker-v2-m3. Multilingual."),
    ("mxbai-rerank-base", "MxbaiRerankBase", "mixedbread mxbai-rerank-base."),
]:
    RERANKERS.register_lazy(
        _name,
        module="contextgrid.rerank.cross_encoder",
        attr=_attr,
        extra="rerank",
        package="sentence-transformers",
        doc=_doc,
    )

RERANKERS.register_lazy(
    "cohere-rerank",
    module="contextgrid.rerank.cohere",
    attr="CohereRerank",
    extra="llm",
    package="cohere",
    shorthand="model",
    doc="Cohere Rerank (bring your own key).",
)


def get_reranker(spec: str | Reranker | None) -> Reranker:
    """Resolve a reranker from a spec, or pass one through. `None` means no reranking."""
    if spec is None:
        return NoReranker()
    return RERANKERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "RERANKERS",
    "LexicalOverlapReranker",
    "MMRReranker",
    "NoReranker",
    "Reranker",
    "get_reranker",
]

"""Rerankers, and the registry of them."""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.rerank.base import (
    LexicalOverlapReranker,
    MMRReranker,
    NoReranker,
    Reranker,
)
from contextgrid.rerank.remote import LiteLLMReranker, RerankerError, TEIReranker

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

# Cross-encoders. The same two backends as the embedders: a TEI server for local models, and
# litellm for hosted ones. TEI needs no dependency at all -- it is reached over urllib -- so
# it is registered eagerly.
RERANKERS.register(
    "tei-rerank",
    shorthand="model",
    doc="A cross-encoder on a local TEI server. No key, no network, no extra dependency.",
)(TEIReranker)

RERANKERS.register_lazy(
    "litellm-rerank",
    module="contextgrid.rerank.remote",
    attr="LiteLLMReranker",
    extra="llm",
    package="litellm",
    shorthand="model",
    doc="A hosted reranker through litellm: Cohere, Jina, Voyage, AWS.",
)


def get_reranker(spec: str | Reranker | None) -> Reranker:
    """Resolve a reranker from a spec, or pass one through. `None` means no reranking."""
    if spec is None:
        return NoReranker()
    return RERANKERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "RERANKERS",
    "LexicalOverlapReranker",
    "LiteLLMReranker",
    "MMRReranker",
    "NoReranker",
    "Reranker",
    "RerankerError",
    "TEIReranker",
    "get_reranker",
]

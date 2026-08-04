"""Embedders, and the registry of them."""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.embed.adapter import (
    AdaptedEmbedder,
    AdapterError,
    AdapterReport,
    LinearAdapter,
    Triplet,
    fit_adapter,
    mine_triplets,
    split_triplets,
)
from contextgrid.embed.base import (
    Embedder,
    EmbeddingResult,
    Vectors,
    normalise,
    truncate,
)
from contextgrid.embed.local import HashEmbedder, TfidfEmbedder, TokenCountEmbedder
from contextgrid.embed.prefixes import Prefixes, for_model
from contextgrid.embed.remote import EmbedderError, LiteLLMEmbedder, TEIEmbedder

EMBEDDERS: Registry[Embedder] = Registry(family="embedder")

EMBEDDERS.register("hash", shorthand="dimensions", doc="Hashed bag of words. No model needed.")(
    HashEmbedder
)
EMBEDDERS.register(
    "tfidf", shorthand="max_features", doc="Classical TF-IDF over the corpus vocabulary."
)(TfidfEmbedder)
EMBEDDERS.register("length", doc="Text length in one dimension. A chance-level control.")(
    TokenCountEmbedder
)

# Real models. Two backends cover the field between them: litellm for anything hosted, TEI for
# anything run locally. Both take the model name as their shorthand, so a sweep across real
# models is one line -- `embedder: [tei:bge-base-en-v1.5, litellm:text-embedding-3-small]`.
EMBEDDERS.register_lazy(
    "litellm",
    module="contextgrid.embed.remote",
    attr="LiteLLMEmbedder",
    extra="llm",
    package="litellm",
    shorthand="model",
    doc="Any hosted model, through litellm. Bring your own key.",
)

# No package= at all: the TEI backend is built on urllib, so a running server plus a bare
# `pip install context-grid` is enough for real embeddings.
EMBEDDERS.register(
    "tei",
    shorthand="model",
    doc="A local text-embeddings-inference server. No key, no network, no extra dependency.",
)(TEIEmbedder)


def get_embedder(spec: str | Embedder) -> Embedder:
    """Resolve an embedder from a spec like `hash:512`, or pass an instance through."""
    return EMBEDDERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "EMBEDDERS",
    "AdaptedEmbedder",
    "AdapterError",
    "AdapterReport",
    "Embedder",
    "EmbedderError",
    "EmbeddingResult",
    "HashEmbedder",
    "LinearAdapter",
    "LiteLLMEmbedder",
    "Prefixes",
    "TEIEmbedder",
    "TfidfEmbedder",
    "TokenCountEmbedder",
    "Triplet",
    "Vectors",
    "fit_adapter",
    "for_model",
    "get_embedder",
    "mine_triplets",
    "normalise",
    "split_triplets",
    "truncate",
]

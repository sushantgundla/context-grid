"""Embedders, and the registry of them."""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.embed.base import (
    Embedder,
    EmbeddingResult,
    Vectors,
    normalise,
    truncate,
)
from contextgrid.embed.local import HashEmbedder, TfidfEmbedder, TokenCountEmbedder

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

# Real models arrive with the extras that provide them.
for _name, _attr, _doc in [
    ("bge-base-en-v1.5", "BgeBaseEn", "BAAI bge-base-en-v1.5 via ONNX. CPU-friendly."),
    ("e5-base-v2", "E5BaseV2", "intfloat/e5-base-v2, with query:/passage: prefixes."),
    ("all-MiniLM-L6-v2", "MiniLmL6", "The speed baseline."),
]:
    EMBEDDERS.register_lazy(
        _name,
        module="contextgrid.embed.onnx",
        attr=_attr,
        extra="embed",
        package="onnxruntime",
        doc=_doc,
    )

EMBEDDERS.register_lazy(
    "text-embedding-3-small",
    module="contextgrid.embed.openai",
    attr="OpenAISmall",
    extra="llm",
    package="openai",
    shorthand="dimensions",
    doc="OpenAI text-embedding-3-small (bring your own key).",
)


def get_embedder(spec: str | Embedder) -> Embedder:
    """Resolve an embedder from a spec like `hash:512`, or pass an instance through."""
    return EMBEDDERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "EMBEDDERS",
    "Embedder",
    "EmbeddingResult",
    "HashEmbedder",
    "TfidfEmbedder",
    "TokenCountEmbedder",
    "Vectors",
    "get_embedder",
    "normalise",
    "truncate",
]

"""Ingestion strategies, and the registry of them.

A chunker produces units where the thing indexed and the thing returned are the same. An
ingestion strategy deliberately breaks that identity -- index a sentence, return the paragraph;
index a question, return the passage that answers it.
"""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.ingest.base import (
    Ingested,
    IngestionContext,
    IngestionError,
    IngestionStrategy,
)
from contextgrid.ingest.generated import (
    ContextualIngestion,
    HypotheticalQuestionsIngestion,
    PropositionsIngestion,
    SummaryIngestion,
)
from contextgrid.ingest.structural import (
    HierarchicalIngestion,
    ParentDocumentIngestion,
    PlainIngestion,
    SentenceWindowIngestion,
)

INGESTERS: Registry[IngestionStrategy] = Registry(family="ingestion")

# Free: structure only, no model, no tokens. The arms the paid ones have to beat.
INGESTERS.register("plain", doc="Index the chunk, return the chunk. The baseline.")(PlainIngestion)
INGESTERS.register(
    "parent-document",
    shorthand="group",
    doc="Index small chunks, return the passage they came from.",
)(ParentDocumentIngestion)
INGESTERS.register(
    "sentence-window",
    shorthand="window",
    doc="Index one chunk, return it with its neighbours either side.",
)(SentenceWindowIngestion)
INGESTERS.register(
    "hierarchical",
    shorthand="group",
    doc="Index leaves; return the parent once enough siblings hit. Decides at query time.",
)(HierarchicalIngestion)

# Paid: a model call at index time, never again. A different bargain from anything per-query.
INGESTERS.register(
    "contextual",
    shorthand="model",
    doc="Prepend an LLM-written note on where the chunk sits, and index that.",
)(ContextualIngestion)
INGESTERS.register(
    "hypothetical-questions",
    shorthand="count",
    doc="Index the questions a chunk answers; return the chunk.",
)(HypotheticalQuestionsIngestion)
INGESTERS.register(
    "propositions",
    shorthand="count",
    doc="Index atomic facts; return the chunk they came from.",
)(PropositionsIngestion)
INGESTERS.register(
    "summary",
    shorthand="model",
    doc="Index a summary of each document; return the document. One call per document.",
)(SummaryIngestion)


def get_ingester(spec: str | IngestionStrategy | None) -> IngestionStrategy:
    """Resolve a strategy from a spec, or pass one through. `None` means plain chunking.

    No `llm` parameter, deliberately, and unlike `get_retriever` and `get_transform`: a strategy
    is handed its model through `IngestionContext` when `ingest()` is called, so building one
    with no model in sight is legitimate and is what every example on `/axes/ingestion` does.
    The refusal for a paid strategy with no model lives where the model is actually known --
    `pipeline.build`, and `contextgrid check` before it.
    """
    if spec is None:
        return PlainIngestion()
    return INGESTERS.create(spec) if isinstance(spec, str) else spec


def model_free_ingesters() -> tuple[str, ...]:
    """The strategies that never call a model, for "use one of these instead".

    Read off `uses_model` on each registered class rather than a hand-kept list, exactly as
    `retrieve.model_free_retrievers` does: a list typed out here goes stale the day a fifth paid
    strategy is registered, and cannot see one that arrived at runtime through `plugins:`.
    Loading a registration imports the class and never builds one.
    """
    free: list[str] = []
    for name in INGESTERS.names():
        try:
            factory = INGESTERS.registration(name).load()
        except Exception:  # pragma: no cover - an uninstallable plugin is neither, usefully
            continue
        if not getattr(factory, "uses_model", False):
            free.append(name)
    return tuple(free)


__all__ = [
    "INGESTERS",
    "ContextualIngestion",
    "HierarchicalIngestion",
    "HypotheticalQuestionsIngestion",
    "Ingested",
    "IngestionContext",
    "IngestionError",
    "IngestionStrategy",
    "ParentDocumentIngestion",
    "PlainIngestion",
    "PropositionsIngestion",
    "SentenceWindowIngestion",
    "SummaryIngestion",
    "get_ingester",
    "model_free_ingesters",
]

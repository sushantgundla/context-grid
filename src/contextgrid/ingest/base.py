"""Ingestion strategies: what goes into the index, and what comes back out of it.

The distinction this whole module rests on, and the one that took a wrong turn to find:

> **A chunker produces units where the thing indexed and the thing returned are the same.
> An ingestion strategy deliberately breaks that identity.**

Chunk size is a compromise nobody is happy with. Small chunks embed precisely -- a 128-token
passage about one thing has a vector that means one thing -- and they arrive at the generator
stripped of the context that made them make sense. Large chunks keep their context and embed
into mush, because a vector averaging six topics is close to nothing in particular.

Every strategy here is a way of refusing that compromise: index one thing, return another.

* `parent-document` indexes small chunks and returns the parent they came from.
* `sentence-window` indexes a sentence and returns the sentences around it.
* `hierarchical` indexes leaves and returns the parent once enough siblings have hit.
* `contextual` indexes the chunk with an LLM-written explanation of where it sits.
* `summary` indexes a summary and returns the document.
* `hypothetical-questions` indexes the questions a chunk answers, and returns the chunk.
* `propositions` indexes atomic facts, and returns the chunk they came from.

It is an axis nobody can currently sweep. Every one of these is a blog post with a favourable
anecdote attached, and choosing between them means building all seven.

**Both sides stay spans into the same parse**, which is what keeps this measurable. Gold
evidence resolves against the *retrievable* units -- the things a generator would actually be
handed -- so a strategy that returns bigger passages is scored on whether the answer is in what
came back, exactly like every other arm. A strategy that rewrites text for the index says so
with `offsets_exact=False` on the indexed side, and the retrievable side keeps its offsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.core.warnings import WarningLog


class IngestionError(ContextGridError, ValueError):
    """A source could not be ingested."""


def needs_a_model(strategy: object) -> bool:
    """Whether this strategy will refuse for want of a model when nothing supplies one.

    True only for a paid strategy whose spec named no model of its own. `contextual` alone is
    the case this exists for; `contextual:model=anthropic:claude-3-5-haiku` is somebody naming
    a provider deliberately, and refusing that would break a documented spec to fix a bug it
    never had -- the same line `retrieve.get_retriever` draws for `agentic:gpt-4o-mini`.

    Asked by `contextgrid check`, which has a config in front of it and can refuse before a
    document is read, and it has to agree with `_GeneratedIngestion._llm`, which is where the
    refusal actually happens.
    """
    if not getattr(strategy, "uses_model", False):
        return False
    from contextgrid.ingest.generated import DEFAULT_MODEL

    return getattr(strategy, "model", None) in (None, DEFAULT_MODEL)


def needs_model_error(name: str) -> Exception:
    """The one message a model-backed ingestion strategy raises when it was handed no model.

    Written once, in the shape `retrieve.base.needs_model_error` already uses, so the three
    places that can refuse -- `contextgrid check`, `contextgrid run`, and `pipeline.build` on
    the `cg.Lab` path -- cannot drift into three wordings for one problem. The list of
    alternatives is read off the registry rather than typed out.

    It names `run.model`, because a config file is how most people reach this axis, and an
    error naming a Python argument they never pass is an error they cannot act on.
    """
    from contextgrid.evalset.llm import LLMError
    from contextgrid.ingest import model_free_ingesters

    return LLMError(
        f"the {name!r} ingestion strategy needs a model. Set `run.model` in your config, or "
        f"use one of the model-free strategies: {', '.join(model_free_ingesters())}"
    )


@dataclass(slots=True)
class Ingested:
    """The two sides of an index, and the map between them.

    `indexed` is embedded and searched. `retrievable` is what a hit turns into: what gets
    reranked, handed to a generator, and scored. For plain chunking they are the same list and
    `parent_of` is empty.
    """

    indexed: list[Chunk]
    retrievable: list[Chunk]
    #: Indexed chunk id -> the retrievable chunk it stands for. Missing means "itself".
    parent_of: dict[str, str] = field(default_factory=dict)
    #: Wider passages a strategy may hand back *instead of* a retrievable unit, mapped to the
    #: units they cover.
    #:
    #: Kept apart from `retrievable` deliberately. A parent and its children are the same
    #: evidence at two granularities, and putting both in the scored set makes gold resolve to
    #: each of them -- so a question with one answer acquires two things to find and recall
    #: halves for a purely structural reason. Measured on this package's demo corpus:
    #: 1.86 relevant units per question against plain chunking's 1.00.
    #:
    #: So retrieval is scored on the units, and presentation shows the passage.
    presentation: dict[str, list[str]] = field(default_factory=dict)
    #: The wider passages themselves, by id, for reranking and generation.
    presented_chunks: dict[str, Chunk] = field(default_factory=dict)
    #: How many model calls building this cost, and what they were for.
    model_calls: int = 0
    notes: dict[str, object] = field(default_factory=dict)

    def resolve(self, indexed_id: str) -> str:
        return self.parent_of.get(indexed_id, indexed_id)

    def scored_ids(self, returned_id: str) -> list[str]:
        """What a returned id counts as, for scoring.

        A presentation passage counts as the units it covers, so showing a generator more
        context never changes what the retrieval was credited with finding.
        """
        return self.presentation.get(returned_id, [returned_id])

    @property
    def expansion(self) -> float:
        """Indexed units per retrievable unit.

        Above 1 means several vectors point at the same passage -- `hypothetical-questions`
        indexes four questions for one chunk -- which multiplies embedding cost and index size
        without multiplying what can be returned. Worth having on the chart beside the recall
        it bought.
        """
        return len(self.indexed) / len(self.retrievable) if self.retrievable else 0.0

    @classmethod
    def plain(cls, chunks: Sequence[Chunk]) -> Ingested:
        return cls(indexed=list(chunks), retrievable=list(chunks))


@runtime_checkable
class IngestionStrategy(Protocol):
    """Decides what is indexed and what a hit on it returns."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def uses_model(self) -> bool:
        """True when building the index costs model calls.

        Paid once at index time rather than per query, which is a genuinely different bargain
        from a query-time strategy and deserves a different column.
        """
        ...

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        """Turn one chunker's output into the two sides of an index."""
        ...


@dataclass(slots=True)
class IngestionContext:
    """What a strategy is allowed to reach for.

    Handed in rather than imported, so a strategy that needs a model gets one that the caller
    chose -- and a test can hand it a scripted one without a key or a network.
    """

    parses: dict[str, object] = field(default_factory=dict)
    warnings: WarningLog = field(default_factory=WarningLog)
    llm: object | None = None
    tokenizer: object | None = None

"""Ingestion strategies that cost nothing but arithmetic.

These four refuse the chunk-size compromise using structure alone: no model, no tokens, no
bill. That makes them the arms the paid strategies have to beat before anyone should pay for
one -- and on a great many corpora they are not beaten, which is worth knowing before spending
a model call per chunk.

Every one of them works by grouping the chunker's output. The chunker still decides where the
cuts fall; these decide which cut is embedded and which is handed back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.ingest.base import Ingested, IngestionContext, IngestionError


@dataclass(frozen=True, slots=True)
class PlainIngestion:
    """Index the chunk, return the chunk. The baseline every other strategy is judged against.

    Not a placeholder. The entire premise of this axis is that the small-versus-large
    compromise is worth escaping, and that premise is a claim -- one that has to be checked
    against the arm that simply accepts the compromise, on the same corpus, with the same
    questions and the same cost columns.
    """

    name: ClassVar[str] = "plain"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        del context
        return Ingested.plain(chunks)


@dataclass(frozen=True, slots=True)
class ParentDocumentIngestion:
    """Index small chunks, return the passage they came from.

    The oldest answer to the compromise and still the strongest free one. A 128-token chunk
    embeds precisely because it is about one thing; the 512-token passage around it is what the
    generator actually needs to answer from.

    `group` chunks are gathered into each parent. The chunker decides the small size, so
    `chunker: recursive:128` with `parent-document:4` indexes 128-token chunks and returns
    roughly 512 tokens of context.
    """

    group: int = 4

    name: ClassVar[str] = "parent-document"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.group < 2:
            raise IngestionError(
                f"parent-document needs to group at least 2 chunks, got {self.group}. "
                "A group of one is plain chunking under a different name."
            )

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        del context
        indexed: list[Chunk] = []
        retrievable: list[Chunk] = []
        parent_of: dict[str, str] = {}

        for group in _consecutive_groups(chunks, self.group):
            parent = _merge(group, suffix="parent")
            retrievable.append(parent)
            for chunk in group:
                indexed.append(chunk)
                parent_of[chunk.id] = parent.id

        return Ingested(indexed=indexed, retrievable=retrievable, parent_of=parent_of)


@dataclass(frozen=True, slots=True)
class SentenceWindowIngestion:
    """Index one chunk, return it with its neighbours.

    The sharpest form of the idea: the embedded unit is as small as the chunker will make it,
    and what comes back is that unit plus `window` chunks either side. Where
    `parent-document` returns a fixed passage whatever matched inside it, this one centres the
    returned context on the match -- so a hit at the end of a passage brings back what follows
    it rather than stopping dead.

    Windows overlap, which is intended. Two adjacent hits return overlapping context and the
    assembler deduplicates.
    """

    window: int = 2

    name: ClassVar[str] = "sentence-window"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.window < 1:
            raise IngestionError(
                f"sentence-window needs a window of at least 1, got {self.window}. "
                "A window of zero is plain chunking under a different name."
            )

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        del context
        by_document = _by_document(chunks)

        indexed: list[Chunk] = []
        retrievable: list[Chunk] = []
        parent_of: dict[str, str] = {}

        for group in by_document.values():
            for position, chunk in enumerate(group):
                low = max(0, position - self.window)
                high = min(len(group), position + self.window + 1)
                window = _merge(group[low:high], suffix=f"window{self.window}")

                indexed.append(chunk)
                retrievable.append(window)
                parent_of[chunk.id] = window.id

        return Ingested(indexed=indexed, retrievable=retrievable, parent_of=parent_of)


@dataclass(frozen=True, slots=True)
class HierarchicalIngestion:
    """Index leaves, and return the parent once enough of its children have hit.

    The one strategy here that decides at *query* time. `parent-document` always returns the
    parent, which wastes context when a single leaf held the whole answer.  This returns the
    leaf when one leaf matched and the parent when several siblings did -- on the reasoning
    that several hits in one passage mean the passage is the answer, not any one line of it.

    `threshold` is the fraction of a parent's children that must hit before it merges.
    """

    group: int = 4
    threshold: float = 0.5

    name: ClassVar[str] = "hierarchical"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.group < 2:
            raise IngestionError(f"hierarchical needs to group at least 2 chunks, got {self.group}")
        if not 0 < self.threshold <= 1:
            raise IngestionError(
                f"hierarchical threshold must be in (0, 1], got {self.threshold}. "
                "At 1 every child must hit before the parent is returned."
            )

    def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested:
        del context
        indexed: list[Chunk] = []
        retrievable: list[Chunk] = []
        children: dict[str, list[str]] = {}
        presented: dict[str, Chunk] = {}

        for group in _consecutive_groups(chunks, self.group):
            parent = _merge(group, suffix="parent")
            # The parent is presentation, not a retrievable unit. Scoring stays on the leaves,
            # so merging shows a generator more context without changing what retrieval is
            # credited with having found.
            presented[parent.id] = parent
            children[parent.id] = [chunk.id for chunk in group]
            for chunk in group:
                indexed.append(chunk)
                retrievable.append(chunk)

        return Ingested(
            indexed=indexed,
            retrievable=retrievable,
            presentation=children,
            presented_chunks=presented,
            notes={"children": children, "threshold": self.threshold},
        )


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def _by_document(chunks: Sequence[Chunk]) -> dict[str, list[Chunk]]:
    """Chunks per document, in reading order.

    Grouping has to stop at a document boundary. A parent spanning the end of one contract and
    the start of another is a passage that does not exist, and any answer read out of it is an
    answer to a question nobody asked.
    """
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    for group in grouped.values():
        group.sort(key=lambda chunk: chunk.span.start)
    return grouped


def _consecutive_groups(chunks: Sequence[Chunk], size: int) -> list[list[Chunk]]:
    """Runs of `size` consecutive chunks, never crossing a document."""
    groups: list[list[Chunk]] = []
    for group in _by_document(chunks).values():
        for start in range(0, len(group), size):
            piece = group[start : start + size]
            if piece:
                groups.append(piece)
    return groups


def _merge(group: Sequence[Chunk], *, suffix: str) -> Chunk:
    """One chunk spanning several, with the text taken from the document rather than joined.

    Taken from the span rather than concatenated, so the merged text is a literal slice of the
    parse -- whitespace and all -- and gold evidence straddling the boundary between two chunks
    still resolves against it. Joining the pieces with a separator would insert characters the
    document does not contain and shift every offset after the first one.
    """
    if not group:  # pragma: no cover - callers never pass an empty group
        raise IngestionError("cannot merge an empty group of chunks")
    if len(group) == 1:
        return group[0]

    first, last = group[0], group[-1]
    start, end = first.span.start, last.span.end
    span = Span(first.doc_id, start, end)

    # Every chunk here came from one parse, so the text of the whole range is recoverable from
    # the pieces only when they tile it. When they do not -- an overlapping or sampling chunker
    # -- the join is the honest reconstruction and the flag says the offsets are approximate.
    tiled = all(a.span.end >= b.span.start for a, b in pairwise(group))
    text = _text_from(group) if tiled else "\n".join(chunk.text for chunk in group)

    return Chunk(
        id=f"{first.doc_id}:{start}-{end}:{suffix}",
        span=span,
        text=text,
        meta={
            **first.meta,
            "merged_from": [chunk.id for chunk in group],
            "merged_count": len(group),
        },
        token_counts=_summed_tokens(group),
        offsets_exact=tiled and all(chunk.offsets_exact for chunk in group),
    )


def _text_from(group: Sequence[Chunk]) -> str:
    """Reconstruct the covered range from chunks that tile it, without re-reading the parse."""
    pieces: list[str] = []
    cursor = group[0].span.start
    for chunk in group:
        if chunk.span.end <= cursor:
            continue
        overlap = max(0, cursor - chunk.span.start)
        pieces.append(chunk.text[overlap:])
        cursor = chunk.span.end
    return "".join(pieces)


def _summed_tokens(group: Sequence[Chunk]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for chunk in group:
        for name, count in chunk.token_counts.items():
            totals[name] = totals.get(name, 0) + count
    return totals

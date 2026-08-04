"""Turning retrieved chunks into the text a generator actually sees.

An entire layer that nothing else sweeps. Between "the retriever returned these five chunks"
and "the model answered" there are decisions worth measuring:

**Order.** Long-context models miss information placed in the middle of their context far
more often than at either end -- a positional bias that comes out of how rotary embeddings
decay. Putting the best evidence at the ends costs nothing and measurably changes answers.

**Budget.** The tokens sent to the generator are the real cost driver, and `k` is a poor
proxy for them: five structural chunks can be four times the text of five sentence windows.

**Deduplication.** Overlapping chunks pay twice for the same sentence, and a top-5 that is
three copies of one paragraph fills a context window with one fact.

None of this changes what was retrieved, so it cannot rescue a bad retriever. It routinely
changes whether a good one produces the right answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.core.protocols import Tokenizer
from contextgrid.core.span import merge_spans
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.tokens import get_tokenizer


class Ordering(str, Enum):
    """Where in the context the best evidence goes.

    RELEVANCE   best first. The obvious choice, and it buries later evidence in the middle.
    ENDS        best first and second-best last, working inwards. The "lost in the middle"
                mitigation: the weakest evidence ends up where the model looks least.
    DOCUMENT    original reading order, ignoring rank. Preserves narrative flow, which
                matters when the answer depends on what a sentence was next to.
    REVERSED    worst first, best last. Puts the strongest evidence closest to the question
                when the question comes after the context.
    """

    RELEVANCE = "relevance"
    ENDS = "ends"
    DOCUMENT = "document"
    REVERSED = "reversed"


class AssemblyError(ContextGridError, ValueError):
    """Context could not be assembled as asked."""


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """What the generator will see, and what it cost to put together."""

    text: str
    chunks: tuple[Chunk, ...]
    tokens: int
    dropped: int = 0
    duplicate_characters: int = 0
    warnings: WarningLog = field(default_factory=WarningLog)

    @property
    def used(self) -> int:
        return len(self.chunks)

    @property
    def characters(self) -> int:
        return len(self.text)

    def as_dict(self) -> dict[str, int]:
        return {
            "tokens": self.tokens,
            "characters": self.characters,
            "chunks_used": self.used,
            "chunks_dropped": self.dropped,
            "duplicate_characters": self.duplicate_characters,
        }


@dataclass(slots=True)
class ContextAssembler:
    """Assembles retrieved chunks into a prompt's context block.

    `budget_tokens` is the honest limit. `k` is not: five structural chunks can be four times
    the text of five sentence windows, so two configurations at the same k can send wildly
    different bills to the generator.
    """

    ordering: Ordering = Ordering.RELEVANCE
    budget_tokens: int | None = None
    deduplicate: bool = True
    include_source: bool = True
    include_heading: bool = False
    separator: str = "\n\n---\n\n"
    tokenizer: str | Tokenizer | None = None

    def assemble(self, chunks: Sequence[Chunk]) -> AssembledContext:
        """Order, trim and format retrieved chunks into one block of text."""
        if not chunks:
            return AssembledContext(text="", chunks=(), tokens=0)

        counter = get_tokenizer(self.tokenizer)
        log = WarningLog()

        kept = list(chunks)
        duplicate_characters = 0
        if self.deduplicate:
            kept, duplicate_characters = _drop_contained(kept)
            if duplicate_characters:
                log.add(
                    WarningCode.CACHE_MISS_STORM,
                    f"{duplicate_characters} characters of the retrieved context were "
                    "duplicated across overlapping chunks. The generator would have paid "
                    "for the same sentences twice",
                    severity=Severity.INFO,
                    stage="assemble",
                    duplicate_characters=duplicate_characters,
                )

        within_budget, dropped = self._apply_budget(kept, counter, log)
        ordered = self._order(within_budget)

        text = self.separator.join(
            self._render(chunk, position) for position, chunk in enumerate(ordered)
        )

        return AssembledContext(
            text=text,
            chunks=tuple(ordered),
            tokens=counter.count(text),
            dropped=dropped,
            duplicate_characters=duplicate_characters,
            warnings=log,
        )

    # -- budget --------------------------------------------------------------

    def _apply_budget(
        self, chunks: list[Chunk], counter: Tokenizer, log: WarningLog
    ) -> tuple[list[Chunk], int]:
        """Keep chunks in rank order until the budget runs out.

        Truncating the *last* chunk to fit would be worse than dropping it: half a passage
        reads as a complete one, and a model given half an answer will confidently give half
        an answer back.
        """
        if self.budget_tokens is None:
            return chunks, 0

        kept: list[Chunk] = []
        used = 0
        for chunk in chunks:
            cost = counter.count(chunk.text)
            if used + cost > self.budget_tokens and kept:
                break
            kept.append(chunk)
            used += cost

        dropped = len(chunks) - len(kept)
        if dropped:
            log.add(
                WarningCode.CHUNK_EXCEEDS_MODEL_CONTEXT,
                f"{dropped} of {len(chunks)} retrieved chunks did not fit in the "
                f"{self.budget_tokens}-token budget and were dropped. If the evidence was in "
                "one of them, this configuration cannot answer whatever the retriever did",
                severity=Severity.CAUTION,
                stage="assemble",
                dropped=dropped,
                budget=self.budget_tokens,
            )
        return kept, dropped

    # -- order ---------------------------------------------------------------

    def _order(self, chunks: list[Chunk]) -> list[Chunk]:
        if self.ordering is Ordering.RELEVANCE:
            return chunks
        if self.ordering is Ordering.REVERSED:
            return list(reversed(chunks))
        if self.ordering is Ordering.DOCUMENT:
            return sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.char_start))
        return _fold_to_ends(chunks)

    # -- rendering -----------------------------------------------------------

    def _render(self, chunk: Chunk, position: int) -> str:
        """One chunk as it appears in the prompt.

        The source label is not decoration: without it a model cannot cite anything, and
        citation accuracy is the metric enterprises care about most.
        """
        parts: list[str] = []
        if self.include_source:
            parts.append(f"[{position + 1}] {chunk.doc_id}")
        if self.include_heading:
            path = chunk.meta.get("heading_path")
            if isinstance(path, (list, tuple)) and path:
                parts.append(" > ".join(str(item) for item in path))
        header = " · ".join(parts)
        return f"{header}\n{chunk.text}" if header else chunk.text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fold_to_ends(chunks: list[Chunk]) -> list[Chunk]:
    """Best first, second-best last, third-best second, working inwards.

    The "lost in the middle" arrangement. Long-context models attend most strongly to the
    start and end of their context, so the weakest evidence is put where it will be missed
    if anything is.
    """
    front: list[Chunk] = []
    back: list[Chunk] = []
    for position, chunk in enumerate(chunks):
        (front if position % 2 == 0 else back).append(chunk)
    return front + list(reversed(back))


def _drop_contained(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Remove chunks whose text is already fully covered by a higher-ranked one.

    Overlapping chunks are not a bug -- overlap is a chunker parameter -- but sending the
    same sentence to a generator twice is paying twice for it.
    """
    kept: list[Chunk] = []
    duplicate = 0

    for chunk in chunks:
        covered = [other.span for other in kept if other.doc_id == chunk.doc_id]
        if not covered:
            kept.append(chunk)
            continue

        overlap = sum(chunk.span.overlap_len(span) for span in merge_spans(covered))
        if overlap >= chunk.span.length:
            duplicate += chunk.span.length
            continue

        duplicate += overlap
        kept.append(chunk)

    return kept, duplicate


def tokens_sent(chunks: Sequence[Chunk], tokenizer: str | Tokenizer | None = None) -> int:
    """What this retrieval will cost the generator, per query.

    The number `k` is a poor proxy for and that determines the monthly bill.
    """
    counter = get_tokenizer(tokenizer)
    return sum(counter.count(chunk.text) for chunk in chunks)

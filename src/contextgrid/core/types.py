"""Core value objects.

The whole package rests on one idea: a piece of text always knows exactly which characters
of which source document it came from. Parsers produce blocks with offsets, chunkers produce
chunks with offsets, and ground truth is stored as offsets too. Nothing is ever identified by
a chunk ID, because chunk IDs change the moment you change the chunker -- which is precisely
the thing this tool exists to compare.

`Span` is where that idea lives, and it holds all the overlap arithmetic. Keeping the maths in
one small, heavily tested value object means the correctness-critical code exists exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from contextgrid.core.errors import DocumentError, EvalSetError, SpanError

# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """A half-open character range `[start, end)` within one document.

    Half-open because it makes the arithmetic clean: adjacent spans share a boundary
    without overlapping, lengths subtract, and `text[start:end]` is the literal Python
    slice with no off-by-one to remember.

    Comparison is by `(doc_id, start, end)`, so spans sort into reading order per document.
    """

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise SpanError(f"span start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise SpanError(f"span end ({self.end}) must be >= start ({self.start})")

    # -- basics --------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of characters covered."""
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        return self.end == self.start

    def same_document(self, other: Span) -> bool:
        return self.doc_id == other.doc_id

    def __len__(self) -> int:
        return self.length

    # -- overlap arithmetic --------------------------------------------------

    def intersection(self, other: Span) -> Span | None:
        """The overlapping range, or None when they do not overlap.

        Spans in different documents never overlap.
        """
        if not self.same_document(other):
            return None
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end <= start:
            return None
        return Span(self.doc_id, start, end)

    def overlap_len(self, other: Span) -> int:
        """Characters shared with `other`. Zero when they do not overlap."""
        if not self.same_document(other):
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def union_len(self, other: Span) -> int:
        """Characters covered by either span.

        For spans in different documents this is simply the sum, since they cannot overlap.
        """
        return self.length + other.length - self.overlap_len(other)

    def overlaps(self, other: Span) -> bool:
        return self.overlap_len(other) > 0

    def contains(self, other: Span) -> bool:
        """True when `other` sits entirely inside this span.

        An empty span is contained if its position falls within this span's range.
        """
        if not self.same_document(other):
            return False
        return self.start <= other.start and other.end <= self.end

    # -- similarity ----------------------------------------------------------

    def iou(self, other: Span) -> float:
        """Intersection over union, in [0, 1].

        Symmetric, and it punishes size differences in both directions. That makes it the
        wrong default for resolving gold spans to chunks -- see `coverage_of` -- but the
        right measure when chunk bloat should count against a configuration.

        Two empty spans give 0.0 rather than a division by zero.
        """
        union = self.union_len(other)
        if union == 0:
            return 0.0
        return self.overlap_len(other) / union

    def coverage_of(self, other: Span) -> float:
        """Fraction of `other` that this span contains, in [0, 1].

        Asymmetric on purpose. `chunk.coverage_of(gold)` asks the question that actually
        matters when scoring retrieval: how much of the evidence is present in this chunk?
        A large chunk holding all of a short gold span scores 1.0, which is correct -- it
        would ground the answer perfectly -- where IoU would have scored it near zero and
        quietly penalised the configuration for using large chunks.

        An empty `other` gives 0.0; there is nothing to cover.
        """
        if other.length == 0:
            return 0.0
        return self.overlap_len(other) / other.length

    # -- construction --------------------------------------------------------

    def clipped_to(self, bound: Span) -> Span | None:
        """This span trimmed to `bound`, or None when it falls entirely outside."""
        return self.intersection(bound)

    def shifted(self, offset: int) -> Span:
        """A copy moved by `offset` characters. Used when re-basing onto a parent document."""
        return Span(self.doc_id, self.start + offset, self.end + offset)

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Span:
        return cls(doc_id=str(data["doc_id"]), start=int(data["start"]), end=int(data["end"]))

    def __repr__(self) -> str:
        return f"Span({self.doc_id!r}, {self.start}, {self.end})"


# ---------------------------------------------------------------------------
# Interval algebra over sets of spans
# ---------------------------------------------------------------------------


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Collapse overlapping and touching spans into a minimal disjoint set.

    Grouped by document, returned in reading order. Touching spans (`[0,5)` and `[5,9)`)
    merge into one, because they cover a contiguous run of characters with no gap.

    This is what makes "how much of the gold did the retrieved set cover?" answerable
    without double-counting characters that appear in two overlapping chunks.
    """
    by_doc: dict[str, list[Span]] = {}
    for span in spans:
        if span.is_empty:
            continue
        by_doc.setdefault(span.doc_id, []).append(span)

    merged: list[Span] = []
    for doc_id in sorted(by_doc):
        ordered = sorted(by_doc[doc_id], key=lambda s: (s.start, s.end))
        current = ordered[0]
        for span in ordered[1:]:
            if span.start <= current.end:  # overlapping or touching
                if span.end > current.end:
                    current = Span(doc_id, current.start, span.end)
            else:
                merged.append(current)
                current = span
        merged.append(current)
    return merged


def total_length(spans: Iterable[Span]) -> int:
    """Characters covered by a set of spans, counting shared characters once."""
    return sum(span.length for span in merge_spans(spans))


def covered_length(target: Span, others: Iterable[Span]) -> int:
    """Characters of `target` that appear anywhere in `others`.

    The core of union recall: a gold span split across two chunks is fully covered when
    both are retrieved, even though neither chunk alone contains enough of it to clear a
    per-chunk threshold.
    """
    if target.is_empty:
        return 0
    clipped = [c for c in (o.intersection(target) for o in others) if c is not None]
    return total_length(clipped)


def coverage_fraction(target: Span, others: Iterable[Span]) -> float:
    """`covered_length` as a fraction of `target`, in [0, 1]."""
    if target.is_empty:
        return 0.0
    return covered_length(target, others) / target.length


def intersection_length(left: Iterable[Span], right: Iterable[Span]) -> int:
    """Characters covered by both sets of spans, counting each character once.

    Both sides are collapsed first, so overlapping chunks on either side do not inflate
    the count. This is what character-level precision and recall are built from.
    """
    merged_right = merge_spans(right)
    return sum(covered_length(span, merged_right) for span in merge_spans(left))


# ---------------------------------------------------------------------------
# Documents and chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """A source document's canonical text. Every offset in the system refers to this.

    The text is whatever the corpus loader produced for the file. Parsers do not redefine
    it -- they produce blocks that point into it. That is what keeps two parsers comparable.
    """

    id: str
    text: str
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)

    def span(self) -> Span:
        """A span covering the whole document."""
        return Span(self.id, 0, len(self.text))

    def slice(self, span: Span) -> str:
        """The text a span refers to.

        Raises rather than silently returning a short string when the span belongs to a
        different document or runs past the end -- a truncated slice would be scored as if
        it were real evidence.
        """
        if span.doc_id != self.id:
            raise DocumentError(
                f"span belongs to document {span.doc_id!r}, cannot slice document {self.id!r}"
            )
        if span.end > len(self.text):
            raise DocumentError(
                f"span {span.start}-{span.end} runs past the end of document "
                f"{self.id!r} (length {len(self.text)})"
            )
        return self.text[span.start : span.end]

    def contains_span(self, span: Span) -> bool:
        return span.doc_id == self.id and span.end <= len(self.text)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text, and where it came from.

    `offsets_exact` is the honesty flag. Most chunkers slice the document, so their text is
    literally `document.text[span]` and the flag is true. Some do not: contextual retrieval
    prepends an LLM-written summary, proposition extraction rewrites sentences into atomic
    facts. Those chunks still carry the span they derive from, but their text is not a slice
    of it, and scoring built on them is approximate. Saying so is better than hiding it.
    """

    id: str
    span: Span
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    offsets_exact: bool = True

    @property
    def doc_id(self) -> str:
        return self.span.doc_id

    @property
    def char_start(self) -> int:
        return self.span.start

    @property
    def char_end(self) -> int:
        return self.span.end

    def token_count(self, tokenizer: str) -> int | None:
        """Length in tokens under a named tokenizer, if it has been measured.

        Chunk size is meaningless without naming the tokenizer -- "512 with 50 overlap"
        describes different text under cl100k_base than under a BERT wordpiece vocabulary.
        Sizes are therefore stored per tokenizer and never as a single number.
        """
        return self.token_counts.get(tokenizer)

    def matches_source(self, document: Document) -> bool:
        """True when this chunk's text is exactly the document text its span points at.

        The invariant every offset-exact chunker must satisfy, and what the conformance
        suite checks.
        """
        if not document.contains_span(self.span):
            return False
        return document.slice(self.span) == self.text


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """A stretch of source text that answers a question, and how well it does so.

    Grades follow the usual IR convention: 2 fully answers, 1 partially relevant,
    0 irrelevant. Graded relevance is what makes nDCG mean anything; binary gold turns it
    into a noisier version of hit rate.
    """

    span: Span
    grade: int = 2

    def __post_init__(self) -> None:
        if self.span.is_empty:
            raise EvalSetError(f"gold span must cover at least one character: {self.span!r}")
        if self.grade < 0:
            raise EvalSetError(f"gold grade must be >= 0, got {self.grade}")

    @property
    def doc_id(self) -> str:
        return self.span.doc_id

    def to_dict(self) -> dict[str, Any]:
        return {**self.span.to_dict(), "grade": self.grade}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoldSpan:
        return cls(span=Span.from_dict(data), grade=int(data.get("grade", 2)))


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One question and the source text that answers it.

    Gold is a set of spans, not one, because a question can legitimately be answered by more
    than one passage -- and because a single piece of evidence can straddle a paragraph break
    and be cleaner to annotate as two spans than one.
    """

    id: str
    question: str
    gold: tuple[GoldSpan, ...] = ()
    qtype: str | None = None
    answer: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise EvalSetError(f"eval item {self.id!r} has an empty question")

    @property
    def is_answerable(self) -> bool:
        """False for questions deliberately included with no supporting evidence.

        Unanswerable questions test whether a system correctly declines to answer, which
        almost no eval set measures and which is a real failure mode in production.
        """
        return bool(self.gold)

    @property
    def gold_spans(self) -> tuple[Span, ...]:
        return tuple(g.span for g in self.gold)

    @property
    def gold_length(self) -> int:
        """Total gold characters, counting overlapping gold spans once."""
        return total_length(self.gold_spans)

    def gold_documents(self) -> set[str]:
        return {g.doc_id for g in self.gold}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold": [g.to_dict() for g in self.gold],
            "qtype": self.qtype,
            "answer": self.answer,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalItem:
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            gold=tuple(GoldSpan.from_dict(g) for g in data.get("gold", ())),
            qtype=data.get("qtype"),
            answer=data.get("answer"),
            meta=dict(data.get("meta", {})),
        )


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A versioned collection of questions with span-level ground truth.

    Versioned because comparing two runs scored against different eval sets is a silent
    correctness bug, and the version is what lets a manifest catch it.
    """

    id: str
    items: tuple[EvalItem, ...]
    version: int = 1
    source: str = "manual"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise EvalSetError(f"duplicate eval item id {item.id!r} in eval set {self.id!r}")
            seen.add(item.id)

    @property
    def answerable(self) -> tuple[EvalItem, ...]:
        return tuple(i for i in self.items if i.is_answerable)

    def by_type(self, qtype: str) -> tuple[EvalItem, ...]:
        return tuple(i for i in self.items if i.qtype == qtype)

    def types(self) -> set[str]:
        return {i.qtype for i in self.items if i.qtype is not None}

    def get(self, item_id: str) -> EvalItem | None:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def __iter__(self) -> Iterator[EvalItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Scoring inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    """A resolved judgement: for this question, this chunk is relevant at this grade."""

    item_id: str
    chunk_id: str
    grade: int


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk a configuration returned for a query, with its position and score."""

    chunk: Chunk
    score: float
    rank: int
    rank_before_rerank: int | None = None

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def moved(self) -> int | None:
        """Positions gained by reranking. Positive means it moved up."""
        if self.rank_before_rerank is None:
            return None
        return self.rank_before_rerank - self.rank


# `ranx` consumes qrels and runs as {query_id: {doc_id: value}}. Naming the shape here keeps
# the boundary with that library in one place.
Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


def chunks_of(retrieved: Sequence[RetrievedChunk]) -> list[Chunk]:
    return [r.chunk for r in retrieved]


def spans_of(chunks: Iterable[Chunk]) -> list[Span]:
    return [c.span for c in chunks]

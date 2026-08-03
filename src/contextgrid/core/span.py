"""Character spans, and the arithmetic over them.

The whole package rests on one idea: a piece of text always knows exactly which characters
of which document it came from. Parsers produce blocks with offsets, chunkers produce chunks
with offsets, and ground truth resolves to offsets too. Nothing is identified by a chunk ID,
because chunk IDs change the moment you change the chunker -- which is precisely the thing
this tool exists to compare.

`Span` is where that idea lives. Keeping the maths in one small, heavily tested value object
means the correctness-critical code exists exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from contextgrid.core.errors import SpanError


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

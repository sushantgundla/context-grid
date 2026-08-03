"""Unit tests for the span algebra.

Every boundary case is written out explicitly rather than left to the property tests,
because when one of these breaks the failure message should say which case it was.
"""

from __future__ import annotations

import pytest

from contextgrid import Span, SpanError, coverage_fraction, covered_length, merge_spans
from contextgrid.core.types import intersection_length, total_length

DOC = "doc-1"
OTHER_DOC = "doc-2"


def s(start: int, end: int, doc: str = DOC) -> Span:
    return Span(doc, start, end)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_rejects_negative_start() -> None:
    with pytest.raises(SpanError, match="must be >= 0"):
        Span(DOC, -1, 5)


def test_rejects_end_before_start() -> None:
    with pytest.raises(SpanError, match="must be >= start"):
        Span(DOC, 10, 4)


def test_empty_span_is_allowed() -> None:
    span = s(5, 5)
    assert span.is_empty
    assert span.length == 0


def test_length_and_len_agree() -> None:
    assert s(10, 25).length == 15
    assert len(s(10, 25)) == 15


def test_spans_sort_into_reading_order() -> None:
    spans = [s(50, 60), s(0, 10), s(0, 5)]
    assert sorted(spans) == [s(0, 5), s(0, 10), s(50, 60)]


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------


def test_disjoint_spans_do_not_overlap() -> None:
    assert s(0, 10).overlap_len(s(20, 30)) == 0
    assert s(0, 10).intersection(s(20, 30)) is None
    assert not s(0, 10).overlaps(s(20, 30))


def test_touching_spans_do_not_overlap() -> None:
    # Half-open ranges: [0,10) and [10,20) share a boundary but no character.
    assert s(0, 10).overlap_len(s(10, 20)) == 0
    assert s(0, 10).intersection(s(10, 20)) is None


def test_partial_overlap() -> None:
    assert s(0, 10).overlap_len(s(5, 15)) == 5
    assert s(0, 10).intersection(s(5, 15)) == s(5, 10)


def test_nested_overlap() -> None:
    outer, inner = s(0, 100), s(40, 50)
    assert outer.overlap_len(inner) == 10
    assert outer.intersection(inner) == inner
    assert outer.contains(inner)
    assert not inner.contains(outer)


def test_identical_spans() -> None:
    span = s(3, 9)
    assert span.overlap_len(span) == 6
    assert span.intersection(span) == span
    assert span.contains(span)


def test_spans_in_different_documents_never_interact() -> None:
    a, b = s(0, 100), s(0, 100, OTHER_DOC)
    assert a.overlap_len(b) == 0
    assert a.intersection(b) is None
    assert not a.contains(b)
    assert a.iou(b) == 0.0
    assert a.coverage_of(b) == 0.0


def test_union_length_excludes_shared_characters() -> None:
    assert s(0, 10).union_len(s(5, 15)) == 15
    assert s(0, 10).union_len(s(20, 30)) == 20


# ---------------------------------------------------------------------------
# iou vs coverage - the distinction the whole design turns on
# ---------------------------------------------------------------------------


def test_iou_is_symmetric() -> None:
    a, b = s(0, 100), s(50, 150)
    assert a.iou(b) == b.iou(a)


def test_iou_of_identical_spans_is_one() -> None:
    assert s(10, 20).iou(s(10, 20)) == 1.0


def test_iou_of_two_empty_spans_is_zero_not_a_crash() -> None:
    assert s(5, 5).iou(s(5, 5)) == 0.0


def test_coverage_is_asymmetric() -> None:
    big, small = s(0, 2000), s(900, 1070)
    assert big.coverage_of(small) == 1.0  # the big chunk holds all the evidence
    assert small.coverage_of(big) == pytest.approx(170 / 2000)


def test_coverage_of_empty_target_is_zero() -> None:
    assert s(0, 100).coverage_of(s(50, 50)) == 0.0


def test_iou_penalises_large_chunks_and_coverage_does_not() -> None:
    """The reason COVERAGE is the default policy.

    A 170-character gold span sits inside both chunks. IoU calls the large chunk a miss at
    any sensible threshold even though it contains every character of the evidence.
    """
    gold = s(900, 1070)
    large_chunk = s(0, 2000)
    small_chunk = s(880, 1130)

    assert large_chunk.iou(gold) == pytest.approx(0.085, abs=1e-3)
    assert small_chunk.iou(gold) == pytest.approx(0.68, abs=1e-2)

    # Coverage sees both for what they are: the evidence is fully present.
    assert large_chunk.coverage_of(gold) == 1.0
    assert small_chunk.coverage_of(gold) == 1.0


def test_containment_implies_full_coverage() -> None:
    outer, inner = s(0, 100), s(20, 30)
    assert outer.contains(inner)
    assert outer.coverage_of(inner) == 1.0


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def test_clipped_to_bound() -> None:
    assert s(0, 100).clipped_to(s(50, 200)) == s(50, 100)
    assert s(0, 10).clipped_to(s(50, 200)) is None


def test_shifted() -> None:
    assert s(10, 20).shifted(5) == s(15, 25)


def test_round_trips_through_dict() -> None:
    span = s(4, 17)
    assert Span.from_dict(span.to_dict()) == span


# ---------------------------------------------------------------------------
# merge_spans
# ---------------------------------------------------------------------------


def test_merge_collapses_overlapping_spans() -> None:
    assert merge_spans([s(0, 10), s(5, 20)]) == [s(0, 20)]


def test_merge_collapses_touching_spans() -> None:
    # [0,5) and [5,9) cover a contiguous run with no gap, so they are one region.
    assert merge_spans([s(0, 5), s(5, 9)]) == [s(0, 9)]


def test_merge_keeps_disjoint_spans_apart() -> None:
    assert merge_spans([s(0, 5), s(10, 15)]) == [s(0, 5), s(10, 15)]


def test_merge_drops_empty_spans() -> None:
    assert merge_spans([s(3, 3), s(0, 5)]) == [s(0, 5)]


def test_merge_handles_nested_spans() -> None:
    assert merge_spans([s(0, 100), s(20, 30), s(40, 50)]) == [s(0, 100)]


def test_merge_separates_documents() -> None:
    merged = merge_spans([s(0, 10), s(0, 10, OTHER_DOC)])
    assert merged == [s(0, 10), s(0, 10, OTHER_DOC)]


def test_merge_of_nothing_is_nothing() -> None:
    assert merge_spans([]) == []


def test_total_length_counts_shared_characters_once() -> None:
    assert total_length([s(0, 10), s(5, 15)]) == 15
    assert total_length([s(0, 10), s(0, 10)]) == 10


# ---------------------------------------------------------------------------
# covered_length / coverage_fraction / intersection_length
# ---------------------------------------------------------------------------


def test_covered_length_of_a_split_target() -> None:
    """The split-gold case: two chunks each holding part of one gold span."""
    gold = s(90, 110)
    chunks = [s(0, 100), s(100, 200)]
    assert covered_length(gold, chunks) == 20
    assert coverage_fraction(gold, chunks) == 1.0


def test_covered_length_ignores_double_counting() -> None:
    gold = s(0, 100)
    overlapping = [s(0, 60), s(40, 100)]
    assert covered_length(gold, overlapping) == 100


def test_covered_length_of_empty_target_is_zero() -> None:
    assert covered_length(s(5, 5), [s(0, 100)]) == 0
    assert coverage_fraction(s(5, 5), [s(0, 100)]) == 0.0


def test_covered_length_with_no_chunks() -> None:
    assert covered_length(s(0, 10), []) == 0


def test_intersection_length_is_symmetric() -> None:
    left = [s(0, 50), s(100, 150)]
    right = [s(25, 125)]
    assert intersection_length(left, right) == intersection_length(right, left) == 50


def test_intersection_length_counts_each_character_once() -> None:
    left = [s(0, 100), s(0, 100)]
    right = [s(0, 100), s(50, 100)]
    assert intersection_length(left, right) == 100

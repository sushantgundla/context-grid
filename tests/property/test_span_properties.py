"""Property tests for the span algebra.

The unit tests pin down the cases we thought of. These check the laws that must hold for
every case, including the ones we did not.
"""

from __future__ import annotations

from itertools import pairwise

from hypothesis import assume, given
from hypothesis import strategies as st

from contextgrid import Span, coverage_fraction, covered_length, merge_spans
from contextgrid.core.types import intersection_length, total_length

DOCS = st.sampled_from(["doc-a", "doc-b"])
POSITIONS = st.integers(min_value=0, max_value=2000)


@st.composite
def spans(draw: st.DrawFn, doc: str | None = None) -> Span:
    doc_id = doc if doc is not None else draw(DOCS)
    start = draw(POSITIONS)
    length = draw(st.integers(min_value=0, max_value=500))
    return Span(doc_id, start, start + length)


@st.composite
def nonempty_spans(draw: st.DrawFn, doc: str | None = None) -> Span:
    doc_id = doc if doc is not None else draw(DOCS)
    start = draw(POSITIONS)
    length = draw(st.integers(min_value=1, max_value=500))
    return Span(doc_id, start, start + length)


@st.composite
def containing_pairs(draw: st.DrawFn) -> tuple[Span, Span]:
    """An (outer, inner) pair where outer definitely contains inner.

    Generated directly rather than filtered with `assume`, because two independently drawn
    spans almost never contain one another and Hypothesis would discard nearly every case.
    """
    doc_id = draw(DOCS)
    outer_start = draw(POSITIONS)
    outer_length = draw(st.integers(min_value=1, max_value=2000))
    outer = Span(doc_id, outer_start, outer_start + outer_length)

    inner_offset = draw(st.integers(min_value=0, max_value=outer_length - 1))
    inner_length = draw(st.integers(min_value=1, max_value=outer_length - inner_offset))
    inner = Span(doc_id, outer.start + inner_offset, outer.start + inner_offset + inner_length)
    return outer, inner


SPAN_LISTS = st.lists(spans(), max_size=12)


# ---------------------------------------------------------------------------
# construction invariants
# ---------------------------------------------------------------------------


@given(spans())
def test_length_is_never_negative(span: Span) -> None:
    assert span.length >= 0
    assert span.length == span.end - span.start


@given(spans(), st.integers(min_value=0, max_value=500))
def test_shifting_preserves_length(span: Span, offset: int) -> None:
    assert span.shifted(offset).length == span.length


# ---------------------------------------------------------------------------
# overlap laws
# ---------------------------------------------------------------------------


@given(spans(), spans())
def test_overlap_is_symmetric(a: Span, b: Span) -> None:
    assert a.overlap_len(b) == b.overlap_len(a)


@given(spans(), spans())
def test_overlap_never_exceeds_either_span(a: Span, b: Span) -> None:
    assert a.overlap_len(b) <= min(a.length, b.length)


@given(spans(), spans())
def test_union_length_identity(a: Span, b: Span) -> None:
    """|A union B| == |A| + |B| - |A intersect B|, always."""
    assert a.union_len(b) == a.length + b.length - a.overlap_len(b)


@given(spans(), spans())
def test_intersection_agrees_with_overlap_length(a: Span, b: Span) -> None:
    intersection = a.intersection(b)
    if intersection is None:
        assert a.overlap_len(b) == 0
    else:
        assert intersection.length == a.overlap_len(b)
        assert a.contains(intersection)
        assert b.contains(intersection)


@given(spans(), spans())
def test_spans_in_different_documents_never_interact(a: Span, b: Span) -> None:
    assume(a.doc_id != b.doc_id)
    assert a.overlap_len(b) == 0
    assert a.intersection(b) is None
    assert a.iou(b) == 0.0
    assert a.coverage_of(b) == 0.0
    assert not a.contains(b)


# ---------------------------------------------------------------------------
# similarity laws
# ---------------------------------------------------------------------------


@given(spans(), spans())
def test_iou_is_symmetric(a: Span, b: Span) -> None:
    assert a.iou(b) == b.iou(a)


@given(spans(), spans())
def test_iou_is_a_fraction(a: Span, b: Span) -> None:
    assert 0.0 <= a.iou(b) <= 1.0


@given(nonempty_spans())
def test_iou_with_self_is_one(span: Span) -> None:
    assert span.iou(span) == 1.0


@given(spans(), spans())
def test_coverage_is_a_fraction(a: Span, b: Span) -> None:
    assert 0.0 <= a.coverage_of(b) <= 1.0


@given(nonempty_spans())
def test_coverage_of_self_is_one(span: Span) -> None:
    assert span.coverage_of(span) == 1.0


@given(containing_pairs())
def test_containment_implies_full_coverage(pair: tuple[Span, Span]) -> None:
    """The property that makes COVERAGE the right default policy.

    However large the containing span is, if it holds all of the target then it holds all
    of the evidence -- and coverage says so, where IoU would not.
    """
    outer, inner = pair
    assert outer.contains(inner)
    assert outer.coverage_of(inner) == 1.0


@given(containing_pairs())
def test_iou_shrinks_as_the_containing_span_grows_but_coverage_does_not(
    pair: tuple[Span, Span],
) -> None:
    """The bias this design exists to avoid, stated as a law.

    Widening a chunk that already holds all the evidence can only lower its IoU, never
    raise it -- so an IoU threshold turns "used bigger chunks" into "missed the answer".
    Coverage stays at 1.0, because the evidence is still all there.
    """
    outer, inner = pair
    wider = Span(outer.doc_id, outer.start, outer.end + 500)
    assert wider.iou(inner) <= outer.iou(inner)
    assert wider.coverage_of(inner) == outer.coverage_of(inner) == 1.0


@given(spans(), spans())
def test_coverage_never_exceeds_iou_scaled_by_size(a: Span, b: Span) -> None:
    """Coverage is at least IoU, because its denominator is never larger than the union."""
    assume(not b.is_empty)
    assert a.coverage_of(b) >= a.iou(b) - 1e-12


# ---------------------------------------------------------------------------
# merge_spans laws
# ---------------------------------------------------------------------------


@given(SPAN_LISTS)
def test_merged_spans_are_disjoint_and_ordered(raw: list[Span]) -> None:
    merged = merge_spans(raw)
    by_doc: dict[str, list[Span]] = {}
    for span in merged:
        by_doc.setdefault(span.doc_id, []).append(span)
    for group in by_doc.values():
        for earlier, later in pairwise(group):
            assert earlier.end < later.start  # strictly apart; touching spans were merged


@given(SPAN_LISTS)
def test_merged_spans_are_never_empty(raw: list[Span]) -> None:
    assert all(not span.is_empty for span in merge_spans(raw))


@given(SPAN_LISTS)
def test_merging_is_idempotent(raw: list[Span]) -> None:
    once = merge_spans(raw)
    assert merge_spans(once) == once


@given(SPAN_LISTS)
def test_merging_does_not_change_the_covered_set(raw: list[Span]) -> None:
    """Every original character is still covered, and no new ones appeared."""
    merged = merge_spans(raw)
    for span in raw:
        assert covered_length(span, merged) == span.length
    assert total_length(merged) == total_length(raw)


@given(SPAN_LISTS, st.lists(spans(), max_size=6))
def test_total_length_is_monotone_under_addition(base: list[Span], extra: list[Span]) -> None:
    assert total_length([*base, *extra]) >= total_length(base)


@given(SPAN_LISTS)
def test_total_length_never_exceeds_the_naive_sum(raw: list[Span]) -> None:
    assert total_length(raw) <= sum(span.length for span in raw)


# ---------------------------------------------------------------------------
# covered_length / coverage_fraction laws
# ---------------------------------------------------------------------------


@given(spans(), SPAN_LISTS)
def test_covered_length_never_exceeds_the_target(target: Span, others: list[Span]) -> None:
    assert 0 <= covered_length(target, others) <= target.length


@given(spans(), SPAN_LISTS)
def test_coverage_fraction_is_a_fraction(target: Span, others: list[Span]) -> None:
    assert 0.0 <= coverage_fraction(target, others) <= 1.0


@given(spans(), SPAN_LISTS, st.lists(spans(), max_size=6))
def test_coverage_only_grows_as_more_spans_are_added(
    target: Span, base: list[Span], extra: list[Span]
) -> None:
    """Union recall is monotone: retrieving more can never cover less.

    This is what makes union-based character recall a safe basis for Recall@k -- increasing
    k cannot make the score go down.
    """
    assert covered_length(target, [*base, *extra]) >= covered_length(target, base)


@given(spans(), SPAN_LISTS)
def test_coverage_ignores_the_order_spans_arrive_in(target: Span, others: list[Span]) -> None:
    assert covered_length(target, others) == covered_length(target, list(reversed(others)))


@given(nonempty_spans())
def test_a_span_fully_covers_itself(span: Span) -> None:
    assert coverage_fraction(span, [span]) == 1.0


# ---------------------------------------------------------------------------
# intersection_length laws
# ---------------------------------------------------------------------------


@given(SPAN_LISTS, SPAN_LISTS)
def test_intersection_length_is_symmetric(left: list[Span], right: list[Span]) -> None:
    assert intersection_length(left, right) == intersection_length(right, left)


@given(SPAN_LISTS, SPAN_LISTS)
def test_intersection_never_exceeds_either_side(left: list[Span], right: list[Span]) -> None:
    assert intersection_length(left, right) <= min(total_length(left), total_length(right))


@given(SPAN_LISTS)
def test_intersection_with_self_is_total_length(raw: list[Span]) -> None:
    assert intersection_length(raw, raw) == total_length(raw)


@given(SPAN_LISTS)
def test_intersection_with_nothing_is_nothing(raw: list[Span]) -> None:
    assert intersection_length(raw, []) == 0

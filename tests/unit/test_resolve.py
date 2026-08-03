"""Unit tests for span-to-chunk resolution and the character-level measures.

The scenarios here are the ones that make span-level ground truth worth the trouble:
the same gold, scored fairly against chunkers that carve the document up differently.
"""

from __future__ import annotations

import pytest

from contextgrid import (
    Chunk,
    EvalItem,
    EvalSet,
    GoldSpan,
    ResolutionError,
    ResolutionPolicy,
    Span,
    SpanResolver,
    WarningCode,
    character_f1,
    character_precision,
    character_recall,
    gold_coverage_by_chunk,
    retrieved_character_count,
)

DOC = "contract"


def span(start: int, end: int, doc: str = DOC) -> Span:
    return Span(doc, start, end)


def chunk(cid: str, start: int, end: int, doc: str = DOC, exact: bool = True) -> Chunk:
    return Chunk(id=cid, span=span(start, end, doc), text="x" * (end - start), offsets_exact=exact)


def item(*gold: GoldSpan, iid: str = "q1", qtype: str | None = None) -> EvalItem:
    return EvalItem(id=iid, question="How long is the notice period?", gold=gold, qtype=qtype)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_default_policy_is_coverage_at_half() -> None:
    resolver = SpanResolver()
    assert resolver.policy is ResolutionPolicy.COVERAGE
    assert resolver.threshold == 0.5


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_threshold_must_be_in_the_open_unit_interval(bad: float) -> None:
    with pytest.raises(ResolutionError, match="threshold must be"):
        SpanResolver(threshold=bad)


# ---------------------------------------------------------------------------
# the central claim: the same gold scores fairly across different chunkers
# ---------------------------------------------------------------------------


def test_same_gold_resolves_under_two_different_chunkers() -> None:
    """Gold at 840-1010. Chunker A splits at 500s, chunker B at 800s.

    Neither is advantaged by how the ground truth was written down, which is the entire
    reason gold is stored as character offsets rather than chunk IDs.
    """
    gold = GoldSpan(span(840, 1010))
    question = item(gold)
    resolver = SpanResolver()

    chunker_a = [chunk("a0", 0, 500), chunk("a1", 500, 1000), chunk("a2", 1000, 1500)]
    chunker_b = [chunk("b0", 0, 800), chunk("b1", 800, 1600)]

    res_a = resolver.resolve_item(question, chunker_a)
    res_b = resolver.resolve_item(question, chunker_b)

    # A: chunk a1 holds 160 of the 170 gold characters -> 94% coverage, relevant.
    assert res_a.relevant_chunk_ids == ("a1",)
    # B: chunk b1 holds all of it.
    assert res_b.relevant_chunk_ids == ("b1",)


def test_a_huge_chunk_containing_the_gold_is_relevant_under_coverage() -> None:
    """Under IoU this chunk would be a miss. It contains every character of the evidence."""
    question = item(GoldSpan(span(900, 1070)))
    huge = [chunk("big", 0, 2000)]

    assert SpanResolver().resolve_item(question, huge).relevant_chunk_ids == ("big",)

    strict = SpanResolver(policy=ResolutionPolicy.IOU, threshold=0.5)
    assert strict.resolve_item(question, huge).relevant_chunk_ids == ()


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def test_coverage_policy_uses_the_gold_span_as_the_denominator() -> None:
    resolver = SpanResolver(policy=ResolutionPolicy.COVERAGE, threshold=0.5)
    gold = span(100, 200)
    assert resolver.is_relevant(span(100, 160), gold)  # 60% of the gold
    assert not resolver.is_relevant(span(100, 140), gold)  # 40% of the gold


def test_iou_policy_is_symmetric_and_punishes_size_difference() -> None:
    resolver = SpanResolver(policy=ResolutionPolicy.IOU, threshold=0.5)
    gold = span(100, 200)
    assert resolver.is_relevant(span(90, 210), gold)  # iou 100/120
    assert not resolver.is_relevant(span(0, 1000), gold)  # iou 100/1000


def test_containment_policy_ignores_the_threshold() -> None:
    resolver = SpanResolver(policy=ResolutionPolicy.CONTAINMENT, threshold=1.0)
    gold = span(100, 200)
    assert resolver.is_relevant(span(0, 1000), gold)
    assert not resolver.is_relevant(span(100, 199), gold)  # one character short


def test_score_reports_the_policy_value() -> None:
    gold = span(0, 100)
    assert SpanResolver().score(span(0, 50), gold) == 0.5
    assert SpanResolver(policy=ResolutionPolicy.IOU).score(span(0, 50), gold) == 0.5
    assert SpanResolver(policy=ResolutionPolicy.CONTAINMENT).score(span(0, 50), gold) == 0.0


# ---------------------------------------------------------------------------
# grades
# ---------------------------------------------------------------------------


def test_a_chunk_takes_the_highest_grade_among_the_gold_it_satisfies() -> None:
    question = item(GoldSpan(span(0, 100), grade=1), GoldSpan(span(100, 200), grade=2))
    resolution = SpanResolver().resolve_item(question, [chunk("c0", 0, 200)])
    assert resolution.as_qrel() == {"c0": 2}


def test_grade_zero_gold_produces_no_label() -> None:
    question = item(GoldSpan(span(0, 100), grade=0))
    resolution = SpanResolver().resolve_item(question, [chunk("c0", 0, 100)])
    assert resolution.labels == ()


# ---------------------------------------------------------------------------
# split gold
# ---------------------------------------------------------------------------


def test_gold_split_evenly_across_two_chunks_is_flagged() -> None:
    """Gold 90-110, chunk boundary at 100. Each chunk holds exactly half.

    At a 0.5 threshold both chunks qualify, so it is not reported as split.
    """
    question = item(GoldSpan(span(90, 110)))
    chunks = [chunk("c0", 0, 100), chunk("c1", 100, 200)]
    resolution = SpanResolver(threshold=0.5).resolve_item(question, chunks)
    assert set(resolution.relevant_chunk_ids) == {"c0", "c1"}


def test_gold_split_unevenly_below_threshold_raises_a_split_warning() -> None:
    """Gold 95-125 across a boundary at 100: 5 chars one side, 25 the other.

    At a 0.9 threshold neither chunk qualifies alone, but together they hold all of it.
    A per-chunk-only scorer would call this a total miss, which is not true.
    """
    question = item(GoldSpan(span(95, 125)))
    chunks = [chunk("c0", 0, 100), chunk("c1", 100, 200)]
    resolution = SpanResolver(threshold=0.9).resolve_item(question, chunks)

    assert resolution.relevant_chunk_ids == ()
    assert resolution.per_gold[0].is_split
    assert resolution.per_gold[0].union_coverage == 1.0
    assert resolution.warnings.of_code(WarningCode.SPLIT_GOLD_SPAN)


def test_unreachable_gold_is_flagged_separately_from_split_gold() -> None:
    question = item(GoldSpan(span(5000, 5100)))
    resolution = SpanResolver().resolve_item(question, [chunk("c0", 0, 100)])

    assert resolution.has_unreachable_gold
    assert not resolution.per_gold[0].is_split
    assert resolution.per_gold[0].union_coverage == 0.0
    assert resolution.warnings.of_code(WarningCode.GOLD_SPAN_UNREACHABLE)


def test_best_score_is_recorded_even_when_nothing_qualifies() -> None:
    question = item(GoldSpan(span(0, 100)))
    resolution = SpanResolver(threshold=0.9).resolve_item(question, [chunk("c0", 0, 40)])
    assert resolution.per_gold[0].best_score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------


def test_approximate_offsets_are_reported() -> None:
    question = item(GoldSpan(span(0, 100)))
    chunks = [chunk("c0", 0, 100, exact=False)]
    resolution = SpanResolver().resolve_item(question, chunks)
    assert resolution.warnings.of_code(WarningCode.APPROXIMATE_RESOLUTION)


def test_exact_offsets_produce_no_approximation_warning() -> None:
    question = item(GoldSpan(span(0, 100)))
    resolution = SpanResolver().resolve_item(question, [chunk("c0", 0, 100)])
    assert not resolution.warnings.of_code(WarningCode.APPROXIMATE_RESOLUTION)


def test_unanswerable_items_are_reported_and_produce_no_labels() -> None:
    question = EvalItem(id="q9", question="Deliberately unanswerable")
    resolution = SpanResolver().resolve_item(question, [chunk("c0", 0, 100)])
    assert resolution.labels == ()
    assert resolution.warnings.of_code(WarningCode.GOLD_SPAN_UNREACHABLE)


# ---------------------------------------------------------------------------
# eval-set level
# ---------------------------------------------------------------------------


def test_qrels_have_the_shape_ranx_expects() -> None:
    evalset = EvalSet(
        id="es",
        items=(
            item(GoldSpan(span(0, 100)), iid="q1"),
            item(GoldSpan(span(100, 200), grade=1), iid="q2"),
        ),
    )
    chunks = [chunk("c0", 0, 100), chunk("c1", 100, 200)]
    assert SpanResolver().qrels(evalset, chunks) == {"q1": {"c0": 2}, "q2": {"c1": 1}}


def test_questions_with_no_resolvable_gold_are_left_out_of_qrels() -> None:
    """Including them empty would score as a legitimate zero and drag the mean down for a
    reason that has nothing to do with the retriever."""
    evalset = EvalSet(
        id="es",
        items=(
            item(GoldSpan(span(0, 100)), iid="q1"),
            item(GoldSpan(span(9000, 9100)), iid="q2"),
            EvalItem(id="q3", question="unanswerable on purpose"),
        ),
    )
    qrels = SpanResolver().qrels(evalset, [chunk("c0", 0, 100)])
    assert set(qrels) == {"q1"}


def test_resolve_collects_warnings_across_the_whole_set() -> None:
    evalset = EvalSet(
        id="es",
        items=(
            item(GoldSpan(span(9000, 9100)), iid="q1"),
            item(GoldSpan(span(8000, 8100)), iid="q2"),
        ),
    )
    resolutions, log = SpanResolver().resolve(evalset, [chunk("c0", 0, 100)])
    assert set(resolutions) == {"q1", "q2"}
    assert len(log.of_code(WarningCode.GOLD_SPAN_UNREACHABLE)) == 2


# ---------------------------------------------------------------------------
# character-level measures
# ---------------------------------------------------------------------------


def test_character_recall_is_union_based() -> None:
    question = item(GoldSpan(span(90, 110)))
    # Neither chunk holds the whole gold span; together they hold all of it.
    assert character_recall(question, [chunk("c0", 0, 100), chunk("c1", 100, 200)]) == 1.0
    assert character_recall(question, [chunk("c0", 0, 100)]) == 0.5


def test_character_recall_without_gold_is_zero() -> None:
    question = EvalItem(id="q", question="no gold")
    assert character_recall(question, [chunk("c0", 0, 100)]) == 0.0


def test_character_precision_exposes_context_waste() -> None:
    """The failure chunk-level Recall@k applauds.

    A 5000-character chunk holds a 200-character gold span. Recall says 1.0. Precision says
    the generator is paying for 25x the text it needed.
    """
    question = item(GoldSpan(span(1000, 1200)))
    retrieved = [chunk("huge", 0, 5000)]
    assert character_recall(question, retrieved) == 1.0
    assert character_precision(question, retrieved) == pytest.approx(0.04)


def test_character_precision_with_a_tight_chunk() -> None:
    question = item(GoldSpan(span(1000, 1200)))
    assert character_precision(question, [chunk("tight", 1000, 1200)]) == 1.0


def test_character_precision_with_nothing_retrieved_is_zero() -> None:
    assert character_precision(item(GoldSpan(span(0, 10))), []) == 0.0


def test_character_measures_ignore_overlapping_retrieved_chunks() -> None:
    """Overlapping chunks must not be charged twice for the same characters."""
    question = item(GoldSpan(span(0, 100)))
    overlapping = [chunk("c0", 0, 100), chunk("c1", 50, 150)]
    assert retrieved_character_count(overlapping) == 150
    assert character_precision(question, overlapping) == pytest.approx(100 / 150)
    assert character_recall(question, overlapping) == 1.0


def test_character_f1_balances_the_two() -> None:
    question = item(GoldSpan(span(0, 100)))
    assert character_f1(question, [chunk("c0", 0, 100)]) == 1.0
    assert character_f1(question, []) == 0.0


def test_gold_coverage_by_chunk_reports_a_fraction_per_chunk() -> None:
    question = item(GoldSpan(span(90, 110)))
    coverage = gold_coverage_by_chunk(question, [chunk("c0", 0, 100), chunk("c1", 100, 200)])
    assert coverage == {"c0": 0.5, "c1": 0.5}


def test_gold_coverage_by_chunk_without_gold_is_empty() -> None:
    assert gold_coverage_by_chunk(EvalItem(id="q", question="none"), [chunk("c0", 0, 10)]) == {}


# ---------------------------------------------------------------------------
# documents do not leak into one another
# ---------------------------------------------------------------------------


def test_chunks_from_another_document_are_never_relevant() -> None:
    question = item(GoldSpan(span(0, 100)))
    elsewhere = [chunk("other", 0, 100, doc="annex")]
    resolution = SpanResolver().resolve_item(question, elsewhere)
    assert resolution.relevant_chunk_ids == ()
    assert character_recall(question, elsewhere) == 0.0

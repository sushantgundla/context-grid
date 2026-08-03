"""Property tests for gold-span resolution.

These check the laws the scorer must obey whatever the corpus, the chunker or the gold looks
like. If one of these breaks, published numbers would be wrong in a way no leaderboard shows.
"""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from contextgrid import (
    Chunk,
    EvalItem,
    GoldSpan,
    ResolutionPolicy,
    Span,
    SpanResolver,
    character_precision,
    character_recall,
    gold_coverage_by_chunk,
)

DOC = "doc"
THRESHOLDS = st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False)
POLICIES = st.sampled_from(list(ResolutionPolicy))


@st.composite
def gold_spans(draw: st.DrawFn) -> GoldSpan:
    start = draw(st.integers(min_value=0, max_value=1500))
    length = draw(st.integers(min_value=1, max_value=300))
    grade = draw(st.integers(min_value=1, max_value=3))
    return GoldSpan(Span(DOC, start, start + length), grade=grade)


@st.composite
def chunk_sets(draw: st.DrawFn) -> list[Chunk]:
    """A contiguous, non-overlapping chunking of a document, as a real chunker produces."""
    size = draw(st.integers(min_value=20, max_value=600))
    count = draw(st.integers(min_value=1, max_value=15))
    return [
        Chunk(id=f"c{i}", span=Span(DOC, i * size, (i + 1) * size), text="x" * size)
        for i in range(count)
    ]


@st.composite
def eval_items(draw: st.DrawFn) -> EvalItem:
    gold = draw(st.lists(gold_spans(), min_size=1, max_size=4))
    return EvalItem(id="q", question="a question", gold=tuple(gold))


# ---------------------------------------------------------------------------
# resolution laws
# ---------------------------------------------------------------------------


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_resolution_does_not_depend_on_chunk_ordering(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    """A chunker that emits the same chunks in a different order must score identically."""
    resolver = SpanResolver(policy=policy, threshold=threshold)
    forward = resolver.resolve_item(item, chunks).as_qrel()
    backward = resolver.resolve_item(item, list(reversed(chunks))).as_qrel()
    assert forward == backward


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_every_label_points_at_a_real_chunk(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    resolver = SpanResolver(policy=policy, threshold=threshold)
    known = {c.id for c in chunks}
    resolution = resolver.resolve_item(item, chunks)
    assert all(label.chunk_id in known for label in resolution.labels)


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_labels_are_unique_per_chunk(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    """A chunk relevant to several gold spans gets one label, at the highest grade."""
    resolution = SpanResolver(policy=policy, threshold=threshold).resolve_item(item, chunks)
    ids = [label.chunk_id for label in resolution.labels]
    assert len(ids) == len(set(ids))


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_every_grade_comes_from_the_gold(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    resolution = SpanResolver(policy=policy, threshold=threshold).resolve_item(item, chunks)
    grades = {g.grade for g in item.gold}
    assert all(label.grade in grades for label in resolution.labels)


@given(eval_items(), chunk_sets(), THRESHOLDS)
def test_lowering_the_threshold_never_loses_a_match(
    item: EvalItem, chunks: list[Chunk], threshold: float
) -> None:
    """Relevance is monotone in the threshold: an easier bar cannot reject more."""
    assume(threshold > 0.1)
    strict = SpanResolver(threshold=threshold).resolve_item(item, chunks)
    lenient = SpanResolver(threshold=threshold / 2).resolve_item(item, chunks)
    assert set(strict.relevant_chunk_ids) <= set(lenient.relevant_chunk_ids)


@given(eval_items(), chunk_sets(), THRESHOLDS)
def test_containment_is_the_strictest_policy(
    item: EvalItem, chunks: list[Chunk], threshold: float
) -> None:
    """Anything a chunk fully contains, it also covers fully. So containment matches are
    always a subset of coverage matches."""
    contained = SpanResolver(policy=ResolutionPolicy.CONTAINMENT).resolve_item(item, chunks)
    covered = SpanResolver(policy=ResolutionPolicy.COVERAGE, threshold=threshold).resolve_item(
        item, chunks
    )
    assert set(contained.relevant_chunk_ids) <= set(covered.relevant_chunk_ids)


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_a_gold_span_is_reachable_or_split_or_absent_but_never_two_of_them(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    resolution = SpanResolver(policy=policy, threshold=threshold).resolve_item(item, chunks)
    for gold in resolution.per_gold:
        assert not (gold.is_reachable and gold.is_split)


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_union_coverage_is_a_fraction(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    resolution = SpanResolver(policy=policy, threshold=threshold).resolve_item(item, chunks)
    assert all(0.0 <= g.union_coverage <= 1.0 for g in resolution.per_gold)


@given(eval_items(), chunk_sets(), POLICIES, THRESHOLDS)
def test_a_matched_gold_span_always_has_some_union_coverage(
    item: EvalItem, chunks: list[Chunk], policy: ResolutionPolicy, threshold: float
) -> None:
    """If a chunk qualified, the chunk set must cover at least part of that gold span."""
    resolution = SpanResolver(policy=policy, threshold=threshold).resolve_item(item, chunks)
    for gold in resolution.per_gold:
        if gold.is_reachable:
            assert gold.union_coverage > 0.0


# ---------------------------------------------------------------------------
# character-level laws
# ---------------------------------------------------------------------------


@given(eval_items(), chunk_sets())
def test_character_measures_are_fractions(item: EvalItem, chunks: list[Chunk]) -> None:
    assert 0.0 <= character_recall(item, chunks) <= 1.0
    assert 0.0 <= character_precision(item, chunks) <= 1.0


@given(eval_items(), chunk_sets(), st.integers(min_value=1, max_value=15))
def test_character_recall_is_monotone_in_k(item: EvalItem, chunks: list[Chunk], k: int) -> None:
    """Retrieving more can never cover less of the gold."""
    assert character_recall(item, chunks[:k]) <= character_recall(item, chunks)


@given(eval_items(), chunk_sets())
def test_character_recall_ignores_retrieval_order(item: EvalItem, chunks: list[Chunk]) -> None:
    assert character_recall(item, chunks) == character_recall(item, list(reversed(chunks)))


@given(eval_items())
def test_retrieving_the_gold_itself_gives_perfect_scores(item: EvalItem) -> None:
    exact = [
        Chunk(id=f"g{i}", span=g.span, text="x" * g.span.length) for i, g in enumerate(item.gold)
    ]
    assert character_recall(item, exact) == 1.0
    assert character_precision(item, exact) == 1.0


@given(eval_items(), chunk_sets())
def test_gold_coverage_by_chunk_sums_to_at_least_total_recall(
    item: EvalItem, chunks: list[Chunk]
) -> None:
    """Per-chunk coverage sums to the union recall for non-overlapping chunks, and to more
    when chunks overlap -- never to less."""
    per_chunk = gold_coverage_by_chunk(item, chunks)
    assert sum(per_chunk.values()) >= character_recall(item, chunks) - 1e-9


@given(eval_items(), chunk_sets())
def test_every_chunk_gets_a_coverage_entry(item: EvalItem, chunks: list[Chunk]) -> None:
    per_chunk = gold_coverage_by_chunk(item, chunks)
    assert set(per_chunk) == {c.id for c in chunks}
    assert all(0.0 <= v <= 1.0 for v in per_chunk.values())

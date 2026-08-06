"""Scoring an embedder against a corpus, with no questions at all.

MTEB says a model scores 63.5 averaged over 56 public datasets. None of them are your
documents. Recall would answer the question properly and needs an eval set -- which nobody has
at the moment they are choosing an embedder.

So these tests are built around synthetic corpora whose *right answer is known*: vectors that
are deliberately clustered, deliberately collapsed, deliberately templated. A diagnostic that
cannot tell those apart is worse than none, because it looks like information.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.embed.quality import (
    EmbeddingQuality,
    EmbeddingQualityError,
    assess,
)


def chunks_for(documents: dict[str, int]) -> list[Chunk]:
    """Chunks laid out in reading order, `count` of them per document."""
    out: list[Chunk] = []
    for doc, count in documents.items():
        for index in range(count):
            start = index * 100
            out.append(
                Chunk(
                    id=f"{doc}:{start}",
                    span=Span(doc, start, start + 100),
                    text=f"{doc} section {index}: "
                    + " ".join(f"word{doc[0]}{index}x{n}" for n in range(12)),
                )
            )
    return out


def clustered(chunks: list[Chunk], dimensions: int = 64, spread: float = 0.05) -> np.ndarray:
    """Vectors where each document sits in its own direction and neighbours are close.

    A well-behaved embedder on a well-behaved corpus: documents separated, adjacent passages
    similar, the space actually used.
    """
    rng = np.random.default_rng(0)
    centres: dict[str, np.ndarray] = {}
    rows = []
    for chunk in chunks:
        if chunk.doc_id not in centres:
            centres[chunk.doc_id] = rng.normal(size=dimensions)
        rows.append(centres[chunk.doc_id] + rng.normal(scale=spread, size=dimensions))
    return np.asarray(rows, dtype=np.float32)


def collapsed(chunks: list[Chunk], dimensions: int = 64) -> np.ndarray:
    """Every chunk on the same point. The worst possible embedder."""
    return np.tile(
        np.array([[1.0, *([0.0] * (dimensions - 1))]], dtype=np.float32), (len(chunks), 1)
    )


def templated(chunks: list[Chunk], dimensions: int = 64) -> np.ndarray:
    """Documents that resemble each other more than they cohere internally.

    The shape a corpus of the same contract with different company names makes: section 1 of
    contract A is far closer to section 1 of contract B than to section 2 of contract A.
    """
    rng = np.random.default_rng(1)
    sections: dict[int, np.ndarray] = {}
    rows = []
    position: dict[str, int] = {}
    for chunk in chunks:
        index = position.get(chunk.doc_id, 0)
        position[chunk.doc_id] = index + 1
        if index not in sections:
            sections[index] = rng.normal(size=dimensions)
        rows.append(sections[index] + rng.normal(scale=0.02, size=dimensions))
    return np.asarray(rows, dtype=np.float32)


CORPUS = {"a.md": 5, "b.md": 5, "c.md": 5, "d.md": 5}


# ---------------------------------------------------------------------------
# it tells a good embedder from a bad one
# ---------------------------------------------------------------------------


def test_a_clustered_embedder_scores_far_above_a_collapsed_one() -> None:
    """The whole claim. If these come out similar, the diagnostic is decoration."""
    chunks = chunks_for(CORPUS)

    good = assess(chunks, clustered(chunks))
    bad = assess(chunks, collapsed(chunks))

    assert good.score > 0.5
    assert bad.score < 0.1


def test_a_collapsed_embedder_is_named_as_degenerate() -> None:
    """Nothing else is worth printing once this is true -- every other number is a restatement
    of it, and reporting them as findings implies they mean something separately."""
    chunks = chunks_for(CORPUS)
    result = assess(chunks, collapsed(chunks))

    assert result.degenerate
    assert result.collapsed == pytest.approx(1.0)
    assert "degenerate" in result.summary()


def test_a_good_embedder_puts_neighbours_closer_than_strangers() -> None:
    """Positive coherence is what "this model understands the corpus" looks like."""
    chunks = chunks_for(CORPUS)
    result = assess(chunks, clustered(chunks))

    assert result.separation > 0
    assert result.measurable
    assert not result.degenerate


def test_anisotropy_notices_a_crowded_space() -> None:
    """Transformer embeddings crowd into a narrow cone, and a model reporting 0.85 between a
    refund policy and a shipping schedule has no room left to express a match."""
    chunks = chunks_for(CORPUS)

    rng = np.random.default_rng(2)
    base = rng.normal(size=64)
    crowded = np.asarray([base + rng.normal(scale=0.1, size=64) for _ in chunks], dtype=np.float32)

    assert assess(chunks, crowded).crowded
    assert not assess(chunks, clustered(chunks)).crowded


# ---------------------------------------------------------------------------
# it tells a bad corpus from a bad model
# ---------------------------------------------------------------------------


def test_a_templated_corpus_is_reported_as_a_fact_about_the_documents() -> None:
    """The single most useful thing to know before blaming an embedder for poor retrieval:
    section 1 of contract A is closer to section 1 of contract B than to section 2 of A, and no
    embedder will separate those cleanly."""
    chunks = chunks_for(CORPUS)
    result = assess(chunks, templated(chunks))

    assert result.templated
    # A difference, not a ratio: cosines go negative, and a ratio with a negative denominator
    # silently inverts the comparison -- which is what it did on the first templated corpus.
    assert result.redundancy > 0
    assert result.separation == pytest.approx(-result.redundancy)
    assert "templated corpus" in result.summary()


def test_a_healthy_corpus_is_not_called_templated() -> None:
    chunks = chunks_for(CORPUS)
    assert not assess(chunks, clustered(chunks)).templated


def test_a_collapsed_model_is_never_blamed_on_the_corpus() -> None:
    """An embedder whose vectors have collapsed reports everything as similar to everything,
    corpus included. Calling that a templated corpus blames the documents for the model."""
    chunks = chunks_for(CORPUS)
    result = assess(chunks, collapsed(chunks))

    assert result.degenerate
    assert not result.templated


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------


def test_it_counts_the_dimensions_actually_carrying_variance() -> None:
    """A 1536-dimension model living in a 40-dimension subspace charges for 1536 and thinks in
    40, and will lose to a smaller model that uses what it has."""
    chunks = chunks_for({"a.md": 20, "b.md": 20})

    rng = np.random.default_rng(3)
    wide = rng.normal(size=(len(chunks), 128)).astype(np.float32)
    # The same vectors squeezed into a three-dimensional subspace of a 128-wide model.
    narrow = np.zeros((len(chunks), 128), dtype=np.float32)
    narrow[:, :3] = rng.normal(size=(len(chunks), 3))

    # Bounded by the sample count as well as the width: 40 points cannot span more than 39
    # dimensions however wide the model is.
    assert assess(chunks, wide).effective_dimensions > 20
    assert assess(chunks, narrow).effective_dimensions < 5
    assert assess(chunks, narrow).dimension_efficiency < 0.05


def test_the_width_paid_for_is_recorded_alongside() -> None:
    chunks = chunks_for(CORPUS)
    assert assess(chunks, clustered(chunks, dimensions=64)).dimensions == 64


# ---------------------------------------------------------------------------
# when it cannot say anything
# ---------------------------------------------------------------------------


def test_one_chunk_per_document_makes_coherence_unmeasurable() -> None:
    """There is no adjacency to measure. Reporting zero would read as "the model found no
    structure" when the truth is that there was none to find."""
    chunks = chunks_for({f"doc{i}.md": 1 for i in range(20)})
    result = assess(chunks, clustered(chunks))

    assert not result.measurable
    assert "unmeasurable" in result.summary()


def test_an_unmeasurable_coherence_is_dropped_from_the_score_not_scored_zero() -> None:
    """The same rule the composite follows for a dimension nobody ran."""
    singles = chunks_for({f"doc{i}.md": 1 for i in range(20)})
    result = assess(singles, clustered(singles))

    assert result.score > 0.0


def test_a_corpus_too_small_to_say_anything_is_refused() -> None:
    """These are statements about the shape of a cloud of points, and three points have no
    shape worth describing."""
    chunks = chunks_for({"a.md": 3})
    with pytest.raises(EmbeddingQualityError, match="too small"):
        assess(chunks, clustered(chunks))


def test_chunks_and_vectors_out_of_step_is_refused() -> None:
    chunks = chunks_for(CORPUS)
    with pytest.raises(EmbeddingQualityError, match="out of step"):
        assess(chunks, clustered(chunks)[:5])


# ---------------------------------------------------------------------------
# the numbers travel
# ---------------------------------------------------------------------------


def test_the_same_corpus_scores_the_same_twice() -> None:
    """It samples, so it seeds. A diagnostic that moves between identical runs cannot compare
    two embedders."""
    chunks = chunks_for(CORPUS)
    vectors = clustered(chunks)
    assert assess(chunks, vectors).score == assess(chunks, vectors).score


def test_it_scales_to_a_corpus_too_large_to_compare_exhaustively() -> None:
    """The full pairwise matrix is quadratic: 50,000 chunks is 1.25 billion pairs to answer a
    question a few thousand samples settle."""
    chunks = chunks_for({f"doc{i}.md": 20 for i in range(60)})
    result = assess(chunks, clustered(chunks), sample=200)

    assert result.sampled == 200
    assert result.score > 0


def test_the_numbers_go_into_a_report() -> None:
    import json

    chunks = chunks_for(CORPUS)
    payload = assess(chunks, clustered(chunks)).as_dict()

    assert json.loads(json.dumps(payload))["embedding_quality"] > 0
    assert "anisotropy" in payload
    assert "redundancy" in payload


def test_it_feeds_the_composite_as_its_own_dimension() -> None:
    """Someone with a corpus and no questions still gets a score, over the one dimension that
    can be measured without them."""
    from contextgrid.report.composite import composite

    chunks = chunks_for(CORPUS)
    metrics = assess(chunks, clustered(chunks)).as_dict()

    result = composite(metrics, dimensions={"embed": ("embedding_quality",)})
    assert result.dimensions == ("embed",)
    assert result.score > 0


def test_the_score_stays_inside_zero_and_one() -> None:
    chunks = chunks_for(CORPUS)
    for vectors in (clustered(chunks), collapsed(chunks), templated(chunks)):
        assert 0.0 <= assess(chunks, vectors).score <= 1.0


def test_a_quality_report_is_worth_inspecting_directly() -> None:
    result = EmbeddingQuality(
        anisotropy=0.2,
        separation=0.3,
        redundancy=0.4,
        effective_dimensions=100.0,
        dimensions=768,
        collapsed=0.0,
        sampled=500,
    )
    assert result.dimension_efficiency == pytest.approx(100 / 768)
    assert not result.crowded

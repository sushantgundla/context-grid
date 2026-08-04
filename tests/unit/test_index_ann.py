"""Approximate indexes: faiss and usearch.

The point of these is the chart that had nothing to plot before them -- what approximation
actually cost. So the tests are mostly about honesty: that `is_exact` tells the truth, that
recall against exhaustive search is measurable and sane, that a small corpus does not quietly
train a bad codebook, and that scores keep one larger-is-better convention across every arm of
the axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.embed.base import Vectors
from contextgrid.index import get_index
from contextgrid.index.dense import ExactDenseIndex, IndexBuildError

pytest.importorskip("faiss")
pytest.importorskip("usearch")

from contextgrid.index.ann import (
    FaissIndex,
    USearchIndex,
    _fit_pq_bits,
    _largest_divisor,
)

DIMENSIONS = 64
COUNT = 400


@pytest.fixture(scope="module")
def corpus() -> tuple[list[Chunk], Vectors, Vectors]:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(COUNT, DIMENSIONS)).astype(np.float32)
    chunks = [
        Chunk(id=f"c{i}", span=Span("doc", i, i + 1), text=f"chunk {i}") for i in range(COUNT)
    ]
    queries = rng.normal(size=(20, DIMENSIONS)).astype(np.float32)
    return chunks, vectors, queries


def recall_of(index: object, corpus: tuple[list[Chunk], Vectors, Vectors], k: int = 10) -> float:
    chunks, vectors, queries = corpus
    exact = ExactDenseIndex()
    exact.build(chunks, vectors)
    index.build(chunks, vectors)  # type: ignore[attr-defined]

    scores = []
    for query in queries:
        truth = {s.chunk_id for s in exact.search("", query, k)}
        found = {s.chunk_id for s in index.search("", query, k)}  # type: ignore[attr-defined]
        scores.append(len(found & truth) / len(truth))
    return float(np.mean(scores))


ANN_INDEXES = [
    FaissIndex(kind="flat"),
    FaissIndex(kind="hnsw"),
    FaissIndex(kind="ivf", nlist=10, nprobe=4),
    FaissIndex(kind="ivfpq", nlist=10, nprobe=10),
    USearchIndex(),
    USearchIndex(dtype="f16"),
    USearchIndex(dtype="i8"),
]
IDS = [
    "faiss:flat",
    "faiss:hnsw",
    "faiss:ivf",
    "faiss:ivfpq",
    "usearch",
    "usearch:f16",
    "usearch:i8",
]


# ---------------------------------------------------------------------------
# honesty about approximation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_approximate_indexes_say_they_are_approximate(index: object) -> None:
    """An approximate index whose numbers are read as exact is worse than not having one."""
    assert index.is_exact == (getattr(index, "kind", "") == "flat")  # type: ignore[attr-defined]


def test_faiss_flat_finds_exactly_what_exhaustive_search_finds(
    corpus: tuple[list[Chunk], Vectors, Vectors],
) -> None:
    """It is the reference the other three are judged against, so it has to be exact -- and
    computed by the same library, or the comparison measures implementation differences."""
    assert recall_of(FaissIndex(kind="flat"), corpus) == 1.0


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_every_index_finds_most_of_what_exact_search_finds(
    index: object, corpus: tuple[list[Chunk], Vectors, Vectors]
) -> None:
    """Not a quality bar -- a sanity bar. An index below this is broken, not approximate."""
    assert recall_of(index, corpus) > 0.1


def test_searching_more_clusters_finds_more(
    corpus: tuple[list[Chunk], Vectors, Vectors],
) -> None:
    """The recall/latency knob has to actually turn, or the axis measures nothing."""
    narrow = recall_of(FaissIndex(kind="ivf", nlist=10, nprobe=1), corpus)
    wide = recall_of(FaissIndex(kind="ivf", nlist=10, nprobe=10), corpus)
    assert wide > narrow


def test_quantized_storage_costs_recall_and_saves_memory(
    corpus: tuple[list[Chunk], Vectors, Vectors],
) -> None:
    """The trade the dtype axis exists to measure, in one test."""
    full, small = USearchIndex(dtype="f32"), USearchIndex(dtype="i8")
    assert recall_of(full, corpus) >= recall_of(small, corpus)
    assert small.size_bytes() < full.size_bytes()


def test_usearch_memory_is_computed_from_the_dtype() -> None:
    """usearch's own `memory_usage` reports the arena it allocated, which barely moves between
    f32 and i8 -- using it would show quantization saving nothing, which is the opposite of
    true and the entire reason the dtype is on the axis."""
    chunks = [Chunk(id=f"c{i}", span=Span("d", i, i + 1), text="x") for i in range(100)]
    vectors = np.ones((100, DIMENSIONS), dtype=np.float32)

    sizes = {}
    for dtype in ("f32", "f16", "i8"):
        index = USearchIndex(dtype=dtype)
        index.build(chunks, vectors)
        sizes[dtype] = index.size_bytes()

    assert sizes["f32"] > sizes["f16"] > sizes["i8"]


# ---------------------------------------------------------------------------
# fitting the parameters to the corpus
# ---------------------------------------------------------------------------


def test_a_codebook_too_large_for_the_corpus_is_shrunk_and_reported(
    corpus: tuple[list[Chunk], Vectors, Vectors],
) -> None:
    """Product quantization learns 2**bits centroids per subspace and faiss wants ~39 points
    for each, so the default 8 bits needs ~10,000 vectors. Training it anyway produces an index
    that returns plausible, wrong neighbours -- and prints a warning nobody reads."""
    chunks, vectors, _ = corpus
    index = FaissIndex(kind="ivfpq", nlist=10, pq_bits=8)
    index.build(chunks, vectors)

    assert index.fitted_to_corpus["pq_bits"] == (8, 3)


def test_a_cluster_count_larger_than_the_corpus_is_shrunk_and_reported() -> None:
    chunks = [Chunk(id=f"c{i}", span=Span("d", i, i + 1), text="x") for i in range(50)]
    vectors = np.random.default_rng(1).normal(size=(50, 8)).astype(np.float32)

    index = FaissIndex(kind="ivf", nlist=100)
    index.build(chunks, vectors)
    assert index.fitted_to_corpus["nlist"][1] < 100


def test_nothing_is_reported_when_nothing_had_to_change(
    corpus: tuple[list[Chunk], Vectors, Vectors],
) -> None:
    chunks, vectors, _ = corpus
    index = FaissIndex(kind="ivf", nlist=10)
    index.build(chunks, vectors)
    assert not index.fitted_to_corpus


@pytest.mark.parametrize(
    ("wanted", "count", "expected"),
    [(8, 100_000, 8), (8, 400, 3), (8, 78, 1), (4, 10, 1)],
)
def test_the_codebook_is_the_largest_the_corpus_can_train(
    wanted: int, count: int, expected: int
) -> None:
    assert _fit_pq_bits(wanted, count) == expected


@pytest.mark.parametrize(
    ("dimensions", "wanted", "expected"),
    [(768, 8, 8), (384, 8, 8), (100, 8, 5), (7, 8, 7), (13, 8, 1)],
)
def test_subquantizers_always_divide_the_vector_width(
    dimensions: int, wanted: int, expected: int
) -> None:
    """PQ splits the vector into equal subspaces, so the count has to divide the width exactly.
    faiss raises otherwise, which would make ivfpq unusable on any model whose width is not a
    multiple of eight."""
    assert _largest_divisor(dimensions, wanted) == expected


def test_ivfpq_works_on_an_awkward_width() -> None:
    chunks = [Chunk(id=f"c{i}", span=Span("d", i, i + 1), text="x") for i in range(200)]
    vectors = np.random.default_rng(2).normal(size=(200, 13)).astype(np.float32)

    index = FaissIndex(kind="ivfpq", nlist=5)
    index.build(chunks, vectors)
    assert index.search("", vectors[0], k=3)


# ---------------------------------------------------------------------------
# the invariants every index shares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_scores_descend(index: object, corpus: tuple[list[Chunk], Vectors, Vectors]) -> None:
    """L2 backends return a distance where smaller is better; everything in this package is
    larger-is-better. One convention across the axis, or the leaderboard sorts backwards."""
    chunks, vectors, queries = corpus
    index.build(chunks, vectors)  # type: ignore[attr-defined]
    scores = [s.score for s in index.search("", queries[0], k=10)]  # type: ignore[attr-defined]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_no_more_than_k_results(
    index: object, corpus: tuple[list[Chunk], Vectors, Vectors]
) -> None:
    chunks, vectors, queries = corpus
    index.build(chunks, vectors)  # type: ignore[attr-defined]
    assert len(index.search("", queries[0], k=5)) <= 5  # type: ignore[attr-defined]


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_asking_for_more_than_the_corpus_holds_is_fine(index: object) -> None:
    """Backends pad short results with -1, which in Python indexes the *last* chunk and puts an
    unrelated passage in the results."""
    chunks = [Chunk(id=f"c{i}", span=Span("d", i, i + 1), text="x") for i in range(3)]
    vectors = np.eye(3, DIMENSIONS, dtype=np.float32)

    index.build(chunks, vectors)  # type: ignore[attr-defined]
    found = index.search("", vectors[0], k=50)  # type: ignore[attr-defined]
    assert len(found) <= 3
    assert {s.chunk_id for s in found} <= {"c0", "c1", "c2"}


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_an_empty_corpus_searches_to_nothing(index: object) -> None:
    index.build([], np.zeros((0, DIMENSIONS), dtype=np.float32))  # type: ignore[attr-defined]
    assert index.search("", np.zeros(DIMENSIONS, dtype=np.float32), k=5) == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_no_vectors_is_a_clear_error(index: object) -> None:
    with pytest.raises(IndexBuildError, match="needs vectors"):
        index.build([Chunk(id="c", span=Span("d", 0, 1), text="x")], None)  # type: ignore[attr-defined]


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_chunks_and_vectors_out_of_step_is_a_clear_error(index: object) -> None:
    with pytest.raises(IndexBuildError, match="out of step"):
        index.build(  # type: ignore[attr-defined]
            [Chunk(id="c", span=Span("d", 0, 1), text="x")],
            np.zeros((3, DIMENSIONS), dtype=np.float32),
        )


@pytest.mark.parametrize("index", ANN_INDEXES, ids=IDS)
def test_a_query_from_a_different_model_is_refused(
    index: object, corpus: tuple[list[Chunk], Vectors, Vectors]
) -> None:
    chunks, vectors, _ = corpus
    index.build(chunks, vectors)  # type: ignore[attr-defined]
    with pytest.raises(IndexBuildError, match="different models"):
        index.search("", np.zeros(7, dtype=np.float32), k=3)  # type: ignore[attr-defined]


@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_every_metric_works_and_ranks_the_obvious_match_first(metric: str) -> None:
    """`cosine` has to mean the same thing here as everywhere else in the package: normalise
    both sides, then inner product. Left to faiss's defaults it would be L2 on raw vectors,
    which ranks differently and turns the index axis into a metric comparison in disguise."""
    chunks = [Chunk(id=f"c{i}", span=Span("d", i, i + 1), text="x") for i in range(4)]
    vectors = np.eye(4, 8, dtype=np.float32) * 2.0

    index = FaissIndex(kind="flat", metric=metric)
    index.build(chunks, vectors)
    assert index.search("", vectors[2], k=1)[0].chunk_id == "c2"


def test_an_unknown_index_type_lists_the_real_ones() -> None:
    with pytest.raises(IndexBuildError, match="flat, hnsw, ivf, ivfpq"):
        FaissIndex(kind="magic")


def test_an_unknown_metric_lists_the_real_ones() -> None:
    with pytest.raises(IndexBuildError, match="cosine, dot, l2"):
        USearchIndex(metric="jaccard")


def test_an_unknown_dtype_lists_the_real_ones() -> None:
    with pytest.raises(IndexBuildError, match="f32, f16, i8, b1"):
        USearchIndex(dtype="f64")


# ---------------------------------------------------------------------------
# reachable from a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "faiss",
        "faiss:flat",
        "faiss:hnsw,ef_search=128",
        "faiss:ivf,nlist=20,nprobe=4",
        "faiss:ivfpq,pq_subquantizers=4",
        "usearch",
        "usearch:f16",
        "usearch:i8,connectivity=32",
    ],
)
def test_every_index_is_reachable_from_one_config_line(spec: str) -> None:
    assert get_index(spec).needs_vectors


def test_the_index_axis_can_now_sweep_exact_against_approximate() -> None:
    """The whole reason for this dimension: both on one axis, in one config."""
    from contextgrid.grid import matrix

    configs = matrix(index=["dense", "faiss:flat", "faiss:hnsw", "usearch:i8"]).expand("factorial")
    assert len({config.index for config in configs}) == 4


def test_recall_against_exact_measures_what_approximation_cost(
    corpus: tuple[list[Chunk], Vectors, Vectors],
) -> None:
    """`recall_against_exact` existed before any approximate index did. This is the first time
    it has something real to measure."""
    from contextgrid.index import recall_against_exact

    chunks, vectors, queries = corpus
    exact, approximate = ExactDenseIndex(), FaissIndex(kind="ivf", nlist=10, nprobe=1)
    exact.build(chunks, vectors)
    approximate.build(chunks, vectors)

    # It compares two result lists, so it works on any pair of indexes -- it was written for
    # quantization and needed no change to measure an ANN index instead.
    measured = [
        recall_against_exact(approximate.search("", query, 10), exact.search("", query, 10), 10)
        for query in queries
    ]
    assert 0.0 < sum(measured) / len(measured) < 1.0

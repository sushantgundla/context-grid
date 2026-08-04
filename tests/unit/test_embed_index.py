"""Unit tests for embedders, indexes and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.core.warnings import WarningCode
from contextgrid.embed import EMBEDDERS, HashEmbedder, TfidfEmbedder, get_embedder
from contextgrid.embed.base import Embedder, normalise, truncate
from contextgrid.embed.local import TokenCountEmbedder
from contextgrid.index import (
    INDEXES,
    BM25Index,
    ExactDenseIndex,
    FusionError,
    HybridIndex,
    IndexBuildError,
    Scored,
    get_index,
    reciprocal_rank_fusion,
    top_k,
    weighted_fusion,
)

DOCS = [
    "The notice period is thirty days for termination by either party.",
    "Fees are payable within thirty days of invoice date.",
    "Send your API key in the X-Api-Key header with every request.",
    "The widget endpoint returns 404 when the identifier is unknown.",
]

EMBEDDERS_UNDER_TEST = [HashEmbedder(dimensions=64), TfidfEmbedder(), TokenCountEmbedder()]
IDS = [e.name for e in EMBEDDERS_UNDER_TEST]


def chunks() -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", span=Span("d", i * 100, i * 100 + len(text)), text=text)
        for i, text in enumerate(DOCS)
    ]


@pytest.fixture(params=EMBEDDERS_UNDER_TEST, ids=IDS)
def embedder(request: pytest.FixtureRequest) -> Embedder:
    model: Embedder = request.param
    model.prepare(DOCS)
    return model


# ---------------------------------------------------------------------------
# the embedder contract
# ---------------------------------------------------------------------------


def test_satisfies_the_protocol(embedder: Embedder) -> None:
    assert isinstance(embedder, Embedder)


def test_one_vector_per_text(embedder: Embedder) -> None:
    result = embedder.embed_documents(DOCS)
    assert result.vectors.shape[0] == len(DOCS)
    assert result.count == len(DOCS)


def test_embedding_is_deterministic(embedder: Embedder) -> None:
    """Non-determinism would make caching and run comparison meaningless."""
    first = embedder.embed_documents(DOCS).vectors
    second = embedder.embed_documents(DOCS).vectors
    assert np.array_equal(first, second)


def test_queries_and_documents_use_the_same_dimensions(embedder: Embedder) -> None:
    assert (
        embedder.embed_queries(["how long is notice?"]).vectors.shape[1]
        == embedder.embed_documents(DOCS).vectors.shape[1]
    )


def test_normalised_models_return_unit_vectors(embedder: Embedder) -> None:
    if not embedder.normalised:
        pytest.skip(f"{embedder.name} does not claim to normalise")
    norms = np.linalg.norm(embedder.embed_documents(DOCS).vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedding_nothing_is_safe(embedder: Embedder) -> None:
    assert embedder.embed_documents([]).count == 0


# ---------------------------------------------------------------------------
# the specific embedders
# ---------------------------------------------------------------------------


def test_hash_embedder_puts_similar_text_closer() -> None:
    model = HashEmbedder(dimensions=512)
    vectors = model.embed_documents(
        ["notice period thirty days", "notice period thirty days termination", "api key header"]
    ).vectors
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_hash_embedder_needs_no_preparation() -> None:
    model = HashEmbedder()
    model.prepare([])  # a no-op, and it must not fail
    assert model.embed_documents(["text"]).count == 1


def test_tfidf_refuses_to_embed_before_it_has_seen_the_corpus() -> None:
    """Returning zero vectors would read as "this embedder is bad" rather than "unfitted"."""
    with pytest.raises(RuntimeError, match="prepare"):
        TfidfEmbedder().embed_documents(DOCS)


def test_tfidf_learns_a_vocabulary_from_the_corpus() -> None:
    model = TfidfEmbedder()
    model.prepare(DOCS)
    assert model.is_prepared
    assert model.dimensions > 10


def test_tfidf_weights_rare_words_above_common_ones() -> None:
    """The whole point of the idf term."""
    model = TfidfEmbedder()
    model.prepare(DOCS)
    common = model._vocabulary["thirty"]  # appears in two of four documents
    rare = model._vocabulary["header"]  # appears in one
    assert model._idf[rare] > model._idf[common]


def test_tfidf_ignores_words_it_has_never_seen() -> None:
    """Exactly as at serving time: an unseen query word contributes nothing."""
    model = TfidfEmbedder()
    model.prepare(DOCS)
    vector = model.embed_queries(["zzzzz qqqqq"]).vectors
    assert float(np.abs(vector).sum()) == 0.0


def test_tfidf_honours_a_feature_cap() -> None:
    model = TfidfEmbedder(max_features=5)
    model.prepare(DOCS)
    assert model.dimensions == 5


def test_the_length_embedder_is_deliberately_useless() -> None:
    """A control that should score near chance. If it does not, the scoring is broken."""
    model = TokenCountEmbedder()
    assert model.dimensions == 1
    assert model.embed_documents(["a b c"]).vectors[0][0] == 3.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_normalising_leaves_a_zero_vector_alone() -> None:
    """Dividing by a zero norm would give NaNs that spread into every similarity score."""
    vectors = normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert np.array_equal(vectors[0], np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(vectors[1], np.array([0.6, 0.8]))


def test_truncation_is_reported_loudly() -> None:
    texts, log, count = truncate(["x" * 10_000], max_tokens=10, model="tiny")
    assert count == 1
    assert len(texts[0]) == 40
    assert log.of_code(WarningCode.INPUT_TRUNCATED)


def test_many_truncations_are_summarised_rather_than_repeated() -> None:
    _, log, count = truncate(["x" * 1000] * 20, max_tokens=10, model="tiny")
    assert count == 20
    assert len(log) < 20
    assert any("20 inputs in total" in warning.message for warning in log)


def test_no_limit_means_no_truncation() -> None:
    texts, log, count = truncate(["x" * 10_000], max_tokens=None, model="big")
    assert count == 0
    assert not log
    assert len(texts[0]) == 10_000


# ---------------------------------------------------------------------------
# dense index
# ---------------------------------------------------------------------------


def build_dense() -> tuple[ExactDenseIndex, TfidfEmbedder]:
    model = TfidfEmbedder()
    model.prepare(DOCS)
    index = ExactDenseIndex()
    index.build(chunks(), model.embed_documents(DOCS).vectors)
    return index, model


def test_dense_search_finds_the_relevant_chunk() -> None:
    index, model = build_dense()
    vector = model.embed_queries(["which header carries the api key?"]).vectors[0]
    assert index.search("which header carries the api key?", vector, 1)[0].chunk_id == "c2"


def test_dense_returns_at_most_k() -> None:
    index, model = build_dense()
    vector = model.embed_queries(["notice"]).vectors[0]
    assert len(index.search("notice", vector, 2)) == 2


def test_dense_without_vectors_says_what_is_missing() -> None:
    index = ExactDenseIndex()
    with pytest.raises(IndexBuildError, match="needs vectors"):
        index.build(chunks(), None)


def test_a_chunk_and_vector_mismatch_is_caught() -> None:
    """Off-by-one here would misattribute every score to the wrong chunk."""
    index = ExactDenseIndex()
    with pytest.raises(IndexBuildError, match="out of step"):
        index.build(chunks(), np.zeros((2, 8), dtype=np.float32))


def test_querying_with_the_wrong_dimensions_says_why() -> None:
    index, _ = build_dense()
    with pytest.raises(IndexBuildError, match="different models"):
        index.search("q", np.zeros(3, dtype=np.float32), 1)


def test_an_unknown_metric_is_refused() -> None:
    with pytest.raises(IndexBuildError, match="Choose 'cosine' or 'dot'"):
        ExactDenseIndex(metric="manhattan")


def test_an_empty_dense_index_returns_nothing() -> None:
    index = ExactDenseIndex()
    index.build([], np.zeros((0, 4), dtype=np.float32))
    assert index.search("q", np.zeros(4, dtype=np.float32), 5) == []


def test_dense_reports_its_size() -> None:
    index, _ = build_dense()
    assert index.size_bytes() > 0


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_finds_the_chunk_with_the_query_terms() -> None:
    index = BM25Index()
    index.build(chunks())
    assert index.search("X-Api-Key header", None, 1)[0].chunk_id == "c2"


def test_bm25_needs_no_vectors_at_all() -> None:
    assert BM25Index.needs_vectors is False
    index = BM25Index()
    index.build(chunks(), None)
    assert index.search("notice period", None, 2)


def test_bm25_scores_nothing_for_a_query_with_no_shared_terms() -> None:
    index = BM25Index()
    index.build(chunks())
    assert index.search("zzzzz qqqqq", None, 5) == []


def test_bm25_parameters_change_the_ranking() -> None:
    """k1 and b are swept rather than assumed: the usual defaults were tuned on TREC news,
    not on 200-token chunks."""
    flat = BM25Index(k1=0.1, b=0.0)
    steep = BM25Index(k1=3.0, b=1.0)
    for index in (flat, steep):
        index.build(chunks())
    assert flat.search("thirty days", None, 4) != steep.search("thirty days", None, 4)


def test_an_empty_bm25_index_returns_nothing() -> None:
    index = BM25Index()
    index.build([])
    assert index.search("anything", None, 5) == []


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------


def test_rrf_uses_rank_and_ignores_score_magnitude() -> None:
    """A cosine of 0.31 and a BM25 score of 14.2 are not on the same scale, and no amount
    of normalisation makes them mean the same thing."""
    left = [Scored("a", 0.31), Scored("b", 0.30)]
    right = [Scored("a", 98.0), Scored("b", 2.0)]
    assert reciprocal_rank_fusion([left, right]) == reciprocal_rank_fusion(
        [[Scored("a", 1.0), Scored("b", 0.9)], [Scored("a", 1.0), Scored("b", 0.9)]]
    )


def test_rrf_rewards_agreement_between_the_two_sides() -> None:
    both = [Scored("a", 1.0)]
    scores = reciprocal_rank_fusion([both, [Scored("b", 1.0), Scored("a", 0.5)]])
    assert scores["a"] > scores["b"]


def test_rrf_rejects_a_nonsense_constant() -> None:
    with pytest.raises(FusionError, match="at least 1"):
        reciprocal_rank_fusion([[Scored("a", 1.0)]], k=0)


def test_weighted_fusion_respects_alpha() -> None:
    dense = [Scored("a", 1.0), Scored("b", 0.0)]
    sparse = [Scored("b", 1.0), Scored("a", 0.0)]
    assert weighted_fusion(dense, sparse, alpha=1.0)["a"] == 1.0
    assert weighted_fusion(dense, sparse, alpha=0.0)["a"] == 0.0


def test_weighted_fusion_keeps_a_document_only_one_side_found() -> None:
    scores = weighted_fusion([Scored("a", 1.0)], [Scored("b", 1.0)], alpha=0.5)
    assert set(scores) == {"a", "b"}


def test_weighted_fusion_handles_a_list_where_every_score_is_equal() -> None:
    scores = weighted_fusion([Scored("a", 5.0), Scored("b", 5.0)], [], alpha=1.0)
    assert scores == {"a": 1.0, "b": 1.0}


def test_hybrid_searches_both_sides() -> None:
    model = TfidfEmbedder()
    model.prepare(DOCS)
    index = HybridIndex(dense=ExactDenseIndex(), sparse=BM25Index())
    index.build(chunks(), model.embed_documents(DOCS).vectors)

    vector = model.embed_queries(["api key header"]).vectors[0]
    assert index.search("api key header", vector, 1)[0].chunk_id == "c2"
    assert len(index) == 4
    assert index.size_bytes() > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fusion": "magic"}, "Choose 'rrf' or 'weighted'"),
        ({"alpha": 1.5}, "between 0 and 1"),
        ({"candidates": 0}, "at least 1"),
    ],
)
def test_hybrid_rejects_nonsense_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(FusionError, match=message):
        HybridIndex(dense=ExactDenseIndex(), sparse=BM25Index(), **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ranking helper
# ---------------------------------------------------------------------------


def test_ties_break_deterministically() -> None:
    """A leaderboard that moves when nothing changed destroys trust in every other number."""
    scores = {"b": 1.0, "a": 1.0, "c": 1.0}
    assert [s.chunk_id for s in top_k(scores, 3)] == ["a", "b", "c"]


def test_top_k_returns_the_highest_first() -> None:
    assert [s.chunk_id for s in top_k({"a": 0.1, "b": 0.9}, 2)] == ["b", "a"]


# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------


def test_registries_know_the_built_ins() -> None:
    assert {"hash", "tfidf", "length"} <= set(EMBEDDERS.names())
    assert {"dense", "bm25", "hybrid"} <= set(INDEXES.names())


def test_spec_strings_configure_plugins() -> None:
    assert get_embedder("hash:512").dimensions == 512
    assert get_index("bm25:0.9").k1 == 0.9  # type: ignore[union-attr]


def test_hybrid_spec_configures_both_sides() -> None:
    index = get_index("hybrid:weighted,alpha=0.8,k1=2.0")
    assert isinstance(index, HybridIndex)
    assert index.fusion == "weighted"
    assert index.alpha == 0.8
    assert index.sparse.k1 == 2.0  # type: ignore[union-attr]

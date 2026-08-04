"""Unit tests for vector quantization.

The interesting assertions are the ones about *loss*. A compression scheme that reports its
ratio and not its recall is telling you half the story, and the half it leaves out is the one
that decides whether to use it.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.core.warnings import WarningCode
from contextgrid.index import ExactDenseIndex, QuantizationError, QuantizedDenseIndex
from contextgrid.index.quantize import (
    BinaryCodec,
    ProductCodec,
    ScalarCodec,
    recall_against_exact,
)

DIMENSIONS = 64
COUNT = 400


def fixture() -> tuple[list[Chunk], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(COUNT, DIMENSIONS)).astype(np.float32)
    chunks = [
        Chunk(id=f"c{i}", span=Span("d", i * 10, i * 10 + 9), text=f"chunk {i}")
        for i in range(COUNT)
    ]
    queries = rng.normal(size=(12, DIMENSIONS)).astype(np.float32)
    return chunks, vectors, queries


def mean_recall(index: QuantizedDenseIndex, exact: ExactDenseIndex, queries: np.ndarray) -> float:
    return float(
        np.mean(
            [
                recall_against_exact(index.search("", q, 10), exact.search("", q, 10), 10)
                for q in queries
            ]
        )
    )


# ---------------------------------------------------------------------------
# the codecs
# ---------------------------------------------------------------------------


def test_scalar_round_trips_closely() -> None:
    _, vectors, _ = fixture()
    codec = ScalarCodec()
    codec.fit(vectors)
    restored = codec.decode(codec.encode(vectors))
    assert np.abs(restored - vectors).max() < 0.02


def test_scalar_survives_a_constant_dimension() -> None:
    """A dimension that never varies would divide by zero, and carries no information anyway."""
    vectors = np.ones((10, 4), dtype=np.float32)
    codec = ScalarCodec()
    codec.fit(vectors)
    assert np.isfinite(codec.decode(codec.encode(vectors))).all()


def test_product_learns_a_codebook_from_the_corpus() -> None:
    _, vectors, _ = fixture()
    codec = ProductCodec(subspaces=8)
    codec.fit(vectors)
    codes = codec.encode(vectors)
    assert codes.shape == (COUNT, 8)
    assert codec.decode(codes).shape == vectors.shape


def test_product_refuses_subspaces_that_do_not_divide() -> None:
    _, vectors, _ = fixture()
    with pytest.raises(QuantizationError, match="do not divide"):
        ProductCodec(subspaces=7).fit(vectors)


def test_product_handles_fewer_points_than_centroids() -> None:
    codec = ProductCodec(subspaces=2)
    tiny = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    codec.fit(tiny)
    assert codec.encode(tiny).shape == (3, 2)


def test_binary_packs_one_bit_per_dimension() -> None:
    _, vectors, _ = fixture()
    codec = BinaryCodec()
    codec.fit(vectors)
    assert codec.encode(vectors).shape == (COUNT, DIMENSIONS // 8)


def test_binary_cannot_be_decoded_and_says_why() -> None:
    """That is the trade. Pretending otherwise would hide where the rescoring pass comes in."""
    _, vectors, _ = fixture()
    codec = BinaryCodec()
    codec.fit(vectors)
    with pytest.raises(QuantizationError, match="rescores its shortlist"):
        codec.decode(codec.encode(vectors))


# ---------------------------------------------------------------------------
# the frontier
# ---------------------------------------------------------------------------


def test_scalar_costs_almost_no_recall_for_four_times_the_compression() -> None:
    chunks, vectors, queries = fixture()
    exact = ExactDenseIndex()
    exact.build(chunks, vectors)

    index = QuantizedDenseIndex(scheme="scalar")
    index.build(chunks, vectors)

    assert mean_recall(index, exact, queries) > 0.95
    assert index.compression().ratio == pytest.approx(4.0, rel=0.1)


def test_product_compresses_hard_and_loses_recall_without_rescoring() -> None:
    chunks, vectors, queries = fixture()
    exact = ExactDenseIndex()
    exact.build(chunks, vectors)

    plain = QuantizedDenseIndex(scheme="product", subspaces=8)
    plain.build(chunks, vectors)

    assert plain.compression().ratio > 20
    assert mean_recall(plain, exact, queries) < 0.8


def test_rescoring_is_what_makes_aggressive_compression_usable() -> None:
    """Leaving it out is the most common way somebody concludes a scheme does not work."""
    chunks, vectors, queries = fixture()
    exact = ExactDenseIndex()
    exact.build(chunks, vectors)

    plain = QuantizedDenseIndex(scheme="product", subspaces=8)
    plain.build(chunks, vectors)
    rescored = QuantizedDenseIndex(scheme="product", subspaces=8, rescore=100)
    rescored.build(chunks, vectors)

    assert mean_recall(rescored, exact, queries) > mean_recall(plain, exact, queries) + 0.2


def test_binary_without_rescoring_warns_that_it_is_being_used_wrong() -> None:
    chunks, vectors, _ = fixture()
    index = QuantizedDenseIndex(scheme="binary")
    index.build(chunks, vectors)

    warnings = index.warnings.of_code(WarningCode.QUANTIZATION_APPLIED)
    assert warnings
    assert "most common mistake" in warnings[0].message


def test_binary_with_rescoring_does_not_warn() -> None:
    chunks, vectors, _ = fixture()
    index = QuantizedDenseIndex(scheme="binary", rescore=100)
    index.build(chunks, vectors)
    assert not index.warnings.of_code(WarningCode.QUANTIZATION_APPLIED)


def test_no_quantization_is_exactly_exact() -> None:
    """The reference row every other one is judged against."""
    chunks, vectors, queries = fixture()
    exact = ExactDenseIndex()
    exact.build(chunks, vectors)

    passthrough = QuantizedDenseIndex(scheme="none")
    passthrough.build(chunks, vectors)

    assert mean_recall(passthrough, exact, queries) == 1.0


def test_the_index_size_includes_the_originals_it_keeps_for_rescoring() -> None:
    """Reporting the compressed size alone would flatter every configuration that rescores,
    which is most of the good ones."""
    chunks, vectors, _ = fixture()

    plain = QuantizedDenseIndex(scheme="binary")
    plain.build(chunks, vectors)
    rescoring = QuantizedDenseIndex(scheme="binary", rescore=100)
    rescoring.build(chunks, vectors)

    assert rescoring.size_bytes() > plain.size_bytes() * 10


def test_the_compression_report_reads_plainly() -> None:
    chunks, vectors, _ = fixture()
    index = QuantizedDenseIndex(scheme="scalar")
    index.build(chunks, vectors)
    assert "smaller" in index.compression().summary()


# ---------------------------------------------------------------------------
# configuration and edges
# ---------------------------------------------------------------------------


def test_an_unknown_scheme_lists_the_real_ones() -> None:
    with pytest.raises(QuantizationError, match="Choose one of"):
        QuantizedDenseIndex(scheme="jpeg")


def test_a_negative_rescore_is_refused() -> None:
    with pytest.raises(QuantizationError, match="rescore must be"):
        QuantizedDenseIndex(rescore=-1)


def test_building_without_vectors_says_so() -> None:
    chunks, _, _ = fixture()
    with pytest.raises(QuantizationError, match="needs vectors"):
        QuantizedDenseIndex().build(chunks, None)


def test_a_chunk_and_vector_mismatch_is_caught() -> None:
    chunks, vectors, _ = fixture()
    with pytest.raises(QuantizationError, match="out of step"):
        QuantizedDenseIndex().build(chunks[:10], vectors)


def test_an_empty_index_returns_nothing() -> None:
    index = QuantizedDenseIndex(scheme="scalar")
    index.build([], np.zeros((0, DIMENSIONS), dtype=np.float32))
    assert index.search("", np.zeros(DIMENSIONS, dtype=np.float32), 5) == []


def test_it_declares_itself_approximate() -> None:
    """An approximate index has to be compared against its exact twin before its numbers
    mean anything."""
    assert QuantizedDenseIndex.is_exact is False


def test_it_is_registered_and_configurable_by_spec() -> None:
    from contextgrid.index import get_index

    index = get_index("quantized:binary,rescore=200")
    assert isinstance(index, QuantizedDenseIndex)
    assert index.scheme == "binary"
    assert index.rescore == 200


def test_recall_against_exact_is_one_when_they_agree() -> None:
    from contextgrid.index.base import Scored

    same = [Scored("a", 1.0), Scored("b", 0.5)]
    assert recall_against_exact(same, same, 2) == 1.0


def test_recall_against_an_empty_reference_is_one() -> None:
    assert recall_against_exact([], [], 5) == 1.0

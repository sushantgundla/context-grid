"""Postgres with pgvector.

Split in two deliberately. The first half needs no database: configuration, validation and the
message somebody gets when there is no server, which is the part most people will actually
meet. The second half needs a real Postgres and skips without one -- there is no in-process
fake, because a fake pgvector would measure nothing and having it pass in CI would be worse
than the arm being absent.

To run the second half:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg17
    PGVECTOR_DSN=postgresql://postgres:pg@localhost:5432/postgres pytest
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from contextgrid.core.documents import Chunk
from contextgrid.core.span import Span
from contextgrid.index import get_index
from contextgrid.index.dense import ExactDenseIndex, IndexBuildError
from contextgrid.index.pgvector import PgVectorIndex

DIMENSIONS = 32


def chunks_and_vectors(count: int = 200) -> tuple[list[Chunk], np.ndarray]:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(count, DIMENSIONS)).astype(np.float32)
    chunks = [
        Chunk(id=f"c{i}", span=Span("doc", i, i + 1), text=f"chunk {i}") for i in range(count)
    ]
    return chunks, vectors


# ---------------------------------------------------------------------------
# no database needed
# ---------------------------------------------------------------------------


def test_only_exact_claims_to_be_exact() -> None:
    assert PgVectorIndex(kind="exact").is_exact
    assert not PgVectorIndex(kind="hnsw").is_exact
    assert not PgVectorIndex(kind="ivfflat").is_exact


def test_an_unknown_index_type_lists_the_real_ones() -> None:
    with pytest.raises(IndexBuildError, match="exact, hnsw, ivfflat"):
        PgVectorIndex(kind="magic")


def test_an_unknown_metric_lists_what_pgvector_offers() -> None:
    with pytest.raises(IndexBuildError, match="cosine, dot, l2"):
        PgVectorIndex(metric="jaccard")


def test_no_dsn_says_how_to_get_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """The likeliest first encounter with this index by a wide margin."""
    monkeypatch.delenv("PGVECTOR_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(IndexBuildError, match="docker run"):
        PgVectorIndex()._resolved_dsn()


def test_the_dsn_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection string holds a password, so it belongs in the environment. `${PGVECTOR_DSN}`
    in a config file expands from there and the file itself stays safe to commit."""
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://from-env")
    assert PgVectorIndex()._resolved_dsn() == "postgresql://from-env"


def test_an_explicit_dsn_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://from-env")
    assert PgVectorIndex(dsn="postgresql://explicit")._resolved_dsn() == "postgresql://explicit"


def test_database_url_is_accepted_as_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGVECTOR_DSN", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://twelve-factor")
    assert PgVectorIndex()._resolved_dsn() == "postgresql://twelve-factor"


def test_no_vectors_is_a_clear_error() -> None:
    with pytest.raises(IndexBuildError, match="needs vectors"):
        PgVectorIndex().build([Chunk(id="c", span=Span("d", 0, 1), text="x")], None)


def test_chunks_and_vectors_out_of_step_is_a_clear_error() -> None:
    with pytest.raises(IndexBuildError, match="out of step"):
        PgVectorIndex().build(
            [Chunk(id="c", span=Span("d", 0, 1), text="x")],
            np.zeros((3, DIMENSIONS), dtype=np.float32),
        )


def test_an_empty_corpus_never_touches_the_database() -> None:
    """Connecting to build nothing would fail a run that should trivially succeed."""
    index = PgVectorIndex(dsn="postgresql://nowhere:1/none")
    index.build([], np.zeros((0, DIMENSIONS), dtype=np.float32))
    assert index.search("", np.zeros(DIMENSIONS, dtype=np.float32), k=5) == []


def test_a_bad_dsn_says_it_could_not_connect() -> None:
    pytest.importorskip("psycopg")
    chunks, vectors = chunks_and_vectors(3)
    index = PgVectorIndex(dsn="postgresql://nobody@127.0.0.1:1/none", table_prefix="t")

    with pytest.raises(IndexBuildError, match="could not connect"):
        index.build(chunks, vectors)


@pytest.mark.parametrize(
    "spec",
    [
        "pgvector",
        "pgvector:exact",
        "pgvector:hnsw,m=32,ef_search=128",
        "pgvector:ivfflat,lists=50,probes=10",
    ],
)
def test_every_variant_is_reachable_from_one_config_line(spec: str) -> None:
    assert get_index(spec).needs_vectors


def test_closing_twice_is_harmless() -> None:
    index = PgVectorIndex()
    index.close()
    index.close()


# ---------------------------------------------------------------------------
# a real database
# ---------------------------------------------------------------------------

DSN = os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")

live = pytest.mark.skipif(
    not DSN,
    reason="needs a Postgres with pgvector; set PGVECTOR_DSN to run",
)


@pytest.fixture
def built() -> object:
    """A live pgvector index, dropped again afterwards."""
    pytest.importorskip("psycopg")
    index = PgVectorIndex(kind="exact", dsn=DSN)
    chunks, vectors = chunks_and_vectors()
    index.build(chunks, vectors)
    yield index
    index.close()


@live
@pytest.mark.parametrize("kind", ["exact", "hnsw", "ivfflat"])
def test_it_finds_the_vector_it_was_given(kind: str) -> None:
    """The most basic thing an index has to get right, and the one that catches a mistake in
    the SQL immediately: search for a stored vector and it should return itself."""
    pytest.importorskip("psycopg")
    chunks, vectors = chunks_and_vectors()
    index = PgVectorIndex(kind=kind, dsn=DSN, lists=10, probes=10)
    try:
        index.build(chunks, vectors)
        assert index.search("", vectors[7], k=1)[0].chunk_id == "c7"
    finally:
        index.close()


@live
def test_exact_search_agrees_with_the_numpy_reference() -> None:
    """Two completely different implementations of the same thing. If they disagree, one of
    them has the metric wrong -- which is exactly the bug that would otherwise be invisible."""
    pytest.importorskip("psycopg")
    chunks, vectors = chunks_and_vectors()

    reference = ExactDenseIndex()
    reference.build(chunks, vectors)
    index = PgVectorIndex(kind="exact", dsn=DSN)
    try:
        index.build(chunks, vectors)
        for query in vectors[:5]:
            assert [s.chunk_id for s in index.search("", query, 5)] == [
                s.chunk_id for s in reference.search("", query, 5)
            ]
    finally:
        index.close()


@live
def test_scores_descend(built: object) -> None:
    """Every pgvector operator returns a distance, smaller-is-better, and everything in this
    package is larger-is-better. One negation keeps a single convention across the axis."""
    _, vectors = chunks_and_vectors()
    scores = [s.score for s in built.search("", vectors[0], k=10)]  # type: ignore[attr-defined]
    assert scores == sorted(scores, reverse=True)


@live
def test_probing_more_lists_finds_more() -> None:
    """`ivfflat.probes` is a *session* setting in Postgres, not an index setting.

    This test caught the parameter being ignored entirely: `SET LOCAL` lasts until the end of
    the current transaction, and on an autocommit connection there is none, so Postgres
    accepted it, did nothing, and every query ran at the default `probes = 1` while the sweep
    reported numbers for whatever was asked for.
    """
    pytest.importorskip("psycopg")
    chunks, vectors = chunks_and_vectors()
    reference = ExactDenseIndex()
    reference.build(chunks, vectors)

    recalls = []
    for probes in (1, 20):  # 1 of 20 lists against all of them: the gap should be large
        index = PgVectorIndex(kind="ivfflat", dsn=DSN, lists=20, probes=probes)
        try:
            index.build(chunks, vectors)
            hits = [
                len(
                    {s.chunk_id for s in index.search("", query, 10)}
                    & {s.chunk_id for s in reference.search("", query, 10)}
                )
                for query in vectors[:10]
            ]
            recalls.append(sum(hits) / (10 * 10))
        finally:
            index.close()

    assert recalls[1] > recalls[0]
    assert recalls[1] == 1.0  # every list probed is exhaustive search


@live
def test_the_table_reports_its_real_size(built: object) -> None:
    """The one index here that can answer honestly, because Postgres keeps the number."""
    assert built.size_bytes() > 0  # type: ignore[attr-defined]


@live
def test_two_indexes_do_not_tread_on_each_other() -> None:
    """Every build gets its own table, so two sweeps against the same database can run at once
    and neither leaves anything behind."""
    pytest.importorskip("psycopg")
    chunks, vectors = chunks_and_vectors(50)

    first, second = PgVectorIndex(dsn=DSN), PgVectorIndex(dsn=DSN)
    try:
        first.build(chunks, vectors)
        second.build(chunks, vectors)
        assert first._table != second._table
        assert first.search("", vectors[3], k=1)[0].chunk_id == "c3"
        assert second.search("", vectors[3], k=1)[0].chunk_id == "c3"
    finally:
        first.close()
        second.close()


@live
def test_the_table_is_gone_after_closing() -> None:
    """A benchmark tool that fills somebody's database with leftover tables is a bad guest."""
    import psycopg

    chunks, vectors = chunks_and_vectors(20)
    index = PgVectorIndex(dsn=DSN)
    index.build(chunks, vectors)
    table = index._table
    index.close()

    with psycopg.connect(DSN, autocommit=True) as connection:
        found = connection.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    assert found[0] is None


@live
def test_a_query_from_a_different_model_is_refused(built: object) -> None:
    with pytest.raises(IndexBuildError, match="different models"):
        built.search("", np.zeros(7, dtype=np.float32), k=3)  # type: ignore[attr-defined]

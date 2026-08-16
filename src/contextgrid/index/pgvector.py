"""Postgres with the pgvector extension.

Every other index here is a library holding vectors in memory. This one is a database, and
that is exactly why it is worth having on the axis: it is what an enormous number of teams
actually deploy on, and "the HNSW settings my Postgres is using" is a different question from
"the HNSW settings faiss would use". Measuring the library and shipping the database is how
the two quietly diverge.

It behaves differently from its neighbours in three ways that are inherent rather than
incidental, and pretending otherwise would make the comparison dishonest:

**It needs a server.** `dsn` points at Postgres, or `PGVECTOR_DSN` in the environment does.
There is no in-process fallback -- a fake would measure nothing and would be worse than
skipping the arm.

**Recall depends on parameters set outside the index.** `ivfflat.probes` and `hnsw.ef_search`
are *session* settings in Postgres, not index settings. They are applied per query here, which
is the only way to sweep them, and it is worth knowing that a production system which never
sets them is running whatever the server's default happens to be.

**Building an index writes to a real database.** Every run creates and drops a table named
from the run, so two sweeps cannot tread on each other and nothing is left behind.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import MissingExtraError
from contextgrid.embed.base import Vectors, normalise
from contextgrid.index.base import Scored
from contextgrid.index.dense import IndexBuildError

#: Distance operators pgvector exposes, and whether smaller means better.
_OPERATORS: dict[str, str] = {
    "cosine": "<=>",  # cosine distance
    "dot": "<#>",  # negative inner product
    "l2": "<->",  # euclidean distance
}

_OPCLASSES: dict[str, str] = {
    "cosine": "vector_cosine_ops",
    "dot": "vector_ip_ops",
    "l2": "vector_l2_ops",
}


@dataclass(slots=True)
class PgVectorIndex:
    """A pgvector table, built and dropped per run.

    `kind="exact"` scans every row -- pgvector's honest baseline, and the reference the two
    approximate types are judged against without leaving the database.
    `kind="hnsw"` and `kind="ivfflat"` are the two index types pgvector offers.
    """

    kind: str = "hnsw"
    metric: str = "cosine"
    dsn: str | None = None
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 40
    lists: int = 100
    probes: int = 8
    table_prefix: str = "contextgrid"

    name: ClassVar[str] = "pgvector"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = True

    KINDS: ClassVar[tuple[str, ...]] = ("exact", "hnsw", "ivfflat")

    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _table: str = field(default="", init=False, repr=False)
    _dimensions: int = field(default=0, init=False, repr=False)
    _connection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise IndexBuildError(
                f"unknown pgvector index {self.kind!r}. Choose one of: {', '.join(self.KINDS)}"
            )
        if self.metric not in _OPERATORS:
            raise IndexBuildError(
                f"unknown metric {self.metric!r}. pgvector offers: {', '.join(_OPERATORS)}"
            )

    @property
    def is_exact(self) -> bool:
        return self.kind == "exact"

    # -- connection ----------------------------------------------------------

    def _resolved_dsn(self) -> str:
        dsn = self.dsn or os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
        if not dsn:
            raise IndexBuildError(
                "the pgvector index needs a running Postgres with the vector extension. Set "
                "PGVECTOR_DSN, or pass dsn=... -- for example "
                "`index: pgvector:hnsw,dsn=${PGVECTOR_DSN}`. To try one quickly:\n"
                "  docker run -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg17"
            )
        return dsn

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as error:
            raise MissingExtraError("the pgvector index", "pgvector", package="psycopg") from error

        try:
            connection = psycopg.connect(self._resolved_dsn(), autocommit=True)
        except Exception as error:
            raise IndexBuildError(
                f"could not connect to Postgres for the pgvector index: {error}"
            ) from error

        try:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as error:
            connection.close()
            raise IndexBuildError(
                "connected to Postgres, but could not enable the vector extension. It has to "
                f"be installed on the server -- the pgvector/pgvector image has it. ({error})"
            ) from error
        return connection

    # -- protocol ------------------------------------------------------------

    def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None:
        if vectors is None:
            raise IndexBuildError(
                f"the {self.name!r} index needs vectors. Give it an embedder, or use a sparse "
                "index that works on text alone."
            )
        if len(chunks) != vectors.shape[0]:
            raise IndexBuildError(
                f"got {len(chunks)} chunks and {vectors.shape[0]} vectors. One is out of step "
                "with the other, and every score would be attached to the wrong text."
            )

        self.close()
        self._ids = [chunk.id for chunk in chunks]
        matrix = (
            normalise(np.asarray(vectors, dtype=np.float32))
            if self.metric == "cosine"
            else (np.asarray(vectors, dtype=np.float32))
        )
        self._dimensions = int(matrix.shape[1]) if matrix.size else 0
        if not self._ids or not self._dimensions:
            return

        self._connection = self._connect()
        # Unique per index instance, so two sweeps against the same database cannot collide.
        self._table = f"{self.table_prefix}_{uuid.uuid4().hex[:12]}"

        self._connection.execute(
            f'CREATE TABLE "{self._table}" '
            f"(row_id integer PRIMARY KEY, embedding vector({self._dimensions}))"
        )
        # COPY rather than INSERT: a corpus is thousands of rows, and one round trip each
        # would make building the index the slowest thing in the sweep by a wide margin.
        with (
            self._connection.cursor() as cursor,
            cursor.copy(f'COPY "{self._table}" (row_id, embedding) FROM STDIN') as copy,
        ):
            for row, vector in enumerate(matrix):
                copy.write_row((row, "[" + ",".join(f"{v:.7g}" for v in vector) + "]"))

        if self.kind != "exact":
            self._create_index()

    def _create_index(self) -> None:
        opclass = _OPCLASSES[self.metric]
        if self.kind == "hnsw":
            statement = (
                f'CREATE INDEX ON "{self._table}" USING hnsw (embedding {opclass}) '
                f"WITH (m = {self.m}, ef_construction = {self.ef_construction})"
            )
        else:
            # pgvector will not build more lists than there are rows, and a list count far
            # above the row count trains on nothing.
            lists = max(1, min(self.lists, len(self._ids)))
            statement = (
                f'CREATE INDEX ON "{self._table}" USING ivfflat (embedding {opclass}) '
                f"WITH (lists = {lists})"
            )
        self._connection.execute(statement)

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        del text
        if vector is None:
            raise IndexBuildError(f"the {self.name!r} index needs a query vector")
        if self._connection is None or not self._ids:
            return []

        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dimensions:
            raise IndexBuildError(
                f"query has {query.shape[1]} dimensions but the index was built with "
                f"{self._dimensions}. The query and the documents were embedded by different "
                "models."
            )
        row = normalise(query)[0] if self.metric == "cosine" else query[0]
        literal = "[" + ",".join(f"{value:.7g}" for value in row) + "]"

        # Session settings, not index settings, which is the only way to sweep them -- and
        # worth knowing, because a production system that never sets them runs on whatever the
        # server defaults to.
        #
        # `SET`, not `SET LOCAL`. `SET LOCAL` lasts until the end of the current transaction,
        # and on an autocommit connection there is no transaction to last for -- Postgres
        # accepts it, does nothing, and every query silently runs at the default `probes = 1`.
        # That is not a small difference: it is the whole parameter being ignored while the
        # sweep reports numbers for it.
        if self.kind == "hnsw":
            self._connection.execute(f"SET hnsw.ef_search = {max(k, self.ef_search)}")
        elif self.kind == "ivfflat":
            self._connection.execute(f"SET ivfflat.probes = {self.probes}")

        operator = _OPERATORS[self.metric]
        found = self._connection.execute(
            f'SELECT row_id, embedding {operator} %s::vector AS distance FROM "{self._table}" '
            f"ORDER BY distance LIMIT %s",
            (literal, k),
        ).fetchall()

        # Every pgvector operator returns a distance, smaller-is-better. Everything in this
        # package is larger-is-better, so one negation keeps a single convention on the axis.
        return [Scored(self._ids[int(row_id)], -float(distance)) for row_id, distance in found]

    def size_bytes(self) -> int:
        """What Postgres says the table and its index occupy, rather than an estimate.

        The one index here that can answer honestly, because the database keeps the number.
        """
        if self._connection is None or not self._table:
            return 0
        try:
            result = self._connection.execute(
                "SELECT pg_total_relation_size(%s)", (self._table,)
            ).fetchone()
        except Exception:  # pragma: no cover - the table has gone
            return 0
        return int(result[0]) if result else 0

    def close(self) -> None:
        """Drop the table and disconnect. Safe to call more than once."""
        if self._connection is None:
            return
        try:
            if self._table:
                self._connection.execute(f'DROP TABLE IF EXISTS "{self._table}"')
        except Exception:  # pragma: no cover - the connection has already gone
            pass
        finally:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._connection.close()
            self._connection = None
            self._table = ""

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown ordering
        with contextlib.suppress(Exception):
            self.close()

    def __len__(self) -> int:
        return len(self._ids)

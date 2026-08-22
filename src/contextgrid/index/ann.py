"""Approximate nearest-neighbour indexes: faiss and usearch.

Until now every index here was exact, which meant the one chart this package most wanted to
draw -- *what did approximation actually cost you?* -- had nothing to plot. Approximate search
is the single largest lever on query latency in a deployed system and the least honestly
reported: `efSearch` and `nprobe` get tuned until the latency looks good, and the recall that
paid for it is rarely measured at all.

So every index here declares `is_exact = False`, and `recall_against_exact` (already in
`quantize.py`) turns that into a number. Tuning an ANN parameter without knowing what recall it
cost is guessing, and it is guessing in the direction that flatters you.

**faiss** gives flat, IVF, HNSW and PQ from one dependency, which is the whole sweep on one
axis. **usearch** is a smaller, quantization-native alternative worth having as a second
implementation -- when two libraries disagree about the recall of "the same" HNSW settings,
that disagreement is itself the finding.

Both are wrapped so that `metric="cosine"` means the same thing it means everywhere else here:
vectors normalised on both sides, then inner product. Left to their own defaults these
libraries would compare L2 distance on unnormalised vectors, which ranks differently and would
make the index axis a metric comparison wearing an index comparison's clothes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import MissingExtraError
from contextgrid.embed.base import Vectors, normalise
from contextgrid.index.base import Scored
from contextgrid.index.dense import IndexBuildError

_METRICS = ("cosine", "dot", "l2")


@dataclass(slots=True)
class _ANNIndex:
    """Shared body: validation, normalising, and turning row numbers back into chunk ids."""

    metric: str = "cosine"

    name: ClassVar[str] = "ann"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = True
    #: The whole reason these exist. An approximate index whose numbers are read as exact is
    #: worse than no approximate index at all.
    is_exact: ClassVar[bool] = False

    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)
    _dimensions: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.metric not in _METRICS:
            raise IndexBuildError(
                f"unknown metric {self.metric!r}. Choose one of: {', '.join(_METRICS)}"
            )

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

        self._ids = [chunk.id for chunk in chunks]
        matrix = self._prepare(np.asarray(vectors, dtype=np.float32))
        self._dimensions = int(matrix.shape[1]) if matrix.size else 0

        if not self._ids:
            self._index = None
            return
        self._index = self._build_backend(matrix)

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        del text
        if vector is None:
            raise IndexBuildError(f"the {self.name!r} index needs a query vector")
        if self._index is None or not self._ids:
            return []

        query: Vectors = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dimensions:
            raise IndexBuildError(
                f"query has {query.shape[1]} dimensions but the index was built with "
                f"{self._dimensions}. The query and the documents were embedded by different "
                "models."
            )

        found = self._search_backend(self._prepare(query), min(k, len(self._ids)))

        # Backends pad short results with -1 rather than returning fewer rows. Left in, that
        # -1 indexes the *last* chunk in Python and puts an unrelated passage in the results.
        results = [
            Scored(self._ids[row], float(score))
            for row, score in found
            if 0 <= row < len(self._ids)
        ]
        # Sorted here rather than trusted: not every backend guarantees an order, and a
        # leaderboard that reshuffles between runs destroys trust in every number on it.
        results.sort(key=lambda scored: (-scored.score, scored.chunk_id))
        return results[:k]

    def size_bytes(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._ids)

    # -- backend hooks -------------------------------------------------------

    def _prepare(self, matrix: Vectors) -> Vectors:
        """Make the vectors mean what our metric names mean.

        Cosine is inner product on unit vectors. Normalising here, on both sides, is what stops
        `metric="cosine"` quietly meaning L2-on-raw-vectors just because that is the library's
        default -- which would rank differently and turn the index axis into a metric
        comparison in disguise.
        """
        if self.metric == "cosine":
            return normalise(matrix)
        return matrix

    def _build_backend(self, matrix: Vectors) -> Any:
        raise NotImplementedError

    def _search_backend(self, query: Vectors, k: int) -> list[tuple[int, float]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# faiss
# ---------------------------------------------------------------------------


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        # `MissingExtraError`, not `IndexBuildError`. The message was always right; the type was
        # not. `IndexBuildError` is a `ValueError`, so the `except MissingExtraError` the docs
        # hand out -- and the `except ImportError` they say also works -- both missed this,
        # which is the worst way to be wrong: the user reads a perfect message, writes the
        # handler the documentation told them to write, and it never fires.
        raise MissingExtraError("The faiss index", "index", package="faiss-cpu") from error
    return faiss


@dataclass(slots=True)
class FaissIndex:
    """faiss, with the index type as a parameter.

    Four types on one axis, which is the sweep worth running:

    * `flat` -- exhaustive. Not useful in production and essential here: it is the reference
      the other three are judged against, computed by the same library so the comparison is
      not confounded by implementation differences.
    * `hnsw` -- a navigable graph. The default choice in most deployments. `m` and
      `ef_construction` set at build time, `ef_search` at query time.
    * `ivf` -- inverted file. Clusters the space, searches `nprobe` clusters. Cheap to build,
      and the recall/latency knob is a single integer.
    * `ivfpq` -- IVF with product quantization on top. The memory answer: a fraction of the
      footprint, and the largest recall cost of the four.

    faiss has an implicit floor on training data that it enforces with a warning on stderr and
    then trains badly anyway: roughly 39 points per cluster for IVF, and 39 per *codebook
    entry* for PQ -- which at the default 8 bits means 256 entries and about 10,000 vectors
    before `ivfpq` is trained properly at all.

    Rather than let a small corpus quietly produce nonsense, `nlist` and `pq_bits` are reduced
    to fit and the reduction is recorded on `fitted_to_corpus`, so a run can say what it
    actually did instead of what was asked for.
    """

    kind: str = "hnsw"
    metric: str = "cosine"
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    nlist: int = 100
    nprobe: int = 8
    pq_subquantizers: int = 8
    pq_bits: int = 8

    name: ClassVar[str] = "faiss"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = True

    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)
    _dimensions: int = field(default=0, init=False, repr=False)
    _effective_nlist: int = field(default=0, init=False, repr=False)
    _effective_pq_bits: int = field(default=0, init=False, repr=False)
    #: What had to be reduced to fit the corpus, so the run can report it rather than a
    #: leaderboard quietly showing the recall of parameters nobody chose.
    fitted_to_corpus: dict[str, tuple[int, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    KINDS: ClassVar[tuple[str, ...]] = ("flat", "hnsw", "ivf", "ivfpq")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise IndexBuildError(
                f"unknown faiss index {self.kind!r}. Choose one of: {', '.join(self.KINDS)}"
            )
        if self.metric not in _METRICS:
            raise IndexBuildError(
                f"unknown metric {self.metric!r}. Choose one of: {', '.join(_METRICS)}"
            )

    @property
    def is_exact(self) -> bool:
        """`flat` really is exhaustive, so claiming otherwise would be a lie in the other
        direction -- and would make it useless as the reference arm."""
        return self.kind == "flat"

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

        self._ids = [chunk.id for chunk in chunks]
        matrix = np.ascontiguousarray(self._prepare(np.asarray(vectors, dtype=np.float32)))
        self._dimensions = int(matrix.shape[1]) if matrix.size else 0

        if not self._ids or not self._dimensions:
            self._index = None
            return

        faiss = _faiss()
        index = self._make(faiss, self._dimensions, len(self._ids))
        if not index.is_trained:
            index.train(matrix)
        index.add(matrix)
        self._index = index

    def _make(self, faiss: Any, dimensions: int, count: int) -> Any:
        metric = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT

        if self.kind == "flat":
            return (
                faiss.IndexFlatL2(dimensions)
                if self.metric == "l2"
                else faiss.IndexFlatIP(dimensions)
            )

        if self.kind == "hnsw":
            index = faiss.IndexHNSWFlat(dimensions, self.m, metric)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index

        # IVF needs roughly 39 training points per cluster before faiss stops complaining, and
        # a cluster count above the corpus size cannot train at all. Shrinking it is better
        # than a warning on stderr nobody reads and an index that quietly returns rubbish.
        self._effective_nlist = max(1, min(self.nlist, count // _POINTS_PER_CENTROID or 1))
        if self._effective_nlist != self.nlist:
            self.fitted_to_corpus["nlist"] = (self.nlist, self._effective_nlist)
        quantizer = (
            faiss.IndexFlatL2(dimensions) if self.metric == "l2" else faiss.IndexFlatIP(dimensions)
        )

        if self.kind == "ivf":
            index = faiss.IndexIVFFlat(quantizer, dimensions, self._effective_nlist, metric)
        else:
            subquantizers = _largest_divisor(dimensions, self.pq_subquantizers)
            self._effective_pq_bits = _fit_pq_bits(self.pq_bits, count)
            if self._effective_pq_bits != self.pq_bits:
                self.fitted_to_corpus["pq_bits"] = (self.pq_bits, self._effective_pq_bits)
            index = faiss.IndexIVFPQ(
                quantizer,
                dimensions,
                self._effective_nlist,
                subquantizers,
                self._effective_pq_bits,
                metric,
            )
        index.nprobe = min(self.nprobe, self._effective_nlist)
        return index

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        del text
        if vector is None:
            raise IndexBuildError(f"the {self.name!r} index needs a query vector")
        if self._index is None or not self._ids:
            return []

        query = np.ascontiguousarray(
            self._prepare(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        )
        if query.shape[1] != self._dimensions:
            raise IndexBuildError(
                f"query has {query.shape[1]} dimensions but the index was built with "
                f"{self._dimensions}. The query and the documents were embedded by different "
                "models."
            )

        distances, rows = self._index.search(query, min(k, len(self._ids)))

        results = [
            # L2 returns a distance, where smaller is better; every score in this package is
            # larger-is-better. Negating keeps one convention across the whole index axis.
            Scored(self._ids[row], -float(score) if self.metric == "l2" else float(score))
            for row, score in zip(rows[0], distances[0], strict=True)
            if 0 <= row < len(self._ids)  # faiss pads short results with -1
        ]
        results.sort(key=lambda scored: (-scored.score, scored.chunk_id))
        return results[:k]

    def _prepare(self, matrix: Vectors) -> Vectors:
        return normalise(matrix) if self.metric == "cosine" else matrix

    def size_bytes(self) -> int:
        """What faiss actually holds, estimated from the index type.

        faiss does not report its own footprint without serialising, and serialising a large
        index to measure it is absurd. These are the documented per-vector costs.
        """
        count, dimensions = len(self._ids), self._dimensions
        if not count or not dimensions:
            return 0

        floats = count * dimensions * 4
        if self.kind == "flat":
            return floats
        if self.kind == "hnsw":
            # Vectors, plus the graph: up to 2*m neighbours at the base layer, 4 bytes each.
            return floats + count * self.m * 2 * 4
        if self.kind == "ivf":
            return floats + count * 8  # vectors plus the id list
        bits = self._effective_pq_bits or self.pq_bits
        codes = _largest_divisor(dimensions, self.pq_subquantizers) * bits / 8
        return int(count * (codes + 8))

    def __len__(self) -> int:
        return len(self._ids)


#: faiss's own rule of thumb for how much training data a centroid needs.
_POINTS_PER_CENTROID = 39


def _fit_pq_bits(wanted: int, count: int) -> int:
    """The largest codebook that this many vectors can actually train.

    Product quantization learns 2**bits centroids per subspace, and faiss wants about 39
    points for each. At the default 8 bits that is ~10,000 vectors before the codebook is
    trained properly -- far more than most corpora anybody points this tool at. Training it
    anyway produces an index that returns plausible, wrong neighbours.
    """
    for bits in range(wanted, 0, -1):
        if count >= (2**bits) * _POINTS_PER_CENTROID:
            return bits
    return 1


def _largest_divisor(dimensions: int, wanted: int) -> int:
    """The largest divisor of `dimensions` no bigger than `wanted`.

    Product quantization splits the vector into equal subspaces, so the count has to divide the
    width exactly. faiss raises otherwise, which would make `ivfpq` unusable on any model whose
    width is not a multiple of eight -- 384 and 768 are, 1536 is, and plenty of others are not.
    """
    for candidate in range(min(wanted, dimensions), 0, -1):
        if dimensions % candidate == 0:
            return candidate
    return 1


# ---------------------------------------------------------------------------
# usearch
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class USearchIndex(_ANNIndex):
    """usearch: HNSW, with quantization built into the index rather than bolted on.

    A second implementation of the same idea as `faiss:hnsw`, and worth having for exactly that
    reason: when two libraries disagree about the recall of nominally identical settings, the
    disagreement is the finding.

    `dtype` is where it differs usefully. Storing vectors as `f16` or `i8` inside the index
    halves or quarters the memory with a recall cost that is measurable here rather than
    assumed.

    `b1` is deliberately absent. usearch's binary mode wants bit-packed input and a Hamming
    metric, not the float vectors every other arm on this axis takes -- it was registered as a
    valid dtype and raised `ValueError: The number of vector dimensions doesn't match!` on the
    first attempt to build one. Binary quantization is available and works: `quantized:binary`
    does it properly, with a rescoring pass to recover the recall it costs.
    """

    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64
    dtype: str = "f32"

    name: ClassVar[str] = "usearch"

    DTYPES: ClassVar[tuple[str, ...]] = ("f32", "f16", "i8")

    def __post_init__(self) -> None:
        # `_ANNIndex.__post_init__(self)`, not `super()`: a `slots=True` dataclass is rebuilt
        # by the decorator, leaving the `__class__` cell zero-argument `super()` reads pointing
        # at the class it replaced.
        _ANNIndex.__post_init__(self)
        if self.dtype not in self.DTYPES:
            raise IndexBuildError(
                f"unknown usearch dtype {self.dtype!r}. Choose one of: {', '.join(self.DTYPES)}"
            )

    def _build_backend(self, matrix: Vectors) -> Any:
        try:
            from usearch.index import Index as USearch
        except ImportError as error:
            raise MissingExtraError("The usearch index", "index", package="usearch") from error

        index = USearch(
            ndim=int(matrix.shape[1]),
            metric={"cosine": "cos", "dot": "ip", "l2": "l2sq"}[self.metric],
            dtype=self.dtype,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
        )
        index.add(np.arange(matrix.shape[0], dtype=np.int64), matrix)
        return index

    def _search_backend(self, query: Vectors, k: int) -> list[tuple[int, float]]:
        matches = self._index.search(query, k)
        # usearch reports distance, smaller-is-better. Everything here is larger-is-better.
        return [
            (int(key), -float(distance))
            for key, distance in zip(matches.keys, matches.distances, strict=True)
        ]

    def size_bytes(self) -> int:
        """Computed from the dtype rather than read from usearch.

        `Index.memory_usage` reports the arena it allocated, which barely moves between `f32`
        and `i8` on a small index -- so using it would show quantization saving nothing, which
        is the opposite of true and the entire reason the dtype is on the axis.
        """
        if self._index is None:
            return 0
        width = {"f32": 4.0, "f16": 2.0, "i8": 1.0}[self.dtype]
        vectors = len(self._ids) * self._dimensions * width
        graph = len(self._ids) * self.connectivity * 2 * 4  # neighbour ids, 4 bytes each
        return int(vectors + graph)

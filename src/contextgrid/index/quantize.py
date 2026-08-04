"""Vector quantization: trading recall for memory, measurably.

Quantization is discussed everywhere as a *memory* decision and almost nowhere as a *quality*
one. The blog posts report compression ratios and queries per second; what they do not report
is how much recall it cost on the corpus in front of you, and that is corpus-specific enough
that the published numbers do not transfer.

Four schemes, in increasing order of how much they throw away:

**None** -- float32. The reference every other row is judged against.
**Scalar** -- each dimension to one byte. 4x smaller, and usually almost free in recall.
**Product** -- the vector split into subspaces, each replaced by the nearest of 256 learned
centroids. Much smaller, and the first scheme where the loss is worth measuring.
**Binary** -- one bit per dimension, with a rescoring pass over the top candidates. 32x
smaller and startlingly good when rescored, startlingly bad when not.

The rescoring pass is what makes binary usable, and leaving it out is the most common way
somebody concludes binary quantization does not work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.embed.base import Vectors, normalise
from contextgrid.index.base import Scored, top_k


class Quantization(str, Enum):
    NONE = "none"
    SCALAR = "scalar"
    PRODUCT = "product"
    BINARY = "binary"


class QuantizationError(ContextGridError, ValueError):
    """A quantized index was configured in a way that cannot work."""


@dataclass(frozen=True, slots=True)
class CompressionReport:
    """What a scheme cost and what it saved."""

    scheme: str
    original_bytes: int
    compressed_bytes: int

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes else 1.0

    def summary(self) -> str:
        return (
            f"{self.scheme}: {self.original_bytes / 1024:.0f} KB to "
            f"{self.compressed_bytes / 1024:.0f} KB ({self.ratio:.1f}x smaller)"
        )


# ---------------------------------------------------------------------------
# the codecs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScalarCodec:
    """Each dimension linearly mapped to one unsigned byte.

    The cheapest real compression, and on normalised embeddings the recall cost is usually
    somewhere near nothing -- which is exactly why it should be measured rather than assumed.
    """

    name: ClassVar[str] = "scalar"

    _low: npt.NDArray[np.float32] = field(
        init=False, repr=False, default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    _scale: npt.NDArray[np.float32] = field(
        init=False, repr=False, default_factory=lambda: np.zeros(0, dtype=np.float32)
    )

    def fit(self, vectors: Vectors) -> None:
        self._low = vectors.min(axis=0).astype(np.float32)
        high = vectors.max(axis=0).astype(np.float32)
        span = high - self._low
        # A dimension that never varies would divide by zero; it also carries no information,
        # so any constant scale is as good as another.
        span[span == 0] = 1.0
        self._scale = span / 255.0

    def encode(self, vectors: Vectors) -> npt.NDArray[np.uint8]:
        scaled = (vectors - self._low) / self._scale
        codes: npt.NDArray[np.uint8] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
        return codes

    def decode(self, codes: npt.NDArray[np.uint8]) -> Vectors:
        return (codes.astype(np.float32) * self._scale + self._low).astype(np.float32)


@dataclass(slots=True)
class ProductCodec:
    """The vector split into `subspaces`, each replaced by the nearest of 256 centroids.

    Centroids are learned by k-means over the corpus itself, so the codebook is fitted to the
    documents being searched rather than to somebody else's.
    """

    subspaces: int = 8
    iterations: int = 12
    seed: int = 0

    name: ClassVar[str] = "product"

    _codebooks: list[npt.NDArray[np.float32]] = field(default_factory=list, init=False, repr=False)
    _width: int = field(default=0, init=False, repr=False)

    def fit(self, vectors: Vectors) -> None:
        dimensions = vectors.shape[1]
        if dimensions % self.subspaces:
            raise QuantizationError(
                f"{dimensions} dimensions do not divide into {self.subspaces} subspaces. "
                "Pick a subspace count that divides the embedding size."
            )

        self._width = dimensions // self.subspaces
        self._codebooks = []
        rng = np.random.default_rng(self.seed)

        for index in range(self.subspaces):
            part = vectors[:, index * self._width : (index + 1) * self._width]
            self._codebooks.append(
                _kmeans(part, k=min(256, len(part)), rng=rng, iterations=self.iterations)
            )

    def encode(self, vectors: Vectors) -> npt.NDArray[np.uint8]:
        codes = np.zeros((vectors.shape[0], self.subspaces), dtype=np.uint8)
        for index, codebook in enumerate(self._codebooks):
            part = vectors[:, index * self._width : (index + 1) * self._width]
            distances = _squared_distances(part, codebook)
            codes[:, index] = np.argmin(distances, axis=1).astype(np.uint8)
        return codes

    def decode(self, codes: npt.NDArray[np.uint8]) -> Vectors:
        parts = [self._codebooks[index][codes[:, index]] for index in range(self.subspaces)]
        decoded: Vectors = np.hstack(parts).astype(np.float32)
        return decoded


@dataclass(slots=True)
class BinaryCodec:
    """One bit per dimension: is this dimension above the corpus mean?

    Thirty-two times smaller, and the similarity it computes is Hamming distance rather than
    anything continuous. Used alone it is crude. Used to shortlist candidates that are then
    rescored against the real vectors, it is remarkably close to exact -- and leaving out that
    rescoring pass is the most common way people conclude binary quantization does not work.
    """

    name: ClassVar[str] = "binary"

    _threshold: npt.NDArray[np.float32] = field(
        init=False, repr=False, default_factory=lambda: np.zeros(0, dtype=np.float32)
    )

    def fit(self, vectors: Vectors) -> None:
        self._threshold = vectors.mean(axis=0).astype(np.float32)

    def encode(self, vectors: Vectors) -> npt.NDArray[np.uint8]:
        bits: npt.NDArray[np.uint8] = np.packbits(vectors > self._threshold, axis=1)
        return bits

    def decode(self, codes: npt.NDArray[np.uint8]) -> Vectors:
        # Binary codes cannot be decoded back to anything useful -- that is the trade. The
        # index keeps the originals for rescoring rather than pretending otherwise.
        raise QuantizationError(
            "binary codes cannot be decoded; a binary index rescores its shortlist against "
            "the original vectors instead"
        )


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuantizedDenseIndex:
    """Exact dense search over compressed vectors, with an optional rescoring pass.

    `rescore` is how many candidates are pulled from the compressed index and re-ranked
    against the original float vectors. It is the parameter that decides whether aggressive
    quantization is a good trade or a bad one, and it costs memory -- the originals have to be
    kept -- which the size report states rather than hides.
    """

    scheme: str = "scalar"
    subspaces: int = 8
    rescore: int = 0
    metric: str = "cosine"

    name: ClassVar[str] = "quantized"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = True
    is_exact: ClassVar[bool] = False

    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _codes: npt.NDArray[np.uint8] = field(
        init=False, repr=False, default_factory=lambda: np.zeros((0, 0), dtype=np.uint8)
    )
    _originals: Vectors = field(
        init=False, repr=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _codec: ScalarCodec | ProductCodec | BinaryCodec | None = field(
        default=None, init=False, repr=False
    )
    warnings: WarningLog = field(default_factory=WarningLog, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scheme not in {q.value for q in Quantization}:
            raise QuantizationError(
                f"unknown quantization {self.scheme!r}. Choose one of: "
                f"{', '.join(q.value for q in Quantization)}"
            )
        if self.rescore < 0:
            raise QuantizationError(f"rescore must be >= 0, got {self.rescore}")

    def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None:
        if vectors is None:
            raise QuantizationError("a quantized index needs vectors")
        if len(chunks) != vectors.shape[0]:
            raise QuantizationError(
                f"got {len(chunks)} chunks and {vectors.shape[0]} vectors, which are out of step"
            )

        self._ids = [chunk.id for chunk in chunks]
        if not self._ids:
            # Nothing to compress. Every codec's fit reduces over the corpus, which is an
            # error on an empty one rather than a sensible identity.
            self._codec = None
            self._originals = np.zeros((0, 0), dtype=np.float32)
            self._codes = np.zeros((0, 0), dtype=np.uint8)
            return

        prepared = (
            normalise(np.asarray(vectors, dtype=np.float32))
            if self.metric == "cosine"
            else np.asarray(vectors, dtype=np.float32)
        )
        self._originals = prepared

        if self.scheme == Quantization.NONE.value:
            self._codec = None
            self._codes = np.zeros((0, 0), dtype=np.uint8)
            return

        self._codec = _make_codec(self.scheme, self.subspaces)
        self._codec.fit(prepared)
        self._codes = self._codec.encode(prepared)

        if self.scheme == Quantization.BINARY.value and self.rescore == 0:
            self.warnings.add(
                WarningCode.QUANTIZATION_APPLIED,
                "binary quantization without a rescoring pass keeps one bit per dimension and "
                "ranks on Hamming distance alone. It will score far below its potential, and "
                "concluding from that that binary quantization does not work is the most "
                "common mistake made with it. Set rescore to 50 or more",
                severity=Severity.CAUTION,
                stage="index",
                subject=self.scheme,
            )

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        del text
        if vector is None:
            raise QuantizationError("a quantized index needs a query vector")
        if not self._ids:
            return []

        # Annotated rather than inferred: reshape gives a two-dimensional shape type and
        # `normalise` returns an any-dimensional one, so the reassignment below would narrow.
        query: Vectors = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self.metric == "cosine":
            query = normalise(query)

        if self._codec is None:
            scores = (self._originals @ query.T).ravel()
            return top_k(dict(zip(self._ids, (float(s) for s in scores), strict=True)), k)

        shortlist = self._shortlist(query, max(k, self.rescore) if self.rescore else k)
        if not self.rescore:
            return shortlist[:k]

        # Rescore the shortlist against the vectors as they actually are. This is what makes
        # aggressive compression usable, and it is why the originals are kept.
        positions = {chunk_id: index for index, chunk_id in enumerate(self._ids)}
        rescored = {
            scored.chunk_id: float(self._originals[positions[scored.chunk_id]] @ query.ravel())
            for scored in shortlist
        }
        return top_k(rescored, k)

    def _shortlist(self, query: Vectors, depth: int) -> list[Scored]:
        assert self._codec is not None

        if isinstance(self._codec, BinaryCodec):
            code = self._codec.encode(query)
            distances = _hamming(self._codes, code[0])
            # Lower Hamming distance is better, so negate to keep "higher is better".
            scores = {
                chunk_id: float(-distance)
                for chunk_id, distance in zip(self._ids, distances, strict=True)
            }
            return top_k(scores, depth)

        reconstructed = self._codec.decode(self._codes)
        similarities = (reconstructed @ query.T).ravel()
        return top_k(dict(zip(self._ids, (float(s) for s in similarities), strict=True)), depth)

    def size_bytes(self) -> int:
        """Memory this index occupies, including the originals kept for rescoring.

        Reporting the compressed size alone would flatter every configuration that rescores,
        which is most of the good ones.
        """
        compressed = int(self._codes.nbytes)
        if self._codec is None:
            return int(self._originals.nbytes)
        return compressed + (int(self._originals.nbytes) if self.rescore else 0)

    def compression(self) -> CompressionReport:
        """What the codes alone cost, against float32."""
        original = int(self._originals.nbytes)
        compressed = original if self._codec is None else int(self._codes.nbytes)
        return CompressionReport(self.scheme, original, compressed)

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# measuring what it cost
# ---------------------------------------------------------------------------


def recall_against_exact(approximate: Sequence[Scored], exact: Sequence[Scored], k: int) -> float:
    """What fraction of exact search's top k the approximate index also found.

    The number that turns a compression ratio into a decision. Tuning quantization without it
    is guessing, and guessing in the direction that looks good.
    """
    if not exact:
        return 1.0
    wanted = {scored.chunk_id for scored in exact[:k]}
    found = {scored.chunk_id for scored in approximate[:k]}
    return len(wanted & found) / len(wanted)


def _make_codec(scheme: str, subspaces: int) -> ScalarCodec | ProductCodec | BinaryCodec:
    if scheme == Quantization.SCALAR.value:
        return ScalarCodec()
    if scheme == Quantization.PRODUCT.value:
        return ProductCodec(subspaces=subspaces)
    return BinaryCodec()


def _squared_distances(points: Vectors, centroids: Vectors) -> Vectors:
    """Pairwise squared euclidean distance, expanded so it stays one matrix multiply."""
    distances: Vectors = (
        (points**2).sum(axis=1)[:, None]
        - 2 * points @ centroids.T
        + (centroids**2).sum(axis=1)[None, :]
    )
    return distances


def _kmeans(
    points: Vectors, *, k: int, rng: np.random.Generator, iterations: int
) -> npt.NDArray[np.float32]:
    """Plain Lloyd's algorithm. Small codebooks over small corpora do not need more."""
    if len(points) <= k:
        # Fewer points than centroids: every point is its own, padded so the codebook is a
        # fixed size and encoding cannot index out of range.
        padded = np.zeros((k, points.shape[1]), dtype=np.float32)
        padded[: len(points)] = points
        return padded

    centroids = points[rng.choice(len(points), size=k, replace=False)].astype(np.float32)
    for _ in range(iterations):
        assignments = np.argmin(_squared_distances(points, centroids), axis=1)
        for index in range(k):
            members = points[assignments == index]
            if len(members):
                centroids[index] = members.mean(axis=0)
    learned: npt.NDArray[np.float32] = centroids
    return learned


def _hamming(codes: npt.NDArray[np.uint8], query: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    """Bits that differ, counted a byte at a time."""
    differing: npt.NDArray[np.int64] = (
        np.unpackbits(codes ^ query[None, :], axis=1).sum(axis=1).astype(np.int64)
    )
    return differing

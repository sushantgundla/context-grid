"""Scoring an embedder against *your* corpus, with no questions at all.

MTEB reports that a model scores 63.5, averaged across 56 public datasets of tweets, medical
abstracts, forum posts and code. None of them are your documents. Choosing an embedder from
that leaderboard is choosing whichever model did best on somebody else's corpus, and the gap
between those two things is exactly the gap this package exists to close.

Recall answers the question properly -- and needs an eval set. Most people have a corpus and no
questions at the moment they are choosing an embedder, which is the worst possible time to have
no signal at all.

Everything here works from the vectors alone. Embed the corpus, look at the geometry, and ask
whether this model has anything useful to say about *these* documents:

* **Anisotropy** -- how similar two unrelated chunks look. Transformer embeddings are famously
  crowded into a narrow cone, and a model reporting 0.85 cosine between a refund policy and a
  shipping schedule has almost no room left to express that a query matches one and not the
  other. It is the single most diagnostic number here and the cheapest to compute.
* **Local coherence** -- how much closer *consecutive* chunks are than random ones. Adjacent
  text really is related, on any corpus, which makes it a far safer signal than "same document":
  a contract's fees clause and its termination clause share a file and little else.
* **Redundancy** -- how similar chunks from *different* documents are, relative to consecutive
  ones. High redundancy means the corpus is full of near-copies -- templated contracts, boilerplate
  policies -- and no embedder will separate them cleanly. It is a fact about the documents rather
  than the model, and knowing it stops a hard corpus being mistaken for a bad embedder.
* **Effective dimensions** -- how many dimensions actually carry variance, by participation
  ratio. A 1536-dimension model whose vectors live in a 40-dimension subspace is charging for
  1536 and thinking in 40, and it will lose to a smaller model that uses what it has.
* **Collapse** -- the fraction of chunks with a near-identical twin in vector space that is not
  near-identical in text. Distinct passages the model cannot tell apart, which no retriever
  downstream can rescue.

**These are diagnostics, not a verdict.** They say whether a model *can* discriminate on this
corpus, never whether it retrieves the right thing -- that is recall's job, and recall needs
questions. A model scoring well here and badly on recall is a real and useful finding; a model
scoring badly here will not be rescued by anything later in the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.embed.base import Vectors, normalise

#: Above this cosine, two chunks are treated as the same point in vector space.
COLLAPSE_THRESHOLD = 0.99

#: Anisotropy at or above this is a model with very little room to discriminate. Not a hard
#: rule -- an honest rule of thumb, from the value at which unrelated pairs and related pairs
#: stop being separable on the corpora this was built against.
CROWDED = 0.6


class EmbeddingQualityError(ContextGridError, ValueError):
    """The corpus or the vectors cannot support a quality assessment."""


@dataclass(slots=True)
class EmbeddingQuality:
    """What the geometry of a corpus's embeddings says about the model that made them."""

    #: Mean cosine between chunks from *different* documents. Lower is better.
    anisotropy: float
    #: Consecutive-chunk mean cosine minus random-pair mean cosine. Higher is better.
    separation: float
    #: How much *more* alike unrelated documents are than consecutive passages. Exactly
    #: `-separation`, kept under its own name because it is the number to read when it is
    #: positive: the corpus is full of near-copies and no embedder will separate them cleanly.
    #:
    #: A difference rather than a ratio. Cosines go negative, and a ratio with a negative
    #: denominator silently inverts the comparison -- which is what it did on the first
    #: templated corpus it met.
    redundancy: float
    #: Dimensions actually carrying variance, by participation ratio.
    effective_dimensions: float
    #: How many the model was paid for.
    dimensions: int
    #: Fraction of chunks with a near-identical twin that is not near-identical in text.
    collapsed: float
    #: Chunks the assessment was computed over.
    sampled: int
    notes: dict[str, float] = field(default_factory=dict)

    @property
    def dimension_efficiency(self) -> float:
        """Effective dimensions over paid-for dimensions."""
        return self.effective_dimensions / self.dimensions if self.dimensions else 0.0

    @property
    def crowded(self) -> bool:
        return self.anisotropy >= CROWDED

    @property
    def degenerate(self) -> bool:
        """Whether the model has stopped discriminating at all.

        Nothing else here means anything once this is true: a model reporting every pair as
        identical produces a corpus diagnosis that is entirely about the model.

        Keyed on anisotropy alone. A high collapse rate looked like the same failure and is
        not: an embedder that puts a document's passages very close together is doing its job,
        and judging it degenerate for that would penalise exactly the behaviour wanted.
        """
        return self.anisotropy >= 0.95

    @property
    def templated(self) -> bool:
        """Whether documents resemble each other more than they cohere internally.

        A fact about the corpus, not the model. It is the shape a templated corpus makes --
        the same contract with a different company name in the header -- and it is the single
        most useful thing to know before blaming an embedder for poor retrieval.

        Only claimed for a model that can still see. An embedder whose vectors have collapsed
        reports everything as similar to everything, corpus included, and calling that a
        templated corpus would blame the documents for the model's failure.
        """
        return self.measurable and self.redundancy > 0 and not self.degenerate

    @property
    def score(self) -> float:
        """A 0-1 blend, for feeding the composite.

        **The weighting is a judgement, not a measurement**, and it is stated here rather than
        buried so it can be argued with. Separation counts double: a model that cannot tell
        related text from unrelated text is useless for retrieval whatever else it does well,
        while poor dimension efficiency merely means paying for width nobody uses.

        `separation` is scaled against 0.25, which is roughly what a healthy model reaches on
        ordinary prose. It is a rule of thumb, and the raw numbers above are the ones to trust.
        """
        headroom = _clamp(1.0 - self.anisotropy)
        efficiency = _clamp(self.dimension_efficiency * 4)
        distinct = _clamp(1.0 - self.collapsed)

        if not self.measurable:
            # No adjacency to measure, so coherence is dropped rather than scored as zero --
            # the same rule the composite follows for a dimension nobody ran.
            return (headroom + efficiency + distinct) / 3

        signal = _clamp(self.separation / 0.25)
        return (2 * signal + headroom + efficiency + distinct) / 5

    @property
    def measurable(self) -> bool:
        """Whether there were enough consecutive pairs to say anything about coherence.

        A corpus of one chunk per document has no adjacency at all, and inventing a zero there
        would read as "the model found no structure" when the truth is that there was none to
        find.
        """
        return bool(self.notes.get("separation_measurable", 0.0))

    def summary(self) -> str:
        if self.degenerate:
            # Nothing else is worth printing. Every other number is a restatement of this one.
            return (
                f"degenerate: anisotropy {self.anisotropy:.3f}, {self.collapsed:.0%} of chunks "
                f"collapsed onto a neighbour, over {self.sampled} chunks"
            )

        crowding = " (crowded)" if self.crowded else ""
        coherence = (
            f"coherence {self.separation:+.3f}" if self.measurable else "coherence unmeasurable"
        )
        templated = " (templated corpus)" if self.templated else ""
        return (
            f"{coherence}{templated}, anisotropy {self.anisotropy:.3f}{crowding}, "
            f"{self.effective_dimensions:.0f}/{self.dimensions} effective dimensions, "
            f"{self.collapsed:.1%} collapsed, over {self.sampled} chunks"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "embedding_quality": self.score,
            "anisotropy": self.anisotropy,
            "separation": self.separation,
            "redundancy": self.redundancy,
            "effective_dimensions": self.effective_dimensions,
            "dimension_efficiency": self.dimension_efficiency,
            "collapsed": self.collapsed,
            **self.notes,
        }


def assess(
    chunks: Sequence[Chunk],
    vectors: Vectors,
    *,
    sample: int = 2000,
    pairs: int = 20_000,
    seed: int = 0,
) -> EmbeddingQuality:
    """Measure what an embedder's geometry says about this corpus.

    `sample` caps the chunks examined and `pairs` the comparisons drawn, because the full
    pairwise matrix is quadratic and nobody needs it -- a few thousand random pairs estimate
    these means to more precision than the conclusions require.
    """
    if len(chunks) != vectors.shape[0]:
        raise EmbeddingQualityError(
            f"got {len(chunks)} chunks and {vectors.shape[0]} vectors. One is out of step with "
            "the other, and every number below would be about the wrong text."
        )
    if len(chunks) < 4:
        raise EmbeddingQualityError(
            f"a corpus of {len(chunks)} chunks is too small to say anything about an embedder. "
            "These are statements about the shape of a cloud of points."
        )

    rng = np.random.default_rng(seed)
    matrix = normalise(np.asarray(vectors, dtype=np.float32))

    if len(chunks) > sample:
        keep = rng.choice(len(chunks), size=sample, replace=False)
        matrix = matrix[keep]
        chosen = [chunks[index] for index in keep]
    else:
        chosen = list(chunks)

    documents = np.array([chunk.doc_id for chunk in chosen])
    left, right = _random_pairs(len(chosen), pairs, rng)
    cosines = np.sum(matrix[left] * matrix[right], axis=1)
    across = cosines[documents[left] != documents[right]]
    anisotropy = float(np.mean(across)) if across.size else float(np.mean(cosines))

    # Adjacency rather than shared document. Consecutive text is genuinely related on any
    # corpus; sharing a file proves very little -- a contract's fees clause and its termination
    # clause sit in one document and have almost nothing to do with each other.
    adjacent = _adjacent_cosines(matrix, chosen)
    measurable = len(adjacent) >= 10
    separation = float(np.mean(adjacent) - anisotropy) if measurable else 0.0
    coherence = float(np.mean(adjacent)) if measurable else 0.0

    return EmbeddingQuality(
        anisotropy=anisotropy,
        separation=separation,
        # Positive means unrelated documents look more alike than consecutive paragraphs do.
        redundancy=(anisotropy - coherence) if measurable else 0.0,
        effective_dimensions=_participation_ratio(matrix),
        dimensions=int(matrix.shape[1]),
        collapsed=_collapse_rate(matrix, chosen),
        sampled=len(chosen),
        notes={
            "adjacent_cosine": coherence,
            "across_document_cosine": anisotropy,
            "separation_measurable": 1.0 if measurable else 0.0,
            "adjacent_pairs": float(len(adjacent)),
        },
    )


def _adjacent_cosines(matrix: Vectors, chunks: Sequence[Chunk]) -> Vectors:
    """Cosines between chunks that follow one another in the same document."""
    order: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        order.setdefault(chunk.doc_id, []).append(index)

    pairs: list[tuple[int, int]] = []
    for indices in order.values():
        indices.sort(key=lambda index: chunks[index].span.start)
        pairs.extend(pairwise(indices))

    if not pairs:
        empty: Vectors = np.zeros(0, dtype=np.float32)
        return empty
    # Typed at the source. `np.array` over a list comprehension has no element type, so
    # everything indexed by it widens to `Any` and the return then fails a typed signature.
    left: npt.NDArray[np.int64] = np.array([a for a, _ in pairs], dtype=np.int64)
    right: npt.NDArray[np.int64] = np.array([b for _, b in pairs], dtype=np.int64)
    cosines: Vectors = (matrix[left] * matrix[right]).sum(axis=1).astype(np.float32)
    return cosines


def _random_pairs(
    count: int, wanted: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Distinct index pairs, drawn rather than enumerated.

    The full matrix is quadratic: a 50,000-chunk corpus is 1.25 billion pairs to answer a
    question a few thousand samples settle.
    """
    left = rng.integers(0, count, size=wanted)
    right = rng.integers(0, count, size=wanted)
    keep = left != right
    return left[keep], right[keep]


def _participation_ratio(matrix: Vectors) -> float:
    """How many dimensions actually carry variance.

    The participation ratio of the eigenvalue spectrum: `(sum l)^2 / sum l^2`. It answers "how
    many dimensions is this model really using?" without a threshold anybody has to defend --
    a model spreading variance evenly over 768 dimensions scores 768, and one putting almost
    all of it in 12 scores about 12.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    # Singular values squared are the covariance eigenvalues, and the SVD avoids forming a
    # dimensions x dimensions matrix for a wide model.
    singular = np.linalg.svd(centred, compute_uv=False)
    eigenvalues = singular.astype(np.float64) ** 2
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 0.0
    return float(total**2 / np.sum(eigenvalues**2))


def _collapse_rate(matrix: Vectors, chunks: Sequence[Chunk]) -> float:
    """Fraction of chunks whose nearest neighbour is the same point but different text.

    Near-identical *text* landing on near-identical vectors is correct and expected -- boilerplate,
    repeated headers. What matters is distinct passages the model cannot tell apart, so text
    that really is nearly the same is excluded before counting.
    """
    if matrix.shape[0] > 4000:  # pragma: no cover - guards a quadratic on large corpora
        return 0.0

    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)
    nearest = np.argmax(similarity, axis=1)
    best = similarity[np.arange(len(chunks)), nearest]

    collapsed = 0
    for index, (neighbour, score) in enumerate(zip(nearest, best, strict=True)):
        if score < COLLAPSE_THRESHOLD:
            continue
        if _texts_differ(chunks[index].text, chunks[int(neighbour)].text):
            collapsed += 1
    return collapsed / len(chunks) if chunks else 0.0


def _texts_differ(left: str, right: str, *, overlap: float = 0.75) -> bool:
    """Whether two passages are genuinely different, by word overlap.

    Deliberately crude. The question is only "did the model collapse two things it should not
    have", and a Jaccard on words answers it without a second embedding pass.

    The threshold is loose on purpose. Passages differing by a single word really are the same
    passage for retrieval purposes, and counting the model as having collapsed them would blame
    it for noticing something true.
    """
    first, second = set(left.lower().split()), set(right.lower().split())
    if not first or not second:
        return left.strip() != right.strip()
    shared = len(first & second) / len(first | second)
    return shared < overlap


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

"""Query-side adapters: a small matrix that moves queries towards their answers.

The most under-used technique in retrieval, and the one this package is best placed to offer,
because it needs exactly the thing the tool already makes you build.

An embedding model maps questions and passages into one space, but not into the *same part*
of it -- a question and its answer are written differently and land apart. An adapter is one
learned matrix applied to the query vector only, nudging it towards where its answer actually
sits. The document embeddings never change, so the index does not have to be rebuilt.

Two properties make this worth having on the grid:

**The training data already exists.** Every eval set is a list of (question, evidence) pairs,
which is precisely a training set of positives. Building an eval set to *measure* retrieval and
then also using it to *improve* retrieval costs nothing extra.

**The hard negatives already exist too.** Every sweep surfaces chunks that ranked highly and
were not the answer, which are exactly the examples worth training against. Random negatives
teach a model almost nothing; near-misses teach it a great deal.

Fitted in closed form by ridge regression rather than by gradient descent. On the few hundred
pairs an eval set contains, a closed-form solution is deterministic, takes milliseconds, and
has one hyperparameter instead of five.

**Two things measurement showed, both worth knowing before using this.**

*A light touch wins and a heavy one is catastrophic.* Measured on a 33-document corpus with
36 training pairs and a held-out half, `strength=0.1` gained +0.081 recall@5 on a dense
embedder; `strength=1.0` lost 0.216, and with a large ridge term lost 0.649. The default is
therefore low. An adapter fitted on a few dozen pairs knows a little about the query
distribution and nothing about the rest of it, so it should be allowed to nudge and not to
decide.

*It is for dense embeddings.* On TF-IDF the same procedure hurt at every setting tried, from
-0.027 to -0.230. A dense linear map over a sparse lexical space destroys the sparsity that
made it work in the first place -- almost every dimension becomes slightly non-zero, and a
lexical match stops being a match. Adapters belong on learned dense vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from contextgrid.core.documents import Chunk
from contextgrid.core.errors import ContextGridError
from contextgrid.core.evalset import EvalSet
from contextgrid.embed.base import Embedder, EmbeddingResult, Vectors, normalise


class AdapterError(ContextGridError, ValueError):
    """An adapter could not be fitted or applied."""


@dataclass(frozen=True, slots=True)
class Triplet:
    """One training example: a question, the passage that answers it, and near misses."""

    query: str
    positive: str
    negatives: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdapterReport:
    """What the adapter was trained on and how far it moved things.

    Reported because an adapter is trained on the eval set it is then measured against, which
    is a real risk of flattering itself. A held-out split is the honest way to use it, and the
    report says plainly when there was not one.
    """

    pairs: int
    negatives: int
    dimensions: int
    mean_shift: float
    held_out: bool
    strength: float = 0.15

    def summary(self) -> str:
        basis = "a held-out split" if self.held_out else "the same questions it is scored on"
        return (
            f"adapter fitted on {self.pairs} pairs and {self.negatives} hard negatives, "
            f"moving query vectors by {self.mean_shift:.3f} on average. Trained on {basis}."
        )

    def warnings(self) -> list[str]:
        """Everything about this fit that should change how its score is read."""
        notes: list[str] = []

        if not self.held_out:
            notes.append(
                "This adapter was fitted on the same questions it is being scored on, so its "
                "score is optimistic and should not be compared with the other arms as though "
                "it were not. Fit it on a held-out split before believing the gain."
            )

        if self.strength > 0.5:
            notes.append(
                f"A strength of {self.strength:g} lets the adapter largely replace the query "
                "vector. Measured on a held-out split, that lost 0.22 recall@5 where 0.1 "
                "gained 0.08 -- a fit on this few pairs should nudge rather than decide."
            )

        if self.pairs < self.dimensions // 4:
            notes.append(
                f"{self.pairs} pairs for {self.dimensions} dimensions is a thin fit. The ridge "
                "term is carrying most of the solution, and the gain may not survive contact "
                "with queries unlike these."
            )

        return notes


@dataclass(slots=True)
class LinearAdapter:
    """One matrix applied to query vectors. Documents are left alone.

    `strength` blends the adapted vector with the original: 1.0 uses the adapter fully, 0.0
    disables it. Blending matters because an adapter fitted on a few dozen pairs can overfit
    badly, and a partial move captures most of the gain with much less of that risk.
    """

    #: Low on purpose. Measured on a held-out split, 0.1 gained +0.08 recall@5 on a dense
    #: embedder while 1.0 lost 0.22 -- an adapter fitted on a few dozen pairs should nudge
    #: rather than decide.
    strength: float = 0.15
    ridge: float = 1.0
    negative_weight: float = 0.5

    name: ClassVar[str] = "linear-adapter"

    _matrix: npt.NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    report: AdapterReport | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise AdapterError(f"strength must be between 0 and 1, got {self.strength}")
        if self.ridge <= 0:
            raise AdapterError(
                f"ridge must be positive, got {self.ridge}. It is what keeps a fit on a few "
                "dozen pairs from being wildly overconfident."
            )

    @property
    def is_fitted(self) -> bool:
        return self._matrix is not None

    def fit(
        self,
        query_vectors: Vectors,
        positive_vectors: Vectors,
        negative_vectors: Vectors | None = None,
        *,
        held_out: bool = False,
    ) -> AdapterReport:
        """Solve for the matrix that best maps queries onto their answers.

        Ridge regression in closed form: `W = (QᵀQ + λI)⁻¹ QᵀP`. The λ term is not optional
        on this much data -- with fewer pairs than dimensions the plain least-squares problem
        is underdetermined and the solution is nonsense.

        Hard negatives are subtracted from the target, which pushes queries away from the
        passages that were nearly right. That is where almost all of the signal is: a random
        negative is trivially far away already.
        """
        queries = np.asarray(query_vectors, dtype=np.float32)
        positives = np.asarray(positive_vectors, dtype=np.float32)

        if queries.shape != positives.shape:
            raise AdapterError(
                f"got {queries.shape[0]} queries and {positives.shape[0]} positives with "
                f"{queries.shape[1]} and {positives.shape[1]} dimensions. Each query needs "
                "exactly one positive, embedded by the same model."
            )
        if queries.shape[0] < 2:
            raise AdapterError(
                f"an adapter needs at least two pairs to fit, got {queries.shape[0]}"
            )

        targets = positives.copy()
        negative_count = 0
        if negative_vectors is not None and len(negative_vectors):
            negatives = np.asarray(negative_vectors, dtype=np.float32)
            if negatives.shape[1] != queries.shape[1]:
                raise AdapterError("negatives were embedded by a different model")
            negative_count = int(negatives.shape[0])
            # Move the target away from the average near-miss. Normalising afterwards keeps
            # the target on the unit sphere where the document vectors live.
            targets = normalise(targets - self.negative_weight * negatives.mean(axis=0))

        dimensions = queries.shape[1]
        gram = queries.T @ queries + self.ridge * np.eye(dimensions, dtype=np.float32)
        self._matrix = np.linalg.solve(gram, queries.T @ targets).astype(np.float32)

        shifted = self.apply(queries)
        shift = float(np.linalg.norm(shifted - normalise(queries), axis=1).mean())

        self.report = AdapterReport(
            pairs=int(queries.shape[0]),
            negatives=negative_count,
            dimensions=dimensions,
            mean_shift=shift,
            held_out=held_out,
            strength=self.strength,
        )
        return self.report

    def apply(self, query_vectors: Vectors) -> Vectors:
        """Move query vectors towards where their answers live."""
        if self._matrix is None:
            raise AdapterError("the adapter has not been fitted yet")

        queries = normalise(np.asarray(query_vectors, dtype=np.float32))
        adapted = normalise(queries @ self._matrix)
        if self.strength >= 1.0:
            return adapted
        blended: Vectors = normalise(self.strength * adapted + (1.0 - self.strength) * queries)
        return blended


@dataclass(slots=True)
class AdaptedEmbedder:
    """An embedder with an adapter on its query side.

    Deliberately a wrapper rather than a change to the embedders themselves. It composes with
    every one of them, it makes the adapter visible in the configuration label, and the
    document side is provably untouched -- which is the property that means the index does not
    have to be rebuilt.
    """

    base: Embedder
    adapter: LinearAdapter

    @property
    def name(self) -> str:
        return f"{self.base.name}+adapter"

    @property
    def version(self) -> str:
        return self.base.version

    @property
    def dimensions(self) -> int:
        return self.base.dimensions

    @property
    def normalised(self) -> bool:
        return True

    @property
    def max_tokens(self) -> int | None:
        return self.base.max_tokens

    def prepare(self, documents: Sequence[str]) -> None:
        self.base.prepare(documents)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        # Untouched, on purpose. An adapter that changed document vectors would mean
        # re-embedding the corpus every time it was refitted.
        return self.base.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        result = self.base.embed_queries(texts)
        if not self.adapter.is_fitted or result.vectors.size == 0:
            return result
        return EmbeddingResult(
            vectors=self.adapter.apply(result.vectors),
            warnings=result.warnings,
            input_tokens=result.input_tokens,
        )


# ---------------------------------------------------------------------------
# building the training set from things that already exist
# ---------------------------------------------------------------------------


def mine_triplets(
    evalset: EvalSet,
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Sequence[str]],
    chunks: Mapping[str, Chunk],
    *,
    negatives_per_query: int = 3,
) -> list[Triplet]:
    """Turn a completed run into adapter training data.

    Positives come from the eval set's own gold. Negatives come from the run: chunks that
    ranked above or alongside the answer and were not it. Those are the near misses, and they
    carry almost all of the training signal -- a randomly chosen negative is already far away
    and teaches nothing.

    Both were produced by measuring. Neither costs anything extra.
    """
    triplets: list[Triplet] = []

    for item in evalset:
        relevant = {cid for cid, grade in qrels.get(item.id, {}).items() if grade > 0}
        if not relevant:
            continue

        positive_id = next((cid for cid in run.get(item.id, ()) if cid in relevant), None)
        positive_id = positive_id or next(iter(sorted(relevant)))
        positive = chunks.get(positive_id)
        if positive is None:
            continue

        near_misses = [
            chunks[cid].text
            for cid in run.get(item.id, ())
            if cid not in relevant and cid in chunks
        ][:negatives_per_query]

        triplets.append(
            Triplet(query=item.question, positive=positive.text, negatives=tuple(near_misses))
        )

    return triplets


def fit_adapter(
    embedder: Embedder,
    triplets: Sequence[Triplet],
    *,
    strength: float = 0.15,
    ridge: float = 1.0,
    held_out: bool = False,
) -> LinearAdapter:
    """Embed the triplets and fit an adapter to them."""
    if len(triplets) < 2:
        raise AdapterError(
            f"an adapter needs at least two training pairs, got {len(triplets)}. Run a sweep "
            "first: its results are the training data."
        )

    adapter = LinearAdapter(strength=strength, ridge=ridge)
    queries = embedder.embed_queries([t.query for t in triplets]).vectors
    positives = embedder.embed_documents([t.positive for t in triplets]).vectors

    negative_texts = [text for triplet in triplets for text in triplet.negatives]
    negatives = embedder.embed_documents(negative_texts).vectors if negative_texts else None

    adapter.fit(queries, positives, negatives, held_out=held_out)
    return adapter


def split_triplets(
    triplets: Sequence[Triplet], *, fraction: float = 0.5, seed: int = 0
) -> tuple[list[Triplet], list[Triplet]]:
    """Split into a training half and a held-out half.

    The honest way to use an adapter. Fitted on everything and scored on everything, it will
    report a gain it has not earned, and that gain will not survive contact with a real query.
    """
    if not 0.0 < fraction < 1.0:
        raise AdapterError(f"fraction must be between 0 and 1, got {fraction}")

    order = np.random.default_rng(seed).permutation(len(triplets))
    cut = max(1, int(len(triplets) * fraction))
    return (
        [triplets[index] for index in order[:cut]],
        [triplets[index] for index in order[cut:]],
    )

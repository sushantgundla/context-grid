"""Semantic chunking: cutting where the meaning changes.

The most-hyped strategy in the field and the one with the least published measurement behind
it. The idea is appealing: embed each sentence, and cut where consecutive sentences stop
being about the same thing, so a chunk boundary lands at a topic change rather than at an
arbitrary token count.

Whether that helps is exactly the sort of question this package exists to answer rather than
assert. It is not free -- it embeds the corpus once to decide where to cut and again to index
the result -- and on corpora with real structure a heading-aware chunker often gets the same
boundaries for nothing.

The threshold is a **percentile of this document's own similarity drops**, not an absolute
similarity. Absolute thresholds do not transfer: 0.7 is a big drop for one embedding model and
noise for another, so a fixed value silently means something different on every arm of a sweep.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from contextgrid.chunk.base import ChunkBuilder, ChunkerError, trim_range
from contextgrid.chunk.sentence import sentence_ranges
from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.protocols import Tokenizer
from contextgrid.embed import Embedder, get_embedder
from contextgrid.tokens import get_tokenizer


@dataclass(slots=True)
class SemanticChunker:
    """Cut where consecutive sentences stop being about the same thing.

    `percentile` is how aggressive to be: at 90, only the largest tenth of this document's
    similarity drops become boundaries, giving few large chunks; at 50, half of them do.

    `buffer_size` groups neighbouring sentences before comparing, which smooths the noise a
    single short sentence introduces. A one-clause sentence between two paragraphs is a blip,
    not a topic change.

    `max_size` is a backstop. Semantic similarity can run for pages without a real break, and
    a chunk nothing will fit in the context window is worse than one cut slightly early.
    """

    embedder: str | Embedder = "tfidf"
    percentile: float = 90.0
    buffer_size: int = 1
    max_size: int = 1024
    min_sentences: int = 1
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "semantic"
    version: ClassVar[str] = "1"

    _embedder: Embedder = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0 < self.percentile < 100:
            raise ChunkerError(
                f"percentile must be between 0 and 100, got {self.percentile}. It is a "
                "percentile of this document's own similarity drops, not a similarity."
            )
        if self.buffer_size < 0:
            raise ChunkerError(f"buffer_size must be >= 0, got {self.buffer_size}")
        if self.max_size <= 0:
            raise ChunkerError(f"max_size must be positive, got {self.max_size}")
        object.__setattr__(self, "_embedder", get_embedder(self.embedder))

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        sentences = sentence_ranges(text)
        if not sentences:
            return []
        if len(sentences) <= self.min_sentences:
            return ChunkBuilder(parsed, [self._counter()]).build_all(
                [trim_range(text, sentences[0][0], sentences[-1][1])]
            )

        boundaries = self._boundaries(text, sentences)
        ranges = self._ranges(text, sentences, boundaries)
        return ChunkBuilder(parsed, [self._counter()]).build_all(ranges)

    # -- where the meaning changes -------------------------------------------

    def _boundaries(self, text: str, sentences: list[tuple[int, int]]) -> set[int]:
        """Indices of sentences that should start a new chunk."""
        windows = [self._window(text, sentences, index) for index in range(len(sentences))]

        self._embedder.prepare(windows)
        vectors = self._embedder.embed_documents(windows).vectors
        if vectors.size == 0:
            return set()

        similarities = [
            float(np.dot(vectors[index], vectors[index + 1])) for index in range(len(windows) - 1)
        ]
        if not similarities:
            return set()

        # A boundary is a *drop* relative to this document, so the cut-off is a percentile of
        # the distances rather than a fixed similarity. Fixed thresholds do not transfer
        # between embedding models and would mean something different on every arm.
        distances = [1.0 - value for value in similarities]
        cutoff = _percentile(distances, self.percentile)

        return {
            index + 1
            for index, distance in enumerate(distances)
            if distance >= cutoff and distance > 0
        }

    def _window(self, text: str, sentences: list[tuple[int, int]], index: int) -> str:
        """One sentence plus its neighbours, so a one-clause sentence is not a topic change."""
        low = max(0, index - self.buffer_size)
        high = min(len(sentences), index + self.buffer_size + 1)
        return text[sentences[low][0] : sentences[high - 1][1]]

    def _ranges(
        self, text: str, sentences: list[tuple[int, int]], boundaries: set[int]
    ) -> list[tuple[int, int]]:
        counter = self._counter()
        ranges: list[tuple[int, int]] = []
        start_index = 0

        for index in range(1, len(sentences) + 1):
            at_boundary = index in boundaries
            at_end = index == len(sentences)
            span = (sentences[start_index][0], sentences[index - 1][1])
            too_large = counter.count(text[span[0] : span[1]]) >= self.max_size

            if at_boundary or at_end or too_large:
                ranges.append(trim_range(text, *span))
                start_index = index

        return [(start, end) for start, end in ranges if end > start]

    def _counter(self) -> Tokenizer:
        return get_tokenizer(self.tokenizer)


def _percentile(values: list[float], percentile: float) -> float:
    """The percentile of a list, without needing numpy's percentile semantics.

    `statistics.quantiles` is exclusive by default, which puts the cut-off slightly inside the
    data and produces marginally more boundaries than asked for. Interpolating the sorted list
    directly is easier to reason about and matches what the parameter name promises.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (percentile / 100) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def similarity_profile(chunker: SemanticChunker, parsed: ParsedDocument) -> list[float]:
    """Sentence-to-sentence similarity down the document.

    For inspecting *why* a semantic chunker cut where it did. A flat profile means the
    strategy had nothing to work with and the boundaries are essentially arbitrary, which is
    worth knowing before believing its score.
    """
    text = parsed.text
    sentences = sentence_ranges(text)
    if len(sentences) < 2:
        return []

    windows = [chunker._window(text, sentences, index) for index in range(len(sentences))]
    chunker._embedder.prepare(windows)
    vectors = chunker._embedder.embed_documents(windows).vectors

    return [float(np.dot(vectors[index], vectors[index + 1])) for index in range(len(windows) - 1)]


def profile_summary(profile: list[float]) -> str:
    """Whether a semantic chunker had anything to work with on this document."""
    if len(profile) < 2:
        return "too few sentences to profile"

    spread = max(profile) - min(profile)
    if spread < 0.05:
        return (
            f"similarity is flat across this document (range {spread:.3f}), so semantic "
            "boundaries here are close to arbitrary and its score should be read that way"
        )
    return (
        f"similarity ranges {min(profile):.2f} to {max(profile):.2f} "
        f"(median {statistics.median(profile):.2f}), so there are real topic shifts to cut on"
    )

"""Embedders that need no model download and no network.

Neither of these will beat a real embedding model, and neither is meant to. They exist so
that the whole pipeline can be exercised, tested and demonstrated with nothing installed --
and because a comparison without a weak baseline in it is hard to read. A reranker that gains
+0.30 over TF-IDF and +0.02 over a good bi-encoder has told you two different things, and
only the second one is interesting.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from contextgrid.embed.base import EmbeddingResult, Vectors, normalise
from contextgrid.tokens import get_tokenizer

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@lru_cache(maxsize=1 << 17)
def _bucket_and_sign(seed: int, word: str, dimensions: int) -> tuple[int, float]:
    """Which dimension a word lands in, and with which sign.

    `hashlib`, never the builtin `hash()`. `hash()` on a `str` is salted by `PYTHONHASHSEED`,
    which Python randomises per process, so the builtin version of this function returned a
    different vector for the same word on every run -- and with it a different score, a
    different ranking and a different report. The whole point of the package is that a run can
    be repeated, so the one embedder that ships with it has to be repeatable first.

    BLAKE2b gives 16 bytes in a single pass. The first 8 choose the bucket, the last 8 choose
    the sign, so the two draws come from disjoint bytes and stay independent of each other --
    the same independence the two separate `hash()` calls used to provide. The seed is folded
    into the message rather than into a salt so that any `int` works, including negatives and
    values wider than BLAKE2b's 16-byte salt.

    The NUL byte separates the seed from the word unambiguously: `_WORD` matches `\\w+`, which
    never contains NUL, so no `(seed, word)` pair can collide with another.
    """
    digest = hashlib.blake2b(f"{seed}\x00{word}".encode(), digest_size=16).digest()
    bucket = int.from_bytes(digest[:8], "big") % dimensions
    # A sign drawn from the hash keeps unrelated words from always adding up, which is what
    # stops every document drifting towards the same direction.
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


@dataclass(frozen=True, slots=True)
class HashEmbedder:
    """Hashed bag of words projected into a fixed number of dimensions.

    The hashing trick, and nothing more: no training, no corpus statistics, no semantics. Two
    documents are close when they share words, and word order is invisible.

    Its job is to be the floor. A dense model that cannot beat this on a corpus is not
    earning its cost, and knowing that is worth the twenty lines it takes to provide.

    Reproducible by construction: `dimensions` and `seed` are the whole state, the digest
    behind them is `hashlib.blake2b`, and nothing in it depends on the process. The same spec
    string on the same text gives the same vectors on any machine, on any run. See
    `_bucket_and_sign` for why that took a deliberate choice.
    """

    dimensions: int = 256
    seed: int = 0

    name: ClassVar[str] = "hash"
    # 2, not 1: version 1 hashed with the builtin `hash()` and so produced different vectors in
    # every process. It is part of the embed cache key, so bumping it retires the vectors that
    # were written by the broken version instead of serving them back as if they were sound.
    version: ClassVar[str] = "2"
    normalised: ClassVar[bool] = True
    max_tokens: ClassVar[int | None] = None

    def prepare(self, documents: Sequence[str]) -> None:
        return None

    def _encode(self, texts: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word, count in Counter(_words(text)).items():
                bucket, sign = _bucket_and_sign(self.seed, word, self.dimensions)
                vectors[row, bucket] += sign * (1.0 + math.log(count))
        return normalise(vectors)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=self._encode(texts),
            input_tokens=sum(len(_words(text)) for text in texts),
        )

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self.embed_documents(texts)


@dataclass(slots=True)
class TfidfEmbedder:
    """Classical TF-IDF over the corpus vocabulary.

    Not a toy. On corpora with distinctive vocabulary -- legal, medical, code -- TF-IDF is
    genuinely competitive with dense retrieval, and the number of teams who pay for an
    embedding API without ever checking that is the reason it belongs on the grid rather
    than in a footnote.

    It is corpus-statistical, so it needs `prepare` before it can embed anything. Queries are
    embedded against the document IDF, never their own, because a query's own statistics are
    meaningless over one sentence.
    """

    max_features: int = 4096
    min_document_frequency: int = 1
    sublinear_tf: bool = True

    name: ClassVar[str] = "tfidf"
    version: ClassVar[str] = "1"
    normalised: ClassVar[bool] = True
    max_tokens: ClassVar[int | None] = None

    _vocabulary: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _idf: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32), init=False, repr=False
    )

    @property
    def dimensions(self) -> int:
        return len(self._vocabulary)

    @property
    def is_prepared(self) -> bool:
        return bool(self._vocabulary)

    def prepare(self, documents: Sequence[str]) -> None:
        """Learn the vocabulary and the inverse document frequencies."""
        document_frequency: Counter[str] = Counter()
        for text in documents:
            document_frequency.update(set(_words(text)))

        kept = [
            word
            for word, count in document_frequency.most_common()
            if count >= self.min_document_frequency
        ][: self.max_features]

        self._vocabulary = {word: index for index, word in enumerate(sorted(kept))}
        total = max(len(documents), 1)
        self._idf = np.array(
            [math.log((1 + total) / (1 + document_frequency[word])) + 1.0 for word in sorted(kept)],
            dtype=np.float32,
        )

    def _encode(self, texts: Sequence[str]) -> Vectors:
        if not self.is_prepared:
            # Falling back to an empty vocabulary would return zero vectors and score zero,
            # which reads as "this embedder is bad" rather than "it was never fitted".
            raise RuntimeError(
                "TfidfEmbedder.prepare() must be called with the corpus before embedding. "
                "It learns its vocabulary from the documents it will search."
            )

        vectors = np.zeros((len(texts), len(self._vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for word, count in Counter(_words(text)).items():
                column = self._vocabulary.get(word)
                if column is None:
                    continue  # out-of-vocabulary, exactly as at serving time
                frequency = 1.0 + math.log(count) if self.sublinear_tf else float(count)
                vectors[row, column] = frequency * self._idf[column]
        return normalise(vectors)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=self._encode(texts),
            input_tokens=sum(len(_words(text)) for text in texts),
        )

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        # The same transform: a query is scored against document statistics, not its own.
        return self.embed_documents(texts)


@dataclass(frozen=True, slots=True)
class TokenCountEmbedder:
    """A deliberately terrible embedder: one dimension, the length of the text.

    Included for the conformance suite and for sanity checks. A metrics pipeline that cannot
    tell this apart from a real model is broken, and it is useful to have something that
    should score near chance to prove that it does.
    """

    tokenizer: str = "regex"

    name: ClassVar[str] = "length"
    version: ClassVar[str] = "1"
    dimensions: ClassVar[int] = 1
    normalised: ClassVar[bool] = False
    max_tokens: ClassVar[int | None] = None

    def prepare(self, documents: Sequence[str]) -> None:
        return None

    def _encode(self, texts: Sequence[str]) -> Vectors:
        counter = get_tokenizer(self.tokenizer)
        return np.array([[float(counter.count(text))] for text in texts], dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=self._encode(texts))

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=self._encode(texts))

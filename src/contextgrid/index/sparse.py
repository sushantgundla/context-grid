"""BM25 lexical search.

Still the baseline that embarrasses people. On keyword-heavy corpora -- error codes, product
names, statute numbers, anything where the query and the document share rare exact tokens --
BM25 regularly beats a dense model that costs real money to run, and the number of stacks
that never checked is the reason it is a first-class arm here rather than a footnote.

No model, no vectors, no network.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from contextgrid.core.documents import Chunk
from contextgrid.embed.base import Vectors
from contextgrid.index.base import Scored, top_k

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25.

    `k1` controls how quickly repeated terms stop helping; `b` how hard long documents are
    penalised. Both are swept rather than assumed, because the usual defaults were tuned on
    TREC news articles and a corpus of 200-token chunks is not that.
    """

    k1: float = 1.5
    b: float = 0.75

    name: ClassVar[str] = "bm25"
    version: ClassVar[str] = "1"
    needs_vectors: ClassVar[bool] = False
    is_exact: ClassVar[bool] = True

    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _postings: dict[str, list[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _average_length: float = field(default=0.0, init=False, repr=False)

    def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None:
        del vectors  # BM25 works on text; vectors are simply not its business.
        self._ids = [chunk.id for chunk in chunks]
        self._lengths = []
        self._postings = {}

        for position, chunk in enumerate(chunks):
            terms = tokenize(chunk.text)
            self._lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self._postings.setdefault(term, []).append((position, count))

        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]:
        del vector
        if not self._ids:
            return []

        total = len(self._ids)
        scores: dict[str, float] = {}

        for term in tokenize(text):
            postings = self._postings.get(term)
            if not postings:
                continue

            # The BM25 idf, with the +0.5 smoothing that keeps a term appearing in more than
            # half the corpus from scoring negative.
            document_frequency = len(postings)
            idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

            for position, frequency in postings:
                length_norm = (
                    1
                    - self.b
                    + self.b
                    * (
                        self._lengths[position] / self._average_length
                        if self._average_length
                        else 0.0
                    )
                )
                weight = (frequency * (self.k1 + 1)) / (frequency + self.k1 * length_norm)
                chunk_id = self._ids[position]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * weight

        return top_k(scores, k)

    def size_bytes(self) -> int:
        # Rough: two ints per posting, plus the term strings themselves.
        postings = sum(len(entries) for entries in self._postings.values())
        terms = sum(len(term) for term in self._postings)
        return postings * 16 + terms

    def __len__(self) -> int:
        return len(self._ids)

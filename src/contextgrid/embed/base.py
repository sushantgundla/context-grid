"""Embedders: turning text into vectors, correctly.

Two details here are the difference between a real comparison and a plausible-looking one,
and almost every homegrown evaluation gets at least one of them wrong.

**Queries and documents are not the same thing.** E5 wants `query:` and `passage:` prefixes,
BGE wants an instruction on the query only, Cohere wants `input_type`. Embed both sides the
same way and the model is being used wrong -- the numbers still look reasonable, they are
just lower than they should be, uniformly and invisibly. So the protocol has two methods and
no default that quietly makes them one.

**Text that exceeds the model's context is silently cut.** The chunk that held the answer
gets truncated to its first 512 tokens, the answer was in the last paragraph, and nothing
says so. Every embedder here reports what it truncated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from contextgrid.core.warnings import Severity, WarningCode, WarningLog

Vectors = np.ndarray


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors, plus everything needed to price and to trust them."""

    vectors: Vectors
    warnings: WarningLog = field(default_factory=WarningLog)
    input_tokens: int = 0
    truncated: int = 0

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0]) if self.vectors.size else 0

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors, with queries and documents handled separately."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def normalised(self) -> bool:
        """True when vectors are unit length, so dot product and cosine agree."""
        ...

    @property
    def max_tokens(self) -> int | None:
        """Context limit, or None when there is none. Anything longer is truncated."""
        ...

    def prepare(self, documents: Sequence[str]) -> None:
        """See the corpus before embedding it.

        A no-op for most models. TF-IDF and other corpus-statistical methods need it, and
        pretending they do not would either give them global statistics they should not have
        or force a different interface for one arm of the comparison.
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult: ...


# ---------------------------------------------------------------------------
# helpers every embedder needs
# ---------------------------------------------------------------------------


def normalise(vectors: Vectors) -> Vectors:
    """Scale each row to unit length, leaving all-zero rows alone.

    A zero vector has no direction, and dividing by its zero norm would produce NaNs that
    propagate into every similarity score as a silently missing document.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    scaled: Vectors = vectors / norms
    return scaled


def truncate(
    texts: Sequence[str],
    max_tokens: int | None,
    *,
    model: str,
    stage: str = "embed",
    approximate_chars_per_token: float = 4.0,
) -> tuple[list[str], WarningLog, int]:
    """Cut texts to a model's context, loudly.

    The character estimate is deliberately crude: an embedder that knows its real tokenizer
    should truncate with it and pass `max_tokens=None` here. What matters is that something
    says a chunk was cut, because the alternative is a config that scores badly for a reason
    no chart will ever show.
    """
    log = WarningLog()
    if max_tokens is None:
        return list(texts), log, 0

    limit = int(max_tokens * approximate_chars_per_token)
    cut: list[str] = []
    truncated = 0
    for index, text in enumerate(texts):
        if len(text) > limit:
            truncated += 1
            cut.append(text[:limit])
            if truncated <= 3:
                log.add(
                    WarningCode.INPUT_TRUNCATED,
                    f"{model} truncated input {index} from {len(text)} to {limit} characters "
                    f"(~{max_tokens} tokens). Anything after the cut cannot be retrieved",
                    severity=Severity.CAUTION,
                    stage=stage,
                    subject=model,
                    original_length=len(text),
                    limit=limit,
                )
        else:
            cut.append(text)

    if truncated > 3:
        log.add(
            WarningCode.INPUT_TRUNCATED,
            f"{model} truncated {truncated} inputs in total. A configuration that cuts this "
            "much of its own corpus is not being measured fairly against one that does not",
            severity=Severity.CAUTION,
            stage=stage,
            subject=model,
            truncated=truncated,
        )

    return cut, log, truncated

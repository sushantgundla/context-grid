"""Real model tokenizers, via tiktoken.

The tokenizers in `tokens.py` are approximations, and they say so: a regex over words is fine
for deciding where to cut text and wrong for deciding what something costs. On English prose a
byte-pair encoder emits roughly a third more tokens than word-splitting does, and on code or
non-Latin scripts the gap is far wider. The cost model refuses to price with an inexact
tokenizer for exactly that reason.

This is the exact one. `cl100k_base` is what GPT-3.5 and GPT-4 use, and it was registered
against this module before the module existed -- asking for it raised an install instruction
for an extra that was also missing, so the honest fix was to write both.

**Character offsets, not just counts.** The `Tokenizer` protocol wants `token_spans`, because a
chunker has to cut at a token boundary *and* report the character offsets it cut at. tiktoken
speaks bytes, so each token's byte length is accumulated and mapped back to characters at the
end. A tokenizer that could only count would force the chunker to guess, and guessed offsets
are the one thing this package refuses to produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from typing import Any

from contextgrid.core.errors import MissingExtraError
from contextgrid.core.protocols import Tokenizer


@lru_cache(maxsize=8)
def _encoding(name: str) -> Any:
    """One encoding per name, built once.

    tiktoken reads a vocabulary of a hundred thousand entries from disk on first use, and a
    chunker asks for a tokenizer per document.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised by the extras test
        raise MissingExtraError("Exact tokenizers", "embed", package="tiktoken") from exc

    try:
        return tiktoken.get_encoding(name)
    except Exception as error:
        raise MissingExtraError(
            f"The {name!r} encoding could not be loaded ({error}). tiktoken downloads its "
            "vocabulary on first use, so this needs network once",
            "embed",
            package="tiktoken",
        ) from error


@dataclass(frozen=True, slots=True)
class TiktokenTokenizer:
    """A byte-pair encoder, with real character offsets.

    `exact` is true, which is what makes it usable for costing. Everything in `tokens.py` is
    an approximation and is barred from the cost model.
    """

    encoding: str = "cl100k_base"

    name: str = field(init=False, default="")
    exact: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(_encoding(self.encoding).encode(text))

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character ranges of each token, in order and non-overlapping.

        tiktoken works in bytes, so this walks the tokens accumulating byte offsets and then
        converts to character offsets once. A multi-byte character can sit inside a single
        token or straddle two, so the conversion has to be done on the accumulated boundary
        rather than per token -- doing it per token drifts on any text that is not ASCII.
        """
        if not text:
            return []

        encoding = _encoding(self.encoding)
        tokens = encoding.encode(text)

        # Byte offset of each token boundary.
        boundaries: list[int] = [0]
        position = 0
        for token in tokens:
            position += len(encoding.decode_single_token_bytes(token))
            boundaries.append(position)

        # Every byte position mapped to the character that contains it. Built for all bytes
        # rather than only boundaries, because a token boundary can land *inside* a multi-byte
        # character -- tiktoken splits on bytes, and "日" is three of them.
        character_of_byte: list[int] = []
        for index, character in enumerate(text):
            character_of_byte.extend([index] * len(character.encode("utf-8")))
        character_of_byte.append(len(text))

        spans: list[tuple[int, int]] = []
        cursor = 0
        for start, end in pairwise(boundaries):
            low = max(cursor, character_of_byte[min(start, len(character_of_byte) - 1)])
            high = character_of_byte[min(end, len(character_of_byte) - 1)]
            if high <= low:
                # Two tokens split one character between them. They become one span rather
                # than two, because half a character is not a range anything can slice.
                continue
            spans.append((low, high))
            cursor = high
        return spans


def cl100k_base() -> Tokenizer:
    """GPT-3.5 and GPT-4's encoding. The one to cost OpenAI models with."""
    return TiktokenTokenizer("cl100k_base")


def o200k_base() -> Tokenizer:
    """GPT-4o's encoding."""
    return TiktokenTokenizer("o200k_base")

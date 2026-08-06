"""Tokenizers, and the registry of them.

Chunk size is the most-discussed parameter in retrieval and the least well specified. "512
with 50 overlap" describes different text under cl100k_base than under a BERT wordpiece
vocabulary than under whitespace splitting -- so a comparison that does not name its
tokenizer is not reproducible, and a comparison that uses *different* tokenizers on different
arms is not fair.

Every tokenizer here therefore reports whether it is `exact`. An approximate tokenizer is
perfectly good for deciding where to cut text. It must never be used to price anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contextgrid.core.protocols import Tokenizer
from contextgrid.core.registry import Registry

TOKENIZERS: Registry[Tokenizer] = Registry(family="tokenizer")

# Words, numbers and standalone punctuation. A deliberately boring approximation.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RegexTokenizer:
    """Words and punctuation marks, found by regex. No dependencies.

    The default, because it works everywhere and produces sensible chunk boundaries. It is
    *not* a model's tokenizer: on English prose a byte-pair encoder emits roughly a third
    more tokens than this does, and on code or non-Latin scripts the gap is much wider. So
    `exact` is false and the cost model refuses to use it.

    Use it to chunk. Use a real model tokenizer to cost.
    """

    name: str = "regex"
    exact: bool = False

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in _WORD_RE.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in _WORD_RE.finditer(text))


@dataclass(frozen=True, slots=True)
class CharacterTokenizer:
    """One token per character. Exact by definition, and unit-free.

    Useful as a control arm: chunk sizes in characters mean the same thing to every model,
    so a sweep measured in characters is the one comparison no tokenizer disagreement can
    distort.
    """

    name: str = "character"
    exact: bool = True

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    def count(self, text: str) -> int:
        return len(text)


@TOKENIZERS.register("regex", doc="Words and punctuation by regex. No dependencies.")
def _regex_tokenizer() -> Tokenizer:
    return RegexTokenizer()


@TOKENIZERS.register("character", doc="One token per character. Exact and unit-free.")
def _character_tokenizer() -> Tokenizer:
    return CharacterTokenizer()


# Real model tokenizers arrive with the extras that need them. Registered lazily so that
# `import contextgrid` stays free of heavy imports.
TOKENIZERS.register_lazy(
    "cl100k_base",
    module="contextgrid.tokenizers_tiktoken",
    attr="cl100k_base",
    extra="embed",
    package="tiktoken",
    doc="OpenAI cl100k_base, via tiktoken. Exact, so the cost model will price with it.",
)
TOKENIZERS.register_lazy(
    "o200k_base",
    module="contextgrid.tokenizers_tiktoken",
    attr="o200k_base",
    extra="embed",
    package="tiktoken",
    doc="GPT-4o's encoding, via tiktoken. Exact.",
)


def get_tokenizer(spec: str | Tokenizer | None) -> Tokenizer:
    """Resolve a tokenizer from a name, a spec string, or an instance.

    `None` gives the default, which is `regex`.
    """
    if spec is None:
        return RegexTokenizer()
    if isinstance(spec, str):
        return TOKENIZERS.create(spec)
    return spec

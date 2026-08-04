"""Chunkers from [chonkie](https://docs.chonkie.ai), adapted onto our `Chunk`.

Chonkie exists to do one thing and does more of it than we ever will: nine strategies
including late chunking, neural boundary detection and AST-aware code splitting, with a Rust
core underneath. Reimplementing that would produce a worse version of it and -- worse -- would
mean this package compares *our* chunkers rather than the ones people actually deploy.

Every chonkie chunk carries `start_index` and `end_index` into the original text, and those
offsets are exact. That is verified in `tests/unit/test_chunk_chonkie.py` and re-checked on
every document at runtime rather than assumed, because the whole scoring model rests on it: a
chunker whose offsets drift silently moves every gold span in the corpus, and the run still
looks fine.

**Chunk sizes are always in our tokens.** Chonkie counts characters by default. This package
counts tokens everywhere, because "512" under a byte-pair encoder and "512" under whitespace
splitting describe different amounts of text, and an axis that sweeps both under one name is
not measuring what it claims to. `_TokenizerBridge` hands our tokenizer down so that
`chonkie:recursive:512` and `recursive:512` mean the same 512.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from contextgrid.chunk.base import ChunkBuilder, ChunkerError
from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.protocols import Tokenizer
from contextgrid.tokens import get_tokenizer


class _ChonkieChunk(Protocol):
    """The part of chonkie's `Chunk` this adapter relies on."""

    text: str
    start_index: int
    end_index: int


def _chonkie() -> Any:
    """Import chonkie, with a message that says what to install."""
    try:
        import chonkie
    except ImportError as error:  # pragma: no cover - exercised by the extras test
        raise ChunkerError(
            "chonkie chunkers need chonkie. Install it with: pip install 'context-grid[chunk]'"
        ) from error
    return chonkie


def _chonkie_class(class_name: str) -> Any:
    module = _chonkie()
    try:
        return getattr(module, class_name)
    except AttributeError as error:
        raise ChunkerError(
            f"the installed chonkie has no {class_name}. This adapter was written against "
            "chonkie 1.7; try: pip install -U chonkie"
        ) from error


def _bridge(tokenizer: Tokenizer) -> Any:
    """Wrap one of our tokenizers so chonkie will accept it.

    Built here rather than at import time because it subclasses a chonkie type, and this
    package must import without chonkie installed.

    The encoding is lossless by construction. Each token id stands for the token *together
    with the text that preceded it*, so the segments cover the string with no gaps and
    `decode(encode(text)) == text` exactly. That matters because chonkie's token chunker
    decodes its own output to work out where a chunk starts -- an encoding that dropped
    whitespace would hand back offsets that are quietly a few characters off, which is the
    single worst failure this package could have.
    """
    base = _chonkie().tokenizer.Tokenizer

    class _TokenizerBridge(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.inner = tokenizer

        def __repr__(self) -> str:
            return f"ContextGridTokenizer({self.inner.name})"

        def _segments(self, text: str) -> list[str]:
            out: list[str] = []
            position = 0
            for _, end in self.inner.token_spans(text):
                out.append(text[position:end])
                position = end
            if position < len(text):
                out.append(text[position:])  # trailing whitespace belongs to somebody
            return out

        def encode(self, text: str) -> list[int]:
            ids: list[int] = []
            for segment in self._segments(text):
                index = self.token2id[segment]
                if index == len(self.vocab):
                    self.vocab.append(segment)
                ids.append(index)
            return ids

        def decode(self, tokens: Any) -> str:
            return "".join(self.vocab[index] for index in tokens)

        def tokenize(self, text: str) -> list[str]:
            return self._segments(text)

        def count_tokens(self, text: str) -> int:
            return self.inner.count(text)

    return _TokenizerBridge()


@dataclass(frozen=True, slots=True)
class _ChonkieChunker:
    """Shared body for every chonkie strategy.

    Subclasses name the chonkie class and add whatever keyword arguments it takes. Turning
    the result into our `Chunk` is identical in every case and lives here.
    """

    size: int = 512
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "chonkie"
    version: ClassVar[str] = "1"
    chonkie_class: ClassVar[str] = ""

    _tokenizer: Tokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ChunkerError(f"chunk size must be positive, got {self.size}")
        object.__setattr__(self, "_tokenizer", get_tokenizer(self.tokenizer))

    def _kwargs(self) -> dict[str, Any]:
        # Subclasses call this as `_ChonkieChunker._kwargs(self)` rather than `super()`.
        # A `slots=True` dataclass is rebuilt by the decorator, which leaves the `__class__`
        # cell that zero-argument `super()` depends on pointing at the class that was
        # replaced -- so `super()` raises TypeError in a subclass method.
        #
        # A fresh bridge per call. Its vocabulary grows as it encodes, and sharing one across
        # documents would make chunking depend on what was chunked before it.
        return {"chunk_size": self.size, "tokenizer": _bridge(self._tokenizer)}

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        if not text.strip():
            return []

        chunker = _chonkie_class(self.chonkie_class)(**self._kwargs())
        try:
            produced: list[_ChonkieChunk] = list(chunker(text))
        except ChunkerError:
            raise
        except Exception as error:
            raise ChunkerError(f"{self.name} failed on {parsed.id}: {error}") from error

        builder = ChunkBuilder(parsed, [self._tokenizer])
        return builder.build_all([self._checked(piece, text) for piece in produced])

    def _checked(self, piece: _ChonkieChunk, text: str) -> tuple[int, int]:
        """Trust nothing about the offsets. Verify them against the text every time."""
        start, end = int(piece.start_index), int(piece.end_index)
        if not (0 <= start <= end <= len(text)):
            raise ChunkerError(
                f"{self.name} returned an out-of-range span ({start}, {end}) for a document "
                f"of {len(text)} characters"
            )
        if text[start:end] != piece.text:
            raise ChunkerError(
                f"{self.name} returned a chunk whose text does not match its own offsets. "
                "Gold evidence in this package is stored as character offsets, so a chunker "
                "that cannot round-trip them cannot be scored against."
            )
        return start, end


@dataclass(frozen=True, slots=True)
class ChonkieTokenChunker(_ChonkieChunker):
    """Fixed token windows with overlap. Chonkie's `TokenChunker`."""

    overlap: int = 0

    name: ClassVar[str] = "chonkie:token"
    chonkie_class: ClassVar[str] = "TokenChunker"

    def _kwargs(self) -> dict[str, Any]:
        return {**_ChonkieChunker._kwargs(self), "chunk_overlap": self.overlap}


@dataclass(frozen=True, slots=True)
class ChonkieRecursiveChunker(_ChonkieChunker):
    """Split on the largest natural boundary that fits. Chonkie's `RecursiveChunker`.

    The head-to-head against our own `recursive`, which is most of the reason for having both
    on the axis: same idea, different implementation, and the tool exists to say which wins on
    a given corpus.
    """

    min_characters: int = 24

    name: ClassVar[str] = "chonkie:recursive"
    chonkie_class: ClassVar[str] = "RecursiveChunker"

    def _kwargs(self) -> dict[str, Any]:
        return {**_ChonkieChunker._kwargs(self), "min_characters_per_chunk": self.min_characters}


@dataclass(frozen=True, slots=True)
class ChonkieSentenceChunker(_ChonkieChunker):
    """Whole sentences packed up to the size limit. Chonkie's `SentenceChunker`."""

    overlap: int = 0
    min_sentences: int = 1

    name: ClassVar[str] = "chonkie:sentence"
    chonkie_class: ClassVar[str] = "SentenceChunker"

    def _kwargs(self) -> dict[str, Any]:
        return {
            **_ChonkieChunker._kwargs(self),
            "chunk_overlap": self.overlap,
            "min_sentences_per_chunk": self.min_sentences,
        }


@dataclass(frozen=True, slots=True)
class ChonkieCodeChunker(_ChonkieChunker):
    """Splits on the syntax tree rather than on the text. Chonkie's `CodeChunker`.

    Nothing hand-written here comes close. A function cut in half is useless to retrieve, and
    only a parser for the language can reliably avoid doing it.
    """

    language: str = "auto"

    name: ClassVar[str] = "chonkie:code"
    chonkie_class: ClassVar[str] = "CodeChunker"

    def _kwargs(self) -> dict[str, Any]:
        return {**_ChonkieChunker._kwargs(self), "language": self.language}

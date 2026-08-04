"""Chunkers from `langchain-text-splitters`, adapted onto our `Chunk`.

These are on the axis for one reason: they are what most deployed RAG systems are actually
running. A comparison that shows chonkie beating our own recursive chunker is interesting;
a comparison that shows either of them beating *LangChain's* recursive splitter tells somebody
whether it is worth changing the code they already have in production.

LangChain's splitters return strings, not offsets, so the adapter asks for
`add_start_index=True` and reads `metadata["start_index"]`. That is the whole reason
`create_documents` is used here instead of the more obvious `split_text`: without the offsets
these chunks could not be scored against character-span gold at all.

Two behaviours worth knowing about, both of them LangChain's rather than ours:

* Its splitters **strip whitespace** by default, so chunks do not tile the document. Offsets
  stay exact -- `start_index` accounts for the stripping -- but a character between two chunks
  can belong to neither. Gold spans in that gap resolve to nothing, and the run reports it.
* **Overlap means chunks share characters.** Ordinary and intended; our scorer handles
  overlapping chunk spans already.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from contextgrid.chunk.base import ChunkBuilder, ChunkerError, validate_size_and_overlap
from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.protocols import Tokenizer
from contextgrid.tokens import get_tokenizer


def _splitter_class(class_name: str) -> Any:
    try:
        import langchain_text_splitters
    except ImportError as error:  # pragma: no cover - exercised by the extras test
        raise ChunkerError(
            "langchain chunkers need langchain-text-splitters. Install with: "
            "pip install 'context-grid[chunk]'"
        ) from error

    try:
        return getattr(langchain_text_splitters, class_name)
    except AttributeError as error:
        raise ChunkerError(
            f"the installed langchain-text-splitters has no {class_name}. "
            "Try: pip install -U langchain-text-splitters"
        ) from error


@dataclass(frozen=True, slots=True)
class _LangChainChunker:
    """Shared body for the LangChain splitters."""

    size: int = 512
    # Zero, not LangChain's own default, so the size shorthand always works on its own:
    # `langchain:recursive:32` with an inherited overlap of 64 would be rejected outright,
    # while `chonkie:token:32` sails through. An axis where one arm refuses a value the others
    # accept is not an axis. Ask for overlap explicitly: `langchain:recursive:512,overlap=64`.
    overlap: int = 0
    tokenizer: str | Tokenizer | None = None

    name: ClassVar[str] = "langchain"
    version: ClassVar[str] = "1"
    splitter_class: ClassVar[str] = ""

    _tokenizer: Tokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_size_and_overlap(self.size, self.overlap)
        object.__setattr__(self, "_tokenizer", get_tokenizer(self.tokenizer))

    def _kwargs(self) -> dict[str, Any]:
        # Subclasses call this as `_LangChainChunker._kwargs(self)`. Zero-argument `super()`
        # does not work inside a `slots=True` dataclass: the decorator rebuilds the class and
        # the `__class__` cell `super()` reads still points at the one it replaced.
        return {
            "chunk_size": self.size,
            "chunk_overlap": self.overlap,
            "add_start_index": True,
            # Sizes are in our tokens, so every arm of the chunker axis means the same 512.
            # LangChain's own default is `len`, which would silently make this axis a
            # characters-versus-tokens comparison instead of a strategy comparison.
            "length_function": self._tokenizer.count,
        }

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        if not text.strip():
            return []

        splitter = _splitter_class(self.splitter_class)(**self._kwargs())
        try:
            documents = splitter.create_documents([text])
        except Exception as error:
            raise ChunkerError(f"{self.name} failed on {parsed.id}: {error}") from error

        builder = ChunkBuilder(parsed, [self._tokenizer])

        ranges: list[tuple[int, int]] = []
        cursor = 0
        for document in documents:
            start, end = self._locate(document, text, cursor)
            ranges.append((start, end))
            cursor = start + 1  # +1, not `end`: chunks overlap by design
        return builder.build_all(ranges)

    def _locate(self, document: Any, text: str, cursor: int) -> tuple[int, int]:
        """Find where this chunk really sits in the document.

        LangChain's `start_index` cannot be taken at face value. It rebuilds each chunk by
        rejoining the pieces it split, then looks the result up in the source -- and when the
        rejoined text differs from the original by so much as a newline, it gives up and
        reports `-1`. On this package's own fixtures that happens to roughly one chunk in
        eight, always on tables.

        The chunk content itself is still a literal slice of the document in those cases, so
        the offset is recoverable: search forward from the last chunk's start, and verify the
        slice matches before using it. Trusting `-1` would drop the chunk; trusting a wrong
        index would silently move every gold span that lands in it.
        """
        content = document.page_content

        claimed = document.metadata.get("start_index")
        if isinstance(claimed, int) and claimed >= 0:
            end = claimed + len(content)
            if end <= len(text) and text[claimed:end] == content:
                return claimed, end

        start = text.find(content, cursor)
        if start < 0:
            start = text.find(content)
        if start < 0:
            raise ChunkerError(
                f"{self.name} returned a chunk that is not a literal slice of the document, so "
                "it has no character offsets to score against. Gold evidence here is stored as "
                "character offsets, and a chunker that rewrites text cannot be scored."
            )
        return start, start + len(content)


@dataclass(frozen=True, slots=True)
class LangChainRecursiveChunker(_LangChainChunker):
    """`RecursiveCharacterTextSplitter`. The default nearly every tutorial reaches for."""

    name: ClassVar[str] = "langchain:recursive"
    splitter_class: ClassVar[str] = "RecursiveCharacterTextSplitter"


@dataclass(frozen=True, slots=True)
class LangChainCharacterChunker(_LangChainChunker):
    """`CharacterTextSplitter`. Splits on one separator, `\\n\\n` by default.

    Kept because it is the naive baseline a great many systems shipped with, and knowing how
    much is lost to it is worth a column on the leaderboard.
    """

    separator: str = "\n\n"

    name: ClassVar[str] = "langchain:character"
    splitter_class: ClassVar[str] = "CharacterTextSplitter"

    def _kwargs(self) -> dict[str, Any]:
        return {**_LangChainChunker._kwargs(self), "separator": self.separator}


@dataclass(frozen=True, slots=True)
class LangChainMarkdownChunker(_LangChainChunker):
    """`MarkdownTextSplitter`: recursive, but with Markdown's own boundaries first."""

    name: ClassVar[str] = "langchain:markdown"
    splitter_class: ClassVar[str] = "MarkdownTextSplitter"

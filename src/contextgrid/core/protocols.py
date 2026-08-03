"""The contracts every plugin family implements.

A plugin here is small on purpose. A parser turns a source file into text plus blocks; a
chunker turns that into chunks; a tokenizer counts. Each one is replaceable, each one is
registered by name, and each one must pass the conformance suite for its family before it
can be trusted in a comparison.

The conformance suites are not a nicety. There will eventually be dozens of plugins, and a
single one that quietly loses character offsets would corrupt every number produced from it
without failing anything.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextgrid.core.documents import Chunk, MediaType, ParsedDocument, SourceFile


@runtime_checkable
class Tokenizer(Protocol):
    """Turns text into token boundaries.

    `token_spans` rather than a token count, because a token chunker has to cut at a token
    boundary *and* report the character offsets it cut at. A tokenizer that can only count
    forces the chunker to guess, and guessed offsets are the thing this package refuses to
    produce.
    """

    @property
    def name(self) -> str:
        """Identifier recorded on every chunk this tokenizer measured."""
        ...

    @property
    def exact(self) -> bool:
        """False when boundaries approximate a real model's tokenizer rather than matching it.

        Approximate tokenizers are fine for chunking and wrong for costing.
        """
        ...

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character ranges of each token, in order, non-overlapping."""
        ...

    def count(self, text: str) -> int:
        """Number of tokens in the text."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Turns a source file into text with structure.

    The parser defines the text that every character offset downstream refers to. Two
    parsers over the same PDF produce different text, which is exactly why parser choice is
    worth measuring -- and why a parse carries a `text_hash` that chunks are checked against.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def supports(self, media_type: MediaType) -> bool:
        """Whether this parser will attempt this format."""
        ...

    def parse(self, source: SourceFile) -> ParsedDocument:
        """Extract text and blocks.

        Must set `offsets_exact=False` on the result rather than silently returning blocks
        whose text is not a literal slice of the document.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Cuts a parsed document into retrievable units.

    Chunkers all cut up the *same* text, which is what makes comparing them fair without any
    re-annotation of ground truth.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        """Produce chunks in reading order.

        Every chunk's span must point into `parsed.document`. A chunker that rewrites text --
        prepending LLM-written context, extracting propositions -- must set
        `offsets_exact=False` on the chunks it returns.
        """
        ...

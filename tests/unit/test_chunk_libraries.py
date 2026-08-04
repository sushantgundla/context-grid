"""The adopted chunker libraries: chonkie and langchain-text-splitters.

The conformance suite already runs every one of these through the same protocol checks as the
hand-written chunkers. What is here is the part specific to adopting somebody else's code:
that the offsets they hand back are real, that "512" means the same 512 it means everywhere
else in the package, and that when a library gets an offset wrong we catch it rather than
score against it.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from contextgrid.chunk import CHUNKERS, get_chunker
from contextgrid.chunk.base import ChunkerError
from contextgrid.core.documents import MediaType, ParsedDocument, SourceFile
from contextgrid.parse import MarkdownParser
from contextgrid.tokens import RegexTokenizer
from tests.support import API_DOCS, CONTRACT

chonkie = pytest.importorskip("chonkie")
langchain_text_splitters = pytest.importorskip("langchain_text_splitters")

from contextgrid.chunk.chonkie import (  # noqa: E402
    ChonkieRecursiveChunker,
    ChonkieSentenceChunker,
    ChonkieTokenChunker,
    _bridge,
)
from contextgrid.chunk.langchain import (  # noqa: E402
    LangChainCharacterChunker,
    LangChainMarkdownChunker,
    LangChainRecursiveChunker,
)

LIBRARY_CHUNKERS = [
    ChonkieTokenChunker(size=64),
    ChonkieRecursiveChunker(size=64),
    ChonkieSentenceChunker(size=64),
    LangChainRecursiveChunker(size=64, overlap=8),
    LangChainCharacterChunker(size=64, overlap=8),
    LangChainMarkdownChunker(size=64, overlap=8),
]
IDS = [chunker.name for chunker in LIBRARY_CHUNKERS]


def parse(text: str, doc_id: str = "doc") -> ParsedDocument:
    source = SourceFile(id=doc_id, raw=text.encode("utf-8"), media_type=MediaType.MARKDOWN)
    return MarkdownParser().parse(source)


# ---------------------------------------------------------------------------
# offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunker", LIBRARY_CHUNKERS, ids=IDS)
@pytest.mark.parametrize("text", [CONTRACT, API_DOCS], ids=["contract", "api-docs"])
def test_every_chunk_is_a_literal_slice(chunker: object, text: str) -> None:
    """The one thing an adopted library must get right.

    Gold evidence in this package is a character range. A chunk whose span points somewhere
    other than its own text moves that evidence, and nothing on the leaderboard would look
    wrong.
    """
    parsed = parse(text)
    for chunk in chunker.chunk(parsed):  # type: ignore[attr-defined]
        assert parsed.text[chunk.span.start : chunk.span.end] == chunk.text


@pytest.mark.parametrize("chunker", LIBRARY_CHUNKERS, ids=IDS)
def test_chunks_come_back_in_reading_order(chunker: object) -> None:
    chunks = chunker.chunk(parse(CONTRACT))  # type: ignore[attr-defined]
    starts = [chunk.span.start for chunk in chunks]
    assert starts == sorted(starts)


@pytest.mark.parametrize("chunker", LIBRARY_CHUNKERS, ids=IDS)
def test_chunking_the_same_document_twice_gives_the_same_chunks(chunker: object) -> None:
    """Caching and diffing both depend on this, and a library with hidden state would break
    both quietly."""
    parsed = parse(CONTRACT)
    first = chunker.chunk(parsed)  # type: ignore[attr-defined]
    second = chunker.chunk(parsed)  # type: ignore[attr-defined]
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_langchain_offsets_are_recomputed_rather_than_trusted() -> None:
    """LangChain reports `start_index: -1` for chunks it cannot find in the source.

    It rebuilds each chunk by rejoining the pieces it split and then looks the result up; on
    tables the rejoined text differs from the original and it gives up. The content is still a
    literal slice, so the offset is recoverable -- and taking the -1 at face value would drop
    the chunk instead.
    """
    text = parse(CONTRACT).text
    splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
        chunk_size=64, chunk_overlap=8, add_start_index=True, length_function=RegexTokenizer().count
    )
    reported = splitter.create_documents([text])
    assert any(d.metadata["start_index"] < 0 for d in reported), (
        "this test is only meaningful while LangChain still loses offsets on this fixture"
    )

    for chunk in LangChainRecursiveChunker(size=64, overlap=8).chunk(parse(CONTRACT)):
        assert chunk.span.start >= 0
        assert text[chunk.span.start : chunk.span.end] == chunk.text


def test_a_chunk_that_is_not_in_the_document_is_refused() -> None:
    """A splitter that rewrites text cannot be scored against character-span gold, so the
    adapter refuses rather than guessing an offset."""

    class Rewriting:
        def __init__(self, **kwargs: object) -> None:
            pass

        def create_documents(self, texts: list[str]) -> list[object]:
            class Doc:
                page_content = "text that never appeared in the document"
                metadata: ClassVar[dict[str, int]] = {"start_index": -1}

            return [Doc()]

    import contextgrid.chunk.langchain as module

    chunker = LangChainRecursiveChunker(size=64)

    original = module._splitter_class
    module._splitter_class = lambda name: Rewriting  # type: ignore[assignment]
    try:
        with pytest.raises(ChunkerError, match="not a literal slice"):
            chunker.chunk(parse(CONTRACT))
    finally:
        module._splitter_class = original  # type: ignore[assignment]


def test_a_chonkie_chunk_whose_offsets_lie_is_refused() -> None:
    """Verified on every document rather than once in a test, because a future chonkie whose
    offsets drift would otherwise move every gold span in the corpus silently."""

    class Lying:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __call__(self, text: str) -> list[object]:
            class Piece:
                text = "Termination"
                start_index = 0
                end_index = 11

            return [Piece()]

    import contextgrid.chunk.chonkie as module

    original = module._chonkie_class
    module._chonkie_class = lambda name: Lying  # type: ignore[assignment]
    try:
        with pytest.raises(ChunkerError, match="does not match its own offsets"):
            ChonkieRecursiveChunker(size=64).chunk(parse(CONTRACT))
    finally:
        module._chonkie_class = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# sizes mean the same thing on every arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunker", LIBRARY_CHUNKERS, ids=IDS)
def test_sizes_are_measured_in_our_tokens(chunker: object) -> None:
    """Chonkie counts characters by default and LangChain counts `len`.

    Left alone, `chonkie:recursive:512` would mean 512 characters while `recursive:512` meant
    512 tokens, and the chunker axis would quietly become a units comparison. Both adapters
    pass our tokenizer down, so a chunk of "size 64" is 64 of the same tokens everywhere.
    """
    tokenizer = RegexTokenizer()
    chunks = chunker.chunk(parse(CONTRACT))  # type: ignore[attr-defined]
    assert chunks

    # Not exact: every splitter here will exceed the limit rather than cut a word in half, and
    # LangChain measures before stripping. What matters is that the number is in tokens at all
    # -- a character-counting chunker would come back around five times over.
    assert max(tokenizer.count(chunk.text) for chunk in chunks) <= 64 * 2


@pytest.mark.parametrize("chunker", LIBRARY_CHUNKERS, ids=IDS)
def test_token_counts_are_recorded_under_our_tokenizer_name(chunker: object) -> None:
    for chunk in chunker.chunk(parse(CONTRACT)):  # type: ignore[attr-defined]
        assert "regex" in chunk.token_counts


def test_a_bigger_size_gives_fewer_chunks() -> None:
    """The parameter has to actually do something, on every arm."""
    parsed = parse(CONTRACT)
    for name in ("chonkie:recursive", "chonkie:token", "langchain:recursive"):
        small = len(get_chunker(f"{name}:32").chunk(parsed))
        large = len(get_chunker(f"{name}:256").chunk(parsed))
        assert large < small, name


# ---------------------------------------------------------------------------
# the tokenizer bridge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [CONTRACT, API_DOCS, "a", "", "   \n\n  ", "héllo wörld"])
def test_the_bridge_round_trips_text_exactly(text: str) -> None:
    """Chonkie's token chunker decodes its own output to work out where a chunk starts.

    An encoding that dropped the whitespace between tokens would hand back offsets a few
    characters off -- correct-looking chunks pointing at slightly the wrong text, which is the
    worst failure this package could have.
    """
    bridge = _bridge(RegexTokenizer())
    assert bridge.decode(bridge.encode(text)) == text


def test_the_bridge_counts_with_the_tokenizer_it_was_given() -> None:
    tokenizer = RegexTokenizer()
    bridge = _bridge(tokenizer)
    assert bridge.count_tokens(CONTRACT) == tokenizer.count(CONTRACT)


def test_each_chunk_call_gets_a_fresh_bridge() -> None:
    """The bridge's vocabulary grows as it encodes. Sharing one across documents would make
    the chunking of a document depend on what was chunked before it."""
    chunker = ChonkieTokenChunker(size=32)
    alone = chunker.chunk(parse(API_DOCS, "api"))

    chunker.chunk(parse(CONTRACT, "contract"))
    after = chunker.chunk(parse(API_DOCS, "api"))

    assert [c.text for c in alone] == [c.text for c in after]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "chonkie:recursive",
        "chonkie:recursive:256",
        "chonkie:token:128,overlap=16",
        "chonkie:sentence:256",
        "langchain:recursive:256,overlap=32",
        "langchain:character",
        "langchain:markdown:128",
    ],
)
def test_every_library_chunker_is_reachable_from_a_config_string(spec: str) -> None:
    """The whole adoption is worthless if it needs new API. One YAML line has to be enough."""
    assert get_chunker(spec).chunk(parse(CONTRACT))


def test_the_library_chunkers_are_documented_in_the_registry() -> None:
    described = CHUNKERS.describe()
    for name in described:
        assert described[name], f"{name} has no description, so `contextgrid plugins` is useless"


def test_a_typo_in_a_namespaced_name_names_the_real_ones() -> None:
    from contextgrid.core.registry import UnknownPluginError

    with pytest.raises(UnknownPluginError, match="chonkie:recursive"):
        get_chunker("chonkie:recursve:256")


def test_size_must_be_positive() -> None:
    with pytest.raises(ChunkerError, match="must be positive"):
        ChonkieRecursiveChunker(size=0)


@pytest.mark.parametrize("chunker", LIBRARY_CHUNKERS, ids=IDS)
@pytest.mark.parametrize("text", ["", "   \n\n \t "], ids=["empty", "whitespace"])
def test_an_empty_document_produces_no_chunks(chunker: object, text: str) -> None:
    assert chunker.chunk(parse(text)) == []  # type: ignore[attr-defined]

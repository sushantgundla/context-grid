"""Profiling a corpus before anything expensive runs.

The matrix builder is a blank form until you know what you are looking at. A corpus that is
40% tables wants a different experiment from one that is 200 short Markdown pages, and the
user usually cannot tell which they have without opening every file.

So the fingerprint is cheap, runs before the sweep, and turns into plain-English hints about
which axes are likely to matter. Nothing else in the field profiles your documents before
asking you to configure a comparison of them.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from contextgrid.core.documents import BlockKind, ParsedDocument
from contextgrid.corpus.loader import Corpus, CorpusError

#: Above this share of characters sitting in tables, table handling dominates everything.
TABLE_HEAVY = 0.15
#: Above this share of characters in code blocks, a code-aware embedder is worth testing.
CODE_HEAVY = 0.20
#: Fewer characters than this per document and large chunk sizes cannot differentiate.
SHORT_DOCUMENT = 2_000
#: More than this and parent-document retrieval starts to matter.
LONG_DOCUMENT = 50_000


@dataclass(frozen=True, slots=True)
class CorpusFingerprint:
    """What this corpus is made of.

    The `source_*` fields need only the raw files. Everything else needs a parse, and is
    therefore a statement about *one parser's reading* of the corpus -- which is itself
    worth knowing, since two parsers can disagree sharply about how much of a document is
    a table.
    """

    file_count: int
    total_bytes: int
    media_types: dict[str, int] = field(default_factory=dict)
    duplicate_groups: tuple[tuple[str, ...], ...] = ()

    parser: str | None = None
    total_characters: int = 0
    document_lengths: tuple[int, ...] = ()
    block_kinds: dict[str, int] = field(default_factory=dict)
    table_characters: int = 0
    code_characters: int = 0
    heading_count: int = 0
    empty_documents: tuple[str, ...] = ()

    # -- derived -------------------------------------------------------------

    @property
    def duplicate_file_count(self) -> int:
        """Files that are byte-identical to another file. Each group contributes n-1."""
        return sum(len(group) - 1 for group in self.duplicate_groups)

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_file_count / self.file_count if self.file_count else 0.0

    @property
    def table_ratio(self) -> float:
        return self.table_characters / self.total_characters if self.total_characters else 0.0

    @property
    def code_ratio(self) -> float:
        return self.code_characters / self.total_characters if self.total_characters else 0.0

    @property
    def mean_length(self) -> float:
        return statistics.fmean(self.document_lengths) if self.document_lengths else 0.0

    @property
    def median_length(self) -> float:
        return statistics.median(self.document_lengths) if self.document_lengths else 0.0

    @property
    def headings_per_document(self) -> float:
        return self.heading_count / self.file_count if self.file_count else 0.0

    @property
    def is_parsed(self) -> bool:
        return self.parser is not None

    # -- the useful bit ------------------------------------------------------

    def hints(self) -> list[str]:
        """Plain-English suggestions about which axes are likely to matter here.

        Deliberately conservative: each hint points at an experiment worth running, never at
        a conclusion. The tool exists because nobody should take advice like this on trust.
        """
        notes: list[str] = []

        if self.duplicate_file_count:
            notes.append(
                f"{self.duplicate_file_count} of {self.file_count} files are byte-identical "
                "to another. Near-duplicate chunks inflate recall without helping anyone, so "
                "de-duplication is worth turning on."
            )

        if not self.is_parsed:
            return notes

        if self.table_ratio >= TABLE_HEAVY:
            notes.append(
                f"{self.table_ratio:.0%} of this corpus is tables. Parser choice will "
                "probably dominate every other axis here, and a chunker that splits a table "
                "in half will lose the answer outright."
            )

        if self.code_ratio >= CODE_HEAVY:
            notes.append(
                f"{self.code_ratio:.0%} of this corpus is code. A code-aware embedder and "
                "AST-based chunking are worth putting on the grid."
            )

        if self.heading_count == 0:
            notes.append(
                "No headings were found, so structural chunking has nothing to work with "
                "and will fall back to recursive splitting. If the documents do have "
                "structure, that is a finding about the parser rather than the corpus."
            )
        elif self.headings_per_document >= 5:
            notes.append(
                f"Documents average {self.headings_per_document:.0f} headings each. "
                "Structural chunking usually wins on corpora like this."
            )

        if self.document_lengths:
            if self.median_length < SHORT_DOCUMENT:
                notes.append(
                    f"The median document is {self.median_length:,.0f} characters. Chunk "
                    "sizes above that cannot differentiate, so sweep small sizes."
                )
            elif self.median_length > LONG_DOCUMENT:
                notes.append(
                    f"The median document is {self.median_length:,.0f} characters. "
                    "Parent-document retrieval and section-scoped search are worth testing."
                )

        if self.empty_documents:
            listed = ", ".join(self.empty_documents[:3])
            more = (
                ""
                if len(self.empty_documents) <= 3
                else f" and {len(self.empty_documents) - 3} more"
            )
            notes.append(
                f"{len(self.empty_documents)} documents came back empty ({listed}{more}). "
                "Either they have no text layer and need OCR, or this parser cannot read them."
            )

        return notes

    def summary(self) -> str:
        """One line, for a log or a CLI header."""
        parts = [f"{self.file_count} files", f"{self.total_bytes:,} bytes"]
        if self.is_parsed:
            parts.append(f"{self.total_characters:,} chars via {self.parser}")
            if self.table_ratio:
                parts.append(f"{self.table_ratio:.0%} tables")
        return ", ".join(parts)


def require_parsed_text(
    corpus: Corpus,
    parses: Mapping[str, ParsedDocument] | Sequence[ParsedDocument],
    *,
    parser: str,
) -> None:
    """Fail here, and say so, when a parser reads no text at all out of a whole corpus.

    A parser pointed at file types it does not handle never announces itself. It declines
    every file, or accepts them and returns nothing, and an empty corpus travels quietly down
    the pipeline until something else fails on its own internal invariant. The first thing to
    notice used to be `TfidfEmbedder`, reporting that it had never been fitted -- true, and
    completely silent about the parser the user actually chose.

    **Only a total wipeout is an error.** A corpus of ten PDFs and one Markdown file read by
    a PDF parser should index the ten and warn about the one; erroring there would break
    sweeps that work. So a single readable document is enough to carry on, and the per-file
    `PARSER_FALLBACK` warning stays the right report for a partial skip. This is the summary
    of that warning when it fired on everything, and deliberately borrows its wording.

    Not a check for any particular parser either: it is the empty result that is wrong,
    whatever produced it. A corpus with no files in it is somebody else's error and passes
    through untouched.
    """
    if not len(corpus):
        return

    documents = list(parses.values()) if isinstance(parses, Mapping) else list(parses)
    if any(parsed.text.strip() for parsed in documents):
        return

    listed = ", ".join(corpus.ids[:3])
    more = "" if len(corpus) <= 3 else f" and {len(corpus) - 3} more"
    kinds = ", ".join(sorted({source.media_type.value for source in corpus}))
    counted = "1 file" if len(corpus) == 1 else f"all {len(corpus)} files"
    raise CorpusError(
        f"the {parser!r} parser read no text from {counted} in corpus "
        f"{corpus.name!r} ({listed}{more}), so none of them are in this index at all and "
        f"nothing can be retrieved. These files are {kinds}. Usually the parser does not "
        f"read the file types in this corpus -- check {parser!r} against them -- or the "
        "files have no text layer and need OCR first."
    )


def fingerprint_sources(corpus: Corpus) -> CorpusFingerprint:
    """Profile a corpus from its bytes alone. Instant, and enough to catch duplicates."""
    media_types: dict[str, int] = {}
    for source in corpus:
        key = source.media_type.value
        media_types[key] = media_types.get(key, 0) + 1

    by_hash: dict[str, list[str]] = {}
    for source in corpus:
        if source.raw is not None:
            by_hash.setdefault(source.content_hash(), []).append(source.id)

    duplicates = tuple(tuple(sorted(ids)) for ids in by_hash.values() if len(ids) > 1)

    return CorpusFingerprint(
        file_count=len(corpus),
        total_bytes=corpus.total_bytes,
        media_types=media_types,
        duplicate_groups=tuple(sorted(duplicates)),
    )


def fingerprint(
    corpus: Corpus, parses: Mapping[str, ParsedDocument] | Sequence[ParsedDocument] | None = None
) -> CorpusFingerprint:
    """Profile a corpus, using a parse for the content statistics when one is available."""
    base = fingerprint_sources(corpus)
    if not parses:
        return base

    documents = list(parses.values()) if isinstance(parses, Mapping) else list(parses)
    if not documents:
        return base

    lengths: list[int] = []
    kinds: dict[str, int] = {}
    table_characters = 0
    code_characters = 0
    headings = 0
    empty: list[str] = []

    for parsed in documents:
        lengths.append(parsed.document.length)
        if not parsed.text.strip():
            empty.append(parsed.id)
        for block in parsed.blocks:
            kinds[block.kind.value] = kinds.get(block.kind.value, 0) + 1
            if block.kind in {BlockKind.TABLE, BlockKind.TABLE_ROW}:
                table_characters += block.span.length
            elif block.kind is BlockKind.CODE:
                code_characters += block.span.length
            elif block.kind is BlockKind.HEADING:
                headings += 1

    return CorpusFingerprint(
        file_count=base.file_count,
        total_bytes=base.total_bytes,
        media_types=base.media_types,
        duplicate_groups=base.duplicate_groups,
        parser=documents[0].parser,
        total_characters=sum(lengths),
        document_lengths=tuple(lengths),
        block_kinds=kinds,
        table_characters=table_characters,
        code_characters=code_characters,
        heading_count=headings,
        empty_documents=tuple(empty),
    )

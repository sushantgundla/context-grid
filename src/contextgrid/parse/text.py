"""Parsers for born-digital text: plain text and Markdown.

These two need no dependencies, which makes them the reference implementations of the
`Parser` protocol and the arms the conformance suite runs against. They are also genuinely
useful: documentation corpora are the most common real input, and a heading-aware Markdown
parse is what makes structural chunking possible.

Both keep the decoded file text exactly as it is, so every block's text is a literal slice
of the document and `offsets_exact` is honestly true. Decoding those bytes is `parse.decode`'s
job, and it is the one place a BOM is dropped and a file that is not UTF-8 is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.core.documents import (
    Block,
    BlockKind,
    Document,
    MediaType,
    ParsedDocument,
    SourceFile,
)
from contextgrid.core.span import Span
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.parse.decode import decode_source

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-{2,})[ \t]*$")
_FENCE = re.compile(r"^([ \t]*)(```|~~~)(.*)$")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_QUOTE = re.compile(r"^[ \t]*>")
_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")


def _note_empty(source: SourceFile, text: str, warnings: WarningLog) -> None:
    """Say a readable file held no text -- but only when it really was readable.

    A file that failed to decode already carries a `PARSER_FALLBACK` warning explaining why.
    Adding `EMPTY_TEXT_LAYER` on top of it would name a second, wrong cause: an empty text
    layer is what a scanned PDF has, and it sends the user looking for OCR.
    """
    if text.strip() or warnings:
        return
    warnings.add(
        WarningCode.EMPTY_TEXT_LAYER,
        f"{source.id!r} contains no text",
        severity=Severity.CAUTION,
        stage="parse",
        subject=source.id,
    )


def _document(source: SourceFile, text: str) -> Document:
    return Document(id=source.id, text=text, source=source.path, meta=dict(source.meta))


@dataclass(frozen=True, slots=True)
class TextParser:
    """Plain text, split into paragraphs on blank lines.

    The simplest thing that can be a parser, and the baseline every other parse is compared
    against on text corpora.
    """

    name: ClassVar[str] = "text"
    version: ClassVar[str] = "1"

    def supports(self, media_type: MediaType) -> bool:
        return media_type in {MediaType.TEXT, MediaType.MARKDOWN, MediaType.UNKNOWN}

    def parse(self, source: SourceFile) -> ParsedDocument:
        text, warnings = decode_source(source)
        blocks: list[Block] = []

        for start, end in _paragraph_ranges(text):
            blocks.append(
                Block(
                    span=Span(source.id, start, end),
                    text=text[start:end],
                    kind=BlockKind.PARAGRAPH,
                )
            )

        _note_empty(source, text, warnings)

        return ParsedDocument(
            document=_document(source, text),
            blocks=tuple(blocks),
            parser=self.name,
            parser_version=self.version,
            offsets_exact=True,
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class MarkdownParser:
    """Markdown, with headings, code fences, lists, quotes and tables identified.

    Structure is the point. A chunk that knows it sits under "Termination > Notice period"
    retrieves far better than a bare paragraph, and a chunker that can see where a table
    starts will not cut it in half.
    """

    name: ClassVar[str] = "markdown"
    version: ClassVar[str] = "1"

    def supports(self, media_type: MediaType) -> bool:
        return media_type in {MediaType.MARKDOWN, MediaType.TEXT, MediaType.UNKNOWN}

    def parse(self, source: SourceFile) -> ParsedDocument:
        text, warnings = decode_source(source)
        blocks = [
            Block(span=Span(source.id, start, end), text=text[start:end], kind=kind, level=level)
            for start, end, kind, level in _markdown_regions(text)
        ]

        _note_empty(source, text, warnings)

        return ParsedDocument(
            document=_document(source, text),
            blocks=tuple(blocks),
            parser=self.name,
            parser_version=self.version,
            offsets_exact=True,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# region finding
# ---------------------------------------------------------------------------


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of blank-line-separated paragraphs, with surrounding space trimmed."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        trimmed = _trim(text, cursor, match.start())
        if trimmed is not None:
            ranges.append(trimmed)
        cursor = match.end()
    trimmed = _trim(text, cursor, len(text))
    if trimmed is not None:
        ranges.append(trimmed)
    return ranges


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink a range past leading and trailing whitespace. None when nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return None if start >= end else (start, end)


def _line_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of each line, excluding the newline itself."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    length = len(text)
    while cursor <= length:
        newline = text.find("\n", cursor)
        if newline == -1:
            ranges.append((cursor, length))
            break
        ranges.append((cursor, newline))
        cursor = newline + 1
    return ranges


def _markdown_regions(text: str) -> list[tuple[int, int, BlockKind, int | None]]:
    """Walk the document line by line, grouping lines into blocks.

    Line-based rather than regex-over-the-whole-text, because a fenced code block can
    contain anything at all -- including lines that look exactly like headings -- and only a
    stateful pass gets that right.
    """
    lines = _line_ranges(text)
    regions: list[tuple[int, int, BlockKind, int | None]] = []

    pending: list[tuple[int, int]] = []
    pending_kind = BlockKind.PARAGRAPH
    fence: str | None = None
    fence_start: int | None = None

    def flush() -> None:
        nonlocal pending, pending_kind
        if pending:
            trimmed = _trim(text, pending[0][0], pending[-1][1])
            if trimmed is not None:
                regions.append((trimmed[0], trimmed[1], pending_kind, None))
        pending = []
        pending_kind = BlockKind.PARAGRAPH

    for index, (start, end) in enumerate(lines):
        line = text[start:end]

        if fence is not None:
            if line.strip().startswith(fence):
                assert fence_start is not None
                regions.append((fence_start, end, BlockKind.CODE, None))
                fence = None
                fence_start = None
            continue

        fence_match = _FENCE.match(line)
        if fence_match:
            flush()
            fence = fence_match.group(2)
            fence_start = start
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            flush()
            regions.append((start, end, BlockKind.HEADING, len(heading.group(1))))
            continue

        # A setext heading underlines the paragraph above it, so it is only a heading if
        # there is exactly one line of text pending.
        if (
            _SETEXT_UNDERLINE.match(line)
            and len(pending) == 1
            and text[pending[0][0] : pending[0][1]].strip()
        ):
            level = 1 if line.strip().startswith("=") else 2
            regions.append((pending[0][0], end, BlockKind.HEADING, level))
            pending = []
            pending_kind = BlockKind.PARAGRAPH
            continue

        if not line.strip():
            flush()
            continue

        kind = _line_kind(line)
        if pending and kind is not pending_kind:
            flush()
        pending_kind = kind
        pending.append((start, end))

        if index == len(lines) - 1:
            flush()

    if fence is not None and fence_start is not None:
        # An unterminated fence runs to the end of the file. Real documents do this.
        regions.append((fence_start, len(text), BlockKind.CODE, None))
    flush()

    return sorted(regions, key=lambda region: region[0])


def _line_kind(line: str) -> BlockKind:
    if _TABLE_ROW.match(line):
        return BlockKind.TABLE
    if _LIST_ITEM.match(line):
        return BlockKind.LIST_ITEM
    if _QUOTE.match(line):
        return BlockKind.QUOTE
    return BlockKind.PARAGRAPH

"""Shared fixtures: sample documents and the plugin instances under conformance test."""

from __future__ import annotations

from dataclasses import dataclass

from contextgrid.chunk import (
    FixedTokenChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    StructuralChunker,
)
from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.protocols import Chunker, Parser
from contextgrid.parse import MarkdownParser, TextParser
from contextgrid.parse.pdfplumber import PDFPlumberParser
from contextgrid.parse.pymupdf import PyMuPDFParser
from tests.pdf_fixtures import contract_pdf, mixed_pdf, prose_pdf, scanned_pdf

# ---------------------------------------------------------------------------
# sample documents
# ---------------------------------------------------------------------------

CONTRACT = """\
# Master Services Agreement

## 1. Term

This agreement begins on the Effective Date and continues for twelve months.

## 2. Termination

### 2.1 Notice period

Either party may terminate this agreement for convenience by giving thirty days
written notice. Notice must be delivered to the address in Schedule A.

### 2.2 Termination for cause

A party may terminate immediately if the other party commits a material breach
and fails to remedy it within fifteen days of written notice.

## 3. Fees

| Service | Monthly fee | Setup fee |
|---|---|---|
| Standard | $1,200 | $500 |
| Premium | $3,400 | $500 |

Fees are payable within 30 days of invoice.
"""

API_DOCS = """\
# Widget API

The Widget API lets you create, read and delete widgets.

## Authentication

Send your key in the `X-Api-Key` header. Requests without a valid key return 401.

## Endpoints

### POST /widgets

Creates a widget. The request body must be JSON.

```python
import requests
requests.post("https://api.example.com/widgets", json={"name": "bolt"})
```

- `name` (string, required) is the widget name
- `colour` (string, optional) defaults to grey

### GET /widgets/{id}

Returns one widget. Returns 404 when the id is unknown.
"""

PLAIN_PROSE = """\
The notice period is thirty days. Either party may terminate for convenience.

Dr. Smith approved the change on 3.5 grounds, i.e. cost and timing. The board
agreed at the meeting on 12 March.

Payment falls due within 30 days of invoice. Late payment attracts interest at
2% per month.
"""

SHORT = "One sentence only."

WHITESPACE_ONLY = "   \n\n  \t \n"

EMPTY = ""


def source(name: str, text: str, media_type: MediaType = MediaType.MARKDOWN) -> SourceFile:
    return SourceFile(
        id=name,
        media_type=media_type,
        path=f"{name}.md",
        raw=text.encode("utf-8"),
    )


def pdf(name: str, data: bytes) -> SourceFile:
    return SourceFile(id=name, media_type=MediaType.PDF, path=f"{name}.pdf", raw=data)


SAMPLE_SOURCES: list[SourceFile] = [
    source("contract", CONTRACT),
    source("api-docs", API_DOCS),
    source("prose", PLAIN_PROSE, MediaType.TEXT),
    source("short", SHORT, MediaType.TEXT),
    source("whitespace", WHITESPACE_ONLY, MediaType.TEXT),
    source("empty", EMPTY, MediaType.TEXT),
]

#: The same content as a PDF, so the parser axis has something to disagree about.
PDF_SOURCES: list[SourceFile] = [
    pdf("contract-pdf", contract_pdf()),
    pdf("prose-pdf", prose_pdf()),
    pdf("scanned-pdf", scanned_pdf()),
    pdf("mixed-pdf", mixed_pdf()),
]

ALL_SOURCES: list[SourceFile] = [*SAMPLE_SOURCES, *PDF_SOURCES]

#: Documents with actual content, for tests that need something to chunk.
CONTENTFUL_SOURCES = [s for s in SAMPLE_SOURCES if s.id not in {"whitespace", "empty"}]


# ---------------------------------------------------------------------------
# plugins under conformance test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkerCase:
    """A chunker configuration, and whether it is allowed to leave text out.

    Most chunkers must cover every non-whitespace character of the document -- silently
    dropping text means evidence that can never be retrieved, whatever the retriever does.
    A sentence window with a stride wider than the window deliberately samples, so it is
    exempt and says so.
    """

    label: str
    chunker: Chunker
    covers_everything: bool = True

    def __str__(self) -> str:
        return self.label


ALL_PARSERS: list[Parser] = [
    TextParser(),
    MarkdownParser(),
    PyMuPDFParser(),
    PDFPlumberParser(),
]

ALL_CHUNKERS: list[ChunkerCase] = [
    ChunkerCase("fixed:64", FixedTokenChunker(size=64, overlap=8)),
    ChunkerCase("fixed:16/no-overlap", FixedTokenChunker(size=16, overlap=0)),
    ChunkerCase("recursive:64", RecursiveChunker(size=64, overlap=8)),
    ChunkerCase("recursive:32/no-overlap", RecursiveChunker(size=32, overlap=0)),
    ChunkerCase("sentence:2/1", SentenceWindowChunker(window=2, stride=1)),
    ChunkerCase("sentence:1/1", SentenceWindowChunker(window=1, stride=1)),
    ChunkerCase(
        "sentence:1/3 (samples)",
        SentenceWindowChunker(window=1, stride=3),
        covers_everything=False,
    ),
    ChunkerCase("structural:64", StructuralChunker(max_size=64, min_size=8)),
    ChunkerCase("structural:128/tables-whole", StructuralChunker(max_size=128, min_size=0)),
]

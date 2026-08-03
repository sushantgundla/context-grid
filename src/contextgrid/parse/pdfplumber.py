"""PDF extraction via pdfplumber.

Slower than PyMuPDF and it finds tables. That single difference is the clearest illustration
of why the parser belongs on the grid: on a corpus of prose the two are nearly
indistinguishable and pdfplumber is simply slower, while on a financial report one of them
returns the number you asked for and the other returns a soup of digits.

Tables are emitted as pipe-delimited rows and marked as tables, so a chunker can be told to
keep them whole.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from contextgrid.core.documents import BlockKind, MediaType, ParsedDocument, SourceFile
from contextgrid.core.errors import DocumentError, MissingExtraError
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.parse.builder import TextAssembler


def _pdfplumber() -> Any:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingExtraError("The pdfplumber parser", "parse", package="pdfplumber") from exc
    return pdfplumber


@dataclass(frozen=True, slots=True)
class PDFPlumberParser:
    """Extract text and tables, in reading order down the page.

    Elements are ordered by vertical position, so a table between two paragraphs comes out
    between them rather than appended at the end. Reading order is one of the things parsers
    most often get wrong, and getting it wrong quietly destroys retrieval on any document
    where the answer depends on what a sentence was next to.
    """

    extract_tables: bool = True
    #: How table cells are joined into text: "pipe", "tsv" or "plain".
    #:
    #: This is a real trade-off, not a formatting preference. Markdown pipes give an embedder
    #: an explicit signal about where one cell ends and the next begins, which usually helps
    #: retrieval. They also mean a row no longer reads as "Premium 3400 500", so ground truth
    #: quoting a table row verbatim will not match. Use "plain" when the eval set was
    #: authored against a reading of the table rather than against this parse.
    table_format: str = "pipe"

    name: ClassVar[str] = "pdfplumber"
    version: ClassVar[str] = "1"

    def supports(self, media_type: MediaType) -> bool:
        return media_type is MediaType.PDF

    def parse(self, source: SourceFile) -> ParsedDocument:
        pdfplumber = _pdfplumber()
        if source.raw is None:
            raise DocumentError(
                f"source file {source.id!r} has no bytes loaded. Read the file before parsing it."
            )

        import io

        started = time.perf_counter()
        warnings = WarningLog()
        assembler = TextAssembler(source.id)
        empty_pages: list[int] = []
        table_count = 0

        with pdfplumber.open(io.BytesIO(source.raw)) as document:
            page_count = len(document.pages)
            for index, page in enumerate(document.pages, start=1):
                elements, tables_here = _page_elements(page, self.extract_tables, self.table_format)
                table_count += tables_here
                if not elements:
                    empty_pages.append(index)
                for top, text, kind in elements:
                    del top
                    assembler.add(text, kind=kind, page=index)

        if empty_pages:
            warnings.add(
                WarningCode.EMPTY_TEXT_LAYER,
                f"{len(empty_pages)} of {page_count} pages in {source.id!r} have no text layer. "
                "They are probably scans, and nothing on them can be retrieved without OCR",
                severity=Severity.CAUTION,
                stage="parse",
                subject=source.id,
                pages=empty_pages[:20],
            )

        return assembler.build(
            parser=self.name,
            version=self.version,
            source=source.path,
            page_count=page_count,
            duration_ms=(time.perf_counter() - started) * 1000,
            warnings=warnings,
            meta={"tables_found": table_count, "table_format": self.table_format},
        )


def _page_elements(
    page: Any, extract_tables: bool, table_format: str
) -> tuple[list[tuple[float, str, BlockKind]], int]:
    """Text and tables on one page, ordered top to bottom."""
    elements: list[tuple[float, str, BlockKind]] = []
    table_boxes: list[tuple[float, float, float, float]] = []
    table_count = 0

    if extract_tables:
        for table in page.find_tables():
            rows = table.extract()
            rendered = _render_table(rows, table_format)
            if rendered:
                elements.append((table.bbox[1], rendered, BlockKind.TABLE))
                table_boxes.append(table.bbox)
                table_count += 1

    for line in _text_lines(page):
        top, text = line
        if any(box[1] <= top <= box[3] for box in table_boxes):
            continue  # already emitted as part of a table
        elements.append((top, text, BlockKind.PARAGRAPH))

    elements.sort(key=lambda element: element[0])
    return elements, table_count


def _text_lines(page: Any) -> list[tuple[float, str]]:
    """Lines of text with their vertical position, tolerating older pdfplumber versions."""
    extract_lines = getattr(page, "extract_text_lines", None)
    if callable(extract_lines):
        return [
            (float(line["top"]), str(line["text"]))
            for line in extract_lines()
            if str(line.get("text", "")).strip()
        ]

    text = page.extract_text() or ""  # pragma: no cover - only on very old pdfplumber
    return [(float(index), line) for index, line in enumerate(text.splitlines()) if line.strip()]


def _render_table(rows: list[list[str | None]], table_format: str) -> str:
    """Turn extracted cells into text an embedder can make sense of."""
    cleaned = [
        [" ".join((cell or "").split()) for cell in row]
        for row in rows
        if any((cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""

    separators = {"pipe": " | ", "tsv": "\t", "plain": " "}
    if table_format not in separators:
        raise DocumentError(
            f"unknown table_format {table_format!r}. Choose one of: {', '.join(sorted(separators))}"
        )

    if table_format == "pipe":
        return "\n".join("| " + " | ".join(row) + " |" for row in cleaned)
    return "\n".join(separators[table_format].join(row) for row in cleaned)

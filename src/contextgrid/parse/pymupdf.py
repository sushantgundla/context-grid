"""PDF extraction via PyMuPDF.

The speed baseline. Very fast, reliable on born-digital text, and it has no idea what a table
is -- a table comes out as a run of loose text in whatever order the content stream happened
to store it. That failure is the point: it is the arm every table-aware parser has to beat,
and the reason the parser belongs on the grid rather than in the setup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from contextgrid.core.documents import BlockKind, MediaType, ParsedDocument, SourceFile
from contextgrid.core.errors import DocumentError, MissingExtraError
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.parse.builder import TextAssembler, infer_heading_levels, round_size


def _pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingExtraError("The pymupdf parser", "parse", package="pymupdf") from exc
    return pymupdf


@dataclass(frozen=True, slots=True)
class PyMuPDFParser:
    """Extract text block by block, inferring headings from font size.

    `detect_headings` is what lets structural chunking work on a PDF at all. A PDF has no
    headings, only text that is larger than the rest, so the parser has to guess -- and two
    parsers guessing differently is a real effect on retrieval, not noise.
    """

    detect_headings: bool = True
    #: Drop text in the top and bottom band of each page, as a fraction of page height.
    #: Repeated page furniture is one of the quietest ways to poison dense retrieval.
    margin_ratio: float = 0.0

    name: ClassVar[str] = "pymupdf"
    version: ClassVar[str] = "1"

    def supports(self, media_type: MediaType) -> bool:
        return media_type is MediaType.PDF

    def parse(self, source: SourceFile) -> ParsedDocument:
        pymupdf = _pymupdf()
        if source.raw is None:
            raise DocumentError(
                f"source file {source.id!r} has no bytes loaded. Read the file before parsing it."
            )

        started = time.perf_counter()
        warnings = WarningLog()
        assembler = TextAssembler(source.id)

        with pymupdf.open(stream=source.raw, filetype="pdf") as document:
            lines = _collect_lines(document, self.margin_ratio)
            # Weighted by characters rather than lines: see infer_heading_levels.
            levels = (
                infer_heading_levels([(size, len(text)) for _, text, size in lines])
                if self.detect_headings
                else {}
            )

            for page_number, text, size in lines:
                level = levels.get(round_size(size))
                assembler.add(
                    text,
                    kind=BlockKind.HEADING if level else BlockKind.PARAGRAPH,
                    page=page_number,
                    level=level,
                )

            page_count = document.page_count
            empty_pages = _empty_pages(document, self.margin_ratio)

        if empty_pages:
            warnings.add(
                WarningCode.EMPTY_TEXT_LAYER,
                f"{len(empty_pages)} of {page_count} pages in {source.id!r} have no text layer "
                f"(pages {_summarise(empty_pages)}). They are probably scans, and nothing on "
                "them can be retrieved without OCR",
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
            meta={"detect_headings": self.detect_headings},
        )


def _collect_lines(document: Any, margin_ratio: float) -> list[tuple[int, str, float]]:
    """Every line of text, with its page and the font size it was set in."""
    collected: list[tuple[int, str, float]] = []
    for index, page in enumerate(document, start=1):
        height = page.rect.height
        top_limit = height * margin_ratio
        bottom_limit = height * (1 - margin_ratio)

        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 is text; 1 is an image
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans)
                if not text.strip():
                    continue
                if margin_ratio > 0:
                    top = line.get("bbox", (0, 0, 0, 0))[1]
                    if top < top_limit or top > bottom_limit:
                        continue
                size = max(float(span.get("size", 0.0)) for span in spans)
                collected.append((index, text, size))
    return collected


def _empty_pages(document: Any, margin_ratio: float) -> list[int]:
    pages: list[int] = []
    for index, page in enumerate(document, start=1):
        if not page.get_text("text").strip():
            pages.append(index)
    return pages


def _summarise(pages: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(page) for page in pages[:limit])
    return shown if len(pages) <= limit else f"{shown}, …"

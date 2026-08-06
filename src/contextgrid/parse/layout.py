"""Layout-model parsers: docling, marker, and pymupdf4llm.

The parser axis is the one nothing else in this field measures, and until now it had four arms
of which two were trivial. These are the three the field actually argues about.

* **docling** (IBM) runs layout and table-structure models over the page. It reads PDF, DOCX,
  PPTX, XLSX and HTML, and its table handling is the reason most people reach for it.
* **marker** runs Surya for layout and OCR across 90+ languages. Reported as the most faithful
  to document structure, and around two orders of magnitude slower than a text extractor --
  which is itself a finding worth having on a chart rather than in a blog post.
* **pymupdf4llm** is Markdown output from the PyMuPDF engine already installed here. It tests
  *output format* rather than extraction quality, which makes it the cheapest possible way to
  ask whether Markdown structure is worth anything to your retriever at all.

All three emit Markdown, and that matters more than it sounds. A table rendered as
`| Premium | 3400 |` no longer reads as `Premium 3400`, so ground truth quoted from a plain
reading of the page will not resolve against it. That is not a bug in the parser or in the
scorer -- it is the parser axis doing its job, and `GoldAnchor` exists precisely so evidence
can be re-resolved against each parse rather than pinned to one of them.

**Offsets stay exact.** Every block is a literal slice of the text this parser produced, and
the text is assembled from the blocks. What changes between parsers is *what the text is*, not
whether the offsets into it can be trusted.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from contextgrid.core.documents import BlockKind, MediaType, ParsedDocument, SourceFile
from contextgrid.core.errors import DocumentError, MissingExtraError
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.parse.builder import TextAssembler

#: A Markdown ATX heading: `## Section`.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
#: A Markdown table row, which is any line that starts and ends with a pipe.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _blocks_from_markdown(markdown: str) -> list[tuple[str, BlockKind, int | None]]:
    """Split Markdown into blocks, keeping tables whole.

    A table split across blocks is a table no chunker can be told to keep together, and on a
    financial document that single difference decides whether the answer is retrievable at all.
    Consecutive pipe rows are therefore gathered into one block, header separator included.

    Page markers written as `<!-- page: 3 -->` are read and dropped -- the comment is
    scaffolding, and leaving it in the text would put it inside chunks and inside embeddings.
    """
    blocks: list[tuple[str, BlockKind, int | None]] = []
    page: int | None = None
    table: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("\n".join(paragraph).strip(), BlockKind.PARAGRAPH, page))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("\n".join(table).strip(), BlockKind.TABLE, page))
            table.clear()

    for line in markdown.splitlines():
        marker = _PAGE_MARKER.match(line)
        if marker:
            flush_paragraph()
            flush_table()
            page = int(marker.group(1))
            continue

        if _TABLE_ROW.match(line):
            flush_paragraph()
            table.append(line)
            continue
        flush_table()

        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            blocks.append((heading.group(2).strip(), BlockKind.HEADING, page))
            # The level is carried separately by the assembler; see `_level_of`.
            continue

        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_table()
    return [(text, kind, page) for text, kind, page in blocks if text]


_PAGE_MARKER = re.compile(r"^\s*<!--\s*page[:\s]+(\d+)\s*-->\s*$", re.IGNORECASE)


def _level_of(line: str) -> int | None:
    match = _HEADING.match(line)
    return len(match.group(1)) if match else None


def _assemble(source: SourceFile, markdown: str) -> tuple[TextAssembler, int]:
    """Turn one Markdown document into blocks, counting the tables found."""
    assembler = TextAssembler(source.id)
    tables = 0

    levels = {
        match.group(2).strip(): len(match.group(1))
        for match in (_HEADING.match(line) for line in markdown.splitlines())
        if match
    }

    for text, kind, page in _blocks_from_markdown(markdown):
        if kind is BlockKind.TABLE:
            tables += 1
        assembler.add(
            text,
            kind=kind,
            page=page,
            level=levels.get(text) if kind is BlockKind.HEADING else None,
        )
    return assembler, tables


@dataclass(frozen=True, slots=True)
class _MarkdownParser:
    """Shared body: run the engine, split its Markdown, report what happened."""

    name: ClassVar[str] = "markdown-engine"
    version: ClassVar[str] = "1"
    #: What this engine will attempt. Overridden by the ones that read more than PDF.
    media_types: ClassVar[tuple[MediaType, ...]] = (MediaType.PDF,)

    def supports(self, media_type: MediaType) -> bool:
        return media_type in self.media_types

    def parse(self, source: SourceFile) -> ParsedDocument:
        if source.raw is None:
            raise DocumentError(
                f"source file {source.id!r} has no bytes loaded. Read the file before parsing it."
            )

        started = time.perf_counter()
        warnings = WarningLog()
        markdown, page_count, meta = self._to_markdown(source)

        assembler, tables = _assemble(source, markdown)
        if not len(assembler):
            warnings.add(
                WarningCode.EMPTY_TEXT_LAYER,
                f"{self.name} produced no text for {source.id!r}. On a scanned document that "
                "means OCR did not run or found nothing, and nothing on it can be retrieved",
                severity=Severity.CAUTION,
                stage="parse",
                subject=source.id,
            )

        return assembler.build(
            parser=self.name,
            version=self.version,
            source=source.path,
            page_count=page_count,
            duration_ms=(time.perf_counter() - started) * 1000,
            warnings=warnings,
            meta={"tables_found": tables, "output_format": "markdown", **meta},
        )

    def _to_markdown(self, source: SourceFile) -> tuple[str, int, dict[str, Any]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# docling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoclingParser(_MarkdownParser):
    """IBM's docling: layout and table-structure models over the page.

    The arm to reach for when the corpus has tables. It reads far more than PDF, which is why
    `media_types` is wide here -- a corpus of DOCX and PPTX has no other parser on this axis.

    `table_structure` and `ocr` are separate switches because they are separate costs. Turning
    OCR off on a corpus with a text layer is most of the speed back for none of the quality,
    and that trade is worth being able to sweep rather than assume.
    """

    table_structure: bool = True
    ocr: bool = False

    name: ClassVar[str] = "docling"
    version: ClassVar[str] = "1"
    media_types: ClassVar[tuple[MediaType, ...]] = (
        MediaType.PDF,
        MediaType.HTML,
        MediaType.DOCX,
        MediaType.MARKDOWN,
    )

    def _to_markdown(self, source: SourceFile) -> tuple[str, int, dict[str, Any]]:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError("The docling parser", "parse-ml", package="docling") from exc

        import io

        from docling_core.types.io import DocumentStream

        options = PdfPipelineOptions()
        options.do_table_structure = self.table_structure
        options.do_ocr = self.ocr

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        stream = DocumentStream(name=source.id, stream=io.BytesIO(source.raw or b""))

        try:
            result = converter.convert(stream)
        except Exception as error:
            raise DocumentError(f"docling could not read {source.id!r}: {error}") from error

        document = result.document
        pages = getattr(document, "pages", None)
        return (
            document.export_to_markdown(),
            len(pages) if pages else 0,
            {"table_structure": self.table_structure, "ocr": self.ocr},
        )


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkerParser(_MarkdownParser):
    """marker, with Surya for layout and OCR.

    The most faithful to structure of the three, and by a wide margin the slowest -- reports of
    roughly a hundred times a plain text extractor are not unusual. That is exactly the sort of
    thing this package exists to put on one chart next to the recall it bought, rather than
    leaving it as folklore.

    Model weights download on first use, which is why the failure message says so: a first run
    that appears to hang for several minutes is otherwise very hard to diagnose.
    """

    languages: str = "en"
    use_llm: bool = False

    name: ClassVar[str] = "marker"
    version: ClassVar[str] = "1"

    def _to_markdown(self, source: SourceFile) -> tuple[str, int, dict[str, Any]]:
        try:
            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError(
                "The marker parser", "parse-marker", package="marker-pdf"
            ) from exc

        import tempfile
        from pathlib import Path

        config = ConfigParser(
            {"output_format": "markdown", "languages": self.languages, "use_llm": self.use_llm}
        )

        # marker takes a path rather than bytes. Corpora here are held in memory, so the file
        # is written out and removed again.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{source.id}.pdf"
            path.write_bytes(source.raw or b"")
            try:
                converter = PdfConverter(
                    artifact_dict=create_model_dict(),
                    config=config.generate_config_dict(),
                    processor_list=config.get_processors(),
                    renderer=config.get_renderer(),
                )
                rendered = converter(str(path))
            except Exception as error:
                raise DocumentError(
                    f"marker could not read {source.id!r}: {error}. marker downloads its Surya "
                    "model weights on first use, so the first run needs network and takes "
                    "several minutes."
                ) from error

        markdown, _, _ = text_from_rendered(rendered)
        pages = getattr(getattr(rendered, "metadata", None), "get", lambda *_: None)("page_stats")
        return (
            markdown,
            len(pages) if isinstance(pages, list) else 0,
            {"languages": self.languages, "use_llm": self.use_llm},
        )


# ---------------------------------------------------------------------------
# pymupdf4llm
# ---------------------------------------------------------------------------


#: The worker that runs one pymupdf4llm conversion and exits.
#:
#: Kept as source rather than a module so it cannot accidentally import the rest of this
#: package -- the point is a process that has touched no other PDF.
_PYMUPDF4LLM_WORKER = """
import json, sys
import pymupdf, pymupdf4llm

raw = sys.stdin.buffer.read()
options = json.loads(sys.argv[1])

document = pymupdf.open(stream=raw, filetype="pdf")
try:
    pages = pymupdf4llm.to_markdown(
        document,
        page_chunks=options["page_chunks"],
        table_strategy=options["table_strategy"],
        show_progress=False,
    )
    page_count = document.page_count
finally:
    document.close()

if options["page_chunks"]:
    parts = []
    for index, chunk in enumerate(pages, start=1):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        parts.append("<!-- page: %d -->\\n%s" % (index, text))
    markdown = "\\n\\n".join(parts)
else:
    markdown = str(pages)

sys.stdout.write(json.dumps({"markdown": markdown, "page_count": page_count}))
"""


@dataclass(frozen=True, slots=True)
class PyMuPDF4LLMParser(_MarkdownParser):
    """Markdown from the PyMuPDF engine already installed here.

    The cheapest arm on this axis, and a genuinely useful control. It shares an extraction
    engine with `pymupdf`, so any difference between the two is *output format alone*:
    headings marked as headings, tables as pipe rows. That isolates a question the heavier
    parsers confound -- is Markdown structure worth anything to your retriever, separately from
    whether the extraction was any good?

    **Each document is parsed in its own process, and that is not optional.** pymupdf4llm's
    output for a document depends on which documents were converted before it in the same
    interpreter: state persists in MuPDF's C layer, below Python, so reloading the module and
    emptying MuPDF's store both fail to clear it. On this package's own fixtures the effect is
    not subtle -- a prose PDF that parses to 1182 characters alone parses to 919 mangled ones
    ("notce perod s trty") after a PDF with a table has gone through.

    For a tool whose entire foundation is the parse, a corpus that parses differently depending
    on file order is disqualifying. A process per document costs about a tenth of a second
    against the several tenths the conversion already takes, and buys back determinism.
    """

    page_chunks: bool = True
    table_strategy: str = "lines_strict"
    #: Escape hatch for anyone who has measured their own corpus and wants the speed. Off by
    #: default because the failure it re-enables is silent.
    isolate: bool = True

    name: ClassVar[str] = "pymupdf4llm"
    version: ClassVar[str] = "1"

    def _to_markdown(self, source: SourceFile) -> tuple[str, int, dict[str, Any]]:
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError(
                "The pymupdf4llm parser", "parse", package="pymupdf4llm"
            ) from exc

        options = {"page_chunks": self.page_chunks, "table_strategy": self.table_strategy}
        markdown, page_count = (
            self._in_subprocess(source, options)
            if self.isolate
            else self._in_process(source, options)
        )
        return (
            markdown,
            page_count,
            {"table_strategy": self.table_strategy, "isolated": self.isolate},
        )

    def _in_subprocess(self, source: SourceFile, options: dict[str, Any]) -> tuple[str, int]:
        import json
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", _PYMUPDF4LLM_WORKER, json.dumps(options)],
            input=source.raw or b"",
            capture_output=True,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
            raise DocumentError(
                f"pymupdf4llm could not read {source.id!r}: "
                f"{detail[-1] if detail else 'the worker exited without a message'}"
            )

        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DocumentError(
                f"pymupdf4llm returned something unreadable for {source.id!r}: {error}"
            ) from error
        return str(payload["markdown"]), int(payload["page_count"])

    def _in_process(self, source: SourceFile, options: dict[str, Any]) -> tuple[str, int]:
        """Only reachable with `isolate=False`. See the class docstring for what it costs."""
        import pymupdf
        import pymupdf4llm

        try:
            # pymupdf ships no type information for these, and the adapter around them is
            # strict -- so the calls are cast at the boundary rather than typed through it.
            opener: Any = pymupdf.open
            document: Any = opener(stream=source.raw, filetype="pdf")
        except Exception as error:
            raise DocumentError(f"pymupdf4llm could not open {source.id!r}: {error}") from error

        try:
            pages = pymupdf4llm.to_markdown(document, show_progress=False, **options)
            page_count = int(document.page_count)
        finally:
            document.close()

        if not options["page_chunks"]:
            return str(pages), page_count

        parts = []
        for index, chunk in enumerate(pages, start=1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            parts.append(f"<!-- page: {index} -->\n{text}")
        return "\n\n".join(parts), page_count


# ---------------------------------------------------------------------------
# agno
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgnoParser(_MarkdownParser):
    """Text extraction through agno's readers.

    On this axis rather than the ingestion one, because turning bytes into text is what a
    parser does -- and what most RAG stacks reach for without noticing they have made a parser
    choice at all. Putting it beside pymupdf, pdfplumber and docling prices that convenience
    against engines built for the job.

    `reader` picks one of agno's readers by name; `auto` chooses by file extension. Its PDF
    reader needs pypdf, and says so rather than failing obscurely.
    """

    reader: str = "auto"

    name: ClassVar[str] = "agno"
    version: ClassVar[str] = "1"
    media_types: ClassVar[tuple[MediaType, ...]] = (
        MediaType.PDF,
        MediaType.MARKDOWN,
        MediaType.HTML,
        MediaType.DOCX,
        MediaType.TEXT,
    )

    def _to_markdown(self, source: SourceFile) -> tuple[str, int, dict[str, Any]]:
        try:
            from agno.knowledge.reader.reader_factory import ReaderFactory
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingExtraError("The agno parser", "agent", package="agno") from exc

        reader = self._reader(ReaderFactory, source)
        if reader is None:
            raise DocumentError(
                f"agno has no reader for {source.id!r}. Its PDF reader needs pypdf: "
                "pip install 'context-grid[agent]'"
            )

        import io

        stream = io.BytesIO(source.raw or b"")
        stream.name = source.id

        try:
            documents = reader.read(stream, name=source.id)
        except TypeError:
            documents = reader.read(stream)
        except Exception as error:
            raise DocumentError(f"the agno reader failed on {source.id!r}: {error}") from error

        text = "\n\n".join(
            str(getattr(document, "content", "")) for document in documents or []
        ).strip()
        return text, 0, {"reader": self.reader}

    def _reader(self, factory: Any, source: SourceFile) -> Any:
        if self.reader != "auto":
            try:
                return factory.create_reader(self.reader, chunk=False)
            except Exception as error:
                raise DocumentError(
                    f"agno has no reader called {self.reader!r}. Available: "
                    f"{', '.join(sorted(factory.get_all_reader_keys()))}"
                ) from error

        suffix = {
            MediaType.PDF: ".pdf",
            MediaType.MARKDOWN: ".md",
            MediaType.HTML: ".html",
            MediaType.DOCX: ".docx",
        }.get(source.media_type, ".txt")
        try:
            reader = factory.get_reader_for_extension(suffix)
        except Exception:
            return None
        if reader is not None:
            # Chunking is its own axis. Letting the reader do it would quietly take that
            # decision away from the thing that measures it.
            reader.chunk = False
        return reader

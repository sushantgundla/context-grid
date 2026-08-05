"""The two ingestion strategies, and the honest difference between them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.ingest.base import IngestionError


@dataclass(frozen=True, slots=True)
class DirectIngestion:
    """Hand the bytes on unchanged. The default, and the arm the other one is judged against.

    Every decision about how those bytes become text is left to the parser axis, which is where
    this package can actually measure them. Nothing is lost and nothing is assumed.
    """

    name: ClassVar[str] = "direct"
    version: ClassVar[str] = "1"
    replaces_parser: ClassVar[bool] = False

    def ingest(self, sources: Sequence[SourceFile], log: WarningLog) -> list[SourceFile]:
        del log
        return list(sources)


@dataclass(frozen=True, slots=True)
class AgnoIngestion:
    """Let an agno reader extract the text, and treat that as the document.

    This is what most RAG stacks actually do: a loader returns strings and everything
    downstream works on those. Putting it on an axis next to `direct` prices the convenience --
    an agno reader has already decided table handling, reading order and whether a heading
    survives as one, and those are exactly the decisions the parser axis exists to measure.

    The text comes back as Markdown, so `parser: markdown` reads it. Pairing this with a PDF
    engine is meaningless and the matrix drops the combination rather than running one.
    """

    #: Which agno reader to use. `auto` picks by file extension, which is what agno itself does.
    reader: str = "auto"
    #: agno readers chunk by default. Off here -- chunking is its own axis, and letting the
    #: reader do it would silently take that decision away from the thing measuring it.
    chunk: bool = False

    name: ClassVar[str] = "agno"
    version: ClassVar[str] = "1"
    replaces_parser: ClassVar[bool] = True

    def ingest(self, sources: Sequence[SourceFile], log: WarningLog) -> list[SourceFile]:
        factory = _reader_factory()
        ingested: list[SourceFile] = []

        for source in sources:
            reader = self._reader_for(factory, source)
            if reader is None:
                log.add(
                    WarningCode.NO_PARSE_FOR_SOURCE,
                    f"agno has no reader for {source.id!r}, so it was left as bytes for the "
                    "parser axis to handle. Its row is a mix of two ingestion strategies",
                    severity=Severity.CAUTION,
                    stage="ingest",
                    subject=source.id,
                )
                ingested.append(source)
                continue

            text = _read_text(reader, source)
            if not text.strip():
                log.add(
                    WarningCode.EMPTY_TEXT_LAYER,
                    f"the agno reader returned no text for {source.id!r}. Nothing in that "
                    "document can be retrieved under this ingestion strategy",
                    severity=Severity.CAUTION,
                    stage="ingest",
                    subject=source.id,
                )

            # The id follows the source file. Gold evidence written against `refunds.pdf` has
            # to resolve whichever strategy produced the text, or changing ingestion would look
            # like changing corpus and the axis could not be measured at all.
            ingested.append(
                SourceFile(
                    id=source.id,
                    raw=text.encode("utf-8"),
                    media_type=MediaType.MARKDOWN,
                    path=source.path,
                )
            )
        return ingested

    def _reader_for(self, factory: Any, source: SourceFile) -> Any:
        if self.reader != "auto":
            try:
                return factory.create_reader(self.reader, chunk=self.chunk)
            except Exception as error:
                raise IngestionError(
                    f"agno has no reader called {self.reader!r}. Available: "
                    f"{', '.join(sorted(factory.get_all_reader_keys()))}"
                ) from error

        suffix = Path(source.path).suffix.lower() if source.path else _suffix_for(source.media_type)
        try:
            reader = factory.get_reader_for_extension(suffix)
        except Exception:
            return None
        if reader is not None:
            reader.chunk = self.chunk
        return reader


def _suffix_for(media_type: MediaType) -> str:
    return {
        MediaType.PDF: ".pdf",
        MediaType.MARKDOWN: ".md",
        MediaType.HTML: ".html",
        MediaType.DOCX: ".docx",
        MediaType.TEXT: ".txt",
    }.get(media_type, ".txt")


def _read_text(reader: Any, source: SourceFile) -> str:
    """Run one agno reader over one source file's bytes.

    agno readers take a path or a file object depending on which reader it is, so the bytes go
    in as a `BytesIO` with a name attached -- which is what the readers that care about the
    filename look at.
    """
    import io

    stream = io.BytesIO(source.raw or b"")
    name = Path(source.path).name if source.path else source.id
    stream.name = name  # BytesIO takes it; the readers that care look at it

    try:
        documents = reader.read(stream, name=source.id)
    except TypeError:
        # Older readers take the object alone.
        documents = reader.read(stream)
    except Exception as error:
        raise IngestionError(f"the agno reader failed on {source.id!r}: {error}") from error

    return "\n\n".join(
        str(getattr(document, "content", "")) for document in documents or []
    ).strip()


def _reader_factory() -> Any:
    try:
        from agno.knowledge.reader.reader_factory import ReaderFactory
    except ImportError as error:  # pragma: no cover - exercised by the extras test
        raise IngestionError(
            "agno ingestion needs agno. Install it with: pip install 'context-grid[agent]'"
        ) from error
    return ReaderFactory

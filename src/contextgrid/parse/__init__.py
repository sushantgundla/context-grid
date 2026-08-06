"""Parsers, and the registry of them.

`unstructured` used to be registered here and was never written. Worse than useless: asking for
it raised a MissingExtraError naming the `parse-ml` extra, which does not contain it and would
not have helped -- an instruction that costs somebody a heavy install and leaves them exactly
where they started. Advertising a plugin that cannot run is the same failure as a config key
nobody reads.

The parser defines the text every character offset downstream refers to, which is why it is
an axis on the grid rather than a setting. Two parsers over the same PDF produce different
text, different chunks and different retrieval -- and nothing else in the field measures that
end to end.

Zero-dependency parsers are registered eagerly. Everything needing a PDF engine or a layout
model is registered lazily, so `import contextgrid` stays cheap and a missing dependency
produces an install instruction rather than a traceback.
"""

from __future__ import annotations

from contextgrid.core.protocols import Parser
from contextgrid.core.registry import Registry
from contextgrid.parse.text import MarkdownParser, TextParser

PARSERS: Registry[Parser] = Registry(family="parser")

PARSERS.register("text", doc="Plain text, split into paragraphs. No dependencies.")(TextParser)
PARSERS.register(
    "markdown", doc="Markdown with headings, code, lists and tables. No dependencies."
)(MarkdownParser)

# Registered before they are written. Asking for one today raises MissingExtraError naming
# the extra to install, which is the honest answer until the module lands in M2.
PARSERS.register_lazy(
    "pymupdf",
    module="contextgrid.parse.pymupdf",
    attr="PyMuPDFParser",
    extra="parse",
    package="pymupdf",
    doc="Fast PDF text extraction. The speed baseline.",
)
PARSERS.register_lazy(
    "pdfplumber",
    module="contextgrid.parse.pdfplumber",
    attr="PDFPlumberParser",
    extra="parse",
    package="pdfplumber",
    doc="Table-aware PDF extraction.",
)
PARSERS.register_lazy(
    "docling",
    module="contextgrid.parse.layout",
    attr="DoclingParser",
    extra="parse-ml",
    package="docling",
    doc="Layout and table-structure models. PDF, DOCX, PPTX, HTML. The table arm.",
)
PARSERS.register_lazy(
    "marker",
    module="contextgrid.parse.layout",
    attr="MarkerParser",
    extra="parse-marker",
    package="marker-pdf",
    shorthand="languages",
    doc="Surya layout and OCR, 90+ languages. The most faithful and by far the slowest.",
)
# Same extraction engine as `pymupdf`, different output. Any difference between the two is
# Markdown structure alone, which is the one question the heavy parsers confound.
PARSERS.register_lazy(
    "agno",
    module="contextgrid.parse.layout",
    attr="AgnoParser",
    extra="agent",
    package="agno",
    shorthand="reader",
    doc="Text through agno's readers. What most RAG stacks use without choosing to.",
)
PARSERS.register_lazy(
    "pymupdf4llm",
    module="contextgrid.parse.layout",
    attr="PyMuPDF4LLMParser",
    extra="parse",
    package="pymupdf4llm",
    doc="Markdown from the PyMuPDF engine. Tests output format, not extraction.",
)


def get_parser(spec: str | Parser) -> Parser:
    """Resolve a parser from a spec string, or pass an instance through."""
    return PARSERS.create(spec) if isinstance(spec, str) else spec


__all__ = ["PARSERS", "MarkdownParser", "TextParser", "get_parser"]

"""Assembling a parsed document so that its offsets cannot be wrong.

For born-digital text the document text already exists and a parser points into it. For a
PDF there is no text until the parser makes some, which is where offsets usually start to
drift: a wrapper extracts text one way, records positions another way, and the two disagree
by a character here and a newline there.

The fix is to make it structurally impossible. The assembler is the only thing that appends
text, and it records each block's span at the moment it appends it. `document.text[span]` is
then equal to `block.text` by construction rather than by careful bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from contextgrid.core.documents import Block, BlockKind, Document, ParsedDocument
from contextgrid.core.span import Span
from contextgrid.core.warnings import WarningLog


@dataclass(slots=True)
class TextAssembler:
    """Builds document text and blocks together, keeping them in step.

    Blocks are separated by `separator` in the assembled text. That whitespace belongs to no
    block, which is correct: it is layout the parser invented, not content it found.
    """

    doc_id: str
    separator: str = "\n\n"
    _parts: list[str] = field(default_factory=list, repr=False)
    _blocks: list[Block] = field(default_factory=list, repr=False)
    _length: int = 0

    def add(
        self,
        text: str,
        *,
        kind: BlockKind = BlockKind.PARAGRAPH,
        page: int | None = None,
        level: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Block | None:
        """Append a block of text and record where it landed.

        Blank text is dropped rather than recorded, because an empty block is not evidence
        of anything and would only clutter the structural view.
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        if self._parts:
            self._parts.append(self.separator)
            self._length += len(self.separator)

        start = self._length
        self._parts.append(cleaned)
        self._length += len(cleaned)

        block = Block(
            span=Span(self.doc_id, start, self._length),
            text=cleaned,
            kind=kind,
            page=page,
            level=level,
            meta=meta or {},
        )
        self._blocks.append(block)
        return block

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def build(
        self,
        *,
        parser: str,
        version: str,
        source: str | None = None,
        page_count: int | None = None,
        duration_ms: float | None = None,
        warnings: WarningLog | None = None,
        offsets_exact: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        return ParsedDocument(
            document=Document(id=self.doc_id, text=self.text, source=source),
            blocks=self.blocks,
            parser=parser,
            parser_version=version,
            offsets_exact=offsets_exact,
            page_count=page_count,
            duration_ms=duration_ms,
            warnings=warnings or WarningLog(),
            meta=meta or {},
        )


# ---------------------------------------------------------------------------
# heading detection by font size
# ---------------------------------------------------------------------------


def infer_heading_levels(
    weighted_sizes: Sequence[tuple[float, int]],
    *,
    tolerance: float = 0.5,
    min_ratio: float = 1.15,
) -> dict[float, int]:
    """Map font sizes to heading levels, given every line's size and how much text it holds.

    A PDF has no headings, only text that happens to be larger. The usual heuristic --
    anything bigger than the body size is a heading, ranked by how much bigger -- is what
    makes structural chunking possible on PDFs at all.

    Sizes are weighted by **characters, not lines**. Weighting by lines looks equivalent and
    is not: a bordered table contributes a dozen two-word cells set slightly smaller than the
    body, which outvotes the prose and drags the inferred body size down. Everything above it
    is then promoted to a heading, including the actual body text. Characters are the honest
    measure of how much of a document is set in a given size.

    `min_ratio` stops 11.5pt being called a heading in an 11pt document. Emphasis is not
    structure.

    It remains a heuristic, and where two parsers disagree about it they produce genuinely
    different structural chunks. That disagreement is a real effect worth measuring, not
    noise to be smoothed away.
    """
    if not weighted_sizes:
        return {}

    weights: dict[float, int] = {}
    for size, weight in weighted_sizes:
        key = round_size(size, tolerance)
        weights[key] = weights.get(key, 0) + max(weight, 1)

    body = max(weights, key=lambda size: (weights[size], size))
    larger = sorted({size for size in weights if size >= body * min_ratio}, reverse=True)
    return {size: level for level, size in enumerate(larger, start=1)}


def round_size(size: float, tolerance: float = 0.5) -> float:
    """Quantise a font size so 11.0 and 11.04 are treated as the same size."""
    return round(size / tolerance) * tolerance

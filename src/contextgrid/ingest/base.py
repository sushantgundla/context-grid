"""Ingestion strategies: how a file becomes something the parser axis can read.

The stage before everything else, and the one with the least written about it. A PDF can reach
a retrieval system as raw bytes for a PDF engine to extract, or as text some reader already
pulled out. Those are different documents by the time anything downstream sees them, and which
one you get is usually decided by whichever loader was easiest to import.

Two strategies to begin with, and they answer a genuinely open question:

* `direct` -- read the bytes and hand them on. The parser axis then decides how they become
  text, which is where this package already does its most distinctive work.
* `agno` -- let an agno reader extract the text, and treat that as the document.

The comparison matters because the second one *skips the parser axis*. An agno reader that
returns text has already made every decision the parser axis exists to measure -- table
handling, reading order, whether a heading survives as a heading. Running both tells you what
that convenience cost, on your corpus, in recall.

**Ingestion never changes what a document is called.** The id follows the source file, so gold
evidence written against `refunds.pdf` resolves whichever strategy produced the text. Anything
else would make the axis unmeasurable, since a change of ingestion would look like a change of
corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contextgrid.core.documents import SourceFile
from contextgrid.core.errors import ContextGridError
from contextgrid.core.warnings import WarningLog


class IngestionError(ContextGridError, ValueError):
    """A source could not be ingested."""


@runtime_checkable
class IngestionStrategy(Protocol):
    """Turns the files found on disk into the source files a parser will read."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def replaces_parser(self) -> bool:
        """True when this strategy extracts text itself.

        A strategy that returns text has already decided everything the parser axis measures,
        so pairing it with a PDF engine is meaningless -- and the runner drops those
        combinations rather than running the engine against text it cannot parse.
        """
        ...

    def ingest(self, sources: Sequence[SourceFile], log: WarningLog) -> list[SourceFile]:
        """Produce the source files to parse. Ids must be preserved."""
        ...

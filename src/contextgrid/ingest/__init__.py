"""Ingestion strategies, and the registry of them.

The stage before everything else: how a file on disk becomes something a parser can read.
"""

from __future__ import annotations

from contextgrid.core.registry import Registry
from contextgrid.ingest.base import IngestionError, IngestionStrategy
from contextgrid.ingest.strategies import AgnoIngestion, DirectIngestion

INGESTERS: Registry[IngestionStrategy] = Registry(family="ingestion")

INGESTERS.register(
    "direct", doc="Hand the bytes to the parser axis. The default, and the honest baseline."
)(DirectIngestion)

# Registered eagerly: the module imports without agno, and asking for it without agno raises
# an IngestionError naming the extra rather than an ImportError from four frames down.
INGESTERS.register(
    "agno",
    shorthand="reader",
    doc="Let an agno reader extract the text. Skips the parser axis, which is the point.",
)(AgnoIngestion)


def get_ingester(spec: str | IngestionStrategy | None) -> IngestionStrategy:
    """Resolve a strategy from a spec, or pass one through. `None` means hand bytes on."""
    if spec is None:
        return DirectIngestion()
    return INGESTERS.create(spec) if isinstance(spec, str) else spec


__all__ = [
    "INGESTERS",
    "AgnoIngestion",
    "DirectIngestion",
    "IngestionError",
    "IngestionStrategy",
    "get_ingester",
]

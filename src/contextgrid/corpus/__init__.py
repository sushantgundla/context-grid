"""Loading and profiling a corpus."""

from __future__ import annotations

from contextgrid.corpus.fingerprint import (
    CorpusFingerprint,
    fingerprint,
    fingerprint_sources,
)
from contextgrid.corpus.loader import DEFAULT_PATTERNS, Corpus, CorpusError

__all__ = [
    "DEFAULT_PATTERNS",
    "Corpus",
    "CorpusError",
    "CorpusFingerprint",
    "fingerprint",
    "fingerprint_sources",
]

"""Content-addressed caching, and the prefix reuse that makes a sweep affordable."""

from __future__ import annotations

from contextgrid.cache.store import (
    CACHE_FORMAT,
    Cache,
    CacheStats,
    DiskCache,
    MemoryCache,
    NullCache,
    cache_key,
    cached,
)

__all__ = [
    "CACHE_FORMAT",
    "Cache",
    "CacheStats",
    "DiskCache",
    "MemoryCache",
    "NullCache",
    "cache_key",
    "cached",
]

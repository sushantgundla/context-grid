"""Unit tests for content-addressed caching.

The failure that matters is not a missed hit -- that only costs time. It is a hit on the
*wrong* entry, which costs correctness and produces a completely believable number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.cache import (
    CacheStats,
    DiskCache,
    MemoryCache,
    NullCache,
    cache_key,
    cached,
)

# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def test_the_same_work_gets_the_same_key() -> None:
    assert cache_key("chunk", "1", {"size": 512}, ["abc"]) == cache_key(
        "chunk", "1", {"size": 512}, ["abc"]
    )


def test_parameter_order_does_not_change_the_key() -> None:
    """Specs come from YAML and dicts, where key order is not meaningful."""
    assert cache_key("chunk", "1", {"size": 512, "overlap": 64}) == cache_key(
        "chunk", "1", {"overlap": 64, "size": 512}
    )


def test_input_order_does_not_change_the_key() -> None:
    assert cache_key("embed", "1", {}, ["a", "b"]) == cache_key("embed", "1", {}, ["b", "a"])


@pytest.mark.parametrize(
    ("stage", "version", "params", "inputs"),
    [
        ("parse", "1", {"size": 512}, ["abc"]),
        ("chunk", "2", {"size": 512}, ["abc"]),
        ("chunk", "1", {"size": 256}, ["abc"]),
        ("chunk", "1", {"size": 512}, ["xyz"]),
        ("chunk", "1", {"size": 512, "overlap": 0}, ["abc"]),
    ],
)
def test_any_difference_changes_the_key(
    stage: str, version: str, params: dict[str, int], inputs: list[str]
) -> None:
    baseline = cache_key("chunk", "1", {"size": 512}, ["abc"])
    assert cache_key(stage, version, params, inputs) != baseline


def test_the_tokenizer_must_change_the_chunk_key() -> None:
    """The collision this design exists to prevent.

    Two embedders with different tokenizers asking for "512-token chunks" want genuinely
    different chunk sets. A key without the tokenizer in it serves one of them the other's
    chunks -- silently, with entirely believable results.
    """
    with_regex = cache_key("chunk", "1", {"size": 512, "tokenizer": "regex"}, ["h"])
    with_bpe = cache_key("chunk", "1", {"size": 512, "tokenizer": "cl100k_base"}, ["h"])
    assert with_regex != with_bpe


def test_a_key_is_a_hex_digest() -> None:
    key = cache_key("parse", "1")
    assert len(key) == 64
    assert all(character in "0123456789abcdef" for character in key)


def test_unhashable_parameters_do_not_crash_the_key() -> None:
    """Plugins hold tokenizer instances and other objects. A key must still be derivable."""

    class Opaque:
        def __repr__(self) -> str:
            return "Opaque(1)"

    assert cache_key("chunk", "1", {"tokenizer": Opaque()})


# ---------------------------------------------------------------------------
# memory cache
# ---------------------------------------------------------------------------


def test_memory_cache_round_trips() -> None:
    cache = MemoryCache()
    cache.put("k", [1, 2, 3])
    assert cache.get("k") == [1, 2, 3]
    assert "k" in cache
    assert len(cache) == 1


def test_a_miss_returns_none() -> None:
    assert MemoryCache().get("absent") is None


def test_clearing_empties_it() -> None:
    cache = MemoryCache()
    cache.put("k", 1)
    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# disk cache
# ---------------------------------------------------------------------------


def test_disk_cache_survives_a_new_instance(tmp_path: Path) -> None:
    DiskCache(tmp_path).put("k", {"value": 42})
    assert DiskCache(tmp_path).get("k") == {"value": 42}


def test_disk_cache_fans_out_into_subdirectories(tmp_path: Path) -> None:
    """A flat directory of fifty thousand files is slow on every filesystem."""
    cache = DiskCache(tmp_path)
    cache.put("abcdef" + "0" * 58, 1)
    assert (tmp_path / "ab" / "cd").is_dir()


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    """Recomputing is cheap. Crashing on somebody's stale cache directory is not."""
    cache = DiskCache(tmp_path)
    key = "a" * 64
    cache.put(key, {"value": 1})
    path = tmp_path / "aa" / "aa" / f"{key}.pkl"
    path.write_bytes(b"not a pickle")
    assert cache.get(key) is None
    assert not path.exists()  # and the bad entry is cleared out of the way


def test_disk_cache_counts_and_clears(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("a" * 64, 1)
    cache.put("b" * 64, 2)
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# the cached() helper
# ---------------------------------------------------------------------------


def test_a_hit_never_pays_for_the_work_it_avoided() -> None:
    cache = MemoryCache()
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        return "value"

    assert cached(cache, "k", "chunk", compute) == "value"
    assert cached(cache, "k", "chunk", compute) == "value"
    assert calls == 1


def test_hits_and_misses_are_recorded_per_stage() -> None:
    cache = MemoryCache()
    stats = CacheStats()

    cached(cache, "k1", "parse", lambda: 1, stats)
    cached(cache, "k1", "parse", lambda: 1, stats)
    cached(cache, "k2", "embed", lambda: 2, stats)

    assert stats.hits == 1
    assert stats.misses == 2
    assert stats.by_stage == {"parse": [1, 1], "embed": [0, 1]}
    assert stats.hit_rate == pytest.approx(1 / 3)


def test_no_cache_still_computes_and_counts() -> None:
    stats = CacheStats()
    assert cached(None, "k", "parse", lambda: 7, stats) == 7
    assert stats.misses == 1


def test_the_null_cache_never_reuses_anything() -> None:
    """For measuring what a stage really costs, with prefix reuse switched off."""
    cache = NullCache()
    calls = 0

    def compute() -> int:
        nonlocal calls
        calls += 1
        return calls

    cached(cache, "k", "parse", compute)
    cached(cache, "k", "parse", compute)
    assert calls == 2
    assert "k" not in cache


def test_stats_summary_reads_like_a_sentence() -> None:
    stats = CacheStats()
    stats.record("parse", hit=True)
    stats.record("parse", hit=False)
    assert "1 of 2 lookups reused" in stats.summary()
    assert CacheStats().summary() == "no cache lookups"


def test_stats_merge() -> None:
    left, right = CacheStats(), CacheStats()
    left.record("parse", hit=True)
    right.record("parse", hit=False)
    right.record("embed", hit=True)
    left.merge(right)
    assert left.hits == 2
    assert left.by_stage == {"parse": [1, 1], "embed": [1, 0]}

"""Content-addressed caching, and the prefix reuse that makes a sweep affordable.

A grid of 48 configurations does not mean 48 parses and 48 embedding runs. Configurations
sharing a parser share its parse; those sharing parser, chunker and embedder share the
embeddings. Sweeping rerankers across twenty configurations should embed exactly once.

Keys are hashes of `(stage, stage version, parameters, input hashes)`. Two properties fall
out of that and both matter: the same work is never done twice, and *different* work never
collides -- which is the failure that would matter, because a cache hit on the wrong entry
produces a plausible number rather than an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Bumped when the pickled shape of a cached value changes. Old entries then miss rather
#: than unpickling into something with the wrong fields.
CACHE_FORMAT = 1


def cache_key(
    stage: str,
    version: str,
    params: Mapping[str, Any] | None = None,
    inputs: Iterable[str] = (),
) -> str:
    """A stable key for one stage's output.

    `params` is canonicalised, so `{"size": 512, "overlap": 64}` and the same dict written
    the other way round produce one key. `inputs` are the content hashes this stage consumed,
    which is what chains the cache together: a chunk set's key contains the parse's hash, so
    changing the parser invalidates everything downstream of it automatically.

    **The tokenizer must be in `params` for any stage that measures size in tokens.** Two
    embedders with different tokenizers asking for "512-token chunks" want genuinely
    different chunk sets, and a key that omits the tokenizer serves one of them the other's
    chunks -- silently, and with entirely believable results.
    """
    payload = {
        "format": CACHE_FORMAT,
        "stage": stage,
        "version": version,
        "params": _canonical(params or {}),
        "inputs": sorted(inputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Reduce parameters to something JSON can hash consistently."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(slots=True)
class CacheStats:
    """Hits, misses and what they saved.

    Reported after a run, because prefix reuse is the engineering decision that makes the
    whole tool affordable and it is completely invisible unless something says so.
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0
    by_stage: dict[str, list[int]] = field(default_factory=dict)

    def record(self, stage: str, *, hit: bool) -> None:
        counts = self.by_stage.setdefault(stage, [0, 0])
        if hit:
            self.hits += 1
            counts[0] += 1
        else:
            self.misses += 1
            counts[1] += 1

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def summary(self) -> str:
        if not self.lookups:
            return "no cache lookups"
        parts = [f"{self.hits} of {self.lookups} lookups reused ({self.hit_rate:.0%})"]
        for stage, (hits, misses) in sorted(self.by_stage.items()):
            parts.append(f"{stage} {hits}/{hits + misses}")
        return ", ".join(parts)

    def merge(self, other: CacheStats) -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.writes += other.writes
        for stage, (hits, misses) in other.by_stage.items():
            counts = self.by_stage.setdefault(stage, [0, 0])
            counts[0] += hits
            counts[1] += misses


@runtime_checkable
class Cache(Protocol):
    """Somewhere to put a stage's output and find it again."""

    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any) -> None: ...

    def __contains__(self, key: str) -> bool: ...


@dataclass(slots=True)
class MemoryCache:
    """In-process cache. The default, and enough for a single sweep."""

    _entries: dict[str, Any] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def get(self, key: str) -> Any | None:
        return self._entries.get(key)

    def put(self, key: str, value: Any) -> None:
        self._entries[key] = value
        self.stats.writes += 1

    def clear(self) -> None:
        self._entries.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(slots=True)
class DiskCache:
    """Cache that survives the process, for re-running a sweep after changing one axis.

    Values are pickled. An entry that cannot be read back -- a partial write, a change to a
    dataclass between versions -- is treated as a miss and deleted, never as a hard failure.
    Recomputing is cheap; crashing on someone's stale cache directory is not.
    """

    root: Path
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Two levels of fan-out: a flat directory of 50,000 files is slow on every filesystem.
        return self.root / key[:2] / key[2:4] / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and move, so a crash mid-write cannot leave a truncated
        # file that later reads as a valid cache hit.
        #
        # The temporary name is unique per writer, which is the part that was missing. Two
        # `contextgrid run` processes over one output directory with a cold cache both wrote
        # `<key>.tmp`, and the loser's rename found nothing: `[Errno 2] No such file or
        # directory: '...tmp' -> '...pkl'`, one sweep dead, an internal cache path in the
        # message and no advice. Sharing a cache directory is a reasonable thing to do and
        # nothing documents otherwise, so it has to work. A warm cache hid it entirely,
        # because nobody writes.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # Atomic, and the last writer wins. Both wrote the same value -- the key is a
            # hash of the inputs -- so which one wins does not matter.
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self.stats.writes += 1

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.pkl"))


@dataclass(slots=True)
class NullCache:
    """Caches nothing. For measuring what a stage really costs."""

    stats: CacheStats = field(default_factory=CacheStats)

    def get(self, key: str) -> Any | None:
        return None

    def put(self, key: str, value: Any) -> None:
        return None

    def __contains__(self, key: str) -> bool:
        return False


def cached(
    cache: Cache | None,
    key: str,
    stage: str,
    compute: Any,
    stats: CacheStats | None = None,
) -> Any:
    """Return a cached value or compute and store it, recording the hit or miss.

    `compute` is a callable rather than a value, so a hit never pays for the work it just
    avoided.
    """
    if cache is None:
        if stats is not None:
            stats.record(stage, hit=False)
        return compute()

    found = cache.get(key)
    if found is not None:
        if stats is not None:
            stats.record(stage, hit=True)
        return found

    value = compute()
    cache.put(key, value)
    if stats is not None:
        stats.record(stage, hit=False)
    return value

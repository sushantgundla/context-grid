# Caching

Source: `src/contextgrid/cache/store.py`. This is the reason a sweep of 48 configurations
doesn't mean 48 parses and 48 embedding runs — configurations that share a parser share its
parse; those that also share a chunker and embedder share the embeddings too.

## Content-addressed: what goes into a key

`cache_key()` hashes four things together:

```python
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from contextgrid.cache.store import CACHE_FORMAT, _canonical


def cache_key(
    stage: str,
    version: str,
    params: Mapping[str, Any] | None = None,
    inputs: Iterable[str] = (),
) -> str:
    payload = {
        "format": CACHE_FORMAT,
        "stage": stage,
        "version": version,
        "params": _canonical(params or {}),
        "inputs": sorted(inputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

- **`stage`** — `"parse"`, `"chunk"`, or `"embed"` (`_parse_all`/`_chunk_all`/`_embed_all` in
  `src/contextgrid/pipeline.py`).
- **`version`** — the plugin's own version string (`parser.version`, `chunker.version`,
  `embedder.version`). Bumping a plugin's internals invalidates every cached entry it wrote,
  without needing to touch anything else.
- **`params`** — every field on the plugin's dataclass (`params_of()` reads `dataclasses.fields`,
  skipping any name starting with `_`), plus the plugin's own name. Two dicts with the same
  keys in a different order produce the same key — `_canonical()` sorts them recursively
  before hashing.
- **`inputs`** — content hashes of whatever this stage consumed. This is what chains the
  cache together: a chunk set's key contains the parse's `text_hash()`, so changing the
  parser invalidates everything downstream of it automatically, without the chunk stage
  needing to know why.

**Different work never collides, and identical work is never redone twice.** A hit on the
wrong entry would produce a plausible-looking number instead of an error, which is the
failure mode this format is built to rule out.

## Why the tokenizer has to be in the chunk key

`params_of()` includes every dataclass field on the chunker — and every token-measuring
chunker (`fixed`, `recursive`, the `chonkie:*` and `langchain:*` wrappers) carries a
`tokenizer` field. That's deliberate, not incidental: **"512" means different text under
different tokenizers.** `regex` (the default, word/punctuation splitting) and `cl100k_base`
(OpenAI's real BPE tokenizer, via `tiktoken`) don't agree on where 512 tokens ends, so
"512-token chunks" under one is a genuinely different chunk set from "512-token chunks"
under the other. A cache key that omitted the tokenizer would serve one of them the other's
chunks — silently, and with entirely believable-looking results.

```
$ .venv/bin/python -c "
from contextgrid.cache.store import cache_key
k_regex   = cache_key('chunk', '1', {'chunker': 'recursive', 'size': 512, 'tokenizer': 'regex'}, ['abc123'])
k_cl100k  = cache_key('chunk', '1', {'chunker': 'recursive', 'size': 512, 'tokenizer': 'cl100k_base'}, ['abc123'])
k_reorder = cache_key('chunk', '1', {'size': 512, 'chunker': 'recursive', 'tokenizer': 'regex'}, ['abc123'])
print('regex  :', k_regex)
print('cl100k :', k_cl100k)
print('key order does not matter:', k_regex == k_reorder)
print('different tokenizer means a different key:', k_regex != k_cl100k)
"
regex  : 44f40d64125ff95aac71706664fa026eb3f23795f2852aa0f3e31128278d44ae
cl100k : 7c33ed72826e8cc8679b8b112deae136457c3ce97fa0e4f02288b9923e4893fb
key order does not matter: True
different tokenizer means a different key: True
```

See [plugins.md](plugins.md) for the tokenizer registry (`regex`, `character`, `cl100k_base`)
and [chunkers.md](../dimensions/chunkers.md) for how `tokenizer:` is set per chunker.

## Where each stage's inputs come from

From `src/contextgrid/pipeline.py`:

| Stage | Params (via `params_of`) | Input hash |
|---|---|---|
| `parse` | the parser's name and its own fields | `source.content_hash()` — the raw file's content |
| `chunk` | the chunker's name, its fields (size, overlap, **tokenizer**, ...) | `parsed.text_hash()` — the parse's output text |
| `embed` | the embedder's name and its fields | `texts_hash(texts)` — a hash of every chunk's text, in order |

That table *is* the reuse chain: change only the embedder and the parse and chunk cache
entries are untouched (their inputs didn't change); change the parser and parse, chunk, and
embed all miss, because the chain runs through `inputs`.

`params_of()` deliberately skips a fitted TF-IDF vocabulary and similar private/derived
state (leading-underscore fields) — that vocabulary is *derived from* the corpus, which is
already represented in the key via the text hash, so including it too would make the key
depend on the very thing it's meant to identify.

## Backends: `MemoryCache`, `DiskCache`, `NullCache`

Set via the config's `run.cache` (`memory` | `disk` | `none`,
`RunConfig.cache` in `src/contextgrid/config/schema.py`; resolved by `build_cache()` in
`src/contextgrid/config/loader.py`):

| `run.cache` | Class | Behaviour |
|---|---|---|
| `memory` (default) | `MemoryCache` | Plain in-process dict. Enough for one sweep; gone when the process exits. |
| `disk` | `DiskCache` | Pickled to `<report.out or the corpus's parent dir>/.contextgrid-cache/`, fanned out two directory levels deep (`key[:2]/key[2:4]/key.pkl`) so the directory never holds tens of thousands of files flat. Survives the process — the point is re-running a sweep after changing one axis without recomputing the rest. |
| `none` | `NullCache` | Caches nothing; every lookup is a miss. For measuring what a stage really costs on its own, not what a sweep costs with reuse. |

`DiskCache` treats a value that fails to unpickle (partial write, a dataclass shape that
changed between versions) as a miss and deletes the file, rather than raising — recomputing
one entry is cheap, and crashing a whole sweep on someone's stale cache directory is not.
Writes go to a `.tmp` file beside the target and get `replace()`d onto it, so a crash
mid-write can never leave a truncated file that later reads back as a false hit.

`CACHE_FORMAT` (currently `1`) is bumped whenever the pickled shape of a cached value
changes, so old on-disk entries miss cleanly instead of unpickling into an object with the
wrong fields.

## Reading the stats back

Every `cached()` call records a hit or a miss on a `CacheStats`, which a sweep reports back
through `Results.cache_summary`:

```python
def summary(self) -> str:
    if not self.lookups:
        return "no cache lookups"
    parts = [f"{self.hits} of {self.lookups} lookups reused ({self.hit_rate:.0%})"]
    for stage, (hits, misses) in sorted(self.by_stage.items()):
        parts.append(f"{stage} {hits}/{hits + misses}")
    return ", ".join(parts)
```

Printed by `contextgrid sweep` as `cache: <summary>` — see
[reports.md](reports.md#leaderboard) for where that line shows up next to the leaderboard.
Prefix reuse is the engineering decision that makes a wide sweep affordable at all, and it's
otherwise completely invisible unless something reports it.

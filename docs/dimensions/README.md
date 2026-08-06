# The axis model

context-grid measures a RAG (retrieval-augmented generation) pipeline by breaking it into
**axes** — independent choices you can swap in and out — and sweeping them. This page explains
what an axis is, the ten of them, how you write one down as a spec string, the three ways to
walk the resulting matrix, and how redundant combinations get dropped before anything runs.

## What an axis is

A pipeline is a sequence of stages: text goes in, an answer comes out. Each stage is a place
where a real system makes a choice — which parser, which chunk size, which embedding model —
and every one of those choices is a variable nobody has actually measured against the others.
Blog posts pick one arm of each axis, ship it, and call the result a best practice. An axis on
this grid is the alternative: hold everything else still, vary one thing, and look at the
number.

The ten axes, **in pipeline order** (`contextgrid.grid.matrix.AXIS_ORDER`):

| # | axis | what it decides | default |
|---|---|---|---|
| 1 | `ingestion` | what gets indexed versus what a hit returns | `None` (plain chunking) |
| 2 | [`parser`](parsers.md) | how raw bytes become text and structure | `markdown` |
| 3 | [`chunker`](chunkers.md) | how that text is cut into retrievable units | `recursive:512` |
| 4 | [`embedder`](embedders.md) | how a chunk becomes a vector | `tfidf` |
| 5 | [`index`](indexes.md) | what data structure the vectors live in | `dense` |
| 6 | [`transform`](transforms.md) | how the query is rewritten before it is searched with | `None` |
| 7 | [`retrieval`](retrieval.md) | how the index is actually queried | `None` (plain search) |
| 8 | [`reranker`](rerankers.md) | how candidates are reordered after retrieval | `None` |
| 9 | `candidates` | how many results the retriever hands the reranker | `50` |
| 10 | [`generator`](generation.md) | how the answer is written from the retrieved context | `None` (retrieval-only) |

This order is deliberate, not alphabetical (`contextgrid.grid.matrix.AXIS_ORDER`). The parser
decides what text exists at all, so it goes first among the content-shaping axes — every later
choice is being made about the real corpus, not a guess. Reranking goes last because it operates
on whatever the rest of the pipeline produced. It is also the order
[staged sweeps](#three-ways-to-walk-it) run in.

`ingestion` sits first in the list above but is unusual enough to have
[its own page](ingestion.md): where the other axes each do one job in sequence, ingestion
reaches back across the chunker's output and changes what "the unit that gets found" even
means. See that page for why it needed a definition of its own, and why it is not just another
setting on the chunker.

`chunker` and `parser` are the two documented here in depth, because between them they decide
what a "chunk" *is* before anything downstream — embedder, index, reranker — ever sees it.
Every axis past `chunker` now has its own page too:
[`embedder`](embedders.md), [`index`](indexes.md), [`transform`](transforms.md),
[`retrieval`](retrieval.md), [`reranker`](rerankers.md), [`generator`](generation.md).

## Spec strings

Every axis takes a **spec string**, never a plugin instance. `contextgrid.grid.matrix.matrix`
says why directly: a configuration has to be writable into a leaderboard row, a cache key and a
YAML file, and a Python object is none of those. `chonkie:recursive:512` survives all three; a
`ChonkieRecursiveChunker()` survives none, and a run nobody can write down is a run nobody can
reproduce. Passing an object in anyway raises immediately, naming the string to use instead —
see `contextgrid.grid.matrix._require_specs`.

A spec string is a plugin name, optionally followed by parameters
(`contextgrid.core.registry.Registry.parse_spec`):

```
"recursive"                 -> RecursiveChunker()                       (defaults)
"recursive:512"             -> RecursiveChunker(size=512)               (the shorthand param)
"recursive:512,overlap=64"  -> RecursiveChunker(size=512, overlap=64)   (shorthand + keywords)
"recursive:overlap=64"      -> RecursiveChunker(overlap=64)             (keywords alone)
```

Only the *first* bare value (no `=`) is allowed, and only when the plugin declares a
`shorthand` — the keyword it stands for. `chunker: recursive:512` means `size=512` because
`recursive` was registered with `shorthand="size"`; `ingestion: parent-document:4` means
`group=4` because `parent-document` was registered with `shorthand="group"`. Every other
parameter must be written as `key=value`. Values are coerced from text automatically:
`true`/`false`/`yes`/`no`/`on`/`off` become booleans, `none`/`null`/empty becomes `None`,
anything that parses as `int` or `float` becomes one, everything else stays a string
(`contextgrid.core.registry._coerce`).

**Namespacing.** A name can itself contain a colon — `chonkie:recursive:512` is the plugin
`chonkie:recursive` with `size=512`, not a plugin called `chonkie`. The registry resolves this
by taking the *longest* registered name the spec starts with
(`Registry._split_name`), so `chonkie:recursive`, `chonkie:sentence`, `langchain:markdown` and
so on all parse correctly even though they share a prefix with something shorter.

**Lazy plugins.** Chunkers and parsers that need an optional dependency — `chonkie:*`,
`langchain:*`, `pymupdf`, `docling`, `marker`, `pymupdf4llm`, `agno` — are registered by module
path and imported only when asked for
(`contextgrid.core.registry.Registry.register_lazy`). Two consequences: `import contextgrid`
never pulls in an ML runtime, and asking for one without its extra installed raises
`MissingExtraError` naming the exact `pip install "context-grid[...]"` to run, not a bare
`ModuleNotFoundError` three frames down. See [parsers.md](parsers.md) for which extra each
parser needs, and [chunkers.md](chunkers.md) for the chunkers.

## Building a matrix

`contextgrid.grid.matrix.matrix(...)` (also reachable as `lab.grid(...)`) takes one value or a
list on any axis — a single value holds that axis still, a list sweeps it:

```python
>>> from contextgrid.grid import matrix
>>> m = matrix(parser=["markdown", "text"], chunker=["recursive:512", "sentence:3"])
>>> m.shape()
'1 × 2 × 2 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = 4'
```

`Matrix.shape()` is the size check before anything runs — four axes with three values each is
81 configurations, and the matrix knows that before spending a single embedding call.
`Matrix.varying_axes` names which axes actually have more than one value; those are the only
ones a sweep can learn anything about.

## Three ways to walk it

Selecting several values on an axis multiplies the run count. `Matrix.expand(mode)` covers the
resulting space three ways (`contextgrid.grid.matrix.SweepMode`):

- **`factorial`** — every combination, including interactions between axes. Measures the most
  and explodes the fastest: four axes with three values each is 81 runs.
- **`ofat`** (one-factor-at-a-time) — holds a baseline (the first value on every axis) and
  varies one axis at a time. Linear rather than exponential, and directly interpretable:
  "switching the chunker gained 0.08." Blind to interactions between axes. **The default**,
  because most of the time axes are close to independent, and when they are not, that is
  itself worth discovering deliberately rather than assumed away.
- **`staged`** — picks the winner on one axis, freezes it, and sweeps the next. The cheapest of
  the three, and the one most practitioners actually reach for. It can be wrong whenever axes
  interact — a chunker that wins under `bm25` might lose under a dense index — and the runner
  says so out loud rather than burying it in a footnote. `Matrix.expand("staged")` returns the
  same configuration set as `ofat`; a staged sweep's later stages depend on the *results* of
  earlier ones, which only the runner (not the matrix) can know, so what staged actually runs
  is decided as it goes, one axis at a time (`Matrix.stage_configs`).

```python
>>> m.count("ofat")
3
>>> m.count("factorial")
4
```

Axes are swept in `AXIS_ORDER` (the table above) for both OFAT and staged, for the same reason
that order matters everywhere else on this page: the parser is fixed before the chunker is
varied, so the chunker sweep is measuring the real corpus rather than a placeholder parse.

## Dropping redundant combinations

`Matrix.expand` and `Matrix.expand_with_dropped` run every configuration through
`contextgrid.grid.matrix.deduplicate`, which does two things before a config is allowed to run:

**Canonicalise** (`canonicalise`) rewrites settings a configuration cannot possibly use, so two
spellings of the same run collapse into one:

- `ingestion: plain` and no ingestion strategy at all are the same run — `plain` is rewritten to
  `None`.
- `transform: none` and `reranker: none` are each rewritten to `None` — `"none"` is the identity
  transform / identity reranker, so it is the same configuration as naming nothing.
- `candidates` is only meaningful when something reranks the candidates. Without a reranker,
  sweeping `candidates` would run identical configurations under different names — it gets
  reset to the default (`50`).
- BM25 never looks at a vector, so `bm25 + tfidf` and `bm25 + hash` are the same run under two
  names. Left alone they would waste two-thirds of the sparse arm of a sweep, and worse, they
  would poison the embedder axis's measured effect — averaging three identical BM25 scores into
  the embedder's record as though it had earned them. When the chosen index doesn't
  `needs_vectors`, `embedder` is rewritten to `None`.

**Drop the impossible** (`is_runnable`): writing `embedder: [tfidf, null]` alongside
`index: [dense, bm25]` obviously means "tfidf with dense, and bm25 with nothing" — but a
factorial expansion also produces `null` with `dense`, which cannot run at all: a dense index
has no vectors to search. Those combinations are dropped rather than errored, because forcing
the two intentions to be written as two separate configs is worse than quietly not running the
one that can't work. `expand_with_dropped` returns how many were dropped, so a matrix that
shrank does not do so silently.

```python
>>> from contextgrid.grid.matrix import deduplicate
>>> from contextgrid.pipeline import Config
>>> kept, dropped = deduplicate([
...     Config(ingestion="plain", chunker="recursive:512"),
...     Config(chunker="recursive:512"),
... ])
>>> len(kept), dropped
(1, 0)
```

Both configs above collapse to the same one — `ingestion="plain"` canonicalises to `None` — so
`deduplicate` keeps a single row rather than two identical ones under different names.

## See also

- [ingestion](ingestion.md) — the axis that changes what gets indexed versus what comes back
- [parsers](parsers.md) — all 8 registered parsers
- [chunkers](chunkers.md) — all 12 registered chunkers
- [configuration reference](../guide/configuration.md) — the YAML/JSON shape every axis lives in

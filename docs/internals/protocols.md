# Protocols: the contracts every plugin implements

`context-grid` has roughly 40 pluggable components. None of them inherit from a base class —
they satisfy a `typing.Protocol`, checked with `@runtime_checkable` so `isinstance(plugin,
Chunker)` works at runtime, not just under a type checker. A `Protocol` was chosen over a base
class deliberately: a plugin wrapping a third-party library (`chonkie`, `langchain`) doesn't
have to inherit from anything of ours, it just has to have the right shape.

Three protocols live in [`core/protocols.py`](../../src/contextgrid/core/protocols.py):
`Tokenizer`, `Parser`, `Chunker`. Two more that follow the identical shape live beside the code
they belong to because they need types (`Chunk`, `Scored`) that would otherwise pull the
`core` package into a dependency cycle: `IngestionStrategy` (`ingest/base.py`) and
`RetrievalStrategy` (`retrieve/base.py`).

Every protocol below is enforced two ways: at runtime via `isinstance()` (cheap, structural,
catches "you forgot a method"), and behaviourally via the conformance suites (see
[conformance.md](conformance.md)), which catch the bugs a shape check cannot — a chunker that
has a `.chunk()` method that returns garbage still satisfies the `Protocol`.

## `Tokenizer`

> Turns text into token boundaries.

```python
@property
def name(self) -> str: ...
@property
def exact(self) -> bool: ...
def token_spans(self, text: str) -> list[tuple[int, int]]: ...
def count(self, text: str) -> int: ...
```

| Member | Must guarantee |
|---|---|
| `name` | Recorded on every chunk this tokenizer measured — `Chunk.token_counts` is keyed by it. |
| `exact` | `False` when boundaries only approximate a real model's tokenizer. Fine for chunking, wrong for costing — the cost model should refuse to trust an inexact count for billing. |
| `token_spans(text)` | Character ranges of each token, **in order, non-overlapping**. This is the method that exists at all: `count()` alone would force a chunker to guess where a `size`-token boundary falls, and a guessed offset is exactly what this package refuses to produce. |
| `count(text)` | Number of tokens. Must agree with `len(token_spans(text))`. |

Why `token_spans` and not just `count`: a chunker that cuts "every 512 tokens" has to know
*which characters* are the 512th token's boundary, not just that there are 512 of them.

## `Parser`

> Turns a source file into text with structure.

```python
from contextgrid.core.documents import MediaType, ParsedDocument, SourceFile


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
def supports(self, media_type: MediaType) -> bool: ...
def parse(self, source: SourceFile) -> ParsedDocument: ...
```

| Member | Must guarantee |
|---|---|
| `name`, `version` | Recorded on the `ParsedDocument` (`parser`, `parser_version`) and folded into the parse cache key, so a version bump correctly invalidates cached parses. |
| `supports(media_type)` | Declared *before* parsing is attempted. The runner uses this to skip a file with a `PARSER_FALLBACK` warning instead of the parser raising three frames into an unrelated library. |
| `parse(source)` | Returns a `ParsedDocument` whose `document.text` is the text every downstream offset refers to. **Must set `offsets_exact=False`** rather than silently returning blocks whose text is not a literal slice of `document.text`. This is the parser half of the one hard invariant — see [design.md §4.7](../design.md#47-the-exactness-flag). |

The conformance suite additionally requires (see [conformance.md](conformance.md)): blocks stay
inside the document, are in reading order, don't overlap, aren't empty, and — when
`offsets_exact` is true — cover every non-whitespace character. None of that is in the
`Protocol` itself; a type checker cannot see it. It's enforced behaviourally.

## `Chunker`

> Cuts a parsed document into retrievable units.

```python
from contextgrid.core.documents import Chunk, ParsedDocument


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
def chunk(self, parsed: ParsedDocument) -> list[Chunk]: ...
```

| Member | Must guarantee |
|---|---|
| `name`, `version` | Same role as `Parser`'s: cache key and provenance. |
| `chunk(parsed)` | Every returned chunk's `span` must point into `parsed.document`. Chunks in reading order. A chunker that rewrites text — prepends an LLM-written summary, extracts propositions — **must set `offsets_exact=False`** on the chunks it returns; it may not claim an exact slice it did not produce. |

Chunkers all cut up the *same* text a given parse produced, which is what makes comparing two
chunkers fair without re-annotating ground truth (`design.md §4.2`). `chunk/base.py`'s
`ChunkBuilder` is what most in-tree chunkers build on — it computes token counts, heading path,
and the `offsets_exact` inheritance from the parse in one place, so an individual chunker
implementation cannot get those wrong even by omission. See [extending.md](extending.md) for a
chunker built from scratch on top of it.

## `IngestionStrategy`

> Decides what is indexed and what a hit on it returns.

```python
from collections.abc import Sequence
from contextgrid.core.documents import Chunk
from contextgrid.ingest.base import Ingested, IngestionContext


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
@property
def uses_model(self) -> bool: ...
def ingest(self, chunks: Sequence[Chunk], context: IngestionContext) -> Ingested: ...
```

| Member | Must guarantee |
|---|---|
| `uses_model` | `True` when *building* the index costs model calls (paid once, at index time). The runner reads this to decide whether a sweep needs a spending cap — see `_warn_if_unbounded` in `grid/runner.py`. |
| `ingest(chunks, context)` | Returns an `Ingested` with `indexed` (embedded and searched) and `retrievable` (what a hit turns into — reranked, scored, handed to a generator). For plain chunking these are the same list. A strategy that rewrites text for the indexed side must mark those chunks `offsets_exact=False`; the retrievable side keeps its real offsets, because gold resolution and scoring both happen against `retrievable`, not `indexed`. |

`IngestionContext` (parses, a `WarningLog`, an `llm`, a `tokenizer`) is handed in rather than
imported, so a strategy that needs a model gets the one the caller chose, and a test can hand it
a scripted one with no network and no key.

The one thing the `Protocol` cannot express: **a parent and its children must never both land
in the scored set.** `Ingested.presentation` exists specifically to keep wider "presentation"
passages out of `retrievable` — putting both in would make gold resolve to each granularity
separately and halve recall for a purely structural reason (measured on this package's demo
corpus at 1.86 relevant units per question, against 1.00 for plain chunking). See the
docstring on `Ingested.presentation` in `ingest/base.py` for the full argument.

## `RetrievalStrategy`

> Turns a question into ranked chunks, using whatever index it was given.

```python
from collections.abc import Sequence
from contextgrid.index.base import Scored
from contextgrid.retrieve.base import RetrievalTrace, Searcher


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
@property
def uses_model(self) -> bool: ...
def retrieve(
    self,
    query: str,
    queries: Sequence[str],
    searcher: Searcher,
    k: int,
    trace: RetrievalTrace,
) -> list[Scored]: ...
```

| Member | Must guarantee |
|---|---|
| `uses_model` | `True` when this strategy calls a model **per query** (not once at index time — that's `IngestionStrategy.uses_model`). The runner refuses to start an unbounded sweep containing one of these without `budget_usd` or `budget_seconds`, because a strategy that decides its own number of calls has no ceiling anybody can eyeball in advance. |
| `retrieve(...)` | `query` is the question as asked; `queries` is what the transform axis made of it (usually one string, sometimes several — a strategy is free to ignore `queries` and work from `query` alone). `searcher` runs one search against whatever index the configuration chose and is the *only* way the strategy may touch the index — it never sees the index type, so every strategy works with every store. Every search performed must be recorded on `trace` via `trace.record_search(text)`, and every model call via `trace.record_model_call()` — two strategies with identical recall and wildly different `model_calls` are a decision, not a tie, and the trace is the only place that shows up. |

`fuse()` (`retrieve/base.py`) is the standard way to combine several ranked lists — reciprocal
rank fusion, not score averaging, because a cosine similarity from one query and one from
another are not on the same scale and averaging lets whichever query produced larger magnitudes
win a result it didn't earn.

See [extending.md](extending.md) for a `RetrievalStrategy` built and exercised from scratch.

## The pattern across all five

Every protocol above shares a spine: `name` and `version` for cache keys and provenance, a
single verb method that does the plugin's actual work, and — for the two strategy protocols —
a `uses_model` flag that lets the runner reason about cost and boundedness without knowing
anything about what the strategy does internally. New plugin families (an `Embedder`, an
`Index`, a `Reranker` — see `embed/base.py`, `index/base.py`, `rerank/base.py`) follow the same
shape even though they live outside `core/protocols.py`.

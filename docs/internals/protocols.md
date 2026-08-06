# Protocols: the contracts every plugin implements

`context-grid` has roughly 40 pluggable components. None of them inherit from a base class —
they satisfy a `typing.Protocol`, checked with `@runtime_checkable` so `isinstance(plugin,
Chunker)` works at runtime, not just under a type checker. A `Protocol` was chosen over a base
class deliberately: a plugin wrapping a third-party library (`chonkie`, `langchain`) doesn't
have to inherit from anything of ours, it just has to have the right shape.

There are **twelve** plugin protocols, one per stage of the pipeline plus `Metric`:

| Protocol | Stage | Lives in |
|---|---|---|
| [`Tokenizer`](#tokenizer) | measuring text | [`core/protocols.py`](../../src/contextgrid/core/protocols.py) |
| [`Parser`](#parser) | source file → text | [`core/protocols.py`](../../src/contextgrid/core/protocols.py) |
| [`Chunker`](#chunker) | text → retrievable units | [`core/protocols.py`](../../src/contextgrid/core/protocols.py) |
| [`IngestionStrategy`](#ingestionstrategy) | what's indexed vs. what's returned | [`ingest/base.py`](../../src/contextgrid/ingest/base.py) |
| [`Embedder`](#embedder) | text → vectors | [`embed/base.py`](../../src/contextgrid/embed/base.py) |
| [`Index`](#index) | storing and searching | [`index/base.py`](../../src/contextgrid/index/base.py) |
| [`QueryTransform`](#querytransform) | rewriting the question | [`transform/query.py`](../../src/contextgrid/transform/query.py) |
| [`RetrievalStrategy`](#retrievalstrategy) | how the index is used | [`retrieve/base.py`](../../src/contextgrid/retrieve/base.py) |
| [`Reranker`](#reranker) | reordering candidates | [`rerank/base.py`](../../src/contextgrid/rerank/base.py) |
| [`Generator`](#generator) | context → answer | [`generate/answer.py`](../../src/contextgrid/generate/answer.py) |
| [`LLM`](#llm) | the model call underneath the four axes above that need one | [`evalset/llm.py`](../../src/contextgrid/evalset/llm.py) |
| [`Metric`](#metric) | judgements + a ranking → one score | [`score/base.py`](../../src/contextgrid/score/base.py) |

Three of them (`Tokenizer`, `Parser`, `Chunker`) live in `core/protocols.py`. The rest live
beside the code they belong to, because each one needs types (`Chunk`, `Scored`, `LLM`...) that
would otherwise pull `core` into a dependency cycle. **The one that surprises people most:**
`LLM` is not in a module called `llm.py` at the top level — it lives in
[`evalset/llm.py`](../../src/contextgrid/evalset/llm.py), because the eval-set tooling
(`evalset/generate.py`, `evalset/filters.py`) was the first thing that needed it. Every other
protocol that needs a model (`QueryTransform`, `IngestionStrategy`, `Generator`) imports `LLM`
from there — `from contextgrid.evalset.llm import LLM`, not `contextgrid.llm` or
`contextgrid.core.llm`. **The second one that surprises people:** `Metric`'s registry
(`METRICS`) lives in `score/base.py`, not `score/__init__.py` like every other family's registry
— see [`Metric`](#metric) below for why.

Every protocol below is enforced two ways: at runtime via `isinstance()` (cheap, structural,
catches "you forgot a method"), and behaviourally via the conformance suites (see
[conformance.md](conformance.md)), which catch the bugs a shape check cannot — a chunker that
has a `.chunk()` method that returns garbage still satisfies the `Protocol`. Conformance suites
currently exist for two families only: `Parser` and `Chunker` (`tests/conformance/`). The other
ten are exercised by hand-written unit tests per plugin; there is nothing to "add a case to"
for them yet.

**Discoverability check, live:** every member listed below was pulled from the source at the
path given. If you only trust one part of this document, trust the import paths — they are
exactly what's in the module, copy-pasted, not reconstructed from memory.

```python
from contextgrid.core.protocols import Tokenizer, Parser, Chunker
from contextgrid.embed.base import Embedder
from contextgrid.index.base import Index
from contextgrid.transform.query import QueryTransform
from contextgrid.retrieve.base import RetrievalStrategy
from contextgrid.rerank.base import Reranker
from contextgrid.ingest.base import IngestionStrategy
from contextgrid.generate.answer import Generator
from contextgrid.evalset.llm import LLM
from contextgrid.score.base import Metric

for protocol in (
    Tokenizer,
    Parser,
    Chunker,
    Embedder,
    Index,
    QueryTransform,
    RetrievalStrategy,
    Reranker,
    IngestionStrategy,
    Generator,
    LLM,
    Metric,
):
    print(f"{protocol.__name__:<20} runtime_checkable={hasattr(protocol, '_is_runtime_protocol')}")
```

Output:

```
Tokenizer            runtime_checkable=True
Parser               runtime_checkable=True
Chunker              runtime_checkable=True
Embedder             runtime_checkable=True
Index                runtime_checkable=True
QueryTransform       runtime_checkable=True
RetrievalStrategy    runtime_checkable=True
Reranker             runtime_checkable=True
IngestionStrategy    runtime_checkable=True
Generator            runtime_checkable=True
LLM                  runtime_checkable=True
Metric               runtime_checkable=True
```

## `Tokenizer`

> Turns text into token boundaries.

```python
from contextgrid.core.protocols import Tokenizer


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
Resolved with `from contextgrid.tokens import get_tokenizer` — `get_tokenizer(None)` returns the
package's default; `get_tokenizer("cl100k")` or an instance both work anywhere a `Tokenizer` is
accepted (see `ContextAssembler.tokenizer` in [`assemble/context.py`](../../src/contextgrid/assemble/context.py)).

## `Parser`

> Turns a source file into text with structure.

```python
from contextgrid.core.documents import MediaType, ParsedDocument, SourceFile
from contextgrid.core.protocols import Parser


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

### Supporting types

All from `from contextgrid.core.documents import ...` unless noted.

**`MediaType`** — a `str` `Enum`: `TEXT`, `MARKDOWN`, `HTML`, `PDF`, `DOCX`, `PPTX`, `XLSX`,
`UNKNOWN`. `MediaType.from_suffix(".pdf")` guesses from a file extension; an unrecognised suffix
returns `UNKNOWN` rather than raising, so a parser's `supports()` can decline it with a proper
warning instead of the pipeline crashing on an unrelated file type.

**`SourceFile`** — one input file, before extraction.

| Field / member | Type | Notes |
|---|---|---|
| `id` | `str` | |
| `media_type` | `MediaType` | default `UNKNOWN` |
| `path` | `str \| None` | |
| `raw` | `bytes \| None` | the bytes, once read |
| `meta` | `dict[str, Any]` | |
| `size_bytes` (property) | `int \| None` | `None` if `raw` is `None` |
| `content_hash()` | `str` | SHA-256 of `raw`; raises `DocumentError` if `raw is None`. The cache key every downstream stage hangs off. |
| `text(encoding="utf-8")` | `str` | decodes `raw`; raises `DocumentError` if `raw is None` |

**`Document`** — the text a parse produced, and the thing spans point into.

| Field / member | Type | Notes |
|---|---|---|
| `id` | `str` | |
| `text` | `str` | |
| `source` | `str \| None` | |
| `meta` | `dict[str, Any]` | |
| `length` (property) | `int` | |
| `span()` | `Span` | a span covering the whole document |
| `slice(span)` | `str` | raises `DocumentError` if `span` belongs to another doc, or runs past the end |
| `contains_span(span)` | `bool` | |
| `text_hash()` | `str` | SHA-256 of `text`; what stops two parses of the same file being mixed up |

**`Block`** — a structural region with its position.

| Field / member | Type | Notes |
|---|---|---|
| `span` | `Span` | |
| `text` | `str` | |
| `kind` | `BlockKind` | default `PARAGRAPH` |
| `page` | `int \| None` | |
| `level` | `int \| None` | heading depth |
| `meta` | `dict[str, Any]` | |
| `doc_id` (property) | `str` | `span.doc_id` |
| `is_heading` (property) | `bool` | `kind is BlockKind.HEADING` |

**`BlockKind`** — a `str` `Enum`: `PARAGRAPH`, `HEADING`, `TABLE`, `TABLE_ROW`, `LIST`,
`LIST_ITEM`, `CODE`, `QUOTE`, `FIGURE`, `CAPTION`, `FOOTNOTE`, `HEADER`, `FOOTER`, `PAGE_BREAK`,
`OTHER`.

**`ParsedDocument`** — one parser's reading of one source file.

| Field / member | Type | Notes |
|---|---|---|
| `document` | `Document` | |
| `blocks` | `tuple[Block, ...]` | default `()` |
| `parser` | `str` | default `"unknown"` |
| `parser_version` | `str` | default `"0"` |
| `offsets_exact` | `bool` | default `True` — the honesty flag |
| `page_count` | `int \| None` | |
| `duration_ms` | `float \| None` | |
| `warnings` | `WarningLog` | |
| `meta` | `dict[str, Any]` | |
| `id` (property) | `str` | `document.id` |
| `text` (property) | `str` | `document.text` |
| `text_hash()` | `str` | delegates to `document.text_hash()` |
| `verify_blocks()` | `list[Block]` | blocks whose text does not match the document at their own span; empty for any parser honestly claiming `offsets_exact=True` |
| `blocks_of(*kinds)` | `tuple[Block, ...]` | filter by `BlockKind` |
| `block_at(position)` | `Block \| None` | the block containing a character position |
| `page_at(position)` | `int \| None` | |
| `heading_path_at(position)` | `tuple[str, ...]` | the chain of headings above a position, outermost first |

`Span` (`from contextgrid.core.span import Span`) is the half-open `[start, end)` character
range every offset in the package is expressed as — see [`core/span.py`](../../src/contextgrid/core/span.py)
for the full arithmetic (`intersection`, `overlap_len`, `coverage_of`, `iou`, and the module-level
`merge_spans`/`covered_length`/`coverage_fraction` helpers used throughout scoring).

## `Chunker`

> Cuts a parsed document into retrievable units.

```python
from contextgrid.core.documents import Chunk, ParsedDocument
from contextgrid.core.protocols import Chunker


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

### Supporting type: `Chunk`

`from contextgrid.core.documents import Chunk` — a unit of retrievable text.

| Field / member | Type | Notes |
|---|---|---|
| `id` | `str` | conventionally `f"{doc_id}:{start}-{end}"`, via `chunk/base.py:chunk_id()` |
| `span` | `Span` | |
| `text` | `str` | |
| `meta` | `dict[str, Any]` | `ChunkBuilder` sets `heading_path`, `page`, `parser`, `index` here |
| `token_counts` | `dict[str, int]` | keyed by `Tokenizer.name` — never a single bare number |
| `offsets_exact` | `bool` | default `True` — the honesty flag |
| `doc_id` (property) | `str` | `span.doc_id` |
| `char_start`, `char_end`, `char_length` (properties) | `int` | delegate to `span` |
| `token_count(tokenizer)` | `int \| None` | `token_counts.get(tokenizer)` |
| `matches_source(document)` | `bool` | `True` when `document.slice(self.span) == self.text` — the invariant every offset-exact chunker must satisfy, and what the conformance suite checks |

`ChunkBuilder` (`from contextgrid.chunk.base import ChunkBuilder, chunk_id, trim_range`) turns
character ranges into `Chunk`s for one parse: `ChunkBuilder(parsed).build(start, end)` for one
chunk, `.build_all(ranges)` for many. It precomputes heading lookup so a chunker doesn't have to
walk the block list per chunk (which would be quadratic on a real document).

## `IngestionStrategy`

> Decides what is indexed and what a hit on it returns.

```python
from collections.abc import Sequence
from contextgrid.core.documents import Chunk
from contextgrid.ingest.base import Ingested, IngestionContext, IngestionStrategy


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
| `uses_model` | `True` when *building* the index costs model calls (paid once, at index time — this is a different bargain from `RetrievalStrategy.uses_model`, which is paid per query). The runner reads this to decide whether a sweep needs a spending cap. |
| `ingest(chunks, context)` | Returns an `Ingested` with `indexed` (embedded and searched) and `retrievable` (what a hit turns into — reranked, scored, handed to a generator). For plain chunking these are the same list. A strategy that rewrites text for the indexed side must mark those chunks `offsets_exact=False`; the retrievable side keeps its real offsets, because gold resolution and scoring both happen against `retrievable`, not `indexed`. |

### Supporting types

**`IngestionContext`** (`from contextgrid.ingest.base import IngestionContext`) — what a
strategy is allowed to reach for. Handed in rather than imported, so a strategy that needs a
model gets the one the caller chose, and a test can hand it a scripted one with no network and
no key.

| Field | Type | Notes |
|---|---|---|
| `parses` | `dict[str, object]` | |
| `warnings` | `WarningLog` | |
| `llm` | `object \| None` | typed loosely to avoid a hard dependency from `ingest/` on `evalset/`; in practice an `LLM` |
| `tokenizer` | `object \| None` | in practice a `Tokenizer` |

**`Ingested`** (`from contextgrid.ingest.base import Ingested`) — the two sides of an index, and
the map between them.

| Field / member | Type | Notes |
|---|---|---|
| `indexed` | `list[Chunk]` | embedded and searched |
| `retrievable` | `list[Chunk]` | what a hit turns into |
| `parent_of` | `dict[str, str]` | indexed chunk id → the retrievable chunk it stands for; a missing key means "itself" |
| `presentation` | `dict[str, list[str]]` | wider passages a strategy may hand back *instead of* a retrievable unit, mapped to the retrievable-unit ids they cover |
| `presented_chunks` | `dict[str, Chunk]` | the wider passages themselves, by id, for reranking and generation |
| `model_calls` | `int` | how many model calls building this cost |
| `notes` | `dict[str, object]` | free-form; e.g. `HierarchicalIngestion` stores `"children"` and `"threshold"` here for `BuiltPipeline._merge_siblings` to read at query time |
| `resolve(indexed_id)` | `str` | `parent_of.get(indexed_id, indexed_id)` |
| `scored_ids(returned_id)` | `list[str]` | what a returned id counts as, for scoring — `presentation.get(returned_id, [returned_id])` |
| `expansion` (property) | `float` | `len(indexed) / len(retrievable)`, or `0.0` if `retrievable` is empty |
| `Ingested.plain(chunks)` (classmethod) | `Ingested` | `indexed=retrievable=list(chunks)` — what `PlainIngestion` returns |

**The trap the `Protocol` cannot express:** a parent and its children must never both land in
the *scored* set. `presentation` exists specifically to keep wider presentation passages out of
`retrievable` — putting both in would make gold resolve to each granularity separately and halve
recall for a purely structural reason (measured on this package's demo corpus at 1.86 relevant
units per question, against 1.00 for plain chunking). A presentation passage's units are what it
is *scored as* — `scored_ids()` above is what implements that, and it's why `BuiltPipeline`
routes every returned id through `pipeline.scored_ids()` before comparing to qrels rather than
comparing raw ids (see `pipeline.py:BuiltPipeline.scored_ids`).

## `Embedder`

> Turns text into vectors, with queries and documents handled separately.

```python
from collections.abc import Sequence
from contextgrid.embed.base import Embedder, EmbeddingResult


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
@property
def dimensions(self) -> int: ...
@property
def normalised(self) -> bool: ...
@property
def max_tokens(self) -> int | None: ...
def prepare(self, documents: Sequence[str]) -> None: ...
def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...
def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult: ...
```

| Member | Must guarantee |
|---|---|
| `name`, `version` | Cache key and provenance, same as every other family. |
| `dimensions` | The width of every row `embed_documents`/`embed_queries` return. |
| `normalised` | `True` when vectors are unit length, so dot product and cosine agree — indexes that assume this (e.g. `ExactDenseIndex(metric="dot")`) trust it rather than re-normalising. |
| `max_tokens` | The context limit, or `None` when there is none. Anything longer gets truncated — see `truncate()` below. |
| `prepare(documents)` | Called once, before either `embed_*` method, with the *whole* corpus about to be embedded. A no-op for most models; TF-IDF and other corpus-statistical embedders fit their vocabulary here. Must not be skipped even for embedders that don't need it — the pipeline always calls it (`pipeline.py:_embed_all`). |
| `embed_documents(texts)` / `embed_queries(texts)` | **Two separate methods, not one method wearing two names.** E5 wants `query:`/`passage:` prefixes, BGE wants an instruction on the query only, Cohere wants `input_type`. An embedder that routes both through the same code path is being used wrong in a way that doesn't raise — the numbers just come out uniformly, invisibly lower. |

### Supporting type: `EmbeddingResult`

`from contextgrid.embed.base import EmbeddingResult, Vectors` — vectors, plus everything needed
to price and to trust them. `Vectors = npt.NDArray[np.float32]` (`from contextgrid.embed.base
import Vectors`) — fully parameterised rather than a bare `np.ndarray`, because an unparameterised
generic is a `mypy --strict` error on some numpy versions and not others.

| Field / member | Type | Notes |
|---|---|---|
| `vectors` | `Vectors` | shape `(n, dimensions)` |
| `warnings` | `WarningLog` | default empty |
| `input_tokens` | `int` | default `0` |
| `truncated` | `int` | default `0` — how many inputs were cut |
| `count` (property) | `int` | `vectors.shape[0]`, or `0` if empty |
| `dimensions` (property) | `int` | `vectors.shape[1]`, or `0` if not 2-D |

Two module-level helpers every embedder reaches for (`from contextgrid.embed.base import
normalise, truncate`):

- **`normalise(vectors)`** — scales each row to unit length; an all-zero row is left alone
  rather than dividing by zero (which would poison every downstream similarity with NaN).
- **`truncate(texts, max_tokens, *, model, stage="embed", approximate_chars_per_token=4.0)`** —
  returns `(cut_texts, WarningLog, truncated_count)`. Uses a **character** estimate
  (`max_tokens * approximate_chars_per_token`), not a real tokenizer — an embedder that knows its
  actual tokenizer should truncate with that instead and pass `max_tokens=None` here. This is the
  function that turns "the chunk holding the answer got silently cut to 512 tokens" into a
  recorded, visible `INPUT_TRUNCATED` warning.

`EMBEDDERS: Registry[Embedder]` lives in `from contextgrid.embed import EMBEDDERS, get_embedder`.
In-tree, model-free embedders worth reading as reference implementations:
`HashEmbedder`/`TfidfEmbedder`/`TokenCountEmbedder` in
[`embed/local.py`](../../src/contextgrid/embed/local.py).

## `Index`

> Holds chunks and finds the ones most like a query.

```python
from collections.abc import Sequence
from contextgrid.core.documents import Chunk
from contextgrid.embed.base import Vectors
from contextgrid.index.base import Index, Scored


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
@property
def needs_vectors(self) -> bool: ...
@property
def is_exact(self) -> bool: ...
def build(self, chunks: Sequence[Chunk], vectors: Vectors | None = None) -> None: ...
def search(self, text: str, vector: Vectors | None = None, k: int = 10) -> list[Scored]: ...
def size_bytes(self) -> int: ...
```

| Member | Must guarantee |
|---|---|
| `needs_vectors` | `False` for indexes that work on text alone (e.g. `BM25Index`), so the pipeline can skip embedding entirely when no index in the sweep needs it. |
| `is_exact` | `False` for approximate search. An approximate index must be compared against its exact twin before its numbers mean anything — tuning `efSearch` without knowing what recall it cost is guessing, in the direction that looks good. |
| `build(chunks, vectors=None)` | Called once, after ingestion and embedding. `vectors` is `None` when the embedder axis is `None` or the index doesn't need them. |
| `search(text, vector=None, k=10)` | Takes **both** the query text and its vector, because sparse and dense indexes need different things and a hybrid needs both — an index ignores whichever it doesn't use. Returns at most `k` results, best first. |
| `size_bytes()` | Roughly how much memory this index occupies — the one-time cost people forget when comparing a 1536-dimension model against a 384-dimension one. |

### Supporting type: `Scored`

`from contextgrid.index.base import Scored` — one result.

| Field | Type |
|---|---|
| `chunk_id` | `str` |
| `score` | `float` |

`top_k(scores: dict[str, float], k: int) -> list[Scored]` (same module) turns a `{chunk_id:
score}` mapping into the top `k`, ties broken by chunk id — deterministic, so two chunks with
identical scores don't swap places between runs and destroy trust in a leaderboard that "moved"
when nothing changed. Almost every `Index` and `Reranker` implementation in this package ends
its scoring method with a call to `top_k`.

`INDEXES: Registry[Index]` lives in `from contextgrid.index import INDEXES, get_index`.

## `QueryTransform`

> Rewrites a question into one or more search queries.

```python
from contextgrid.transform.query import QueryTransform, TransformedQuery


@property
def name(self) -> str: ...
def transform(self, query: str) -> TransformedQuery: ...
```

| Member | Must guarantee |
|---|---|
| `name` | Provenance only — there is no `version` on this protocol (unlike every other family). A transform that changes behaviour needs a new `name`. |
| `transform(query)` | Returns a `TransformedQuery`. Every LLM call the transform makes must be reflected in `llm_calls`/`llm_tokens` on the result — this is what lets `describe_cost()` (below) say a transform costs a model call **on every query, forever**, which is the number that decides whether it's worth using at all. |

### Supporting type: `TransformedQuery`

`from contextgrid.transform.query import TransformedQuery` (also re-exported from
`contextgrid.transform`).

| Field / member | Type | Notes |
|---|---|---|
| `original` | `str` | the question as asked |
| `queries` | `tuple[str, ...]` | what actually gets searched with |
| `llm_calls` | `int` | default `0` |
| `llm_tokens` | `int` | default `0` |
| `is_identity` (property) | `bool` | `queries == (original,)` |
| `fan_out` (property) | `int` | `len(queries)` — how many searches this question now costs |

Model-free transforms (`NoTransform`, `ExpandAcronyms`) register normally in `TRANSFORMS:
Registry[QueryTransform]` (`from contextgrid.transform import TRANSFORMS, get_transform`).
Model-backed ones (`HyDE`, `MultiQuery`, `Decompose`, `StepBack`) **cannot** go through the
registry — building one needs an `LLM` a spec string alone can't supply — so they're built by
`get_transform(spec, llm)` instead and listed separately in `MODEL_BACKED: tuple[str, ...]`
(same module) purely so `contextgrid plugins` and the config template can say they exist.
`available_transforms()` returns the union of both.

`describe_cost(transformed: Sequence[TransformedQuery]) -> str` (same module) summarises what a
transform cost across an eval set in exactly the terms that decide whether to use it.

## `RetrievalStrategy`

> Turns a question into ranked chunks, using whatever index it was given.

```python
from collections.abc import Sequence
from contextgrid.index.base import Scored
from contextgrid.retrieve.base import Lookup, RetrievalStrategy, RetrievalTrace, Searcher


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
    lookup: Lookup = ...,
) -> list[Scored]: ...
```

| Member | Must guarantee |
|---|---|
| `uses_model` | `True` when this strategy calls a model **per query** (not once at index time — that's `IngestionStrategy.uses_model`). The runner refuses to start an unbounded sweep containing one of these without `budget_usd` or `budget_seconds`. |
| `retrieve(...)` | `query` is the question as asked; `queries` is what the transform axis made of it (usually one string, sometimes several — a strategy is free to ignore `queries` and work from `query` alone). `searcher` runs one search against whatever index the configuration chose and is the *only* way the strategy may touch the index — it never sees the index type, so every strategy works with every store. `lookup` reads the text of a chunk a `searcher` call already returned an id for — the *only* way a strategy may read what it found (see below). Every search performed must be recorded on `trace` via `trace.record_search(text)`, and every model call via `trace.record_model_call()`. |

### Supporting types

**`Searcher`** (`from contextgrid.retrieve.base import Searcher`) — a type alias, not a class:
`Callable[[str, int], Sequence[Scored]]`. Given query text and a result count, returns a ranked
list. The strategy neither knows nor cares whether that was BM25, HNSW or Postgres underneath.

**`Lookup`** (`from contextgrid.retrieve.base import Lookup`) — a type alias:
`Callable[[str], Chunk | None]`. Given a `chunk_id` a `searcher` call already returned, hands
back the `Chunk` behind it — text, span, everything a chunk carries — or `None` if the id isn't
recognised. Backed by `BuiltPipeline.chunk_by_id()`, so it resolves an id to exactly what that
id would mean anywhere else in the pipeline: the *retrievable* chunk for a plain id, or the
wider *presentation* passage for an id a sibling-merge produced (see `Ingested.presentation` in
`ingest/base.py`). There is no way to enumerate or browse through it — a strategy can only look
up an id it already has, which is what keeps `Lookup` from being the index in disguise.

Defaults to a function that always returns `None`, so a strategy that has no use for chunk
text — `simple`, `widened`, `decomposed`, `agentic` — doesn't need to know the parameter
exists, and every call site written before it did (including every strategy called directly in
a test) keeps compiling. `BuiltPipeline.search()` always passes a real one. `relevance-feedback`
(`retrieve/strategies.py`) is the strategy built on it: it searches once, reads the best hit's
text through `lookup`, pulls out the words that don't already appear in the question — the
rarest ones first, since a strategy that never sees the index has no real document frequencies
to weight by — and searches again with those added. See
[dimensions/retrieval.md](../dimensions/retrieval.md#relevance-feedback--read-the-best-hit-search-again)
for what that buys on a real corpus.

**`RetrievalTrace`** (`from contextgrid.retrieve.base import RetrievalTrace`) — what a strategy
actually did, for the columns a recall number can't carry.

| Field / member | Type | Notes |
|---|---|---|
| `searches` | `int` | default `0` |
| `model_calls` | `int` | default `0` |
| `queries` | `list[str]` | every query text actually searched with |
| `notes` | `dict[str, object]` | free-form; e.g. `SecondChanceRetrieval` (extending.md) sets `notes["widened"]` |
| `record_search(query)` | — | increments `searches`, appends to `queries` |
| `record_model_call(count=1)` | — | increments `model_calls` |
| `merge(other)` | — | folds another trace's counters and notes into this one |

`fuse(results: Sequence[Sequence[Scored]], k: int) -> list[Scored]` (`from
contextgrid.retrieve.base import fuse`) is the standard way to combine several ranked lists —
reciprocal rank fusion, not score averaging, because a cosine similarity from one query and one
from another are not on the same scale and averaging lets whichever query produced larger
magnitudes win a result it didn't earn.

See [extending.md](extending.md) for a `RetrievalStrategy` built and exercised from scratch.

## `Reranker`

> Reorders a candidate list using the query and the passage together.

```python
from collections.abc import Sequence
from contextgrid.core.documents import Chunk
from contextgrid.index.base import Scored
from contextgrid.rerank.base import Reranker


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
def rerank(self, query: str, candidates: Sequence[Chunk], k: int) -> list[Scored]: ...
```

| Member | Must guarantee |
|---|---|
| `name`, `version` | Cache key and provenance. |
| `rerank(query, candidates, k)` | Returns at most `k` results, best first. Unlike a retriever, a reranker sees the query and *every* passage together, so it can judge whether a passage answers **this** question rather than whether it's nearby in a vector space — which is why it helps, and why it costs more per candidate. |

`candidates` — how many results the retriever hands the reranker — is the parameter most
reranking advice omits, and where most of the effect lives: over the top 10 a reranker can only
reorder what was already found; over the top 100 it can rescue evidence ranked 47th. It's an
ordinary field on `Config` (`Config.candidates`, default `50`), not part of this protocol, so
every reranker gets the same knob for free.

`RERANKERS: Registry[Reranker]` lives in `from contextgrid.rerank import RERANKERS,
get_reranker`. `NoReranker` (the baseline every reranker must beat) and `LexicalOverlapReranker`
/ `MMRReranker` (model-free, in [`rerank/base.py`](../../src/contextgrid/rerank/base.py)) are
worth reading before writing a new one — see [extending.md](extending.md) for a worked example.

## `Generator`

> Turns a question and its context into an answer.

```python
from contextgrid.assemble.context import AssembledContext
from contextgrid.generate.answer import Answer, Generator


@property
def name(self) -> str: ...
def answer(self, question: str, context: AssembledContext) -> Answer: ...
```

| Member | Must guarantee |
|---|---|
| `name` | Provenance only — like `QueryTransform`, there is no `version` on this protocol. |
| `answer(question, context)` | Returns an `Answer`. Not required to call a model at all — `ExtractiveGenerator` (below) doesn't. |

### Supporting types

**`Answer`** (`from contextgrid.generate.answer import Answer`).

| Field / member | Type | Notes |
|---|---|---|
| `text` | `str` | |
| `prompt_tokens` | `int` | default `0` |
| `completion_tokens` | `int` | default `0` |
| `citations` | `tuple[int, ...]` | the `[N]` markers the answer cited, parsed out by the caller — see `LLMGenerator.answer` |
| `is_abstention` (property) | `bool` | `True` when the text matches one of a broad, deliberately over-inclusive set of refusal phrases (`_REFUSALS` in `generate/answer.py`) |

**`AssembledContext`** (`from contextgrid.assemble.context import AssembledContext,
ContextAssembler, Ordering`) — what the generator actually sees, built by `ContextAssembler` from
retrieved chunks (ordering, token budget, deduplication — none of it changes what was retrieved,
but it routinely changes whether a good retrieval produces the right answer).

| Field / member | Type | Notes |
|---|---|---|
| `text` | `str` | the rendered prompt context |
| `chunks` | `tuple[Chunk, ...]` | what survived ordering/budget/dedup, in the order rendered |
| `tokens` | `int` | |
| `dropped` | `int` | default `0` — chunks that didn't fit the budget |
| `duplicate_characters` | `int` | default `0` |
| `warnings` | `WarningLog` | |
| `used` (property) | `int` | `len(chunks)` |
| `characters` (property) | `int` | `len(text)` |

Two in-tree generators worth reading: `ExtractiveGenerator` (no model — returns the top passage
verbatim, the ceiling retrieval alone can reach) and `LLMGenerator` (needs an `LLM`, hence not in
the plain `GENERATORS: Registry[Generator]` — resolved via `get_generator(spec, llm)` from
`contextgrid.generate`, the same "model-backed things can't live in a spec-string-only registry"
pattern as `QueryTransform`).

`score_answer(item, answer, context, gold_chunks=())` → `AnswerScore`, and `GenerationReport`
(both `contextgrid.generate.answer`) turn a batch of `Answer`s into groundedness, citation
accuracy and abstention numbers without a second model — see the module docstring for why an LLM
judge was deliberately not the default.

## `LLM`

> Text in, text out. The narrowest interface in the package.

```python
from contextgrid.evalset.llm import LLM


@property
def name(self) -> str: ...
def complete(self, prompt: str, *, max_tokens: int = 512) -> str: ...
```

| Member | Must guarantee |
|---|---|
| `name` | Provenance — recorded wherever a model call is logged. |
| `complete(prompt, *, max_tokens=512)` | One call, one string back. Anything richer (structured output, function calling) is layered on top by the caller — `parse_json_reply()` (below) is how the model-backed `QueryTransform`s pull JSON out of a plain string reply. Must raise `LLMError` on failure rather than returning an empty string or `None` silently. |

**Import path, worth stating plainly:** `LLM` lives in `contextgrid.evalset.llm`, not
`contextgrid.llm`, `contextgrid.core.llm`, or `contextgrid.generate.llm`. Every module that needs
it imports from there — `transform/query.py`, `generate/answer.py`, `ingest/generated.py` all do
`from contextgrid.evalset.llm import LLM`.

Two implementations worth knowing:

- **`RecordingLLM`** (`from contextgrid.evalset.llm import RecordingLLM`) — returns scripted
  replies from a list and remembers every prompt it was asked. Not a mock hidden in the tests:
  it's how generation and query-transform code gets exercised without a network or an API key.
  `RecordingLLM(replies=["first reply", "second reply"])` pops one reply per call, falling back
  to `default=""` once the list is empty.
- **`LiteLLMChat`** (`from contextgrid.evalset.llm import LiteLLMChat`) — any chat model, through
  [litellm](https://github.com/BerriAI/litellm): `openai/gpt-4o-mini`,
  `anthropic/claude-sonnet-5`, `gemini/gemini-2.0-flash`, `ollama/llama3`, one call shape for all
  of them. `transport: Callable[[str, int], str] | None` replaces the network call entirely — set
  it to exercise anything that calls a model without a key or a network, the same idea as
  `RecordingLLM` but for code that specifically needs a `LiteLLMChat`.

`LLMS: Registry[LLM]` lives in the same module (`from contextgrid.evalset.llm import LLMS,
get_llm`). `parse_json_reply(reply: str) -> Any` pulls JSON out of a model's answer whether it's
fenced in triple-backticks, prefixed with prose, or both. `answerer_from(llm)` builds a
closed-book, guess-don't-decline answerer for the general-knowledge eval-set filter.
`batched(items, size)` is a plain sequence-chunking helper used when a caller needs to keep model
calls under a batch-size limit.

## `Metric`

> Scores one query: relevance judgements in, a ranked list in, one float out.

```python
from collections.abc import Mapping, Sequence
from contextgrid.score.base import Metric


@property
def name(self) -> str: ...
@property
def version(self) -> str: ...
def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float: ...
```

| Member | Must guarantee |
|---|---|
| `name`, `version` | Same role as everywhere else. `name` is also half of the metric's *reported* name — `evaluate()`/`per_query()` (`score/metrics.py`) call `.evaluate(...)` once per cut-off and build the `{name}@{k}`-shaped column themselves (`recall@5`, `weighted_recall@5`), rather than a metric choosing its own cut-offs. |
| `evaluate(judgements, ranked, k)` | `judgements` maps chunk id → grade, where `grade > 0` means relevant; the grade itself only matters to metrics that use it (`ndcg` does, `recall` doesn't). `ranked` is the retriever's ordered chunk ids for this query. **Must never raise** for empty or missing `judgements`/`ranked` — every built-in in `score/metrics.py` returns `0.0` rather than raising on an unjudged or empty query, and a custom metric is expected to do the same. |

**Why the registry lives in `score/base.py` and not `score/__init__.py`, unlike every other
family.** Every other family's registry (`CHUNKERS`, `RETRIEVERS`, `RERANKERS`, `EMBEDDERS`...)
is built in that family's `__init__.py`: it imports the concrete plugin classes and registers
them there, and nothing *inside* e.g. `chunk/fixed.py` ever needs to look another chunker up by
name. Metrics are different — `evaluate()` and `per_query()` in `score/metrics.py` have to
resolve a metric *by name* at call time, including custom ones they know nothing about at
import time, so they need the registry object itself, not just the ability to populate it.
Putting `METRICS` in `score/__init__.py` the way every other family does would make
`score/metrics.py` import its own package's `__init__.py` — which runs `metrics.py` in the first
place, a real import cycle, not a stylistic one. Living in `score/base.py`, a module
`metrics.py` already has no reason not to import, avoids it. `score/__init__.py` still does the
actual *registering* of the six built-ins, exactly like every other family's `__init__.py` does
for its plugins — it's only the registry object's *home* that differs.

```python
from contextgrid.score import METRICS, get_metric

print(f"METRICS lives in score/base.py; get_metric resolves through it: {'recall' in METRICS}")
built = get_metric("recall")
print(f"get_metric('recall'): {built.name!r} v{built.version}")
```

Output:

```
METRICS lives in score/base.py; get_metric resolves through it: True
get_metric('recall'): 'recall' v1
```

`get_metric(spec: str) -> Metric` and `METRICS: Registry[Metric]` are both re-exported from
`contextgrid.score` (`from contextgrid.score import METRICS, get_metric, Metric`) — the same
`get_metric` name pattern as every other family (`get_chunker`, `get_embedder`, ...), just
resolving through a registry object that happens to live one module over. Six built-ins ship in
`score/metrics.py` (`RecallMetric`, `PrecisionMetric`, `HitRateMetric`, `MRRMetric`, `MAPMetric`,
`NDCGMetric`) and register into `METRICS` from `score/__init__.py`.
`DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)` (`contextgrid.score.metrics`) is the default
set of cut-offs `evaluate()`/`per_query()` report at; `available_metrics()` (same module) lists
every registered name. See [extending.md](extending.md) for a `Metric` written, registered into
the real `METRICS`, and swept with `Runner` end to end.

## Pipeline helpers a plugin author reaches for

Not protocols — plain functions and one method on `BuiltPipeline` — but every one of them was a
place the docs previously said nothing and an implementer had to read source to find.

```python
from contextgrid.pipeline import BuiltPipeline, Config, build, resolve_evalset, build_qrels
```

| Name | Signature | Notes |
|---|---|---|
| `BuiltPipeline.chunk_by_id` | `chunk_by_id(self) -> dict[str, Chunk]` | **A method, not a property.** `pipeline.chunk_by_id()["some-id"]`, not `pipeline.chunk_by_id["some-id"]` — the latter raises `AttributeError: 'function' object has no attribute 'items'` (or `'__getitem__'`, depending on how it's used) three frames away from the actual mistake. Returns every chunk that can come back, including presentation passages from `ingested.presented_chunks`, merged in — reranking and generation both look chunks up here. |
| `resolve_evalset` | `resolve_evalset(evalset: EvalSet, parses: Mapping[str, ParsedDocument], resolver: AnchorResolver \| None = None) -> tuple[EvalSet, WarningLog]` | Lives in `contextgrid.pipeline`, **not** `contextgrid.score.anchor` or any `contextgrid.score.*` module, despite being about resolving gold evidence. Locates an eval set's evidence inside one configuration's parse; a no-op (returns the eval set unchanged) when every item already carries spans rather than anchors. |
| `build_qrels` | `build_qrels(evalset: EvalSet, chunks: Sequence[Chunk], resolver: SpanResolver \| None = None) -> tuple[Qrels, WarningLog]` | Also in `contextgrid.pipeline`, not `contextgrid.score.resolve` (where `SpanResolver` itself actually lives — `from contextgrid.score.resolve import SpanResolver`). Turns span-level gold into chunk-level judgements for one chunking. |
| `Config` | `Config(parser="markdown", chunker="recursive:512", embedder="tfidf", index="dense", transform=None, retrieval=None, reranker=None, k=10, candidates=50, ingestion=None, generator=None)` | One point in the grid, in `contextgrid.pipeline`. Field order is public API — `ingestion` and `generator` were added last specifically so positional `Config("markdown", "recursive:512", ...)` calls already written wouldn't silently shift. |
| `build` | `build(config: Config, corpus: Corpus, *, cache: Cache \| None = None, stats: CacheStats \| None = None, llm: LLM \| None = None) -> BuiltPipeline` | Runs one configuration's indexing side end to end: parse, chunk, ingest, embed, index. |

`BuiltPipeline.search(query, k=None) -> list[str]` and `.answer(question, chunk_ids) ->
tuple[Answer, AssembledContext]` are the two methods that actually exercise a built
configuration at query time — see `pipeline.py` directly for how they compose `transform`,
`retrieval`, `reranker` and `generator` in order.

## The pattern across all twelve

Every protocol above shares a spine: `name` for provenance (plus `version` on everything except
`QueryTransform` and `Generator`, which have nothing that needs cache-invalidating), a small
number of verb methods that do the plugin's actual work, and — for the model-cost-aware ones
(`IngestionStrategy`, `RetrievalStrategy`) — a `uses_model` flag that lets the runner reason about
cost and boundedness without knowing anything about what the strategy does internally. A new
plugin family that doesn't exist yet would follow the same shape: `name`, `version` if there's
anything worth cache-keying, one or two verb methods, and honest reporting of whatever it costs.

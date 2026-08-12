# Extending: seven new plugins, worked end to end

This walks through writing, registering, and testing a new plugin for seven of the twelve
plugin families in [protocols.md](protocols.md): `Chunker`, `RetrievalStrategy`, `Embedder`,
`IngestionStrategy`, `Reranker`, `QueryTransform`, and `Metric`. Every code block below actually
runs — output is pasted from `.venv/bin/python`, not typed by hand. To reproduce it yourself:

```bash no-run: a template -- path/to/the/script.py is a placeholder, not a real file
PYTHONPATH=. .venv/bin/python path/to/the/script.py
```

(`PYTHONPATH=.` because the chunker and retrieval-strategy scripts import from `tests.support`
and `tests.conformance.test_chunker_conformance` directly, the same modules the real test suite
uses — see [Why direct function calls instead of `pytest`](#why-direct-function-calls-instead-of-pytest)
at the end. The other five examples don't touch `tests/` at all.)

Every example registers into a **local** `Registry`, not the real registry
(`CHUNKERS`/`RETRIEVERS`/`EMBEDDERS`/`INGESTERS`/`RERANKERS`/`TRANSFORMS`) in the plugin's own
package, so this doc doesn't need to touch anything under `src/`. Registering for real is the
same call — in practice you'd add `CHUNKERS.register("paragraph", doc="...")(ParagraphChunker)`
to `chunk/__init__.py`, and the equivalent line to the `__init__.py` of whichever family you're
extending, next to the other in-tree plugins. **`Metric` is the one exception** — see
[Part 7](#part-7-a-new-metric) for why it has to register into the real `METRICS`.

## Part 1: a new `Chunker`

### The plugin

The simplest structural chunker there is: one chunk per paragraph, splitting on blank lines. No
size limit, no overlap — it exists to show the shape every chunker has, not to be a good idea
on a real corpus.

```python
import re
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.chunk.base import ChunkBuilder, trim_range
from contextgrid.core.documents import Chunk, ParsedDocument

_BLANK_LINE = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class ParagraphChunker:
    """One chunk per paragraph, splitting on blank lines."""

    name: ClassVar[str] = "paragraph"
    version: ClassVar[str] = "1"

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.text
        if not text.strip():
            return []

        builder = ChunkBuilder(parsed)
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for match in _BLANK_LINE.finditer(text):
            start, end = trim_range(text, cursor, match.start())
            if end > start:
                ranges.append((start, end))
            cursor = match.end()
        start, end = trim_range(text, cursor, len(text))
        if end > start:
            ranges.append((start, end))

        return builder.build_all(ranges)
```

Everything the offset invariant needs is handled by `ChunkBuilder.build_all()` — exact
character ranges, per-tokenizer sizes, heading-path metadata, and inheriting `offsets_exact`
from the parse. `trim_range()` (`chunk/base.py`) just shrinks a range past surrounding
whitespace without ever inverting it, so a paragraph followed by three blank lines doesn't
produce a chunk starting or ending on whitespace. This chunker never *rewrites* text, so it
never has to set `offsets_exact=False` itself — it inherits whatever the parse says.

### Registering it

```python
from contextgrid.core.protocols import Chunker
from contextgrid.core.registry import Registry

local_chunkers: Registry[Chunker] = Registry(family="chunker")
local_chunkers.register("paragraph", doc="One chunk per paragraph.")(ParagraphChunker)

built = local_chunkers.create("paragraph")
print(f"built via spec string: {built.name!r} v{built.version}")
assert isinstance(built, Chunker)
print(f"isinstance(built, Chunker): {isinstance(built, Chunker)}")
```

Output:

```
built via spec string: 'paragraph' v1
isinstance(built, Chunker): True
```

`Registry.register()` is a decorator (see [registry.md](registry.md)); `create("paragraph")`
resolves the spec string, finds no parameters after the name, and calls `ParagraphChunker()`.

### Passing it through the conformance suite

`ALL_CHUNKERS` in `tests/support.py` is a fixed list — the normal way to conformance-test an
in-tree chunker is adding a `ChunkerCase` entry there (see
[conformance.md](conformance.md#running-a-new-plugin-through-conformance-without-touching-testssupportpy)).
For a self-contained doc example, the individual `test_*` functions are called directly
instead — they're plain functions taking `(case, parsed)`, nothing pytest-specific about
calling them by hand:

```python
from contextgrid.parse import MarkdownParser
from tests.conformance.test_chunker_conformance import (
    test_a_document_shorter_than_one_chunk_still_produces_one,
    test_carries_the_heading_path_where_there_is_one,
    test_chunk_ids_are_unique,
    test_chunking_twice_gives_the_same_chunks,
    test_chunks_are_in_reading_order,
    test_chunks_cover_every_non_whitespace_character,
    test_chunks_stay_inside_the_document,
    test_empty_document_produces_no_chunks,
    test_every_chunk_is_a_literal_slice_of_the_document,
    test_ids_are_stable_across_instances,
    test_inherits_exactness_from_the_parse,
    test_no_chunk_contains_another,
    test_no_chunk_is_empty,
    test_records_a_token_count_under_a_named_tokenizer,
)
from tests.support import CONTENTFUL_SOURCES, ChunkerCase

case = ChunkerCase("paragraph", ParagraphChunker())
parser = MarkdownParser()

checks = [
    test_every_chunk_is_a_literal_slice_of_the_document,
    test_chunks_stay_inside_the_document,
    test_no_chunk_is_empty,
    test_chunks_are_in_reading_order,
    test_chunk_ids_are_unique,
    test_no_chunk_contains_another,
    test_chunks_cover_every_non_whitespace_character,
    test_records_a_token_count_under_a_named_tokenizer,
    test_inherits_exactness_from_the_parse,
    test_chunking_twice_gives_the_same_chunks,
    test_ids_are_stable_across_instances,
]

for source_file in CONTENTFUL_SOURCES:
    parsed = parser.parse(source_file)
    for check in checks:
        check(case, parsed)
    print(f"conformance ok on {source_file.id!r}: {len(built.chunk(parsed))} chunks")

test_carries_the_heading_path_where_there_is_one(case)
print("conformance ok: heading path is carried")

test_a_document_shorter_than_one_chunk_still_produces_one(case)
print("conformance ok: short document still produces one chunk")

test_empty_document_produces_no_chunks(case, "")
test_empty_document_produces_no_chunks(case, "   \n\n \t ")
print("conformance ok: empty and whitespace-only documents produce no chunks")

print()
print("ALL CONFORMANCE CHECKS PASSED for ParagraphChunker")
```

Real output:

```
conformance ok on 'contract': 11 chunks
conformance ok on 'api-docs': 11 chunks
conformance ok on 'prose': 3 chunks
conformance ok on 'short': 1 chunks
conformance ok: heading path is carried
conformance ok: short document still produces one chunk
conformance ok: empty and whitespace-only documents produce no chunks

ALL CONFORMANCE CHECKS PASSED for ParagraphChunker
```

Every check from [conformance.md](conformance.md)'s table passed for real, against the same
sample documents (`tests/support.py:CONTENTFUL_SOURCES`) every in-tree chunker is tested
against.

## Part 2: a new `RetrievalStrategy`

### The idea

`SimpleRetrieval` does one search. `WidenedRetrieval` always does one search but asks the index
for more results than needed. `SecondChanceRetrieval` sits between them: search once at the
requested depth, and only pay for a second, wider search when the *first* one looks weak (its
top score is below `threshold`). Most of the recall `widened` buys, without doubling every
query's cost — only the questions that actually struggle pay for it.

```python
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.index.base import Scored
from contextgrid.retrieve.base import RetrievalTrace, Searcher, fuse


@dataclass(frozen=True, slots=True)
class SecondChanceRetrieval:
    """Search once. If the top result looks weak, search again, wider."""

    threshold: float = 0.3
    factor: int = 4

    name: ClassVar[str] = "second-chance"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def retrieve(
        self,
        query: str,
        queries: Sequence[str],
        searcher: Searcher,
        k: int,
        trace: RetrievalTrace,
    ) -> list[Scored]:
        del query
        first: list[Sequence[Scored]] = []
        for text in queries:
            trace.record_search(text)
            first.append(searcher(text, k))

        weak = all(not hits or hits[0].score < self.threshold for hits in first)
        if not weak:
            trace.notes["widened"] = False
            return fuse(first, k)

        trace.notes["widened"] = True
        depth = k * self.factor
        second: list[Sequence[Scored]] = []
        for text in queries:
            trace.record_search(text)
            second.append(searcher(text, depth))
        return fuse(second, k)
```

This satisfies `RetrievalStrategy` (see [protocols.md](protocols.md)): `name`, `version`,
`uses_model = False` (it never calls a model — only the index, harder or softer), and
`retrieve()` with exactly the signature the protocol declares. Every search — first pass and
second — is recorded on `trace` via `trace.record_search()`, which is what lets the leaderboard
show that this strategy costs more searches on hard questions without costing a single model
call, ever.

### Registering it

```python
from contextgrid.core.registry import Registry
from contextgrid.retrieve.base import RetrievalStrategy

local_retrievers: Registry[RetrievalStrategy] = Registry(family="retrieval")
local_retrievers.register(
    "second-chance", shorthand="threshold", doc="Widen the search only when the top hit is weak."
)(SecondChanceRetrieval)

built = local_retrievers.create("second-chance:0.5")
print(f"built via spec string: {built.name!r} v{built.version} threshold={built.threshold}")
assert isinstance(built, RetrievalStrategy)
print(f"isinstance(built, RetrievalStrategy): {isinstance(built, RetrievalStrategy)}")
```

Output:

```
built via spec string: 'second-chance' v1 threshold=0.5
isinstance(built, RetrievalStrategy): True
```

`"second-chance:0.5"` resolves to `threshold=0.5` because `shorthand="threshold"` was declared
at registration — the same bare-value-in-first-position rule `RecursiveChunker`'s `"512"` uses
(see [registry.md](registry.md)).

### Exercising it

There is no conformance suite for `RetrievalStrategy` yet (see
[conformance.md](conformance.md)) — the existing built-in strategies are tested the way
`tests/unit/test_retrieve.py` does it: a fake index that records what it was asked for, so the
strategy can be checked without a real store. `Searcher` is just a `(text, k) -> Sequence[Scored]`
callable — a strategy never sees the index itself, so a plain class with a `__call__` is a
complete stand-in for a real one.

```python
class FakeIndex:
    def __init__(self, ranking: dict[str, list[tuple[str, float]]]) -> None:
        self.ranking = ranking
        self.calls: list[tuple[str, int]] = []

    def __call__(self, text: str, k: int) -> Sequence[Scored]:
        self.calls.append((text, k))
        hits = self.ranking.get(text, [])
        return [Scored(chunk_id, score) for chunk_id, score in hits[:k]]


print("-- case 1: a confident top hit --")
confident = FakeIndex({"q": [("a", 0.91), ("b", 0.4), ("c", 0.2)]})
trace = RetrievalTrace()
result = SecondChanceRetrieval(threshold=0.3).retrieve("q", ["q"], confident, 2, trace)
print(f"results: {[(r.chunk_id, r.score) for r in result]}")
print(f"searches: {trace.searches}, widened: {trace.notes['widened']}")
assert trace.searches == 1
assert trace.notes["widened"] is False


class WideningFakeIndex(FakeIndex):
    """A real index returns hits already ranked -- `fuse` trusts that ordering rather than
    re-sorting, so a fake index has to keep the same contract."""

    def __call__(self, text: str, k: int) -> Sequence[Scored]:
        self.calls.append((text, k))
        if k <= 2:
            return [Scored("a", 0.1), Scored("b", 0.05)]
        return [Scored("c", 0.44), Scored("d", 0.3), Scored("a", 0.1), Scored("b", 0.05)]


print()
print("-- case 2: a weak top hit triggers a second, wider search --")
widening = WideningFakeIndex({})
trace2 = RetrievalTrace()
result2 = SecondChanceRetrieval(threshold=0.3, factor=4).retrieve("q", ["q"], widening, 2, trace2)
print(f"results: {[(r.chunk_id, r.score) for r in result2]}")
print(f"searches: {trace2.searches}, widened: {trace2.notes['widened']}")
print(f"depths asked for: {[k for _, k in widening.calls]}")
assert trace2.searches == 2
assert trace2.notes["widened"] is True
assert widening.calls[-1][1] == 8
```

Real output:

```
-- case 1: a confident top hit --
results: [('a', 0.91), ('b', 0.4)]
searches: 1, widened: False

-- case 2: a weak top hit triggers a second, wider search --
results: [('c', 0.44), ('d', 0.3)]
searches: 2, widened: True
depths asked for: [2, 8]
```

Case 2 is the point of the strategy: at depth 2 the index only has `a` (0.1) and `b` (0.05) —
both weak. `SecondChanceRetrieval` notices the top score is below `threshold=0.3`, searches
again at `depth = k * factor = 2 * 4 = 8`, and finds `c` (0.44) and `d` (0.3) — genuinely
better evidence that a plain `SimpleRetrieval` at k=2 would never have seen. `trace.searches ==
2` and `trace.notes["widened"] is True` are exactly the columns a leaderboard would need to show
this strategy cost one extra search on this question and no model calls at all.

## Part 3: a new `Embedder`

### The idea

`embed_documents` and `embed_queries` are two separate methods on purpose (see
[protocols.md](protocols.md#embedder)): real models like E5 want a `query:`/`passage:` prefix on
the text before encoding, and an embedder that routes both through the same code silently gets
worse numbers with nothing pointing at why. `PrefixedHashEmbedder` fakes that asymmetry with a
hashed bag of words — no model, no network — so the split is visible without downloading
anything.

```python
import re
import zlib
from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from contextgrid.embed.base import Embedder, EmbeddingResult, Vectors, normalise, truncate

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class PrefixedHashEmbedder:
    """A hashed bag-of-words embedder that prefixes queries and documents differently.

    Demonstrates the one thing every real embedder has to get right: `embed_queries` and
    `embed_documents` are not the same method under two names. Real models like E5 prepend
    `query:` or `passage:` to the text before encoding -- same vocabulary, same vector space,
    a different marker token. This does the same thing: shared words still land in the same
    hash buckets (so a query and its matching document are still close), but the prefix
    itself occupies a bucket too, exactly the way a real prefixed model behaves.
    """

    dimensions: int = 4096
    max_tokens: int | None = None

    name: ClassVar[str] = "prefixed-hash"
    version: ClassVar[str] = "1"
    normalised: ClassVar[bool] = True

    def prepare(self, documents: Sequence[str]) -> None:
        return None

    def _encode(self, texts: Sequence[str], *, prefix: str) -> Vectors:
        # zlib.crc32 rather than the builtin hash(): PYTHONHASHSEED randomises str hashing
        # per process, which would make every bucket -- and this example's assertions --
        # different on every run.
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _WORD.findall(f"{prefix} {text}".lower())
            for word, count in Counter(words).items():
                bucket = zlib.crc32(word.encode()) % self.dimensions
                vectors[row, bucket] += count
        return normalise(vectors)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        cut, log, truncated = truncate(texts, self.max_tokens, model=self.name)
        return EmbeddingResult(
            vectors=self._encode(cut, prefix="passage:"),
            warnings=log,
            input_tokens=sum(len(text.split()) for text in cut),
            truncated=truncated,
        )

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        cut, log, truncated = truncate(texts, self.max_tokens, model=self.name)
        return EmbeddingResult(
            vectors=self._encode(cut, prefix="query:"),
            warnings=log,
            input_tokens=sum(len(text.split()) for text in cut),
            truncated=truncated,
        )
```

`truncate()` is the helper every real embedder reaches for to cut text to `max_tokens` and say so
(`INPUT_TRUNCATED` on the returned `EmbeddingResult.warnings`) rather than silently dropping the
paragraph that held the answer. `normalise()` scales rows to unit length without dividing by zero
on an all-zero row.

### Registering and exercising it

```python
built = PrefixedHashEmbedder()
assert isinstance(built, Embedder)
print(f"isinstance(built, Embedder): {isinstance(built, Embedder)}")

docs = [
    "Either party may terminate this agreement for convenience by giving thirty days notice.",
    "Setup fee for the premium plan is due upon signature.",
]
result = built.embed_documents(docs)
print(f"embed_documents: count={result.count} dimensions={result.dimensions}")

queries = built.embed_queries(["How do I terminate the agreement?"])
sims = queries.vectors @ result.vectors.T
print(f"cosine to doc 0 (termination): {sims[0, 0]:.3f}")
print(f"cosine to doc 1 (fees):        {sims[0, 1]:.3f}")
assert sims[0, 0] > sims[0, 1]

# embed_queries and embed_documents prefix the text differently before hashing, so running
# the same string through the wrong method produces a different vector, not the same one --
# exactly the mistake this protocol's two-method split exists to make impossible to make
# silently.
wrong_side = built.embed_documents(["How do I terminate the agreement?"])
same_side = built.embed_queries(["How do I terminate the agreement?"])
assert not np.allclose(wrong_side.vectors, same_side.vectors)
print("embed_documents(query text) != embed_queries(query text): confirmed")

# truncation: an embedder with a small max_tokens cuts anything longer, and says so.
short_context = PrefixedHashEmbedder(dimensions=64, max_tokens=8)
long_doc = " ".join(f"word{i}" for i in range(40))
truncated_result = short_context.embed_documents([long_doc])
print(f"truncated: {truncated_result.truncated}, warnings: {truncated_result.warnings.summary()}")
assert truncated_result.truncated == 1
```

Real output:

```
isinstance(built, Embedder): True
embed_documents: count=2 dimensions=4096
cosine to doc 0 (termination): 0.202
cosine to doc 1 (fees):        0.114
embed_documents(query text) != embed_queries(query text): confirmed
truncated: 1, warnings: input_truncated x1
```

The query is closer to the document that actually shares its words ("terminate", "agreement")
than the one that shares only "the" — normal, expected bag-of-words behaviour, and worth seeing
work end to end before trusting the same shape against a real model. `zlib.crc32` is used instead
of the builtin `hash()` deliberately: `hash()` on strings is salted per process by
`PYTHONHASHSEED`, which would make the bucket assignments — and therefore these numbers — change
on every run. (The in-tree `HashEmbedder` in
[`embed/local.py`](../../src/contextgrid/embed/local.py) does use the builtin `hash()`, which is
fine for its purpose — nothing compares its output across processes — but would make a poor
choice for a doc example whose asserts have to hold every time `check-docs.sh` runs it.)

## Part 4: a new `IngestionStrategy`

### The idea

An `IngestionStrategy` decides what's indexed and what a hit on it returns
([protocols.md](protocols.md#ingestionstrategy)) — for plain chunking those are the same list.
`HeadingPrefixIngestion` is a free, model-free cousin of the in-tree `contextual` strategy
(`ingest/generated.py:ContextualIngestion`, which prepends an LLM-written note on where a chunk
sits): instead of asking a model, it reuses the heading path the parser already extracted and
`ChunkBuilder` already carried onto `Chunk.meta["heading_path"]`. What gets embedded and searched
is the heading-prefixed text; what a hit turns into is the original, unprefixed chunk.

```python
from dataclasses import dataclass, replace
from typing import ClassVar

from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.ingest.base import Ingested, IngestionContext, IngestionStrategy
from contextgrid.parse import MarkdownParser
from contextgrid.chunk import get_chunker

CONTRACT = """\
# Master Services Agreement

## 1. Term

This agreement begins on the Effective Date and continues for twelve months.

## 2. Termination

### 2.1 Notice period

Either party may terminate this agreement for convenience by giving thirty days
written notice.

### 2.2 Termination for cause

A party may terminate immediately if the other party commits a material breach.
"""


@dataclass(frozen=True, slots=True)
class HeadingPrefixIngestion:
    """Index each chunk with its heading path prepended; return the chunk unprefixed."""

    separator: str = " > "

    name: ClassVar[str] = "heading-prefix"
    version: ClassVar[str] = "1"
    uses_model: ClassVar[bool] = False

    def ingest(self, chunks, context: IngestionContext) -> Ingested:
        del context
        indexed = []
        parent_of: dict[str, str] = {}
        for chunk in chunks:
            path = chunk.meta.get("heading_path")
            if not path:
                indexed.append(chunk)
                continue
            prefix = self.separator.join(path) + ": "
            prefixed_id = f"{chunk.id}:prefixed"
            # The indexed text is no longer a literal slice of the parse -- it has a prefix
            # glued on -- so this chunk must say so. The chunk being *returned* to the caller
            # (in `retrievable`, below) is the original, untouched one and keeps its real
            # offsets.
            indexed.append(
                replace(chunk, id=prefixed_id, text=prefix + chunk.text, offsets_exact=False)
            )
            parent_of[prefixed_id] = chunk.id
        return Ingested(indexed=indexed, retrievable=list(chunks), parent_of=parent_of)
```

### Registering and exercising it

```python
built = HeadingPrefixIngestion()
assert isinstance(built, IngestionStrategy)
print(f"isinstance(built, IngestionStrategy): {isinstance(built, IngestionStrategy)}")

parser = MarkdownParser()
parsed = parser.parse(
    SourceFile(id="contract", media_type=MediaType.MARKDOWN, raw=CONTRACT.encode())
)
chunker = get_chunker("recursive:20")
chunks = chunker.chunk(parsed)
print(f"chunker produced {len(chunks)} chunks")

ingested = built.ingest(chunks, IngestionContext())
print(f"indexed: {len(ingested.indexed)}, retrievable: {len(ingested.retrievable)}")

for chunk in ingested.indexed[:3]:
    print(f"  indexed {chunk.id!r} offsets_exact={chunk.offsets_exact}")
    print(f"    text: {chunk.text[:70]!r}")

# every indexed chunk resolves back to a retrievable chunk with the plain, un-prefixed text
by_id = {chunk.id: chunk for chunk in chunks}
retrievable_ids = {c.id for c in ingested.retrievable}
for indexed_chunk in ingested.indexed:
    resolved = ingested.resolve(indexed_chunk.id)
    assert resolved in retrievable_ids
    original = by_id[resolved]
    assert indexed_chunk.text.endswith(original.text)
    assert indexed_chunk.text != original.text  # the prefix really was added

print()
print(f"expansion: {ingested.expansion:.2f} (indexed chunks per retrievable chunk)")
assert ingested.expansion == 1.0  # one indexed unit per retrievable unit -- only the text changed
print("resolve() maps every prefixed id back to its plain chunk: confirmed")
```

Real output:

```
isinstance(built, IngestionStrategy): True
chunker produced 6 chunks
indexed: 6, retrievable: 6
  indexed 'contract:0-39:prefixed' offsets_exact=False
    text: '# Master Services Agreement: # Master Services Agreement\n\n## 1. Term'
  indexed 'contract:33-136:prefixed' offsets_exact=False
    text: '# Master Services Agreement > ## 1. Term: . Term\n\nThis agreement begin'
  indexed 'contract:123-159:prefixed' offsets_exact=False
    text: '# Master Services Agreement > ## 2. Termination: . Termination\n\n### 2.'

expansion: 1.00 (indexed chunks per retrievable chunk)
resolve() maps every prefixed id back to its plain chunk: confirmed
```

`expansion == 1.0` here because this strategy doesn't multiply the number of indexed units the
way `hypothetical-questions` does (four questions indexed per chunk, `expansion == 4.0`) — it
only rewrites the text of each one, which is exactly why `offsets_exact=False` is the field that
matters here rather than `Ingested.presentation` (which is for strategies that hand back a wider
passage than what was indexed — see `ParentDocumentIngestion` and
[protocols.md](protocols.md#ingestionstrategy) for that trap).

## Part 5: a new `Reranker`

### The idea

`HeadingMatchReranker` blends two signals: the retriever's own rank (an untouched candidate keeps
its position) and how many of the query's words appear in the chunk's heading path
(`Chunk.meta["heading_path"]` again). Genuinely useful on structured documents, where the section
title is often more informative than whatever score the retriever attached to the body text.

```python
import re
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.index.base import Scored, top_k
from contextgrid.rerank.base import Reranker

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class HeadingMatchReranker:
    """Boosts a candidate whose heading path mentions a word from the query."""

    boost: float = 3.0

    name: ClassVar[str] = "heading-match"
    version: ClassVar[str] = "1"

    def rerank(self, query: str, candidates, k: int) -> list[Scored]:
        terms = {word.lower() for word in _WORD.findall(query)}
        scores: dict[str, float] = {}
        for position, chunk in enumerate(candidates):
            base = len(candidates) - position  # the incoming rank, as a score
            path_words = {
                word.lower()
                for text in chunk.meta.get("heading_path", ())
                for word in _WORD.findall(text)
            }
            matches = len(terms & path_words)
            scores[chunk.id] = base + self.boost * matches
        return top_k(scores, k)
```

### Registering and exercising it

```python
built = HeadingMatchReranker()
assert isinstance(built, Reranker)
print(f"isinstance(built, Reranker): {isinstance(built, Reranker)}")

parser2 = MarkdownParser()
parsed2 = parser2.parse(
    SourceFile(
        id="contract2",
        media_type=MediaType.MARKDOWN,
        raw=(
            CONTRACT
            + "\n## 3. Fees\n\nThe monthly fee for the standard plan is one thousand two hundred dollars.\n"
        ).encode(),
    )
)
chunks2 = get_chunker("recursive:20").chunk(parsed2)
by_id2 = {chunk.id: chunk for chunk in chunks2}


def under(chunk, heading: str) -> bool:
    return any(heading in text for text in chunk.meta.get("heading_path", ()))


# Pretend the retriever ranked the fee clause first, purely on lexical noise -- the "notice
# period" clause that actually answers the question came back fourth.
fee_chunk = next(c for c in chunks2 if under(c, "3. Fees"))
notice_chunk = next(
    c for c in chunks2 if under(c, "2.1 Notice period") and "Either party" in c.text
)
other_chunks = [c for c in chunks2 if c.id not in {fee_chunk.id, notice_chunk.id}]
candidates = [fee_chunk, *other_chunks[:2], notice_chunk]
print("retriever order:", [c.meta["heading_path"][-1] for c in candidates])

query = "What is the notice period for termination?"
reranked = built.rerank(query, candidates, k=4)
reranked_headings = [by_id2[s.chunk_id].meta["heading_path"][-1] for s in reranked]
print("reranked order: ", reranked_headings)

assert reranked[0].chunk_id == notice_chunk.id
print(
    f"top result after rerank: {reranked_headings[0]!r} (was rank {candidates.index(notice_chunk) + 1})"
)
```

Real output:

```
isinstance(built, Reranker): True
retriever order: ['## 3. Fees', '# Master Services Agreement', '## 1. Term', '### 2.1 Notice period']
reranked order:  ['### 2.1 Notice period', '## 3. Fees', '# Master Services Agreement', '## 1. Term']
top result after rerank: '### 2.1 Notice period' (was rank 4)
```

The notice-period clause was ranked last by the (simulated) retriever and first after reranking,
because its heading path — `"## 2. Termination" > "### 2.1 Notice period"` — shares three words
with the query ("notice", "period", "termination") that no other candidate's heading path does.
`top_k()` (`contextgrid.index.base`) is the same deterministic, tie-broken-by-id sort every
`Index` and `Reranker` in the package ends its scoring with.

## Part 6: a new `QueryTransform`

### The idea

`SynonymExpand` sits between `NoTransform` (free, stuck with whatever vocabulary mismatch the
corpus has) and the model-backed transforms like `MultiQuery` (a model call and several searches
on *every* query, forever): a fixed synonym table, applied with no model at all, that doubles
`fan_out` only on the queries the table actually touches.

```python
from dataclasses import dataclass, field
from typing import ClassVar

from contextgrid.core.registry import Registry
from contextgrid.transform.query import QueryTransform, TransformedQuery


@dataclass(frozen=True, slots=True)
class SynonymExpand:
    """Search with the question as asked, plus a copy with known synonyms substituted."""

    synonyms: dict[str, str] = field(default_factory=dict)

    name: ClassVar[str] = "synonym-expand"

    def transform(self, query: str) -> TransformedQuery:
        words = query.split()
        substituted = " ".join(self.synonyms.get(word.lower().strip("?.,"), word) for word in words)
        if substituted == query:
            return TransformedQuery(original=query, queries=(query,))
        return TransformedQuery(original=query, queries=(query, substituted))
```

### Registering and exercising it

```python
local_transforms: Registry[QueryTransform] = Registry(family="transform")
local_transforms.register("synonym-expand", doc="Add a synonym-substituted copy of the query.")(
    SynonymExpand
)

default_built = local_transforms.create("synonym-expand")
print(f"built via spec string: {default_built.name!r}, synonyms={default_built.synonyms}")
assert isinstance(default_built, QueryTransform)
print(f"isinstance(default_built, QueryTransform): {isinstance(default_built, QueryTransform)}")

# the registry builds it with an empty table by default; a real config would pass one --
# constructed directly here, the same object `Registry.create(..., synonyms={...})` would hand
# back.
built = SynonymExpand(synonyms={"terminate": "end", "notice": "warning"})

result = built.transform("How do I terminate this agreement?")
print(f"queries: {result.queries}")
print(f"fan_out: {result.fan_out}, is_identity: {result.is_identity}")
assert result.fan_out == 2
assert result.queries[0] == "How do I terminate this agreement?"
assert "end" in result.queries[1]

unmatched = built.transform("What is the fee schedule?")
print(f"unmatched query: {unmatched.queries}, fan_out: {unmatched.fan_out}")
assert unmatched.is_identity
assert unmatched.fan_out == 1
```

Real output:

```
built via spec string: 'synonym-expand', synonyms={}
isinstance(default_built, QueryTransform): True
queries: ('How do I terminate this agreement?', 'How do I end this agreement?')
fan_out: 2, is_identity: False
unmatched query: ('What is the fee schedule?',), fan_out: 1
```

`TransformedQuery.fan_out` is what a leaderboard's cost panel reads to say a transform costs "N
searches per question" (`describe_cost()` in `transform/query.py`) — here it's 2 only when the
table actually rewrote something, 1 (the identity case) otherwise, which is the honest number for
a transform that doesn't touch every query the same way. Unlike the four model-backed transforms
(`HyDE`, `MultiQuery`, `Decompose`, `StepBack`), `SynonymExpand` needs no `LLM` to build, so it
registers into an ordinary `Registry` and resolves from a plain spec string — no `MODEL_BACKED`
special-casing required.

## Part 7: a new `Metric`

### The idea

Plain `recall_at_k` (`score/metrics.py`) treats every relevant chunk the same regardless of
grade — finding the one chunk graded "fully answers" and missing two graded "partially
relevant" scores identically to finding all three, as long as the *count* of chunks found
matches. `WeightedRecall` answers a different question: what fraction of a query's total
relevance — grades summed, not chunks counted — actually landed in the top k.

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.score.base import Metric


@dataclass(frozen=True, slots=True)
class WeightedRecall:
    """Recall, weighted by grade rather than by chunk count."""

    name: ClassVar[str] = "weighted_recall"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        total = sum(grade for grade in judgements.values() if grade > 0)
        if total == 0:
            return 0.0
        top = set(ranked[:k])
        found = sum(grade for cid, grade in judgements.items() if grade > 0 and cid in top)
        return found / total
```

`name`, `version`, `evaluate(judgements, ranked, k) -> float` — the same three things every
metric in `score/metrics.py` has, just not wrapped in a module-level function. `k` is handled
by the caller, not the metric: `evaluate()`/`per_query()` (`score/metrics.py`) call
`.evaluate(...)` once per cut-off and build the `weighted_recall@5`-shaped name themselves, the
same as `recall@5`.

### Registering it — into the real `METRICS`, unlike every plugin above

Every other family in this doc registers into a **local** `Registry` (see the intro) so the
example doesn't touch real, shared state. A metric can't take that path and still prove
anything: `Runner.run_one` (`grid/runner.py`) calls the module-level `evaluate()` and
`per_query()` in `score/metrics.py`, and those resolve a metric name through exactly one
registry — `contextgrid.score.METRICS`. A metric that only exists in a local `Registry` is
invisible to `run.headline`, `run.metrics`, and every sweep, which defeats the entire point of
sweeping with a custom one. So this is the one family here that registers for real:

```python
from contextgrid.score import METRICS, Metric

if "weighted_recall" not in METRICS:
    METRICS.register("weighted_recall", doc="Recall weighted by grade, not by chunk count.")(
        WeightedRecall
    )

built = METRICS.create("weighted_recall")
print(f"registered: {built.name!r} v{built.version}")
print(f"isinstance(built, Metric): {isinstance(built, Metric)}")
```

Real output:

```
registered: 'weighted_recall' v1
isinstance(built, Metric): True
```

`if "weighted_recall" not in METRICS` guards against re-registering — `Registry.register` (and
therefore `METRICS.register`) raises if the name is already taken (see
[registry.md](registry.md)), which matters here because, unlike a local `Registry` created
fresh for the example, `METRICS` is shared and this file might run more than once in the same
process.

### A real sweep: `run.headline`, the leaderboard, and `RunResult.has()`

Two chunkers over a two-document corpus, one question whose evidence is split across a
strongly-relevant quote (grade 2) and a weakly-relevant one (grade 1), scored on
`weighted_recall@5` end to end through the ordinary `Runner`:

```python
from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.corpus import Corpus
from contextgrid.grid import Runner, matrix

CONTRACT = """\
# Master Services Agreement

## 2. Termination

### 2.1 Notice period

Either party may terminate this agreement for convenience by giving thirty days
written notice. Notice must be delivered to the address in Schedule A.

### 2.2 Termination for cause

A party may terminate immediately if the other party commits a material breach
and fails to remedy it within fifteen days of written notice.
"""

API_DOCS = """\
# Widget API

## Authentication

Every request needs an `X-Api-Key` header. Requests without one return 401.
"""

corpus = Corpus.from_texts(
    {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
)

evalset = EvalSet(
    id="es",
    items=(
        EvalItem(
            id="q1",
            question="How much notice is needed to terminate for convenience, and where must "
            "it be sent?",
            anchors=(
                GoldAnchor(source_id="contract.md", quote="thirty days\nwritten notice", grade=2),
                GoldAnchor(source_id="contract.md", quote="Schedule A", grade=1),
            ),
        ),
        EvalItem(
            id="q2",
            question="What happens on a material breach?",
            anchors=(
                GoldAnchor(
                    source_id="contract.md", quote="fifteen days of written notice", grade=2
                ),
            ),
        ),
        EvalItem(
            id="q3",
            question="Which header carries the API key?",
            anchors=(GoldAnchor(source_id="api.md", quote="X-Api-Key", grade=2),),
        ),
    ),
)

runner = Runner(corpus=corpus, headline="weighted_recall@5")
results = runner.run(
    matrix(chunker=["sentence:1", "fixed:20,overlap=0"], embedder="tfidf", k=1),
    evalset,
    mode="factorial",
)

for row in results.leaderboard("weighted_recall@5", extra=["recall@5"]):
    print(f"{row['config']:52} {row['weighted_recall@5']:6.3f} {row['recall@5']:6.3f}")

winner = results.best("weighted_recall@5")
print()
print(f"winner.has('weighted_recall@5'): {winner.has('weighted_recall@5')}")
print(f"winner.has('made_up_metric@5'): {winner.has('made_up_metric@5')}")
print(results.summary("weighted_recall@5"))
```

Real output:

```
markdown · fixed:20,overlap=0 · tfidf · dense         1.000  1.000
markdown · sentence:1 · tfidf · dense                 0.778  0.833

winner.has('weighted_recall@5'): True
winner.has('made_up_metric@5'): False
markdown · fixed:20,overlap=0 · tfidf · dense scored best on weighted_recall@5 at 1.000, across 2 configurations, scored on 3 questions. markdown · fixed:20,overlap=0 · tfidf · dense and markdown · sentence:1 · tfidf · dense are not distinguishable on this eval set (n=3). The gap of +0.222 on weighted_recall@5 sits inside the confidence interval +0.000 to +0.667, so it is consistent with no difference at all. About 80 questions would be needed to settle a gap this size. It runs locally at no cost per query, answering at under 1 ms p95.
```

`sentence:1` scores lower on `weighted_recall@5` (0.778) than on plain `recall@5` (0.833) —
the two metrics genuinely disagree, which is the point of writing a second one rather than
reusing the first. On `q1`, `sentence:1`'s one retrieved chunk holds only the grade-1
"Schedule A" evidence, not the grade-2 notice-period evidence: two relevant chunks exist for
that question and one was found, so plain `recall_at_k` — which counts *chunks*, not weight —
scores it 0.5. `weighted_recall` scores the same question 0.333, because the chunk it found
carries only 1 of the 3 grade-points available (2 + 1) for `q1`. `recall_at_k` says "found half
of what mattered"; `weighted_recall` says "found the smaller half" — and only the second one is
true here. That's the whole reason to write a metric instead of reading `recall_at_k` and
`ndcg_at_k` (which is graded too, but discounts by *rank* — a different question again) and
calling it close enough.

`Runner(headline="weighted_recall@5")` works with no other change anywhere — `RunConfig.validate`
(`config/schema.py`) checks a config's `run.headline` against `available_metrics()` the same
way, so `headline: weighted_recall@5` in a YAML config is equally valid once `weighted_recall`
is registered. `winner.has('weighted_recall@5')` is `True` because it was actually computed;
`winner.has('made_up_metric@5')` is `False` because nothing computed it — never `0.0` for a
metric nobody ran, which is what makes `RunResult.metric()` safe to feed into `composite()`
(see [composite.md](../scoring/composite.md)) without a silent zero corrupting the score.

## Why direct function calls instead of `pytest`

Both worked examples call test functions and assertions directly rather than shelling out to
`pytest`, so the exact commands above are reproducible with nothing but `.venv/bin/python` and
paste cleanly into a doc as ordinary Python output. The real workflow for a plugin you intend to
ship is different and described in [conformance.md](conformance.md): add it to
`tests/support.py:ALL_CHUNKERS` (or the equivalent for other families) and let `pytest` run the
whole suite, so every invariant added to the suite in the future covers it automatically without
anyone remembering to re-run it by hand.

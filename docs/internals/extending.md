# Extending: a new chunker and a new retrieval strategy, worked end to end

This walks through writing, registering, and testing two new plugins: a `Chunker` and a
`RetrievalStrategy`. Every code block below actually runs — output is pasted from
`.venv/bin/python`, not typed by hand. To reproduce it yourself:

```bash no-run: a template -- path/to/the/script.py is a placeholder, not a real file
PYTHONPATH=. .venv/bin/python path/to/the/script.py
```

(`PYTHONPATH=.` because the scripts import from `tests.support` and
`tests.conformance.test_chunker_conformance` directly, the same modules the real test suite
uses — see the last section for why.)

Both examples register into a **local** `Registry`, not the real `CHUNKERS` / `RETRIEVERS` in
`contextgrid.chunk` / `contextgrid.retrieve`, so this doc doesn't need to touch anything under
`src/`. Registering for real is the same call — in practice you'd add
`CHUNKERS.register("paragraph", doc="...")(ParagraphChunker)` to `chunk/__init__.py` and
`RETRIEVERS.register("second-chance", ...)(SecondChanceRetrieval)` to `retrieve/__init__.py`,
next to the other in-tree plugins.

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

## Why direct function calls instead of `pytest`

Both worked examples call test functions and assertions directly rather than shelling out to
`pytest`, so the exact commands above are reproducible with nothing but `.venv/bin/python` and
paste cleanly into a doc as ordinary Python output. The real workflow for a plugin you intend to
ship is different and described in [conformance.md](conformance.md): add it to
`tests/support.py:ALL_CHUNKERS` (or the equivalent for other families) and let `pytest` run the
whole suite, so every invariant added to the suite in the future covers it automatically without
anyone remembering to re-run it by hand.

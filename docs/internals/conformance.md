# Conformance suites

A `Protocol` checks shape: does this object have a `.chunk()` method with the right signature.
It cannot check behaviour: does that method actually keep chunks inside the document, or
silently drop the last paragraph. With ~40 plugins and counting, and several of them wrapping
third-party libraries whose behaviour changes between releases, shape-checking alone is not
enough — a single plugin that quietly loses character offsets would corrupt every number
produced from it without failing anything.

The conformance suites are the behavioural check. Every plugin of a family is run through the
*same* parameterised test module, so an invariant written once is enforced on every
implementation, forever, including ones not written yet.

Two suites exist today, both under [`tests/conformance/`](../../tests/conformance/):

- [`test_parser_conformance.py`](../../tests/conformance/test_parser_conformance.py) — every
  `Parser` in `tests/support.py:ALL_PARSERS`.
- [`test_chunker_conformance.py`](../../tests/conformance/test_chunker_conformance.py) — every
  `ChunkerCase` in `tests/support.py:ALL_CHUNKERS`.

`RetrievalStrategy` and `IngestionStrategy` don't have a conformance suite yet — they're
exercised by ordinary unit tests instead (`tests/unit/test_retrieve.py`,
`tests/unit/test_ingest.py`). Worth knowing if you're adding one of those: there's no shared
module to plug a new strategy into today, only the pattern those unit tests establish.

## How a suite is wired up

Both suites use the same shape. Taking the chunker one
(`test_chunker_conformance.py`) as the example:

```python no-run: excerpt from tests/conformance/test_chunker_conformance.py, not standalone
@pytest.fixture(params=ALL_CHUNKERS, ids=CASE_IDS)
def case(request: pytest.FixtureRequest) -> ChunkerCase:
    return request.param


@pytest.fixture(params=CONTENTFUL_SOURCES, ids=SOURCE_IDS)
def parsed(request: pytest.FixtureRequest) -> ParsedDocument:
    return PARSER.parse(request.param)
```

Every `test_*` function in the file takes `case` and (usually) `parsed` as arguments. Pytest's
fixture parametrization means each test runs once per `(chunker, sample document)` pair —
adding a chunker to `ALL_CHUNKERS` in `tests/support.py` is the only step needed to run every
existing invariant against it; nothing in the test file itself changes.

## What the chunker suite enforces, and why each check exists

| Check | Catches |
|---|---|
| `test_satisfies_the_protocol` | Not behavioural — the `isinstance(..., Chunker)` floor. |
| `test_every_chunk_is_a_literal_slice_of_the_document` | The offset invariant itself: when `offsets_exact`, `chunk.matches_source(document)` must hold. Skipped (not failed) for chunks that honestly declare `offsets_exact=False`. |
| `test_chunks_stay_inside_the_document` | A span past the end of the text, or into a different document. |
| `test_no_chunk_is_empty` | An empty chunk is retrievable text that answers nothing and inflates chunk counts. |
| `test_chunks_are_in_reading_order` | Downstream code (heading-path lookup, `ends` ordering in assembly) assumes it. |
| `test_chunk_ids_are_unique` | Two chunks sharing an id silently collapse into one entry in every qrel and every run — see `CollidingChunker` below. |
| `test_no_chunk_contains_another` | A chunk fully inside another is redundant text, and it inflates character-level precision by counting the same evidence twice. |
| `test_chunks_cover_every_non_whitespace_character` | The most damaging, least visible bug: text in no chunk is evidence no retriever can ever return, and the leaderboard just shows slightly worse recall with nothing pointing at the cause. Skipped for chunkers that sample on purpose (`ChunkerCase.covers_everything=False`, e.g. a strided sentence window). |
| `test_records_a_token_count_under_a_named_tokenizer` | Chunk size means nothing without naming the tokenizer that measured it (`design.md §6`). |
| `test_carries_the_heading_path_where_there_is_one` | Heading path is one of the cheapest retrieval gains available; a chunker that drops it silently is worth catching. |
| `test_inherits_exactness_from_the_parse` | A chunk cannot be *more* exact than the parse it was cut from — feeding in a `ParsedDocument` with `offsets_exact=False` must produce chunks that are also `offsets_exact=False`, regardless of what the chunker itself did. |
| `test_chunking_twice_gives_the_same_chunks`, `test_ids_are_stable_across_instances` | Determinism. Non-deterministic chunking makes the cache wrong and makes two runs incomparable; ids must derive from position, not a counter, so a cached run can be reused. |
| Degenerate-input tests | Empty document → no chunks. Document shorter than one chunk → still one chunk. No structural markers → structural chunkers must still produce *something*, not an empty index. |

The parser suite (`test_parser_conformance.py`) mirrors this at the block level: blocks are
literal slices (when `offsets_exact`), stay inside the document, are ordered and non-overlapping,
cover every non-whitespace character, and parsing is deterministic. It adds a few
parser-specific checks: `supports()` is declared honestly, an unread `SourceFile` fails with a
clear `DocumentError` rather than an `AttributeError` three frames down, and
`text_hash()` differs when the text differs (the whole guard that stops chunks from one parse
being scored against another).

## Proof the suites can fail: `test_conformance_catches_bugs.py`

A suite that passes every plugin might be checking nothing. This file exists to prove
otherwise: it builds plugins with one specific, realistic bug apiece, and asserts the matching
invariant actually catches it. None of the bugs are invented — each is described in its
docstring as a mistake that's easy to make and produces plausible-looking output.

| Broken plugin | The bug | What catches it |
|---|---|---|
| `OffByOneParser` | Every block span's `end` is off by one character — from treating a range as inclusive at both ends. | `blocks_are_literal_slices()` — `document.slice(span) != block.text` once the boundary is wrong. |
| `NormalisingParser` | Returns cleaned-up text (`" ".join(text.split())`) but keeps the original offsets. Common in real wrappers: the library tidies whitespace, the wrapper doesn't notice its recorded position drifted. | Same literal-slice check. |
| `TableLosingParser` | Drops table blocks entirely — what a fast PDF extractor does to a financial report. | `covers_all_content()` — the content is simply gone, and coverage over non-whitespace characters catches the hole. |
| `GappyChunker` | Advances the cursor by `size` while only emitting `size - overlap` characters per chunk, so a run of text between chunks is silently skipped. Individual chunks look fine. | `covers_all_content()` again — the gap is invisible chunk-by-chunk and only shows up as missing coverage. |
| `LyingChunker` | Prepends an LLM-written summary to the chunk text while claiming `offsets_exact=True`. What a contextual-retrieval chunker does if it forgets to set the flag. | `chunks_are_literal_slices()` fails; `test_the_same_chunker_passes_once_it_tells_the_truth` shows the *same* chunker output passes once `offsets_exact=False` is set honestly — rewriting is allowed, lying about it is the bug. |
| `CollidingChunker` | Both halves of a split document get the id `f"{doc_id}:chunk"`. | Plain `len(ids) != len(set(ids))` — no invariant needed, just proves duplicate ids are actually produced by a realistic mistake. |

Each broken plugin also has `test_a_broken_parser_is_still_shaped_like_a_parser` /
`test_broken_chunkers_still_satisfy_the_type` — confirming `isinstance(broken, Parser)` /
`isinstance(broken, Chunker)` still holds. That's the point of the whole file: these are
**behavioural** bugs, invisible to a type checker or a shape check, which is the argument for
having conformance suites at all rather than relying on `Protocol` conformance.

## Running a new plugin through conformance without touching `tests/support.py`

`tests/support.py:ALL_CHUNKERS` is a fixed list, so the normal way to conformance-test a new
in-tree chunker is to add a `ChunkerCase` entry there. But the individual `test_*` functions in
`test_chunker_conformance.py` are plain functions taking `(case, parsed)` — nothing stops
calling them directly against a chunker that was never added to that list, which is how
[extending.md](extending.md)'s worked example proves a brand-new chunker passes conformance
without modifying any file under `tests/`. That's a convenience for a doc example, not the
normal workflow — a real new chunker belongs in `ALL_CHUNKERS` so every future invariant added
to the suite covers it automatically.

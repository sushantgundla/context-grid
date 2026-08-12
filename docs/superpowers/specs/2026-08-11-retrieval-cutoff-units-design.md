# Cut-offs count units, users receive items

**Status:** design, approved 11 Aug 2026. Not yet implemented.

## The bug

`k` is applied to returned *items*, then an ingestion strategy expands one item into several
scored *ids*, and `evaluate()` then applies `@1`/`@3`/`@5` to the expanded list.

`src/contextgrid/pipeline.py:345`:

```python
run[item.id] = self.scored_ids(self.search(item.question, k))
```

`search(question, k)` returns `k` items. `scored_ids` replaces each item with the units it
covers, in order, deduped. So a `sentence-window:2` hit — one passage covering five chunks —
becomes five entries, and `recall@1` reads the first of them. The question `recall@1` answers
is "was the centre chunk of the first window gold?", while the user received the whole window.

Measured on `examples/lab_demo.py`'s `markdown_corpus()` and `build_evalset()` at
`chunker: recursive:96`, `embedder: tfidf`, `index: dense`:

| ingestion | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| `plain` | 0.568 | 0.603 | 0.658 |
| `sentence-window:2` | **0.295** | 0.836 | 0.863 |
| `hierarchical:4` | **0.274** | 0.582 | 0.603 |
| `parent-document:4` | 0.829 | 0.884 | 0.884 |

`sentence-window` returns strictly more text than `plain` and scores half as well at `@1`.

It hits any strategy whose returned item covers more than one scored id — `sentence-window`
and `hierarchical`. `parent-document` is unaffected because its returned passage maps to a
single id. At `@5` the cut happens to land near an item boundary for a window of 5, which is
why only the small cut-offs look wrong; the defect is present at every `k`, not just `@1`.

`SentenceWindowIngestion` already orders its covered ids centre-first
(`src/contextgrid/ingest/structural.py`), which is why the number is 0.295 rather than 0.029.
That is a floor under the symptom, not a fix for the cause.

## Decisions

Both were live design choices with more than one defensible answer. Recorded here because the
numbers move and somebody will ask why.

### 1. `k` counts result slots

`recall@5` means "the five passages the user actually received". A `sentence-window` counts as
one slot however wide it is.

The alternative — `k` counts chunks of context, so a 5-chunk window consumes the whole `@5`
budget in one slot — is defensible and was rejected. It compares strategies at equal generator
cost, but it makes `@1` meaningless for any strategy whose items are wider than one chunk, and
`k` is documented as the number of results reaching the generator, not a text budget.

### 2. The extra context becomes visible

With `k` counting slots, a wide-passage strategy buys its score with context that nothing
currently measures. The leaderboard is `configuration / metric / p95 ms / $/1k`; nothing records
how much text a configuration hands over.

Leaving that unmeasured would make `sentence-window` look strictly better than `plain` when it
is actually trading context for recall — the same failure class this tool exists to catch. So a
`ctx/q` column (characters of context per query, mean) joins `p95 ms` and `$/1k`, for the same
reason those two are there: a metric win that costs 5x the context is not a free lunch.

### 3. Cut and rank in item space; relevance in id space

| Quantity | Space |
|---|---|
| the cut-off `k` | items |
| rank, for `mrr` and `ndcg` | items — a window at slot 1 ranks 1, not 4 |
| relevance, and the recall denominator | chunk ids — unchanged |

Scoring wholly in item space was rejected: it would rebuild the qrels per configuration and
break the agreement between `char_recall@5` and `recall@5` that `docs/dimensions/ingestion.md`
relies on as its cross-check that the two measure the same retrieval.

## Approach

**A parallel item-index list.** `run` keeps its shape, `dict[str, list[str]]`. A second mapping
records which slot each id came from:

```python
run = {"q1": ["c7", "c8", "c9", "c3"]}  # unchanged
item_of = {"q1": [0, 0, 0, 1]}  # c7/c8/c9 are all slot 0
```

`plain` produces `[0, 1, 2, ...]` and behaves exactly as it does today, which is what keeps this
additive: every existing reader of `run` — significance, diagnostics, `results.json` — works
unchanged, and a caller that does not pass `item_of` gets current behaviour.

Two alternatives were rejected. One ranked list per cut-off is exact but changes the `run` shape
everywhere it is read. Carrying boundaries inside the `run` structure itself is the same
semantics with a larger edit.

## What changes

| File | Change |
|---|---|
| `src/contextgrid/pipeline.py` | `run_queries` returns the item index alongside the ids; `scored_ids` reports which slot each id came from |
| `src/contextgrid/score/metrics.py` | `evaluate()` and `per_query()` take an optional item index and truncate by item; `mrr`/`ndcg` rank by item |
| `src/contextgrid/score/base.py` | the `Metric` protocol, if `evaluate()`'s signature moves |
| `src/contextgrid/report/results.py` | carry context-per-query on `RunResult` |
| `src/contextgrid/report/export.py` | `ctx/q` column in `format_leaderboard`; the figure in `results.json` |
| `src/contextgrid/grid/runner.py` | measure context per query while running |

Custom metrics are a public plugin family (`METRICS`, `docs/scoring/metrics.md`). If
`evaluate()` gains a required parameter, every third-party metric breaks. It must be optional,
with the current behaviour as its default.

## Migration

The published numbers move, and they are quoted in more places than the one table.

- `docs/dimensions/ingestion.md` — the table above, and the prose reading of it
- `docs/COVERAGE.md`, `docs/roadmap.md`, `docs/adoption-backlog.md`,
  `docs/recipes/choose-an-embedder.md`, `docs/reference/plugins.md`, `README.md` — all quote
  ingestion figures or `sentence-window` behaviour

Regenerate every one from a real run rather than editing digits by hand. `README.md`'s claim
that `parent-document` beats plain by `+0.247` and `hierarchical` scores *below* plain both need
re-measuring: the `hierarchical` reading in particular is currently explained as a property of
the strategy, and part of it is this bug.

`./scripts/check-docs.sh` executes many of these pages, so a stale figure in a doctest fails the
gate rather than rotting silently.

## Testing

- A window whose gold sits in a non-centre chunk must score 1.0 at `@1`. That is the bug, stated
  as a test, and it must fail against the current code.
- `plain` must be unchanged at every cut-off — same numbers, before and after. This is the
  regression that matters most.
- `parent-document` must be unchanged; it was never affected.
- `char_recall@k` must still agree with `recall@k` on every row.
- A metric registered through `plugins:` with the old single-argument `evaluate()` must still
  work.
- Property test: for any ranked list and any `k`, truncating by item never returns more items
  than `k`, and never splits an item.

The handoff's warning applies directly here: several bugs in this repo survived because the test
built its inputs with the same wrong assumption the code had. Each test above must be checked to
fail against the current implementation before it is trusted.

## Out of scope

- RAPTOR and GraphRAG, still deliberately not built.
- Pricing context by tokens rather than characters. `ctx/q` is characters because the core has no
  tokenizer dependency; a token figure would need `tiktoken` from `[embed]`.

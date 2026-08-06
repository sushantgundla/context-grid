# Architecture: how a run flows end to end

This page is for someone extending or reviewing `context-grid`, not using it. It traces one
configuration (`Config`) from raw files to a scored leaderboard row, naming the real function
at every step.

## The shape

```
Corpus ──parse──▶ ParsedDocument ──chunk──▶ Chunk[] ──ingest──▶ Ingested
                                                                    │
                                                          ┌─────────┴─────────┐
                                                      indexed              retrievable
                                                          │
                                                        embed
                                                          │
                                                        index (build)
                                                          │
                                              ── query time, per question ──
                                                          │
                                  transform ──▶ retrieve ──▶ (rerank) ──▶ chunk ids
                                                          │
                                              ── scoring, per run ──
                                                          │
                          anchor-resolve evalset ──▶ span-resolve to qrels ──▶ metrics
                                                          │
                                       (config.generator set) ──▶ assemble ──▶ generate ──▶ score
```

A tenth, optional stage sits on top of scoring: when `config.generator` is set, `Runner.run_one`
assembles the retrieved chunks and generates an answer, in addition to (not instead of) the
retrieval metrics above. See [§7](#7-generation-optional-wired-into-scoring).

Everything from parse through scoring happens inside two files:
[`pipeline.py`](../../src/contextgrid/pipeline.py) builds and queries one configuration;
[`grid/runner.py`](../../src/contextgrid/grid/runner.py) does it for every configuration in a
matrix and turns the result into a scored `RunResult`.

## 1. Corpus

A `Corpus` (`corpus/loader.py:Corpus`) is a tuple of `SourceFile` — raw bytes plus an id and a
`MediaType`. `Corpus.from_dir(path)` walks a directory and reads matching files. The corpus is
deliberately dumb: everything about *content* (length, structure, table density) depends on
which parser reads it, so that lives in `corpus/fingerprint.py`, not here.

## 2. Parse

`pipeline._parse_all(parser, corpus, ...)` loops over every `SourceFile` and calls
`Parser.parse(source) -> ParsedDocument` (the `Parser` protocol is in
[`core/protocols.py`](../../src/contextgrid/core/protocols.py), see
[protocols.md](protocols.md)). Each parse is cached under a key built from the parser's name,
version, params, and the source's content hash (`cache_key("parse", ...)`).

A `ParsedDocument` carries `document.text` — the canonical text every character offset
downstream refers to — plus `blocks` (paragraphs, headings, tables, ...) and
`offsets_exact`. If a source's media type isn't supported, the file is skipped with a
`PARSER_FALLBACK` warning rather than failing the whole run.

## 3. Chunk

`pipeline._chunk_all(chunker, parses, ...)` calls `Chunker.chunk(parsed) -> list[Chunk]` for
every parsed document, again cached — this time keyed on the chunker's params plus
`parsed.text_hash()`, so two parses of the same source never share a chunk cache entry.

Every `Chunk` is built through `ChunkBuilder.build()` / `.build_all()`
(`chunk/base.py`), which is what guarantees the offset invariant, per-tokenizer sizes, and
heading-path metadata are computed the same way by every chunker. See
[extending.md](extending.md) for a chunker written from scratch using it.

## 4. Ingest

`pipeline.build()` calls `get_ingester(config.ingestion).ingest(chunks, IngestionContext(...))
-> Ingested` (`ingest/base.py`). This is the step most pipelines skip: it decides what gets
**indexed** versus what a hit on it **returns**. For plain chunking (`config.ingestion is
None`), `Ingested.plain(chunks)` makes the two sides identical. Strategies like
`parent-document` or `hierarchical` index small units and return larger ones — see
[registry.md](registry.md) and [extending.md](extending.md) for how a strategy is registered
and what it must guarantee.

If ingestion produces zero chunks, an `EMPTY_CHUNK_SET` warning is raised at
`Severity.INVALID` — every query on this configuration will score zero, and the warning says
why before anyone has to guess.

## 5. Embed

`pipeline._embed_all(config, ingested.indexed, ...)` resolves an `Embedder`
(`get_embedder(config.embedder)`), calls `embedder.prepare(texts)` (lets a local model warm up
or a remote one batch), then `embedder.embed_documents(texts) -> EmbeddingResult`. Skipped
entirely when `config.embedder is None` — a sparse index like `bm25` never needs vectors. The
result is cached on a hash of the ordered text list (`texts_hash`), and any warnings the
embedder raised (truncated input, missing query prefix, unnormalised vectors) are folded into
the run's `WarningLog`.

## 6. Index

`pipeline._make_index(config)` resolves an `Index` (`get_index(config.index)`,
`index/base.py`) and calls `index.build(ingested.indexed, vectors)`. The result is a
`BuiltPipeline` — a configuration that has read the corpus and is ready to answer queries.

## Query time: transform → retrieve → rerank

Everything above happens once per configuration. `BuiltPipeline.search(query, k)`
(`pipeline.py:173`) happens once per question:

1. **Transform** — `self.transform.transform(query)` rewrites the question into one or more
   query strings (HyDE, multi-query, decomposition, ...). `QueryTransform` lives in
   `transform/query.py`; `NoTransform` is the identity.
2. **Retrieve** — `self.retrieval.retrieve(query, rewritten.queries, searcher, depth, trace)`
   runs the `RetrievalStrategy` (`retrieve/base.py`, see [protocols.md](protocols.md)). The
   strategy is handed a `Searcher` closure — `search()`'s inner `searcher()` function — and
   never sees the index directly, which is what lets every strategy work with every store.
   What the strategy actually did (how many searches, how many model calls) accumulates in a
   `RetrievalTrace` across the whole eval set, not just one question.
3. **De-duplicate to passages** — `BuiltPipeline._to_retrievable()` maps indexed-unit hits back
   to the retrievable passages `Ingested` says they stand for (`ingested.resolve()`), keeping
   the best score per passage, then `_merge_siblings()` gives an ingestion strategy its one
   query-time decision: promote a run of sibling leaves to their shared parent once enough of
   them hit.
4. **Rerank** — if `config.reranker` is set, the retriever is asked for `config.candidates`
   results and `Reranker.rerank(query, candidates, limit)` (`rerank/base.py`) cuts them to `k`.
   Without a reranker, the retriever is asked for `k` directly, so the no-reranker arm never
   pays for candidates it would have thrown away.

`run_queries(evalset)` calls `search()` once per `EvalItem` and records wall-clock per query in
`Timings.query_ms`.

## Scoring one configuration

This is `Runner.run_one()` (`grid/runner.py:101`), called once per `Config` in a matrix.

1. **`build(config, corpus, ...)`** — everything above, indexing side.
2. **`resolve_evalset(evalset, pipeline.parses, anchor_resolver)`** — ground truth is
   re-resolved *against this parse*. Two parsers produce different text, so a `GoldAnchor`
   (a quoted sentence) has to be relocated in each one; `AnchorResolver.resolve()`
   (`score/anchor.py`) does that and turns anchors into concrete `GoldSpan`s. Skipped when the
   eval set already carries spans rather than anchors.
3. **`build_qrels(resolved, pipeline.chunks, span_resolver)`** — `SpanResolver.resolve()`
   (`score/resolve.py`) turns span-level gold into chunk-level relevance judgements (`qrels`),
   using the `coverage` / `iou` / `containment` policy from [design.md](../design.md) §4.4.
4. **`pipeline.run_queries(resolved)`** — the search loop above, for every question.
5. **`evaluate(qrels, run, ks=...)`** (`score/metrics.py`) — recall@k, nDCG@k, MAP, MRR,
   hit-rate, precision@k, over `ranx`-shaped qrels/run dicts.
6. **`_character_metrics(...)`** — `character_precision` / `character_recall`
   (`score/resolve.py`) over the actual retrieved text, the check that catches a configuration
   scoring recall@5 = 1.0 while burying the evidence in 25× its weight of irrelevant text.
7. **`diagnose(resolved, qrels, run, k)`** (`diagnose/taxonomy.py`) — classifies failing
   questions into the Seven Failure Points (Barnett et al., 2024): FP1–FP3 from retrieval data
   alone, FP4–FP7 need a generator this stage doesn't have, and it says so rather than guessing.
8. **`self.cost_model.estimate(...)`** (`cost/model.py`) — dollars for this configuration, from
   embed tokens, query tokens and wall-clock.

The result is a `RunResult` (`report/results.py`) — metrics, timings, cost, warnings, the raw
per-query run, and the failure report — one row of the eventual leaderboard.

## A whole matrix

`Runner.run(matrix, evalset, mode=...)` expands a `Matrix` (see [registry.md](registry.md) for
spec strings, `grid/matrix.py` for `factorial` / `ofat` / `staged` expansion) into a list of
`Config`s and calls `run_one()` for each, honouring a `Budget` (seconds or dollars) checked
*between* configurations — never predicted before they run, because an agentic strategy's cost
depends on how many model calls it decides to make. Caching is what makes a sweep affordable:
configurations sharing a parser share its parse; those additionally sharing a chunker and
embedder share the embeddings too, so sweeping four rerankers across twenty configurations
embeds nothing at all. `CacheStats` (`cache/store.py`) counts hits and misses so that claim can
be checked, not just believed — see `results.cache_summary`.

## 7. Generation: optional, wired into scoring

`grid.generator` is the tenth axis (`AXIS_ORDER`, last — it runs on whatever reranking
produced; see [generation](../dimensions/generation.md)). `None` (the default) means exactly
what every config meant before this axis existed: no assembly, no model call, no cost.

When it is set, `Runner.run_one()` calls `self._score_generation(...)` after retrieval scoring
(`diagnose()` has already run, using retrieval data alone — see step 7 below), which for every
question calls `BuiltPipeline.answer()`:

```python no-run: excerpt from inside BuiltPipeline.answer() -- self/retrieved are method context, not standalone
# BuiltPipeline.answer(), pipeline.py -- assembles then generates, in one call
context = self.assembler.assemble(retrieved)      # assemble/context.py:ContextAssembler
return self.generator.answer(question, context), context   # the configured Generator
```

`score_answer()` (`generate/answer.py`) scores the reply lexically against the gold chunks —
`groundedness`, `citation_accuracy`, `evidence_overlap`, `abstention_accuracy`. When `run.model`
is set and `deepeval` is installed, a `GenerationJudge` additionally scores `faithfulness` and
`answer_relevancy`. Both fold into the same metrics dict every other stage reports into, which
is what lets `DIMENSION_METRICS["generation"]` in
[`report/composite.py`](../../src/contextgrid/report/composite.py) find them — see
[composite score](../scoring/composite.md). A generator that fails on one question is recorded
and skipped, the same rule an agentic retrieval planner follows for a refusal; the rest of the
eval set still gets scored.

**`diagnose()` still cannot see any of this.** It classifies FP1–FP3 from retrieval data alone
(step 7 above) regardless of whether a generator ran in the same configuration — FP4–FP7, the
failure points about what the generator did with the context, are not fed generation output and
stay unclassified. `FailureReport.observed_generation` is hardcoded `False`. Watching what the
generator actually did with a bad context is future work, not something this codebase does yet.

## See also

- [protocols.md](protocols.md) — the contracts each stage's plugins implement.
- [registry.md](registry.md) — how `get_parser("markdown")` / `"recursive:512"` etc. resolve to
  a class.
- [extending.md](extending.md) — writing and registering a new chunker and retrieval strategy.
- [conformance.md](conformance.md) — what stops a broken plugin from silently corrupting a run.
- [design.md](../design.md) — why offsets are the one hard invariant everything above depends on.

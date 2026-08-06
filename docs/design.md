# context-grid — Design

**Package:** `context-grid` on PyPI · `import contextgrid` · **Owner:** Sushant Gundla
**Status:** Design v1 · **Date:** 2026-08-03

Companion documents: the product rationale lives in the blog repo at
`docs/prd/rag-retrieval-lab-prd-v2.md`, and the full feature universe with stable IDs
(`A1`, `K6`, `L13`…) lives at `docs/prd/rag-retrieval-lab-features.md`. This document is the
engineering design for the SDK. Feature IDs referenced below point at that catalogue.

---

## 1. What this is

A Python SDK for running controlled experiments across the whole grounding pipeline —
ingestion, parsing, chunking, embedding, indexing, retrieval, reranking, context assembly and
generation — and getting back ranked, reproducible, cost-aware results on **your own documents**.

```python no-run: pre-implementation design sketch (cg.Lab, .grid(), .pareto() were never shipped this way -- see guide/getting-started.md for the real API)
import contextgrid as cg

lab = cg.Lab(corpus="./contracts")

lab.grid(
    parser=["pymupdf", "docling"],
    chunker=["recursive:512", "semantic"],
    embedder=["bge-base-en-v1.5"],
    reranker=[None, "bge-reranker-v2-m3"],
)

results = lab.run(evalset)
results.leaderboard()
results.pareto("recall@5", "cost_per_1k")
```

The web UI at `sushantgundla.com/lab` comes later and is a *consumer* of this package, not the
other way round.

## 2. Design principles

1. **Character offsets are a hard invariant.** Every chunk knows the exact character range of the
   source document it came from. Everything valid about this tool follows from that one property.
2. **Fair by construction.** Where a comparison can be silently unfair — different tokenizers,
   truncated inputs, approximate search compared against exact — the SDK makes the unfairness
   visible rather than trusting the user to notice.
3. **Thin core.** `pip install context-grid` installs a handful of pure-Python dependencies.
   Every parser, model and index backend is an optional extra. Nobody should install CUDA to try
   a chunker.
4. **Everything is a plugin, plugins prove themselves.** ~40 pluggable components behind a small
   set of protocols, each of which must pass a shared conformance suite.
5. **The config is the manifest.** One serialisable object drives the run and pins its
   reproducibility. There is no second source of truth.
6. **Warnings are data.** Truncation, approximate offsets, ANN recall loss and cache misses ride
   on the result object as structured records, never as log lines.
7. **Cheap by default.** Content-addressed caching with prefix reuse, local CPU models as the
   default, and a cost estimate before anything expensive runs.

## 3. Package layout

```
src/contextgrid/
  core/          protocols, types, registry, warnings, errors
  cache/         content-addressed store, prefix reuse
  corpus/        loading, normalisation, fingerprint            (A1–A6)
  parse/         parser plugins → ParsedDocument                (B1–B15)
  chunk/         chunker plugins → list[Chunk]                  (C1–C21)
  embed/         embedder plugins                               (D1–D17)
  index/         dense, sparse, hybrid, quantized               (E1–E23)
  transform/     query transforms and routing                   (F1–F15)
  retrieve/      retrieval strategies                           (G1–G16)
  rerank/        reranker plugins                               (H1–H13)
  assemble/      context ordering, compression, budgeting       (I1–I10)
  generate/      generator plugins and answer scoring           (J1–J13)
  evalset/       generation, filters, review, import            (K1–K19)
  score/         span resolution, metrics, significance         (L1–L21)
  cost/          pricing tables, cost model                     (M1–M15)
  diagnose/      failure taxonomy, clustering, inspection       (N1–N11)
  grid/          matrix builder, sweep modes, runner
  report/        results objects, exports, manifest             (P1–P16)
  cli/           typer CLI
```

Each plugin family follows the same shape: a `Protocol` in `core/protocols.py`, a registry, one
module per implementation, and a conformance test module that every implementation is run through.

## 4. The offset spine

The single most important design decision, and the reason day one is spent here.

### 4.1 The problem

Ground truth in retrieval evaluation is normally stored as a **chunk ID**. That works exactly as
long as you never change the chunker. The moment you compare two chunking strategies, they produce
different chunks, so a gold chunk ID from strategy A means nothing under strategy B. Every
cross-chunker and cross-parser comparison built this way is invalid, and the invalidity is silent.

### 4.2 The fix

Gold is stored as a **character span in the source document**: `(doc_id, start, end)`. Chunks also
carry `(doc_id, start, end)`. At scoring time, gold spans are resolved to whichever chunks a given
configuration happens to have produced, by measuring overlap. The eval set is therefore
independent of every configuration it scores.

```
Document text ─────────────────────────────────────────────────────
                        ╔═══════════════╗
gold span               ║  chars 840–1010║
                        ╚═══════════════╝
chunker A     [ 0–500 ][ 500–1000 ][ 1000–1500 ]   → gold straddles two chunks
chunker B     [ 0–800 ][ 800–1600 ]                → gold sits inside one
```

Both are scored against the same gold. Neither is advantaged by the labelling.

### 4.3 Why IoU alone is the wrong criterion

The PRD says "resolve by IoU overlap". Implementing that literally introduces the exact bias the
design exists to remove.

Take a 170-character gold span. Under a 2000-character chunk that fully contains it, IoU is
`170 / 2000 = 0.085`. Under a 250-character chunk that contains it, IoU is `0.68`. With any
sensible IoU threshold the large chunk is scored as a miss — even though it contains every
character of the evidence and would ground the answer perfectly.

IoU therefore **systematically penalises large-chunk configurations for being large**. Since chunk
size is one of the axes under test, that is a bias in the measuring instrument itself.

### 4.4 Resolution policies

The resolver supports three policies. `coverage` is the default and the one used for headline
metrics.

| Policy | A chunk is relevant when | Use |
|---|---|---|
| **`coverage`** *(default)* | it contains at least `threshold` of the **gold span's** characters | Headline metrics. Asks the question that matters: is the evidence present? |
| **`iou`** | intersection over union ≥ `threshold` | Strict, symmetric. For precision-focused analysis where chunk bloat should be punished |
| **`containment`** | the chunk fully contains the gold span | Strictest. Useful for citation-accuracy work |

Threshold defaults to `0.5` for `coverage` — half the evidence present in one chunk. Configurable,
recorded in the manifest, and reported in the results header so no chart is ever shown without the
policy that produced it.

### 4.5 Split gold and union recall

A gold span can straddle a chunk boundary such that no single chunk clears the threshold, while the
retrieved set together contains every character of the evidence. Scoring per-chunk only would
report a miss on a retrieval that would in fact ground the answer perfectly.

So the resolver produces two things:

1. **Per-chunk relevance labels** (`qrels`) — what `ranx` needs for nDCG, MAP, MRR and precision.
2. **Union coverage** over the retrieved set — the fraction of gold characters present anywhere in
   the retrieved context, which is the honest basis for Recall@k and for character-level recall
   (`L8`).

Reporting both, and explaining the difference on the methodology page, is a correctness feature
other tools do not have because they never stored offsets in the first place.

### 4.6 Graded relevance

Gold spans carry a grade (`2` fully answers, `1` partially relevant, `0` irrelevant). A resolved
chunk inherits the highest grade among the gold spans it satisfies. `ranx` needs graded qrels for
nDCG to mean anything.

### 4.7 The exactness flag

Some pipelines cannot honestly report offsets. An LLM that rewrites a chunk (contextual retrieval,
proposition extraction) produces text that does not appear in the source. A parser that reflows
columns may lose the mapping.

Every `ParsedDocument` and every `Chunk` therefore carries `offsets_exact: bool`. When false, a
structured warning is attached and the results header says so. Being visibly honest about a
limitation is worth more than hiding it — and it is what stops a user drawing a confident
conclusion from an approximate comparison.

## 5. Core types

```python no-run: pre-implementation sketch of the core types -- real signatures now differ (see core/span.py, core/documents.py, core/evalset.py)
Span(doc_id, start, end)          # half-open [start, end)
Document(id, text, source, meta)  # the canonical text offsets refer to
Block(span, text, kind, page, meta)          # parser output unit
ParsedDocument(doc_id, blocks, parser, offsets_exact, warnings)
Chunk(id, doc_id, span, text, meta, token_counts, offsets_exact)
GoldSpan(span, grade)
EvalItem(id, question, qtype, gold: list[GoldSpan], meta)
EvalSet(id, items, version, source)
RelevanceLabel(item_id, chunk_id, grade)
```

`Span` carries the whole overlap algebra: `length`, `intersection`, `overlap_len`, `iou`,
`coverage_of`, `contains`. Keeping it in one small, heavily tested value object means the
correctness-critical arithmetic exists in exactly one place.

## 6. Caching

Content-addressed, keyed on the SHA-256 of `(stage name, stage version, params, input hash)`.

The cache key for a chunk set **must include the tokenizer**, because `C14` reports chunk size in
tokens per tokenizer. Two embedders with different tokenizers asking for "512-token chunks" want
genuinely different chunk sets; a key that omits the tokenizer silently serves one the other's
chunks. This is a real bug the design has to prevent, not a detail to add later.

Prefix reuse is what makes sweeps affordable: configurations sharing a parser share its parse
artefacts; configurations sharing parser + chunker + embedder share the embeddings. Sweeping
rerankers across 20 configurations should embed nothing at all.

## 7. Dependencies

**Core** (always installed): `pydantic`, `numpy`, `ranx`, `platformdirs`, `typing-extensions`.

**Extras:**

| Extra | Pulls in |
|---|---|
| `parse` | pymupdf, pdfplumber |
| `parse-ml` | unstructured, docling, marker |
| `chunk` | chonkie |
| `embed` | sentence-transformers / infinity + onnxruntime |
| `index` | lancedb, rank-bm25 |
| `rerank` | cross-encoder models |
| `llm` | openai, anthropic, cohere clients |
| `all` | everything above |

A missing extra raises `MissingExtraError` naming the exact `pip install` command. Never an
`ImportError` traceback.

## 8. Testing strategy

Four layers, because ~40 plugins cannot be tested one at a time by hand.

| Layer | What | Runs |
|---|---|---|
| **Conformance suites** | Every plugin of a family is run through the same parameterised tests. A parser must round-trip offsets; a chunker must produce chunks whose text matches `document.text[start:end]` exactly when `offsets_exact` is true; an embedder must be deterministic and correctly normalised | Every PR |
| **Property tests** (Hypothesis) | The span algebra and the resolver. Generated spans, generated chunk boundaries, invariants asserted — IoU symmetric, coverage in [0,1], union recall monotone in the retrieved set, resolution invariant to chunk ordering | Every PR |
| **Golden files** | Committed fixture documents with committed expected parse and chunk output. Catches silent behaviour changes when a parser dependency updates | Every PR |
| **Benchmark validation** | The full scorer run against a LegalBench-RAG slice, asserted to reproduce published numbers within tolerance | Nightly |

Unit tests never touch the network. Anything that calls a model or an API is marked `integration`
and skipped by default.

Coverage target is 90% on `core/` and `score/`, which is where a bug is unrecoverable, and 75%
elsewhere.

## 9. Public API sketch

```python no-run: pre-implementation API sketch (cg.Lab, evalset.review(), .estimate() were never shipped this way -- see guide/getting-started.md and guide/evalsets.md for the real API)
import contextgrid as cg

# 1. Corpus
corpus = cg.Corpus.from_dir("./contracts")
print(corpus.fingerprint())  # table density, dup rate, language mix   (A5)

# 2. Eval set — generate, filter, review, or import
evalset = cg.EvalSet.generate(corpus, n=100, model="gpt-4o-mini")  # (K1–K5)
evalset = evalset.filter(cg.filters.default())
evalset.review()  # opens the keyboard queue         (K9)
evalset.save("evalset.jsonl")

# 3. Grid
lab = cg.Lab(corpus)
lab.grid(parser=[...], chunker=[...], embedder=[...], reranker=[...])
print(lab.estimate())  # configs, wall-clock, dollars      (M3)

# 4. Run
results = lab.run(evalset, mode="ofat", budget_usd=2.00)

# 5. Read
results.leaderboard()
results.pareto("recall@5", "cost_per_1k")  # (M7)
results.compare("config_a", "config_b")  # paired significance (L13)
results.inspect(item_id="q_042")  # (N1)
results.failures().by_taxonomy()  # (N2)
results.export_config("winner.yaml")  # (P3)
results.manifest().save("manifest.json")  # (P4)
```

Every result object is plain data with a `.warnings` list. Nothing important is ever printed and
lost.

## 10. What day one delivers

The offset spine and the resolver — `core/types.py`, `core/warnings.py`, `score/resolve.py` — with
unit tests, property tests and full type coverage. No parsers, no models, no I/O.

This is the critical path. Every number the SDK will ever produce depends on the span algebra being
right, and a mistake here is not recoverable later without invalidating published results.

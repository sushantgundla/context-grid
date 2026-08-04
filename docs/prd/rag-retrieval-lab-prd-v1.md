# RAG Retrieval Lab — Product Requirements Document

**Owner:** Sushant Gundla
**Status:** Draft v1
**Type:** Public portfolio project + open-source tool
**Target home:** sushantgundla.com/lab (or /retrieval-lab)

---

## 1. One-liner

Upload a document, define a set of questions, and get a ranked, reproducible comparison of every retrieval pipeline configuration — parser × chunker × embedder × index × reranker — scored on retrieval quality, latency, and cost.

## 2. Why this project

Most RAG content on the internet is anecdote. "Semantic chunking is better." "Use a reranker." "512 tokens with 50 overlap is the sweet spot." Almost nobody publishes the measurement that backs it, and almost nobody can reproduce someone else's setup on their own corpus.

The value of this tool is that it turns those opinions into numbers on *your* documents. The value of it as a portfolio piece is that building it correctly requires you to have solved the parts of RAG that are actually hard: ground truth construction, metric selection, fair comparison under different tokenizers, cost accounting, and orchestration of long-running jobs.

A demo chatbot proves you can call an API. This proves you can run an experiment.

### Why it fits you specifically

You have already done the underlying research in private: PDF parser evaluation for RAG pipelines, open-source embedding models for CPU inference, Infinity with ONNX runtime, bge-base-en-v1.5 for text and CodeRankEmbed for code. This project is that research made public, reusable, and defensible. The build cost is low relative to the credibility return because you are not learning the domain, you are packaging it.

---

## 3. Goals and non-goals

### Goals

- **G1.** Let a user run a full factorial (or sampled) sweep across retrieval configurations on their own corpus, without writing code.
- **G2.** Produce ground truth cheaply. Auto-generate a labelled eval set from the corpus, then let the user correct it. This is the single biggest adoption blocker in retrieval evaluation and the place where the product earns its keep.
- **G3.** Report retrieval quality, p50/p95 latency, and cost per 1k queries together, on one screen. Quality alone is a misleading axis.
- **G4.** Make every result reproducible: a shareable permalink and an exportable config that runs the same pipeline in the user's own codebase.
- **G5.** Work in a zero-setup demo mode with precomputed results so a visitor gets value in under ten seconds without keys, uploads, or waiting.

### Non-goals

- **N1.** Not a RAG framework. It does not replace LlamaIndex, Haystack, or your own stack. It emits configs *for* them.
- **N2.** Not a production serving layer. No SLAs, no persistent user corpora at scale, no multi-tenant vector hosting.
- **N3.** Not primarily an answer-quality evaluator. Generation metrics are a secondary panel, deliberately. Retrieval is the bottleneck and the differentiator; if you evaluate end-to-end answers first, generation noise swamps retrieval signal.
- **N4.** No fine-tuning of embedding models in v1. Tempting, expensive, and a different product.

---

## 4. Users

| Persona | What they want | What they do in the tool |
|---|---|---|
| **The engineer picking a stack** | "Which chunker and embedder should I use for these contracts?" | Uploads 10 representative docs, runs a sweep, exports the winning config |
| **The engineer debugging bad retrieval** | "Why does this one question always fail?" | Uses the per-query inspector to see what got retrieved and where the gold chunk actually ranked |
| **The skeptic / hiring manager** | "Does this person know what they're talking about?" | Opens demo mode, reads the methodology page, leaves impressed in 90 seconds |
| **The LinkedIn reader** | "Interesting chart" | Clicks a permalink from a post, sees a real leaderboard, follows |

The third and fourth personas are the actual business case for a portfolio project. Design demo mode for them first, not last.

---

## 5. Core concept: the experiment matrix

Every run is a point in a configuration space. The UI is essentially a matrix builder over these axes:

```
Corpus  ×  Parser  ×  Chunker  ×  Embedder  ×  Index/Search  ×  Query Transform  ×  Reranker
```

Selecting multiple values on any axis multiplies the run count. The UI must show the combinatorial count live ("You have selected 3 × 4 × 2 × 2 = 48 configurations, est. 14 min, est. $0.62") and warn before expensive sweeps. Offer three sweep modes:

- **Full factorial** — every combination. Fine for small selections.
- **One-factor-at-a-time (OFAT)** — hold a baseline, vary one axis at a time. Cheap, interpretable, the right default.
- **Staged/greedy** — pick the best chunker, freeze it, then sweep embedders, then rerankers. Cheapest path to a good config, with a clear caveat in the UI that it can miss interactions.

The staged mode is the one most practitioners actually want, and offering it (with the honest caveat) is a small detail that signals maturity.

---

## 6. Feature specification

Priority key: **P0** = required for launch, **P1** = fast follow, **P2** = later.

### 6.1 Corpus ingestion — P0

- Upload PDF, DOCX, HTML, Markdown, TXT. Multi-file up to a configured cap (suggest 25 files / 50 MB in hosted mode, unlimited self-hosted).
- URL ingestion with a crawler depth of 0 or 1 (P1).
- Paste-text corpus for quick tests.
- Built-in sample corpora so users can start instantly:
  - A messy scanned-ish PDF (financial report with tables)
  - A technical docs set (API reference, heading-heavy Markdown)
  - A legal/insurance policy document (dense, cross-referential, long)
  - A code + SQL corpus (exercises a different embedder class)
- Per-file preview with detected page count, language, and whether the text layer is present or OCR will be needed.

### 6.2 Parsing layer — P0

Selectable parsers, each producing a normalized intermediate representation (text blocks + structure metadata + page/offset provenance):

- PyMuPDF (fast baseline)
- pdfplumber (table-aware)
- Unstructured (hi-res and fast strategies)
- Docling
- Marker or MinerU (layout/ML-based)
- LlamaParse (hosted, BYOK) — P1
- Azure Document Intelligence / AWS Textract / Google Document AI (hosted, BYOK) — P1
- OCR fallback: Tesseract or PaddleOCR for image-only pages

**Parser diff view (P1, and a genuine differentiator):** render the same page side by side under two parsers with the extracted text overlaid, so the user can see which one lost the table or merged two columns. Almost no tool shows this. It is also the most screenshot-able feature you will build.

Parser quality sub-metrics to report where computable: table cell recall on a labelled sample, reading-order correctness, characters extracted per page, wall-clock per page.

### 6.3 Chunking layer — P0

Each chunker is parameterized and every parameter is sweepable.

| Strategy | Key params |
|---|---|
| Fixed-size token | size, overlap, tokenizer |
| Recursive character | size, overlap, separator hierarchy |
| Sentence window | window size, stride |
| Structural / heading-aware | max depth, min/max chunk size, keep-heading-path flag |
| Semantic (embedding breakpoint) | percentile threshold, buffer size, breakpoint method |
| Proposition-based (LLM-extracted atomic facts) | model, max propositions per block |
| Parent-document / small-to-big | child size, parent size |
| Contextual retrieval (LLM-prepended chunk context) | context model, context length budget |
| Late chunking (embed long, pool per chunk) | requires long-context embedder |

**Critical implementation note:** chunk size must be reported in a consistent unit across models with different tokenizers. Store both character count and token count per tokenizer, and expose a unit toggle. Getting this wrong is the most common way these comparisons become invalid, and calling it out explicitly in your methodology page is a credibility marker.

Also expose: metadata attachment (heading path, page number, source file, section), and de-duplication of near-identical chunks.

### 6.4 Embedding layer — P0

**Local / open (CPU-first, served via Infinity + ONNX):**
- bge-base-en-v1.5, bge-large-en-v1.5, bge-m3 (multilingual + multi-vector)
- e5-base/large-v2, multilingual-e5
- gte-base/large
- nomic-embed-text-v1.5 (Matryoshka dimensions — expose the dimension truncation as a sweepable param, it is a great cost/quality lever)
- all-MiniLM-L6-v2 (speed baseline)
- jina-embeddings-v3
- CodeRankEmbed and a general model, for the code corpus

**Hosted (BYOK):**
- OpenAI text-embedding-3-small / -large (with `dimensions` param)
- Cohere embed-v3 (with input_type distinction)
- Voyage
- Google Gemini embeddings

Per-model handling that must be correct or results are garbage: query vs document prefixes/instructions (E5's `query:`/`passage:`, BGE's query instruction, Cohere's `input_type`), normalization, and max sequence length truncation behaviour. Surface a warning when a chunk exceeds a model's context and is being truncated — this silently destroys results in most homegrown evaluations.

Report per model: dimensions, index size on disk, embed throughput (chunks/sec), cost per million tokens.

### 6.5 Index and search layer — P0

- **Dense** — cosine/dot, exact and ANN (HNSW params `M`, `efConstruction`, `efSearch` sweepable; include an exact-search reference run so users can see ANN recall loss).
- **Sparse** — BM25 (k1, b tunable), and learned sparse (SPLADE) as P1.
- **Hybrid** — fusion via Reciprocal Rank Fusion (k param) or weighted score normalization (alpha param). Both, because they behave differently and the choice matters.
- **Multi-vector / late interaction** — ColBERT-style, P1.
- Metadata filtering and section-scoped search — P1.
- Backends: LanceDB embedded by default (zero-ops, fits the hosted model), Qdrant and pgvector as configurable alternatives for self-hosters.

### 6.6 Query transformation layer — P1 (P0 if scope allows)

- None (baseline)
- HyDE (hypothetical document embedding)
- Multi-query expansion (n variants, fused)
- Query decomposition for multi-hop questions
- Step-back / abstraction prompting
- Query rewriting for conversational context

Each carries an LLM call cost and latency penalty that must be attributed to the config in the cost panel. The whole point is showing whether the quality gain justifies it. Frequently it does not, and demonstrating that with data is a strong post.

### 6.7 Reranking layer — P0

- None (baseline)
- bge-reranker-base / v2-m3 (local cross-encoder)
- mxbai-rerank
- jina-reranker-v2
- Cohere Rerank (BYOK)
- LLM-as-reranker (pointwise or listwise, BYOK) — P1

Sweepable: candidate depth fed to the reranker (top 20/50/100) and final k. The candidate-depth curve is one of the most useful charts the tool can produce and nobody publishes it.

### 6.8 Eval set builder — P0, and the heart of the product

Three paths, all supported:

1. **Auto-generated (default).** Sample chunks from the corpus, use an LLM to write questions answerable *only* from that chunk, and record the source chunk as the gold passage. Then run automatic quality filters:
   - Reject questions answerable from general knowledge (test by asking the LLM without context).
   - Reject questions containing pronouns or references with no antecedent.
   - Reject near-duplicate questions.
   - Flag questions whose gold chunk also appears verbatim in other chunks.
2. **Human-in-the-loop review.** A fast keyboard-driven queue: question on the left, gold chunk on the right, keys for accept / reject / edit / mark-multi-hop. Target under 5 seconds per item. This screen is where the tool becomes genuinely usable rather than a toy.
3. **Import.** CSV/JSONL of `question, answer, gold_chunk_ids` or `question, gold_doc_id`. Support BEIR-format import (P1) so users can sanity-check the tool against a known benchmark.

**Question type tagging** — auto-classify each question as factoid, multi-hop, comparative, numeric/tabular, or summarization-style. Then report metrics *per question type*. This is where the real insights live: semantic chunking might win overall while losing badly on tabular questions. Slicing by type is what separates this from a leaderboard toy.

**Gold set integrity features:**
- Multiple gold chunks per question (a question can be legitimately answered by more than one passage).
- Graded relevance (2 = fully answers, 1 = partially relevant, 0 = irrelevant), needed for nDCG to mean anything.
- A "chunk-boundary-agnostic" gold representation: store gold as a **character span in the source document**, not a chunk ID. This is essential. If gold is a chunk ID, you cannot fairly compare two different chunking strategies, because they produce different chunks. Resolve gold-to-chunk at scoring time by span overlap with a configurable IoU threshold. **This is the single most important design decision in the document.**

### 6.9 Metrics engine — P0

Retrieval metrics, computed at multiple k (1, 3, 5, 10, 20):

- **Recall@k** — fraction of gold spans retrieved. The headline metric for RAG, since generation only needs the evidence present.
- **Precision@k**
- **nDCG@k** — with graded relevance
- **MRR** — rank of first relevant
- **MAP**
- **Hit rate@k** — at least one gold retrieved
- **Mean rank of first gold** — more intuitive than MRR for debugging
- **Context precision / context recall** (Ragas-style, LLM-judged) — P1
- **Chunk attribution rate** — of retrieved chunks, what fraction contain any gold span. Measures wasted context.

Operational metrics, always shown alongside:

- Index build time and index size
- Query latency: p50, p95, p99, broken down by stage (embed → search → rerank)
- Cost per 1,000 queries and one-time indexing cost, itemized by call type
- Tokens sent to the generator at the chosen k (this is the real downstream cost driver)

**Statistical honesty — P1 but do it:**
- Bootstrap confidence intervals on the headline metric.
- Paired significance test (paired bootstrap or randomization test) between two configs, since both run on identical queries.
- A visible "these two configs are not statistically distinguishable on this eval set (n=87)" banner. Publishing that banner is worth more to your reputation than any leaderboard.

### 6.10 Run orchestration — P0

- Async job queue; runs survive page refresh.
- Live progress: per-config status, ETA, running cost meter with a hard budget cap the user sets up front.
- Aggressive caching keyed on content hash: parse results, chunk sets, and embeddings are all reused across configs that share a prefix of the pipeline. Without this, a 48-config sweep is unaffordable. With it, sweeping rerankers is nearly free.
- Cancel / pause / resume.
- Deterministic seeds recorded for anything stochastic.

### 6.11 Results explorer — P0

- **Leaderboard table** — sortable, with columns for each metric, latency, cost. Pareto-optimal rows highlighted.
- **Quality vs cost scatter** and **quality vs latency scatter**, with the Pareto frontier drawn. This is the chart people will screenshot.
- **Axis-effect view** — marginal effect of each axis holding others fixed ("switching to a reranker gained +0.11 Recall@5 on average across all embedders"). Presented as small multiples.
- **Per-question-type breakdown** — heatmap of config × question type.
- **Config diff** — pick two configs, see a side-by-side of every parameter difference and every question where they disagree.

### 6.12 Per-query inspector — P0

Click any question, see:
- The retrieved chunks in rank order, with gold spans highlighted inside them.
- Where the gold chunk actually ranked (or "not in top 100"), before and after reranking.
- The rank movement caused by the reranker, as an arrow diagram.
- The chunk boundaries drawn on the original document page, so the user can see when a chunker split a table in half.

That last one is a strong, visual, "oh, *that's* the problem" feature and worth prioritizing.

### 6.13 Export and reproducibility — P0

- **Permalink** for every run, with a public/private toggle. Public permalinks render an OG image with the headline chart — this is what makes the tool spread from a LinkedIn post.
- **Config export** as YAML/JSON.
- **Code export** — generate a working snippet for the winning config in LlamaIndex, LangChain, Haystack, and plain Python. P1 for the framework variants, P0 for plain Python.
- **Report export** — a one-page PDF/Markdown with methodology, matrix, leaderboard, and top charts. Engineers will paste this into their team's decision doc, which is exactly the adoption path you want.
- **Full run bundle** download (JSONL of every query, every retrieved chunk, every score) for offline analysis.

### 6.14 Demo mode — P0, and do not treat this as optional

A precomputed run on each sample corpus, loaded instantly with no keys and no compute. Every chart, the inspector, the diff view — all fully interactive on cached data. A visitor should reach an interesting insight within ten seconds of landing.

Pair it with a **Methodology** page that states plainly how gold spans are defined, how tokenization is normalized, what the confidence intervals mean, and what the tool cannot tell you. That page is the thing a senior reviewer reads to decide whether you are serious.

### 6.15 Bring-your-own-key — P0

Keys entered client-side, held in session, never persisted server-side, never logged. State this clearly in the UI. Offer a small free allowance on local CPU models so the tool works with no keys at all. Rate-limit and cap per session.

### 6.16 Later / P2

- Corpus-level diagnostics before any run: duplicate content, average passage length distribution, OCR-quality score, language mix, table density. Give the user a "your corpus is 40% tables, structural chunking is likely to matter here" hint.
- Generation panel: run the top-3 configs through an LLM and score faithfulness and answer relevance, to confirm retrieval gains actually translate.
- Multi-turn / conversational retrieval evaluation.
- Cost-constrained auto-tuner: "find the best config under $X per 1k queries and 300ms p95."
- Public community leaderboard by corpus type, opt-in.
- CLI + Python package, so the same sweep runs in CI against a team's real corpus.
- GitHub Action that fails a PR if retrieval quality regresses. This is the enterprise version of the idea and a natural sequel post.

---

## 7. Screens

1. **Landing** — one-sentence value prop, live demo-mode chart above the fold, "Try with your own docs" as secondary CTA.
2. **New experiment** — corpus upload → matrix builder → cost/time estimate → confirm.
3. **Eval set review** — the keyboard-driven accept/reject queue.
4. **Run monitor** — progress, live cost meter, cancel.
5. **Results** — leaderboard + Pareto charts + axis effects.
6. **Inspector** — per-query drilldown.
7. **Diff** — two-config comparison.
8. **Methodology** — static, written by you, the credibility page.
9. **Shared run** — read-only permalink view.

Design direction: dense, monospace-inflected, data-tool aesthetic. Think a research dashboard, not a SaaS landing page. Charts should be legible in a screenshot at LinkedIn dimensions, because that is where most people will first see them.

---

## 8. Architecture

**Frontend:** Next.js + Tailwind, deployed on Vercel (consistent with your existing site, so `/lab` can live on the same domain). Charts via Recharts or Observable Plot. Server-sent events for run progress.

**Backend:** FastAPI. Job queue via Redis + RQ or Celery; SQLite for metadata in single-node mode, Postgres if it grows. Deployed on Fly.io or Railway with a persistent volume.

**Embeddings:** Infinity server with ONNX runtime for CPU inference — you have already validated this path. One container, batched, dynamic batching on. Hosted models called directly with the user's key.

**Vector store:** LanceDB embedded, one table per (corpus, chunker, embedder) triple. Zero-ops, file-based, cheap to throw away.

**Cache:** content-addressed store keyed by SHA of `(stage inputs + params)`. Parse cache, chunk cache, embedding cache. Store embeddings as float16 to halve disk.

**Object storage:** S3-compatible for uploads and run bundles, with a TTL (suggest 7 days for anonymous uploads, stated in the UI).

**Isolation:** parsing runs untrusted user files. Sandbox the parse workers, cap memory and wall-clock per file, disable network in the parse container.

### Data model (sketch)

```
Corpus(id, name, created_at, file_meta[], hash)
Document(id, corpus_id, filename, page_count, bytes)
ParseRun(id, document_id, parser, params, artifact_ref, duration_ms)
ChunkSet(id, parse_run_id, chunker, params, chunk_count, hash)
Chunk(id, chunk_set_id, text, char_start, char_end, page, heading_path, token_counts{})
EmbeddingSet(id, chunk_set_id, model, dims, index_ref, build_ms, cost_usd)
EvalSet(id, corpus_id, source[auto|manual|import], version)
EvalItem(id, eval_set_id, question, qtype, gold_spans[{doc_id, start, end, grade}], status)
Config(id, parser, chunker, embedder, index, transform, reranker, k, params_hash)
Run(id, corpus_id, eval_set_id, config_id, status, metrics{}, cost_usd, latency{}, seed)
QueryResult(id, run_id, eval_item_id, retrieved[{chunk_id, score, rank, post_rerank_rank}], hit_ranks[])
```

The `gold_spans` as character offsets, resolved to chunks at scoring time by IoU overlap, is the schema decision that makes cross-chunker comparison valid. Everything else follows from it.

---

## 9. Constraints and budgets

- Hosted default caps: 25 files, 50 MB, 100 eval questions, 60 configs per sweep, $2 hard cost ceiling per session on BYOK.
- A 50-question × 20-config sweep on local CPU models should complete in under 10 minutes on a 4-core box. If it does not, the caching is wrong.
- Cold-start demo mode must render in under 1.5s.
- Monthly infra target under $30 for the hosted instance, since local models on CPU do the heavy lifting and heavy users self-host.

---

## 10. Roadmap

| Phase | Scope | Rough effort |
|---|---|---|
| **0. Spike** | Hardcoded pipeline, 2 chunkers × 2 embedders, metrics in a notebook, no UI. Validates the gold-span scoring approach. | 1 weekend |
| **1. Core loop** | Upload → auto eval set → OFAT sweep → leaderboard → per-query inspector. 3 parsers, 4 chunkers, 4 local embedders, BM25 + dense + hybrid, 1 reranker. Demo mode with 2 sample corpora. | 2–3 weekends |
| **2. Credibility** | Confidence intervals, paired significance tests, question-type slicing, methodology page, permalinks with OG images, config + code export. | 1–2 weekends |
| **3. Depth** | Query transforms, hosted embedders/rerankers via BYOK, parser diff view, chunk boundaries drawn on the page, report export. | 2 weekends |
| **4. Distribution** | CLI + pip package, CI action, community leaderboard. | Open-ended |

Ship phase 1 publicly. Do not wait for phase 3.

---

## 11. Success metrics

**As a portfolio piece (primary):**
- Reaches the methodology page — indicates a serious reader, not a bouncer.
- Inbound mentions of the project in conversations you did not initiate.
- Permalinks shared by other people.

**As a tool (secondary):**
- Demo-mode → own-corpus conversion rate.
- Completed sweeps per week.
- Eval sets reviewed by hand (proxy for real usage rather than curiosity).
- GitHub stars and, more meaningfully, issues filed by people running it on their own data.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Auto-generated eval sets are weak, so all conclusions are suspect | Filter aggressively, expose the review queue prominently, publish the filter criteria, allow import of a real gold set, validate the whole scorer against a known BEIR slice |
| Cost blows up on a big sweep | Hard budget cap, cost estimate before run, prefix caching, local models as default |
| Unfair comparisons quietly invalidate results (tokenizer mismatch, silent truncation, ANN vs exact) | Normalize units, surface truncation warnings, always include an exact-search reference row, document all of it |
| Scope creep into a full RAG framework | The non-goals section is load-bearing. Export configs, do not execute production pipelines |
| Nobody uses it | The portfolio value is realized at publication, not adoption. Demo mode plus the methodology page delivers that regardless of usage |
| Hosting abuse (large uploads, key scraping fears) | Caps, TTL deletion, client-side keys, sandboxed parse workers, clear privacy copy |

---

## 13. Content plan

The project is also a content engine. Each of these is a post backed by your own tool, with a permalink:

1. Chunk size versus recall, and why the sweet spot moved when the corpus changed.
2. What a reranker is actually worth, plotted against the candidate depth you feed it.
3. Hybrid search versus dense-only, sliced by question type — the tabular questions tell a different story.
4. Contextual retrieval: the quality gain, and the indexing cost nobody mentions.
5. Matryoshka dimension truncation — how much quality you lose cutting 1536 to 256, and what you save.
6. Two configs, identical scores, and why the difference was not statistically significant.
7. The parser diff: three parsers on the same financial table, three different answers.

Post six is the one that will earn the most respect from the people whose respect is worth having.

---

## 14. Open questions

- Do you host it with compute, or ship it primarily as a self-hosted tool with a read-only hosted demo? The latter is dramatically cheaper and arguably reads as *more* serious.
- Is code/SQL retrieval a first-class corpus type in v1, given you have CodeRankEmbed context? It is a differentiated angle almost no retrieval tool covers.
- Should the eval-set builder be spun out as its own small open-source package? It is the most reusable component and would likely get more stars standalone than the whole app.
- How much of your existing private research can be published as the seed methodology page without any employer-sensitive material?

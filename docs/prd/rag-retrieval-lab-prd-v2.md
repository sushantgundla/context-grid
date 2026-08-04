# RAG Retrieval Lab — Product Requirements Document

**Owner:** Sushant Gundla
**Status:** Draft v2 — supersedes [v1](./rag-retrieval-lab-prd-v1.md)
**Type:** Public portfolio project + open-source tool
**Target home:** sushantgundla.com/lab (or /retrieval-lab)

**See also:** [Feature Catalogue](./rag-retrieval-lab-features.md) — the full universe of features
across the whole RAG field, with stable IDs and priorities. That document also **widens the scope
from a retrieval bench to a full RAG Lab** (generation is now first-class) and revises the non-goals
in its §0.2. Where the two documents disagree on scope, the catalogue is newer.

**What changed in v2.** v1 was written without knowing what already exists. v2 adds a competitive
landscape (§3), decides which parts to *buy rather than build* (§4), sharpens the product down to
the four things nobody else does (§5), promotes and demotes features accordingly (§8), and answers
the open questions v1 left dangling (§16). Sections marked **[NEW]** did not exist in v1.
Sections marked **[CHANGED]** exist in v1 but the substance has moved.

---

## 1. One-liner

Upload a document, define a set of questions, and get a ranked, reproducible comparison of every
retrieval pipeline configuration — **parser** × chunker × embedder × index × reranker — scored on
retrieval quality, latency, and cost.

The bolded word is the whole differentiator. Read §3 before §5.

## 2. Why this project

Most RAG content on the internet is anecdote. "Semantic chunking is better." "Use a reranker."
"512 tokens with 50 overlap is the sweet spot." Almost nobody publishes the measurement that backs
it, and almost nobody can reproduce someone else's setup on their own corpus.

The value of this tool is that it turns those opinions into numbers on *your* documents. The value
of it as a portfolio piece is that building it correctly requires you to have solved the parts of
RAG that are actually hard: ground truth construction, metric selection, fair comparison under
different tokenizers, cost accounting, and orchestration of long-running jobs.

A demo chatbot proves you can call an API. This proves you can run an experiment.

### Why it fits you specifically

You have already done the underlying research in private: PDF parser evaluation for RAG pipelines,
open-source embedding models for CPU inference, Infinity with ONNX runtime, `bge-base-en-v1.5` for
text and `CodeRankEmbed` for code. This project is that research made public, reusable, and
defensible. The build cost is low relative to the credibility return because you are not learning
the domain, you are packaging it.

---

## 3. Competitive landscape **[NEW]**

This space is not empty. Six or seven projects overlap. None of them do the same job, and knowing
exactly where each one stops is what lets this project be positioned honestly rather than as
"another RAG eval tool."

### 3.1 What exists

| Project | What it does | Where it stops |
|---|---|---|
| [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) (Marker-Inc-Korea) | Closest open-source match. AutoML-style sweep over RAG modules, YAML-driven, auto eval-data creation from raw docs, deploys the winner | **No parser axis.** No cost model. Config-file driven — the artefact is a `summary.csv`, not something you can show anyone. Node/module abstraction is heavy to extend |
| [RAGBuilder](https://github.com/kruxai/ragbuilder) (KruxAI) | Bayesian hyperparameter search over chunker / chunk size / embedder / retriever; synthetic test data; exports the winning pipeline as code | It *optimises for* you and hides the comparison. You get a winner, not a landscape. No parser axis, no latency/cost axis, no per-query inspection |
| [Ragas](https://www.ragas.io/) | The de-facto metrics library. Synthetic test-set generation via a knowledge graph over source docs. LLM-judged context precision/recall | A library, not a bench. No sweep, no UI, no human correction of the generated set. Generation-centric metrics |
| [open-rag-eval](https://github.com/vectara/open-rag-eval) (Vectara) | Evaluation without golden answers; can compare chunking strategies and name a winner | Answer-quality first, retrieval second — the inverse of this project's N3. Vectara-flavoured |
| [reranker-eval](https://github.com/agentset-ai/reranker-eval) | nDCG, Recall **and latency** for rerankers | One axis only. Fixed corpus. Exactly the shape of one row of our leaderboard |
| [rag-chunk](https://github.com/messkan/rag-chunk), [ChunkViz](https://testdev.tools/chunk-viz/), [Ailog Chunking Simulator](https://app.ailog.fr/en/tools/chunking-simulator), RAG Chunking Lab | Interactive chunk visualisers. Pretty, instant, client-side | No retrieval, no scoring, no ground truth. They show you *what* a chunker did, never whether it helped |
| [pdf-parser-benchmark](https://github.com/applied-artificial-intelligence/pdf-parser-benchmark), Docling/Marker/Unstructured blog benchmarks | Parser accuracy numbers on a fixed set of PDFs | Static, published-once, on someone else's documents, and **never connected to downstream retrieval quality**. They tell you Docling extracted the table; they never tell you it changed Recall@5 |
| Arize Phoenix, DeepEval, TruLens, LangSmith, Braintrust | Tracing + metrics + CI gates for a pipeline you already run | They observe **one** pipeline in production. They do not *compare fifty candidate pipelines before you pick one*. Different moment in the lifecycle |
| [Vectorize.io RAG Evaluation](https://docs.vectorize.io/learn/getting-started/rag-evaluation-quick-start/) | **The closest commercial product.** Upload docs, pick embedders + chunking strategies, it runs parallel pipelines and reports nDCG and relevancy in under a minute. Plus a "RAG Sandbox" for interactive testing | Closed-source SaaS, funnels into their platform. Fixed at ~4 pipelines. No parser axis, no cost-per-1k-queries, no statistical significance, no span-level gold, no exportable config for someone else's stack |

### 3.2 The gaps — and they are real

1. **Nobody sweeps the parser.** Parser benchmarking and retrieval benchmarking are two separate
   literatures. The parser benchmarks measure character accuracy and table cell recall; the
   retrieval tools start from clean text and assume parsing is solved. Joining them — *"pdfplumber
   beats PyMuPDF by 11 points of Recall@5 on these contracts"* — is genuinely unclaimed territory
   and is the single most defensible thing this project can own.
2. **Nobody shows cost and latency beside quality.** Every open-source tool ranks on quality alone.
   `reranker-eval` includes latency for rerankers only. Cost per 1,000 queries on the same chart as
   quality, with the Pareto frontier drawn, does not exist anywhere.
3. **Everything is a Python config file.** AutoRAG and RAGBuilder both need a repo, a YAML, keys and
   a terminal before you learn anything. The visualisers are instant but shallow. A zero-setup web
   demo that reaches a real insight in ten seconds sits in the empty middle.
4. **Eval sets are generated, never corrected.** Ragas, AutoRAG and RAGBuilder all generate
   synthetic questions. None ship a review queue to fix the bad ones. Everyone in the field knows
   auto-generated ground truth is the weak link and nobody has built the obvious fix.
5. **Gold is stored as a chunk ID, so cross-chunker comparison is quietly invalid.** Only
   [LegalBench-RAG](https://arxiv.org/abs/2408.10343) does it properly — expert-annotated **character
   spans**, with character-level precision and recall. That is the same design decision as §9 of this
   document, which means the approach is externally validated *and* there is a ready-made corpus to
   prove the scorer is correct.
6. **Tokenizer normalisation is unaddressed.** Nothing found reports chunk size in a consistent unit
   across models with different tokenizers. Most published comparisons of "512 vs 1024" are
   comparing different things.
7. **Retrieval regression gating is described but not productised.** The 2026 consensus stack is
   "Ragas for metric science, DeepEval for CI gates." Nobody gates on *retrieval* specifically, with
   a manifest diff naming which axis changed.

### 3.3 Positioning sentence

> AutoRAG and RAGBuilder find you a config. Phoenix and DeepEval watch the config you chose.
> Retrieval Lab is the **bench you stand at before either** — including the parsing step everyone
> else skips, priced and timed, with ground truth you can actually trust.

---

## 4. Build vs. buy **[NEW]**

v1 implied building everything. Several parts are solved better elsewhere and using them is both
cheaper and a maturity signal — the methodology page saying "metrics computed by `ranx`, not by me"
is worth more than a hand-rolled nDCG.

| Component | Decision | Why |
|---|---|---|
| Ranking metrics (nDCG, MAP, MRR, Recall@k) | **Use [`ranx`](https://github.com/AmenRa/ranx)** | Numba-fast, peer-reviewed (ECIR 2022), and the reference implementation people trust |
| Paired significance tests | **Use `ranx`** | It ships paired statistical comparison and LaTeX/table export. §11's statistical honesty goal becomes ~20 lines instead of a week |
| Fusion (RRF, weighted score normalisation) | **Use `ranx`** | It has both, plus automatic fusion-weight optimisation — which quietly gives us a sweepable hybrid-alpha axis for free |
| Chunkers | **Use [`chonkie`](https://docs.chonkie.ai/common/open-source)** as the engine, wrap for sweeping | Nine strategies including late chunking, neural (BERT boundary detection) and **AST-based code chunking**; Rust-backed, ~33× faster on token chunking. Building nine chunkers by hand is a month we do not need to spend |
| Parsers | **Wrap directly** (PyMuPDF, pdfplumber, Unstructured, Docling, Marker/MinerU) | No abstraction layer exists that preserves character offsets across all of them, and offsets are non-negotiable (§9). This wrapper *is* our contribution |
| Synthetic question generation | **Own it**, but read Ragas' knowledge-graph approach | The filters and the review queue are the product; generation is the commodity part |
| Vector store | LanceDB embedded | Unchanged from v1 |
| Metric of "did the chunk contain the gold span" | **Own it** — this is the core scorer | Nothing off-the-shelf resolves character spans to chunks by IoU |

Net effect: phase 0 and phase 1 get materially shorter, and the parts we build are the parts that
are actually novel.

---

## 5. The four things that make this different **[NEW]**

Everything in §8 serves one of these. If a feature serves none of them, it is P2.

1. **The parser is an axis.** No other tool has it. It is also the axis with the largest and least
   understood effect, especially on table-heavy documents.
2. **Quality, latency and cost on one Pareto chart.** The screenshot that travels.
3. **Ground truth as character spans, with a human review queue.** Makes cross-chunker comparison
   valid, and makes the eval set trustworthy rather than plausible.
4. **Statistical honesty as a visible feature.** A banner reading "these two configs are not
   distinguishable on this eval set (n=87)" is the thing a senior reviewer will remember.

---

## 6. Goals and non-goals **[CHANGED]**

### Goals

- **G1.** Run a full-factorial, OFAT or staged sweep across retrieval configurations on the user's
  own corpus, without writing code.
- **G2.** Produce ground truth cheaply — auto-generate, filter, then let the user correct it.
- **G3.** Report retrieval quality, p50/p95 latency, and cost per 1k queries together, on one screen.
- **G4.** Make every result reproducible: shareable permalink, exportable config, and a **run
  manifest** (§14) that pins every version and hash.
- **G5.** Work in a zero-setup demo mode with precomputed results — value in under ten seconds.
- **G6. [NEW]** Prove the scorer is correct by reproducing a published benchmark. Ship a
  "validation" page showing our numbers on a LegalBench-RAG slice next to the paper's. No other
  tool in §3 does this, and it converts "trust me" into "check me."
- **G7. [NEW]** Make the parsing step measurable end-to-end — parser choice scored on downstream
  retrieval quality, not on character accuracy in isolation.

### Non-goals

- **N1.** Not a RAG framework. It emits configs *for* LlamaIndex, LangChain, Haystack.
- **N2.** Not a production serving layer.
- **N3.** Not primarily an answer-quality evaluator. Generation metrics are a secondary panel.
- **N4.** No fine-tuning of embedding models in v1.
- **N5. [NEW]** Not an auto-optimiser. RAGBuilder does Bayesian search and hides the landscape;
  we show the landscape. Staged/greedy mode is as close as we get, and it is labelled as a shortcut.
- **N6. [NEW]** Not an observability tool. No tracing, no production monitoring, no live traffic.
  That is Phoenix and TruLens, and competing there is a losing fight.

---

## 7. Users **[CHANGED]**

| Persona | What they want | What they do in the tool |
|---|---|---|
| **The engineer picking a stack** | "Which parser, chunker and embedder for these contracts?" | Uploads 10 representative docs, runs an OFAT sweep, exports the winning config |
| **The engineer debugging bad retrieval** | "Why does this one question always fail?" | Per-query inspector: what got retrieved, where the gold span actually ranked, and whether a chunker cut a table in half |
| **The engineer defending a choice internally** *(new)* | "I need to show my team why we are switching parsers" | Runs two configs, exports the one-page report with the significance test, pastes it into the decision doc |
| **The skeptic / hiring manager** | "Does this person know what they're talking about?" | Demo mode → methodology page → validation page. Impressed in 90 seconds |
| **The LinkedIn reader** | "Interesting chart" | Clicks a permalink, sees a real Pareto chart, follows |

Personas 4 and 5 are the actual business case. Design demo mode for them first, not last.

---

## 8. Core concept: the experiment matrix

Every run is a point in a configuration space:

```
Corpus × Parser × Chunker × Embedder × Index/Search × Query Transform × Reranker
```

The UI shows the combinatorial count live — "3 × 4 × 2 × 2 = 48 configurations, est. 14 min, est.
$0.62" — and warns before expensive sweeps. Three sweep modes:

- **Full factorial** — every combination. Fine for small selections.
- **One-factor-at-a-time (OFAT)** — hold a baseline, vary one axis. Cheap, interpretable, the default.
- **Staged/greedy** — best chunker, freeze, then embedders, then rerankers. Cheapest path to a good
  config, with a visible caveat that it can miss interactions.

**[NEW] Cost-aware sweep suggestion.** Before the run, propose a reduced matrix that keeps the
informative comparisons and drops the redundant ones, with the saving stated ("48 → 19 configs,
$0.62 → $0.24, you lose only the parser × reranker interaction"). Accept or ignore.

---

## 9. Feature specification **[CHANGED]**

Priority key: **P0** = required for launch, **P1** = fast follow, **P2** = later.
Changes from v1 are flagged inline.

### 9.1 Corpus ingestion — P0

- Upload PDF, DOCX, HTML, Markdown, TXT. Cap at 25 files / 50 MB hosted; unlimited self-hosted.
- Paste-text corpus for quick tests. URL ingestion with crawl depth 0 or 1 — P1.
- Built-in sample corpora:
  - A messy financial report PDF with tables (the parser-axis showcase)
  - A technical docs set (heading-heavy Markdown)
  - A legal/insurance policy document (dense, cross-referential)
  - A code + SQL corpus (**promoted to P0** — see §16)
  - **[NEW]** A LegalBench-RAG slice, used for the validation page (G6)
- Per-file preview: page count, language, text-layer present or OCR needed.

### 9.2 Parsing layer — P0 — **the headline axis, promoted throughout**

Selectable parsers, each producing a normalised intermediate representation: text blocks +
structure metadata + **page and character-offset provenance**. The offsets are mandatory; a parser
that cannot give us offsets cannot be an axis, because §9.8's gold spans depend on them.

- PyMuPDF (fast baseline)
- pdfplumber (table-aware)
- Unstructured (`hi_res` and `fast` strategies as separate points on the axis)
- Docling
- Marker or MinerU (layout/ML-based)
- OCR fallback: Tesseract or PaddleOCR for image-only pages
- LlamaParse, Azure Document Intelligence / AWS Textract / Google Document AI (hosted, BYOK) — P1

**Parser diff view — promoted from P1 to P0.** Same page, side by side, under two parsers, with
extracted text overlaid on the rendered page. Nothing in §3 shows this. It is the most
screenshot-able feature in the product and it is the visual proof of differentiator #1. Launching
without it wastes the one thing nobody else has.

**[NEW] Parser → retrieval attribution.** The chart that does not exist anywhere: parser on the
x-axis, Recall@5 on the y-axis, chunker/embedder held fixed. Published parser benchmarks stop at
character accuracy; this connects the parse to the outcome. Make it the default view of the sample
financial corpus in demo mode.

Parser sub-metrics where computable: table cell recall on a labelled sample, reading-order
correctness, characters extracted per page, wall-clock per page, **cost per page** for hosted parsers.

### 9.3 Chunking layer — P0 — **built on `chonkie`**

| Strategy | Key params |
|---|---|
| Fixed-size token | size, overlap, tokenizer |
| Recursive character | size, overlap, separator hierarchy |
| Sentence window | window size, stride |
| Structural / heading-aware | max depth, min/max chunk size, keep-heading-path |
| Semantic (embedding breakpoint) | percentile threshold, buffer size, breakpoint method |
| **Neural (BERT boundary detection)** **[NEW]** | model, threshold |
| Proposition-based (LLM-extracted atomic facts) | model, max propositions per block |
| Parent-document / small-to-big | child size, parent size |
| Contextual retrieval (LLM-prepended context) | context model, context length budget |
| Late chunking (embed long, pool per chunk) | requires long-context embedder |
| **Code / AST-aware** **[NEW, P0]** | language, max node size — pairs with `CodeRankEmbed` |

**Critical implementation note (unchanged, and now known to be unaddressed by anyone else):** chunk
size must be reported in a consistent unit across models with different tokenizers. Store character
count *and* token count per tokenizer; expose a unit toggle. State it on the methodology page.

**[NEW] Every chunk carries `char_start` / `char_end` back to the source document.** This is not
optional metadata — the scorer depends on it. Any chunker that cannot preserve offsets (some
LLM-rewriting strategies) must record a best-effort alignment and be **flagged in the UI as
approximate**. Being honest about that limitation is better than hiding it.

Also expose: metadata attachment (heading path, page, source file, section) and near-duplicate
chunk de-duplication.

### 9.4 Embedding layer — P0

**Local / open (CPU-first, served via Infinity + ONNX):** `bge-base-en-v1.5`, `bge-large-en-v1.5`,
`bge-m3`, `e5-base/large-v2`, `multilingual-e5`, `gte-base/large`, `nomic-embed-text-v1.5`
(Matryoshka dimensions as a sweepable param), `all-MiniLM-L6-v2` (speed baseline),
`jina-embeddings-v3`, `CodeRankEmbed`.

**Hosted (BYOK):** OpenAI `text-embedding-3-small`/`-large` (with `dimensions`), Cohere `embed-v3`
(with `input_type`), Voyage, Google Gemini embeddings.

Per-model handling that must be correct or results are garbage: query vs document prefixes (E5's
`query:`/`passage:`, BGE's query instruction, Cohere's `input_type`), normalisation, and max
sequence length truncation. **Surface a loud warning when a chunk exceeds a model's context and is
being truncated.** Report per model: dimensions, index size on disk, embed throughput, cost per
million tokens.

### 9.5 Index and search layer — P0

- **Dense** — cosine/dot, exact and ANN (HNSW `M`, `efConstruction`, `efSearch` sweepable; always
  include an exact-search reference row so ANN recall loss is visible).
- **Sparse** — BM25 (`k1`, `b` tunable); learned sparse (SPLADE) — P1.
- **Hybrid** — RRF (`k`) and weighted score normalisation (`alpha`). **[CHANGED]** Both come from
  `ranx`, including its automatic fusion-weight optimisation, which turns "best alpha" from a manual
  sweep into a solved sub-problem.
- **Multi-vector / late interaction** — ColBERT-style — P1.
- Metadata filtering and section-scoped search — P1.
- Backends: LanceDB embedded by default; Qdrant and pgvector for self-hosters.

### 9.6 Query transformation layer — P1

None (baseline), HyDE, multi-query expansion, query decomposition, step-back prompting,
conversational rewriting. Each carries an LLM cost and latency penalty attributed to the config in
the cost panel. **Demoted firmly to P1** — the interesting result here ("it usually does not pay")
is a *content* outcome, and it needs the cost panel (P0) far more than it needs breadth of methods.

### 9.7 Reranking layer — P0

None (baseline), `bge-reranker-base` / `v2-m3`, `mxbai-rerank`, `jina-reranker-v2`, Cohere Rerank
(BYOK), LLM-as-reranker (P1).

Sweepable: candidate depth fed to the reranker (top 20/50/100) and final k. The candidate-depth
curve is one of the most useful charts the tool can produce. `reranker-eval` benchmarks rerankers
with latency but on a fixed corpus — **on your own corpus, with cost attached, is still open.**

### 9.8 Eval set builder — P0, the heart of the product

Three paths:

1. **Auto-generated (default).** Sample chunks, LLM writes questions answerable *only* from that
   chunk, source span recorded as gold. Then filter automatically:
   - Reject questions answerable from general knowledge (ask the LLM with no context).
   - Reject questions with pronouns or dangling references.
   - Reject near-duplicates.
   - Flag questions whose gold text also appears verbatim elsewhere in the corpus.
   - **[NEW]** Reject questions the *baseline* config answers with rank 1 trivially — they have no
     discriminating power and inflate every score.
2. **Human-in-the-loop review.** Keyboard-driven queue: question left, gold span right, keys for
   accept / reject / edit / mark-multi-hop. Target under 5 seconds per item. **This is the gap
   nobody has filled** (§3.2 #4) — Ragas, AutoRAG and RAGBuilder all generate and none let you fix.
3. **Import.** CSV/JSONL of `question, answer, gold_spans` or `question, gold_doc_id`. BEIR-format
   import — P1. **[NEW] LegalBench-RAG import — P0**, because it drives the validation page.

**Question type tagging** — auto-classify each question as factoid, multi-hop, comparative,
numeric/tabular, or summarisation-style, and report metrics *per type*. Semantic chunking may win
overall while losing badly on tabular questions. This slicing is what separates the tool from a
leaderboard toy.

**Gold set integrity:**
- Multiple gold spans per question.
- Graded relevance (2 fully answers / 1 partial / 0 irrelevant) so nDCG means something.
- **Gold stored as a character span in the source document, never as a chunk ID.** Resolved to
  chunks at scoring time by span overlap with a configurable IoU threshold. **This remains the
  single most important design decision in the document**, and §3.2 #5 confirms only
  LegalBench-RAG does it this way — which is precisely why it is also our external validation set.

**[NEW] Eval-set quality score.** Show the user a one-line health read on their own eval set:
n questions, % human-reviewed, type distribution, mean discriminating power, and whether n is large
enough to detect the differences they are asking about. Nothing else in §3 tells a user their ground
truth is too weak to support their conclusion.

### 9.9 Metrics engine — P0 — **now `ranx`-backed**

Retrieval metrics at k ∈ {1, 3, 5, 10, 20}: Recall@k (headline), Precision@k, nDCG@k with graded
relevance, MRR, MAP, Hit rate@k, mean rank of first gold, chunk attribution rate (fraction of
retrieved chunks containing any gold span — measures wasted context). Ragas-style LLM-judged context
precision/recall — P1.

**[NEW] Character-level precision and recall**, following LegalBench-RAG: of the characters
retrieved, what fraction are gold, and of the gold characters, what fraction were retrieved. This is
the metric that exposes a config returning huge chunks to game Recall@k. Chunk-level Recall@5 can be
1.0 while character precision is 0.04 — and that config is burning your context window. Nobody
surfaces this in a comparison UI.

Operational metrics, always alongside: index build time and size; query latency p50/p95/p99 broken
down by stage (parse → embed → search → rerank); cost per 1,000 queries and one-time indexing cost,
itemised; tokens sent to the generator at the chosen k.

**Statistical honesty — promoted from P1 to P0**, because `ranx` makes it nearly free and because it
is differentiator #4:
- Bootstrap confidence intervals on the headline metric.
- Paired significance test between any two configs (they run on identical queries).
- A visible "these two configs are not statistically distinguishable on this eval set (n=87)"
  banner, rendered as a first-class result state rather than a footnote.

### 9.10 Run orchestration — P0

Async job queue; runs survive refresh. Live per-config progress, ETA, running cost meter with a hard
user-set budget cap. Cancel / pause / resume. Deterministic seeds recorded.

**Aggressive prefix caching keyed on content hash.** Parse results, chunk sets and embeddings are
reused across configs sharing a pipeline prefix. Without it a 48-config sweep is unaffordable; with
it, sweeping rerankers is nearly free. **[NEW]** Show the cache saving in the UI — "37 of 48 configs
reused cached embeddings, saved ~$0.41 and 6 minutes." It makes an invisible engineering decision
visible, which is exactly what a portfolio piece should do.

### 9.11 Results explorer — P0

- **Leaderboard** — sortable, every metric plus latency and cost, Pareto-optimal rows highlighted.
- **Quality vs cost** and **quality vs latency** scatter with the Pareto frontier drawn. The
  screenshot chart.
- **Axis-effect view** — marginal effect of each axis holding others fixed ("adding a reranker gained
  +0.11 Recall@5 on average across all embedders"), as small multiples.
- **Per-question-type breakdown** — heatmap of config × question type.
- **Config diff** — two configs, every parameter difference and every question where they disagree.
- **[NEW] Recommendation card.** One plain-English paragraph at the top: what won, by how much,
  whether it is significant, what it costs relative to the baseline, and what it is worst at. Every
  tool in §3 outputs a table and leaves the reader to interpret it. Writing the conclusion in words
  is the thing a hiring manager actually reads.

### 9.12 Per-query inspector — P0

Click any question: retrieved chunks in rank order with gold spans highlighted inside them; where
the gold ranked before and after reranking (or "not in top 100"); rank movement from the reranker as
an arrow diagram; **chunk boundaries drawn on the original document page**, so a chunker splitting a
table in half is visible rather than inferred. That last one is the strongest "oh, *that's* the
problem" moment in the product — keep it P0.

### 9.13 Export and reproducibility — P0

- **Permalink** per run, public/private toggle. Public permalinks render an OG image with the
  headline chart — this is how it spreads from a LinkedIn post.
- **Config export** as YAML/JSON.
- **Code export** — working snippet for the winning config in plain Python (P0); LlamaIndex,
  LangChain, Haystack variants (P1).
- **Report export** — one-page PDF/Markdown with methodology, matrix, leaderboard, significance and
  top charts. Engineers paste this into a decision doc, which is the adoption path we want.
- **Full run bundle** — JSONL of every query, retrieved chunk and score.
- **[NEW] Run manifest.** A single hash-pinned record: parser versions, chunker params, embedding
  model revisions, tokenizer, index params, corpus content hash, eval-set version, seeds, library
  versions. Two runs with the same manifest hash must produce identical numbers. This is what makes
  §9.15's regression gate possible and is the difference between "reproducible" as a claim and as a
  property.

### 9.14 Demo mode — P0, not optional

Precomputed runs on every sample corpus, loaded instantly, no keys, no compute. Every chart, the
inspector, the parser diff — fully interactive on cached data. A visitor reaches an interesting
insight within ten seconds.

Default landing view: **the parser-attribution chart on the financial corpus**, because it is the
one result no other tool can show.

Paired with a **Methodology** page stating plainly how gold spans are defined, how tokenisation is
normalised, what the confidence intervals mean, which metrics come from `ranx`, and what the tool
cannot tell you.

**[NEW] Validation page (G6).** Our numbers on a LegalBench-RAG slice next to the published ones,
with the delta and an explanation of any gap. This is the single cheapest credibility feature in the
document and nothing in §3 offers an equivalent.

### 9.15 CLI, package and CI gate — **promoted from P2 to P1**

v1 parked this at "later." The research says retrieval regression gating is widely *described* and
not productised, and it is the natural enterprise sequel.

- `pip install retrieval-lab`; `retrieval-lab sweep config.yaml`.
- **Regression gate:** run the pinned config against the pinned eval set; fail if Recall@5 drops
  more than a threshold, and **diff the run manifest against the last passing run to name the axis
  that changed**. That manifest diff is the part nobody ships.
- GitHub Action wrapper.

### 9.16 Bring-your-own-key — P0

Keys entered client-side, held in session, never persisted server-side, never logged, stated clearly
in the UI. A free allowance on local CPU models so the tool works with no keys at all. Rate-limit
and cap per session.

### 9.17 Corpus diagnostics — **promoted from P2 to P0 (lite version)**

Before any run: duplicate content, passage length distribution, OCR-quality score, language mix,
**table density**, code/prose ratio. Then a hint: "your corpus is 40% tables — parser choice and
structural chunking are likely to dominate here; we have pre-selected a matrix that tests that."

Cheap to compute, it makes the empty state useful, and it turns the matrix builder from a blank form
into a guided decision. Full diagnostics stay P2; the fingerprint plus hint is P0.

### 9.18 Later / P2

- Generation panel: run the top-3 configs through an LLM, score faithfulness and answer relevance,
  to confirm retrieval gains translate.
- Multi-turn / conversational retrieval evaluation.
- Cost-constrained auto-tuner: "best config under $X per 1k queries and 300 ms p95."
- Public community leaderboard by corpus type, opt-in.
- **[NEW]** Multimodal corpora (charts, figures) — the parser axis is the natural entry point, but
  the gold-span model needs rethinking for non-text evidence.

---

## 10. Screens **[CHANGED]**

1. **Landing** — one-sentence value prop, live parser-attribution chart above the fold, "Try with
   your own docs" secondary.
2. **New experiment** — upload → **corpus fingerprint + suggested matrix** → matrix builder →
   cost/time estimate → confirm.
3. **Eval set review** — the keyboard accept/reject queue, with the eval-set quality score.
4. **Run monitor** — progress, live cost meter, cache-saving readout, cancel.
5. **Results** — recommendation card, leaderboard, Pareto charts, axis effects.
6. **Inspector** — per-query drilldown with chunk boundaries on the page.
7. **Diff** — two-config comparison with the significance verdict.
8. **Parser diff** — *(new screen, was a sub-view)* side-by-side page rendering.
9. **Methodology** — static, written by you.
10. **Validation** — *(new)* our numbers vs LegalBench-RAG's published numbers.
11. **Shared run** — read-only permalink view.

Design direction: dense, monospace-inflected, research-dashboard aesthetic. Charts must be legible
in a screenshot at LinkedIn dimensions, because that is where most people will first see them.

---

## 11. Architecture **[CHANGED]**

**Frontend:** Next.js + Tailwind on Vercel, so `/lab` lives on the existing domain. Charts via
Recharts or Observable Plot. Server-sent events for run progress.

**Backend:** FastAPI. Job queue via Redis + RQ or Celery. SQLite for metadata single-node, Postgres
if it grows. Fly.io or Railway with a persistent volume.

**Embeddings:** Infinity + ONNX runtime for CPU inference — already validated. One container,
dynamic batching on. Hosted models called directly with the user's key.

**Metrics:** `ranx` for all ranking metrics, fusion and paired significance tests. **[NEW]**

**Chunking:** `chonkie` wrapped in an offset-preserving adapter. **[NEW]**

**Vector store:** LanceDB embedded, one table per (corpus, parser, chunker, embedder) tuple.
**[CHANGED — parser added to the key, since parser is now an axis.]**

**Cache:** content-addressed, keyed by SHA of (stage inputs + params). Parse, chunk and embedding
caches. Embeddings stored as float16.

**Object storage:** S3-compatible, TTL of 7 days for anonymous uploads, stated in the UI.

**Isolation:** parsing runs untrusted user files. Sandbox parse workers, cap memory and wall-clock
per file, no network in the parse container.

### Data model

```
Corpus(id, name, created_at, file_meta[], hash, fingerprint{})
Document(id, corpus_id, filename, page_count, bytes)
ParseRun(id, document_id, parser, params, artifact_ref, duration_ms, cost_usd, offsets_exact:bool)
ChunkSet(id, parse_run_id, chunker, params, chunk_count, hash, offsets_exact:bool)
Chunk(id, chunk_set_id, text, char_start, char_end, page, heading_path, token_counts{})
EmbeddingSet(id, chunk_set_id, model, dims, index_ref, build_ms, cost_usd)
EvalSet(id, corpus_id, source[auto|manual|import], version, quality{})
EvalItem(id, eval_set_id, question, qtype, gold_spans[{doc_id, start, end, grade}], status, discriminating_power)
Config(id, parser, chunker, embedder, index, transform, reranker, k, params_hash)
Run(id, corpus_id, eval_set_id, config_id, status, metrics{}, cost_usd, latency{}, seed, manifest_hash)
QueryResult(id, run_id, eval_item_id, retrieved[{chunk_id, score, rank, post_rerank_rank}], hit_ranks[], char_precision, char_recall)
Manifest(hash, versions{}, params{}, corpus_hash, evalset_version, seeds{})
```

`gold_spans` as character offsets, resolved to chunks at scoring time by IoU overlap, is the schema
decision that makes cross-chunker *and* cross-parser comparison valid. `offsets_exact` is the new
honesty flag: when a parser or chunker cannot guarantee offsets, the UI says so.

---

## 12. Constraints and budgets

- Hosted caps: 25 files, 50 MB, 100 eval questions, 60 configs per sweep, $2 hard ceiling per BYOK
  session.
- A 50-question × 20-config sweep on local CPU models completes in under 10 minutes on a 4-core box.
  If it does not, the caching is wrong.
- Cold-start demo mode renders in under 1.5 s.
- Monthly infra target under $30, since local CPU models do the heavy lifting and heavy users
  self-host.

---

## 13. Roadmap **[CHANGED]**

Shorter than v1 in phases 0–1 because `ranx` and `chonkie` remove real work; reordered so the
differentiators ship in phase 1 rather than phase 3.

| Phase | Scope | Rough effort |
|---|---|---|
| **0. Spike** | Hardcoded pipeline, 2 parsers × 2 chunkers × 2 embedders, `ranx` metrics, gold-span IoU scorer validated against a LegalBench-RAG slice. No UI. **Ship nothing until the scorer is proven.** | 1 weekend |
| **1. Core loop + the differentiators** | Upload → corpus fingerprint → auto eval set + review queue → OFAT sweep → leaderboard with cost/latency Pareto → per-query inspector → **parser diff view**. 4 parsers, 5 chunkers, 4 local embedders, BM25 + dense + hybrid, 1 reranker. Demo mode on 2 sample corpora. | 3–4 weekends |
| **2. Credibility** | Confidence intervals, paired significance, question-type slicing, character-level precision/recall, methodology page, **validation page**, permalinks with OG images, config + code + manifest export. | 1–2 weekends |
| **3. Depth** | Query transforms, hosted embedders/rerankers via BYOK, chunk boundaries drawn on the page, report export, code/AST corpus. | 2 weekends |
| **4. Distribution** | CLI + pip package, CI regression gate with manifest diff, community leaderboard. | Open-ended |

Ship phase 1 publicly. The parser diff view moved into phase 1 deliberately — launching without the
one feature nobody else has would waste the launch.

---

## 14. Success metrics

**As a portfolio piece (primary):** reaches the methodology or validation page; inbound mentions you
did not initiate; permalinks shared by other people.

**As a tool (secondary):** demo-mode → own-corpus conversion; completed sweeps per week; eval sets
reviewed by hand (the real usage proxy); GitHub issues filed by people running it on their own data.

---

## 15. Risks **[CHANGED]**

| Risk | Mitigation |
|---|---|
| Auto-generated eval sets are weak, so all conclusions are suspect | Filter aggressively, expose the review queue prominently, publish the filter criteria, allow import, **and validate the whole scorer against LegalBench-RAG on the public validation page** |
| Cost blows up on a big sweep | Hard budget cap, pre-run estimate, prefix caching, local models by default, cost-aware matrix suggestion |
| Unfair comparisons quietly invalidate results (tokenizer mismatch, silent truncation, ANN vs exact) | Normalise units, surface truncation warnings, always include an exact-search reference row, `offsets_exact` flag, document all of it |
| Scope creep into a full RAG framework | The non-goals are load-bearing. N5 and N6 exist because AutoRAG and Phoenix are the two gravity wells |
| **[NEW] "This is just AutoRAG"** | Lead with the parser axis and the cost Pareto everywhere — landing page, README, first post. Link to AutoRAG and say plainly what it does better. Generosity about prior art reads as confidence |
| **[NEW] Vectorize.io or another vendor ships the same thing** | They will not open-source it, and they will not add a parser axis that makes their own parsing look replaceable. Open source, self-hostable, exportable-to-anyone's-stack is the defensible position |
| Nobody uses it | Portfolio value is realised at publication, not adoption. Demo mode + methodology + validation delivers that regardless |
| Hosting abuse | Caps, TTL deletion, client-side keys, sandboxed parse workers, clear privacy copy |

---

## 16. Answers to v1's open questions **[NEW]**

v1 §14 left four questions open. The research answers three of them.

**Host with compute, or self-host with a read-only demo?**
→ **Read-only hosted demo + BYOK-light hosted mode.** Precomputed demo runs cost nothing and serve
personas 4 and 5, who are the actual business case. A capped BYOK mode lets a curious engineer try
their own docs without you paying for it. Everything heavy is self-hosted. This is also how AutoRAG
and RAGBuilder distribute, so it reads as normal rather than cheap.

**Is code/SQL retrieval first-class in v1?**
→ **Yes.** `chonkie` ships AST-based code chunking, you already have `CodeRankEmbed` context, and
nothing in §3 covers code retrieval evaluation at all. It is a differentiated angle for roughly a
weekend of extra work, and it makes the tool relevant to every engineer building code search.

**Should the eval-set builder be spun out as its own package?**
→ **Yes, but after phase 2, not before.** It is the most reusable component and the gap in §3.2 #4
is real — Ragas generates and does not correct. A standalone `span-eval` package (character-span
ground truth + generation + filters + review UI + IoU scorer) would likely get more stars than the
whole app. Spinning it out early splits your attention before either thing is good.

**How much private research can be published?**
→ Still yours to answer. Practical framing: publish *method* and *public-model results*, never
employer corpora or internal numbers. Every parser and embedder in this document is publicly
available, so a clean-room rerun on the sample corpora produces publishable numbers that owe nothing
to prior work.

---

## 17. Content plan **[CHANGED]**

Each item is a post backed by your own tool, with a permalink. Reordered so the unclaimed results
come first.

1. **The parser diff: three parsers on the same financial table, three different Recall@5 scores.**
   Nobody has published the parser → retrieval attribution. Lead with this.
2. **Chunk-level recall said 1.0; character-level precision said 0.04.** Why your retrieval metric is
   lying to you about context waste.
3. What a reranker is actually worth, plotted against the candidate depth you feed it.
4. Chunk size versus recall, and why the sweet spot moved when the corpus changed.
5. Hybrid versus dense-only, sliced by question type — the tabular questions tell a different story.
6. Contextual retrieval: the quality gain, and the indexing cost nobody mentions.
7. Matryoshka dimension truncation — what you lose cutting 1536 to 256, and what you save.
8. **Two configs, identical scores, and why the difference was not statistically significant.**
9. "512 with 50 overlap" means nothing until you say whose tokenizer.

Posts 1, 2 and 8 are the ones that earn respect from the people whose respect is worth having.

---

## 18. Sources

Competitive and technical research behind §3, §4 and the new features:

- [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) · [RAGBuilder](https://github.com/kruxai/ragbuilder) · [Ragas](https://www.ragas.io/) · [open-rag-eval](https://github.com/vectara/open-rag-eval) · [reranker-eval](https://github.com/agentset-ai/reranker-eval)
- [rag-chunk](https://github.com/messkan/rag-chunk) · [ChunkViz](https://testdev.tools/chunk-viz/) · [Ailog Chunking Simulator](https://app.ailog.fr/en/tools/chunking-simulator)
- [pdf-parser-benchmark](https://github.com/applied-artificial-intelligence/pdf-parser-benchmark) · [Docling paper](https://arxiv.org/pdf/2501.17887)
- [Vectorize RAG Evaluation](https://docs.vectorize.io/learn/getting-started/rag-evaluation-quick-start/) · [Vectorize RAG Sandbox](https://docs.vectorize.io/build-deploy/test-improve/rag-sandbox/)
- [ranx](https://github.com/AmenRa/ranx) · [chonkie](https://docs.chonkie.ai/common/open-source)
- [LegalBench-RAG](https://arxiv.org/abs/2408.10343)

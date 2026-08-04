# RAG Lab — Feature Catalogue

**Companion to** [PRD v2](./rag-retrieval-lab-prd-v2.md) · **Owner:** Sushant Gundla · **Status:** Draft 1

This is the full universe of features for the project, across the whole retrieval-augmented
generation field. It is deliberately larger than what will ever be built. The point is to have the
map, so that every build decision is a *choice* rather than an omission.

**Scope decision (changed from PRD v2):** this is a **RAG Lab**, not a retrieval-only bench.
Generation is first-class — answer quality, prompt strategy, citation behaviour and agentic loops
are sweepable axes with their own metrics, not a secondary panel. PRD v2's non-goals N3 (retrieval
only), N5 (no auto-optimiser) and N6 (no observability) are superseded by §0.2 below.

**How to read this.** Features are grouped by pipeline layer, A–R. Each has an ID, a priority, and a
one-line reason it earns its place. IDs are stable — reference them in plans and commits.

Priority key:

| | Meaning |
|---|---|
| **P0** | Launch. Without it the project does not make its point. |
| **P1** | Fast follow, within the first two months of launch. |
| **P2** | Real feature, no committed date. |
| **P3** | Catalogued so the map is complete. May never be built. |

Markers: **★** = nothing found in the field does this (see [PRD v2 §3](./rag-retrieval-lab-prd-v2.md#3-competitive-landscape-new)) · **↑** = promoted after research · **NEW** = did not exist in PRD v1 or v2.

---

## 0. What the research changed

### 0.1 The eight findings that reshaped the feature set

1. **ColPali and page-as-image retrieval** ([ViDoRe benchmark](https://arxiv.org/pdf/2510.03663)) means "no parser at all" is a legitimate row on the parser axis. Roughly 80% of enterprise PDFs contain a table, chart or complex layout; parser-free visual retrieval sidesteps the corruption entirely. Our headline axis gets a control row that beats every parser on some corpora. → **B12**
2. **Query-side linear adapters** train a small matrix on top of a *frozen* embedding model from (query, gold chunk, hard negative) triplets. The eval set we already build is exactly that training data. A near-free quality lever that reuses the product's core asset. → **D14**
3. **Vector quantization is a cost axis nobody sweeps.** Scalar, product, binary, RaBitQ, rotational — each trades recall for memory and QPS, and the trade is corpus-specific. Cost per 1k queries is meaningless without it. → **E10–E14**
4. **The [Seven Failure Points](https://arxiv.org/abs/2401.05856) paper gives us a diagnostic vocabulary.** Every failing query can be auto-classified FP1–FP7. That turns a leaderboard into a debugger. → **N2**
5. **There is a whole layer between retrieval and generation that nobody sweeps:** context ordering ("lost in the middle"), compression (LongLLMLingua: ~21% accuracy gain on ¼ the tokens), packing and token budgeting. → **Section I**
6. **Adaptive routing is the 2026 consensus** — classify query complexity, route cheap or expensive. Whether routing actually beats always-heavy is an unanswered empirical question we are uniquely placed to answer. → **F10–F13**
7. **Corpus poisoning is measurable.** PoisonedRAG showed five documents in a corpus of millions achieving ~90% attack success. Retrieval robustness can be scored like any other metric. → **Section O**
8. **Semantic caching changes the cost picture by up to 85%** in production reports. Any cost model that ignores cache hit rate is wrong by an order of magnitude. → **M10–M13**

### 0.2 Revised non-goals

| | Non-goal |
|---|---|
| **N1** | Not a RAG framework. We emit configs *for* LlamaIndex, LangChain, Haystack; we do not replace them. |
| **N2** | Not a production serving layer. No SLAs, no multi-tenant hosting, no live traffic. |
| **N3** ~~retrieval only~~ | **Superseded.** Generation is first-class. But retrieval stays the *default* view and the first thing measured, because generation noise swamps retrieval signal when you look at answers first. |
| **N4** | No full fine-tuning of embedding models. Frozen-base adapters (D14) are explicitly allowed — different cost class, different risk. |
| **N5** ~~no auto-optimiser~~ | **Softened.** Optimisation is allowed *if it shows its work*. RAGBuilder's sin is hiding the landscape, not searching it. |
| **N6** ~~no observability~~ | **Softened.** Offline replay of production traces is in (N8). Live tracing, dashboards and alerting stay out — that is Phoenix's job. |
| **N7** | Not a data labelling platform. The review queue serves eval sets only. |
| **N8** | Not a production monitor. We import traces to build eval sets from them; we do not watch traffic. |

---

## A. Corpus and ingestion

*What goes in. Everything downstream inherits its quality.*

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| A1 | Multi-file upload: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, EPUB | P0 | The front door |
| A2 | Paste-text corpus | P0 | Ten-second path to a first result |
| A3 | Built-in sample corpora — financial report, technical docs, legal policy, code+SQL, LegalBench-RAG slice | P0 | Demo mode is the business case |
| A4 | Per-file preview: pages, language, text-layer present or OCR needed, table density | P0 | Sets expectations before a slow run |
| A5 | **Corpus fingerprint ★↑** — duplicate rate, passage length distribution, OCR quality, language mix, table density, code/prose ratio, entity density | P0 | Turns a blank matrix builder into a guided decision. Nothing else profiles your corpus before running |
| A6 | **Fingerprint-driven matrix suggestion NEW** | P0 | "40% tables → parser choice will dominate here; matrix pre-selected" |
| A7 | URL ingestion, crawl depth 0–1 | P1 | Docs sites are the most common corpus |
| A8 | Connectors: GitHub repo, Notion, Confluence, Google Drive, S3 | P2 | Where real corpora actually live |
| A9 | Near-duplicate detection and removal (MinHash / SimHash) at ingest | P1 | Deferring dedup to post-chunking loses the chunk→document link and inflates the index |
| A10 | Metadata extraction and propagation — source URI, version hash, author, date, section | P0 | Cannot be added retroactively once indexed |
| A11 | LLM-extracted metadata — entities, topics, doc type, summary | P2 | Powers filtering (E15) and routing (F12) |
| A12 | **Incremental indexing and freshness NEW** — process only new/changed docs, measure staleness | P2 | Full re-index cost grows with corpus; incremental does not. A real production concern nobody benches |
| A13 | Corpus versioning and snapshot hashing | P1 | Required for the run manifest (P4) to mean anything |
| A14 | PII detection and redaction at ingest | P2 | Makes the hosted mode defensible for real documents |
| A15 | Language detection and per-language routing | P2 | Multilingual corpora silently break monolingual embedders |
| A16 | **Corpus difficulty score NEW ★** — predicted retrieval difficulty from lexical overlap, redundancy, entity density | P3 | "Your corpus is intrinsically hard; expect Recall@5 near 0.6, not 0.9." Manages expectations honestly |

---

## B. Parsing and document understanding

*The headline axis. Nobody else sweeps this.*

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| B1 | Parser axis: PyMuPDF, pdfplumber, Unstructured (`fast` / `hi_res` as separate points), Docling, Marker, MinerU | P0 | The differentiator |
| B2 | Normalised intermediate representation — text blocks + structure + **character offsets** | P0 | Offsets are non-negotiable; the span scorer (K6) depends on them |
| B3 | **Parser diff view ★↑** — same page, side by side, two parsers, text overlaid on the render | P0 | The most screenshot-able feature in the product |
| B4 | **Parser → retrieval attribution chart ★ NEW** — parser on x, Recall@5 on y, everything else fixed | P0 | Published parser benchmarks stop at character accuracy. This connects parse to outcome. The single unclaimed result in the field |
| B5 | OCR fallback: Tesseract, PaddleOCR, per-page auto-trigger | P0 | Scanned pages are otherwise silently empty |
| B6 | Hosted parsers, BYOK: LlamaParse, Azure Document Intelligence, AWS Textract, Google Document AI, Reducto | P1 | The "is paid parsing worth it" question, answered with cost attached |
| B7 | VLM-as-parser: page image → markdown via a vision model | P1 | Increasingly the strongest option and the most expensive. Belongs on the cost/quality frontier |
| B8 | Parser sub-metrics: table cell recall, reading-order correctness, chars/page, ms/page, $/page | P0 | Lets a user see *why* a parser won, not just that it did |
| B9 | Table-specific handling — extract as markdown, HTML, or linearised text, as a sweepable sub-axis | P1 | Table representation changes retrieval more than chunk size does |
| B10 | Layout element classification — heading, paragraph, table, figure, caption, footnote, header/footer | P1 | Feeds structural chunking (C4) and section filters (E15) |
| B11 | Header/footer/boilerplate stripping | P1 | Repeated page furniture poisons dense retrieval quietly |
| B12 | **Parser-free row: ColPali / ColQwen page-as-image retrieval ★ NEW** | P1 | Turns the parser axis into "which parser — or none at all?". No other tool can pose that question |
| B13 | Formula and equation handling (LaTeX extraction) | P3 | Scientific corpora are otherwise unservable |
| B14 | **Offset-exactness flag NEW** — mark parsers/chunkers that cannot guarantee character offsets as approximate | P0 | Honesty about a limitation beats hiding it. Feeds the methodology page |
| B15 | Speech/audio and video transcript ingestion | P3 | Completes the map; no near-term demand |

---

## C. Chunking and representation

*Built on `chonkie`, wrapped in an offset-preserving adapter.*

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| C1 | Fixed-size token — size, overlap, tokenizer | P0 | The baseline everyone claims to have tuned |
| C2 | Recursive character — size, overlap, separator hierarchy | P0 | The de-facto default in every framework |
| C3 | Sentence window — window size, stride | P0 | Cheap, and often beats the fancy options |
| C4 | Structural / heading-aware — max depth, min/max size, keep heading path | P0 | Wins decisively on technical docs |
| C5 | Semantic breakpoint — percentile threshold, buffer, breakpoint method | P0 | The most-hyped strategy. Deserves a fair measurement |
| C6 | **Neural chunking (BERT boundary detection) NEW** | P1 | Newest strategy class; nobody has benchmarked it against the classics |
| C7 | Proposition-based (LLM-extracted atomic facts) | P1 | High quality, high cost — exactly the trade the cost panel exists to show |
| C8 | Parent-document / small-to-big | P0 | Retrieve precise, generate with context. Common and rarely measured |
| C9 | Contextual retrieval (LLM-prepended chunk context) | P1 | Big claimed gains, big unmentioned indexing bill |
| C10 | Late chunking (embed long, pool per chunk) | P1 | Requires long-context embedder; a genuinely different mechanism |
| C11 | **Code / AST-aware chunking ↑** — language-aware, preserves function and class boundaries | P0 | Pairs with `CodeRankEmbed`. No retrieval eval tool covers code at all |
| C12 | Markdown/HTML structural chunking honouring the DOM | P1 | Web and docs corpora are the most common real input |
| C13 | Slide and spreadsheet chunking (per-slide, per-sheet, per-table) | P2 | PPTX and XLSX break every text chunker |
| C14 | **Tokenizer-normalised size reporting ★** — char count and token count per tokenizer, with a unit toggle | P0 | "512 with 50 overlap" means nothing until you say whose tokenizer. Nothing in the field does this |
| C15 | Chunk metadata attachment — heading path, page, file, section, entities | P0 | Free quality, and required for filtering |
| C16 | Near-duplicate chunk de-duplication with a similarity threshold | P1 | Duplicate chunks silently inflate Recall@k |
| C17 | Multi-granularity indexing — sentence + paragraph + section, retrieved together | P2 | Strong on visually-rich and long documents |
| C18 | Auto-merging retrieval — adjacent retrieved chunks merged into their parent | P1 | Cheap trick, meaningful gain, rarely benchmarked |
| C19 | **Chunk-boundary overlay on the source page ★** | P0 | The "oh, *that's* the problem" moment — you see a table cut in half |
| C20 | Overlap strategy as a first-class param — none / token / sentence / semantic | P1 | Overlap is usually a magic number nobody justifies |
| C21 | **Chunk quality diagnostics NEW ★** — orphan fragments, mid-sentence splits, split tables, size distribution | P1 | Scores the chunker independently of retrieval, the way B8 scores the parser |

---

## D. Embedding

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| D1 | Local CPU models via Infinity + ONNX: `bge-base/large-en-v1.5`, `bge-m3`, `e5-base/large-v2`, `multilingual-e5`, `gte-base/large`, `nomic-embed-text-v1.5`, `all-MiniLM-L6-v2`, `jina-embeddings-v3` | P0 | Zero marginal cost is what makes free sweeps possible |
| D2 | `CodeRankEmbed` and a general model on the code corpus | P0 | Your existing research, and an uncontested angle |
| D3 | Hosted BYOK: OpenAI `text-embedding-3-small/-large`, Cohere `embed-v3`, Voyage, Gemini | P0 | The models most readers actually use |
| D4 | **Correct query/document asymmetry** — E5 `query:`/`passage:`, BGE query instruction, Cohere `input_type` | P0 | Get this wrong and every number is garbage. Most homegrown evals get it wrong |
| D5 | Normalisation and similarity metric as explicit params (cosine / dot / L2) | P0 | Silent mismatches produce plausible, wrong rankings |
| D6 | **Truncation warning ★** — loud flag when a chunk exceeds the model's context and is being cut | P0 | Silently destroys results in most homegrown evaluations |
| D7 | Matryoshka dimension truncation as a sweepable param (1536 → 768 → 512 → 256) | P0 | The best cost/quality lever in the whole pipeline, and a great post |
| D8 | Per-model reporting: dims, index size, embed throughput, $/M tokens | P0 | Half of the cost model |
| D9 | Instruction-tuned embedders with task prefixes as a sweepable param | P2 | Prefix wording measurably moves scores; nobody reports it |
| D10 | Multimodal embedders — Cohere Embed 4, voyage-multimodal, ColQwen | P1 | Required for B12 and Section G's visual retrieval |
| D11 | Embedding cache keyed on (chunk hash, model, params) | P0 | Without it a 48-config sweep is unaffordable |
| D12 | float16 / int8 embedding storage | P1 | Halves disk; measurable recall cost |
| D13 | Batch size and dynamic batching as throughput params | P2 | Throughput claims need the batch size stated |
| D14 | **Query-side linear adapter ★ NEW** — train a small matrix on frozen embeddings from the user's own eval set | P2 | Reuses the eval set as training data. Big gains, minutes of CPU, no model change. The most under-used technique in the field |
| D15 | **Hard-negative mining from sweep results NEW ★** | P2 | Every sweep already surfaces high-ranking non-gold chunks — free triplets for D14. Plateaus around 40 negatives per query |
| D16 | LoRA / DoRA embedding fine-tuning | P3 | Explicitly beyond v1 (N4). Catalogued as the natural sequel to D14 |
| D17 | Embedding drift detection across model versions | P3 | A real production problem, wrong moment in the lifecycle for us |

---

## E. Indexing, vector store and search

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| E1 | Dense exact search (flat) | P0 | The reference row every ANN result is judged against |
| E2 | HNSW with `M`, `efConstruction`, `efSearch` sweepable | P0 | The default index everywhere |
| E3 | IVF / IVF-PQ with `nlist`, `nprobe` | P1 | The right choice past ~50M vectors |
| E4 | DiskANN / StreamingDiskANN | P2 | Where the memory-constrained frontier actually is |
| E5 | **ANN recall-loss row ★** — every ANN config reported against its exact-search twin | P0 | People tune `efSearch` blind. Showing the recall they gave up is a small, sharp contribution |
| E6 | BM25 with `k1`, `b` tunable | P0 | Still beats dense on keyword-heavy corpora and everyone forgets |
| E7 | Learned sparse — SPLADE, uniCOIL | P1 | The strongest sparse option; rarely compared fairly |
| E8 | Hybrid fusion — RRF (`k`) and weighted score normalisation (`alpha`), both via `ranx` | P0 | They behave differently and the choice matters |
| E9 | **Automatic fusion-weight optimisation** (`ranx` built-in) | P1 | Turns "what alpha?" from a manual sweep into a solved sub-problem, free |
| E10 | **Scalar quantization NEW** | P1 | 4× memory cut, small recall loss |
| E11 | **Product quantization NEW** | P1 | The classic memory/recall trade |
| E12 | **Binary quantization + rescoring NEW ★** | P1 | 32× compression, ~30× QPS gains reported. The largest cost lever in the index layer |
| E13 | **RaBitQ / rotational quantization NEW** | P2 | Current state of the art; beats scalar on most datasets |
| E14 | **Quantization cost/recall frontier chart ★ NEW** | P1 | Quantization is discussed as a memory decision and never as a *quality* decision. This is a whole post on its own |
| E15 | Metadata filtering — pre-filter, post-filter, and filtered-HNSW as separate strategies | P1 | Filtering strategy changes recall dramatically and is always glossed over |
| E16 | Section-scoped and document-scoped search | P1 | The cheapest large gain on structured corpora |
| E17 | ColBERT-style late interaction / multi-vector | P1 | Strong quality, awkward cost. Belongs on the frontier |
| E18 | Backends: LanceDB (default), Qdrant, pgvector, Chroma, Weaviate, Milvus | P0 LanceDB / P2 others | LanceDB is zero-ops and fits the hosted model; the rest serve self-hosters |
| E19 | **Backend as a sweepable axis ★ NEW** — same vectors, same params, different engine | P2 | Vendor benchmarks are all self-published. A neutral, reproducible comparison on *your* data does not exist |
| E20 | Index build time, size on disk and memory footprint reported per config | P0 | The half of cost everyone forgets |
| E21 | Sharding and multi-index federation | P3 | Scale beyond this tool's remit |
| E22 | **Graph index — entity/relation extraction, community summaries, graph traversal ★ NEW** | P2 | GraphRAG as *an index type on the same axis*, priced and timed against dense. Every published comparison is vendor marketing |
| E23 | Temporal/recency-weighted scoring | P2 | News and changelog corpora rank badly without it |

---

## F. Query understanding, transformation and routing

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| F1 | None (baseline) | P0 | Every transform is judged against it |
| F2 | HyDE — hypothetical document embedding | P1 | Most-cited transform; rarely measured with cost attached |
| F3 | Multi-query expansion — n variants, fused | P1 | Reliable small gain, n× the embedding cost |
| F4 | Query decomposition for multi-hop | P1 | The only thing that works on genuinely multi-hop questions |
| F5 | Step-back / abstraction prompting | P2 | Helps on conceptual questions, hurts on factoids — a nice sliced result |
| F6 | Conversational query rewriting with history | P1 | Every real chatbot needs it; no bench measures it |
| F7 | Spelling correction and acronym expansion | P2 | Unglamorous, and it moves BM25 more than anything else |
| F8 | Keyword extraction for the sparse leg of hybrid | P2 | Hybrid quality is bottlenecked by a bad sparse query |
| F9 | **Per-transform cost and latency attribution ★** | P0 | The whole point: does the gain justify the LLM call? Usually not, and proving that is a strong post |
| F10 | **Query complexity classifier NEW** | P2 | The core of adaptive RAG. Even length + keyword heuristics reach ~80% effectiveness |
| F11 | **Adaptive routing — cheap path vs heavy path vs graph path NEW ★** | P2 | The 2026 consensus architecture, and nobody has published whether routing actually beats always-heavy on cost-adjusted quality |
| F12 | Metadata-based routing (route by doc type, language, section) | P2 | Cheaper and more reliable than LLM routing, and less discussed |
| F13 | **Router quality metrics NEW ★** — routing accuracy, cost saved, quality lost | P2 | A router is a classifier and should be scored like one. Nobody does |
| F14 | Query intent classification — factoid, multi-hop, comparative, tabular, summarisation | P0 | Drives the per-question-type slicing that makes results interesting |
| F15 | Multilingual query translation before retrieval | P3 | Completes the map |

---

## G. Retrieval strategies

*Whole architectures, compared on equal footing.*

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| G1 | Single-shot dense | P0 | Baseline |
| G2 | Single-shot sparse | P0 | Baseline |
| G3 | Hybrid | P0 | The realistic default |
| G4 | Parent-document / small-to-big retrieval | P0 | Retrieve narrow, generate wide |
| G5 | Auto-merging / hierarchical retrieval | P1 | Cheap gain on structured docs |
| G6 | Sentence-window retrieval with expansion | P1 | Precise retrieval, contextual generation |
| G7 | **Iterative / multi-hop retrieval loops NEW** | P2 | Retrieve, read, re-query. The only thing that handles compositional questions |
| G8 | **Self-RAG — model decides whether to retrieve at all NEW** | P2 | Skipping retrieval is a legitimate strategy and never appears on a comparison chart |
| G9 | **Corrective RAG (CRAG) — grade retrieved docs, fall back to web/re-query NEW** | P2 | The standard recovery pattern; unmeasured |
| G10 | **Agentic retrieval — plan, tool-call, reflect, bounded self-correction NEW ★** | P2 | 2–10 s per query, 3–10× the token cost, real accuracy gains. The cost/quality trade nobody has plotted |
| G11 | **GraphRAG local search (entity neighbourhood) NEW** | P2 | Multi-hop questions vector search structurally cannot do |
| G12 | **GraphRAG global search (community summaries) NEW ★** | P2 | Dataset-wide questions — "what are the main themes?" — which chunk similarity cannot answer at all |
| G13 | **Visual / page-image retrieval (ColPali family) NEW** | P1 | Pairs with B12; the parser-free architecture end to end |
| G14 | Multi-index federation — search several corpora, merge | P3 | Enterprise shape, out of scope for now |
| G15 | Recursive / small-to-big over summaries | P2 | Strong on very long documents |
| G16 | **Architecture-level comparison view ★ NEW** — naive vs advanced vs agentic vs graph vs visual, on one Pareto chart | P2 | Every blog post compares these in prose. Nobody has ever priced them side by side on the same corpus |

---

## H. Reranking and result selection

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| H1 | None (baseline) | P0 | Half of "use a reranker" advice is untested |
| H2 | Local cross-encoders — `bge-reranker-base`, `bge-reranker-v2-m3`, `mxbai-rerank`, `jina-reranker-v2` | P0 | Free to run, big gains, the honest default |
| H3 | Hosted rerankers BYOK — Cohere Rerank, Voyage rerank | P1 | The paid comparison people actually want |
| H4 | LLM-as-reranker — pointwise, pairwise, listwise | P1 | Best quality, worst cost. Belongs on the frontier |
| H5 | **Candidate-depth sweep (top 20/50/100/200) with the depth curve ★** | P0 | The single most useful chart in the reranking layer and nobody publishes it |
| H6 | Final-k sweep | P0 | Determines the generator's bill |
| H7 | Score-threshold cutoff instead of fixed k | P1 | Adaptive k is better practice and never benchmarked |
| H8 | MMR / diversity reranking with a λ param | P1 | Redundant top-5 results are a common silent failure |
| H9 | Post-rerank de-duplication | P1 | Overlapping chunks waste context |
| H10 | **Reranker latency and cost per 1k queries ★** | P0 | `reranker-eval` does latency on a fixed corpus. On *your* corpus with cost attached is still open |
| H11 | **Rank-movement diagram ★** — arrows showing what the reranker actually did | P0 | Makes an invisible step visible; strong inspector feature |
| H12 | Two-stage reranking (cheap then expensive) | P2 | The production pattern; unbenchmarked |
| H13 | Reciprocal rank fusion of multiple rerankers | P3 | Completes the map |

---

## I. Context assembly

*Between retrieval and generation. An entire layer nobody sweeps.* **NEW section**

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| I1 | **Context ordering strategies ★** — relevance-descending, ascending, "lost in the middle" reorder (best at the ends), original document order | P1 | RoPE decay makes mid-context evidence get missed. Ordering is free and changes answer accuracy measurably |
| I2 | **Token budget as a first-class param** | P1 | The real cost driver downstream, and the constraint everything else negotiates with |
| I3 | **Prompt compression — LongLLMLingua ★ NEW** | P2 | Reported ~21% accuracy gain at ¼ the tokens. Nothing compares compressed vs uncompressed on the same retrieval |
| I4 | Extractive compression — sentence-level selection from retrieved chunks | P2 | Cheaper than I3, often as good |
| I5 | Context de-duplication before assembly | P1 | Overlapping chunks pay twice for the same sentence |
| I6 | Citation formatting strategy — inline markers, footnotes, structured JSON | P1 | Changes citation accuracy far more than people expect |
| I7 | Metadata injection into context — heading path, page, date | P1 | Cheap grounding gain |
| I8 | **Tokens-sent-to-generator reported per config ★** | P0 | The number that actually determines the monthly bill |
| I9 | Context window overflow policy — truncate, drop lowest, compress | P1 | Silent truncation is a real, invisible failure |
| I10 | **Position-sensitivity probe ★ NEW** — same evidence, different positions, measure the answer delta | P3 | Quantifies lost-in-the-middle on *your* model and corpus. A genuinely novel diagnostic |

---

## J. Generation

*First-class as of this document, not a secondary panel.*

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| J1 | Generator model axis, BYOK — GPT, Claude, Gemini, plus local via Ollama/vLLM | P1 | "Does a better retriever matter more than a better model?" is the question everyone asks |
| J2 | Prompt template as a sweepable axis | P1 | Prompt changes routinely beat retrieval changes, which is an uncomfortable and valuable result |
| J3 | Answer faithfulness / groundedness scoring | P1 | The primary generation metric |
| J4 | Answer relevance scoring | P1 | Faithful but useless answers exist |
| J5 | Citation accuracy — do cited spans actually support the claim? | P1 | The metric enterprises care about most and tools measure least |
| J6 | **Abstention / refusal behaviour ★** — does it say "I don't know" when evidence is absent? | P2 | FP1 in the failure taxonomy. Almost nothing measures a correct refusal as a *success* |
| J7 | Structured output extraction accuracy (JSON/schema) | P2 | Half of real RAG is extraction, not chat |
| J8 | Temperature, top-p, seed as recorded params | P1 | Reproducibility |
| J9 | **Retrieval→generation lift chart ★ NEW** — does +0.1 Recall@5 actually become a better answer? | P1 | The question the whole project implicitly promises to answer. Nobody plots it |
| J10 | Generation cost and latency per config | P1 | Completes the cost model end to end |
| J11 | Answer-length and verbosity control | P3 | Completes the map |
| J12 | Multi-turn conversational evaluation | P2 | Real assistants are conversational; almost no bench is |
| J13 | Hallucination detection on the generated answer | P2 | Standard panel, well-served elsewhere — we integrate rather than invent |

---

## K. Ground truth and eval sets

*Still the heart of the product.*

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| K1 | LLM auto-generation of questions from sampled chunks | P0 | Removes the biggest adoption blocker |
| K2 | Filter: reject questions answerable from general knowledge | P0 | Otherwise you measure the model, not the retriever |
| K3 | Filter: reject dangling pronouns and unresolvable references | P0 | Auto-generated questions are full of them |
| K4 | Filter: reject near-duplicates | P0 | Duplicate questions fake statistical power |
| K5 | **Filter: reject non-discriminating questions ★** — ones the baseline already answers at rank 1 | P0 | They inflate every score and separate nothing |
| K6 | **Gold as character spans in the source document, resolved to chunks by IoU at scoring time ★** | P0 | **The single most important decision in the project.** Chunk-ID gold makes cross-chunker and cross-parser comparison invalid. Only LegalBench-RAG does it this way |
| K7 | Multiple gold spans per question | P0 | Questions legitimately have several supporting passages |
| K8 | Graded relevance (2 / 1 / 0) | P0 | nDCG is meaningless without it |
| K9 | **Human review queue ★** — keyboard-driven, under 5 s per item, accept/reject/edit/tag | P0 | Ragas, AutoRAG and RAGBuilder all generate and none let you fix. The clearest gap in the field |
| K10 | Question-type tagging — factoid, multi-hop, comparative, numeric/tabular, summarisation | P0 | Per-type slicing is what separates this from a leaderboard toy |
| K11 | Import CSV/JSONL, BEIR format, **LegalBench-RAG format** | P0 | Drives the validation page |
| K12 | **Eval-set quality score ★** — n, % reviewed, type mix, mean discriminating power, statistical power for the deltas being claimed | P1 | Nothing tells a user their ground truth is too weak to support their conclusion |
| K13 | **Adversarial / distractor question generation NEW** | P2 | Questions with near-miss passages elsewhere in the corpus. Where retrievers actually break |
| K14 | Unanswerable questions with correct-answer "no evidence" | P1 | Tests FP1 and abstention (J6). Almost no eval set has them |
| K15 | Multi-hop question generation across two chunks | P1 | Needed to make G7–G12 measurable at all |
| K16 | **Eval sets from production traces NEW** | P2 | Import real queries, label the good retrievals. The highest-quality ground truth there is |
| K17 | Eval set versioning and diff | P1 | Comparing runs across eval-set versions is a silent correctness bug |
| K18 | Inter-annotator agreement when several people review | P3 | The rigorous version; overkill for one person |
| K19 | **Spin-out as a standalone `span-eval` package ★** | P2 | The most reusable component. Likely more stars alone than the whole app — but only after phase 2 |

---

## L. Metrics

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| L1 | Recall@k for k ∈ {1,3,5,10,20} | P0 | The headline. Generation only needs the evidence present |
| L2 | Precision@k | P0 | Standard |
| L3 | nDCG@k with graded relevance | P0 | The metric IR people trust |
| L4 | MRR | P0 | Standard |
| L5 | MAP | P0 | Standard |
| L6 | Hit rate@k | P0 | The most intuitive number for non-specialists |
| L7 | Mean rank of first gold | P0 | More useful than MRR when debugging |
| L8 | **Character-level precision and recall ★ NEW** | P1 | Chunk Recall@5 can be 1.0 while character precision is 0.04 — that config is burning your context window. Nothing surfaces this in a comparison UI |
| L9 | **Chunk attribution rate ★** — fraction of retrieved chunks containing any gold | P1 | Measures wasted context directly |
| L10 | Context precision / context recall (LLM-judged, Ragas-style) | P1 | The bridge to generation metrics |
| L11 | **All metrics via `ranx`** | P0 | Peer-reviewed, Numba-fast, and "not computed by me" is a credibility statement |
| L12 | **Bootstrap confidence intervals ↑** | P0 | A number without an interval is an opinion |
| L13 | **Paired significance testing between configs ↑ ★** | P0 | They run on identical queries, so the paired test is both valid and nearly free via `ranx` |
| L14 | **"Not statistically distinguishable (n=87)" as a first-class result state ★** | P0 | Publishing that banner is worth more than any leaderboard |
| L15 | Per-question-type metric breakdown | P0 | Semantic chunking can win overall and lose badly on tables |
| L16 | Per-document and per-section metric breakdown | P2 | Locates which part of the corpus is failing |
| L17 | **Metric-correlation panel ★ NEW** — do Recall@5, nDCG@10 and answer faithfulness actually agree on this corpus? | P2 | When they disagree, that is the most interesting finding in the run |
| L18 | Latency p50 / p95 / p99, broken down by stage | P0 | An average latency hides the whole problem |
| L19 | Index build time and size | P0 | One-time cost people forget entirely |
| L20 | Throughput — queries/sec at fixed concurrency | P2 | The serving-shaped question |
| L21 | Effect size, not just significance | P2 | "Significant but tiny" is the most common misreading in the field |

---

## M. Cost, performance and optimization

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| M1 | **Cost per 1,000 queries, itemised by call type ★** | P0 | Every open-source tool ranks on quality alone. This is differentiator #2 |
| M2 | One-time indexing cost per config | P0 | Contextual retrieval looks free until you see it |
| M3 | Pre-run cost and time estimate with a hard budget cap | P0 | Nobody starts a sweep they cannot price |
| M4 | Live cost meter with automatic stop | P0 | Trust |
| M5 | **Prefix caching across configs** — parse, chunk and embedding caches keyed on content hash | P0 | Without it a 48-config sweep is unaffordable; with it, sweeping rerankers is nearly free |
| M6 | **Cache-saving readout ★** — "37 of 48 configs reused cached embeddings, saved $0.41 and 6 min" | P1 | Makes an invisible engineering decision visible, which is what a portfolio piece should do |
| M7 | **Quality vs cost and quality vs latency Pareto charts ★** | P0 | The screenshot that travels |
| M8 | **Cost-aware matrix suggestion ★** — "48 → 19 configs, you lose only the parser × reranker interaction" | P1 | Respects the user's money and shows you understand experiment design |
| M9 | Self-hosted vs hosted cost modelling (amortised CPU/GPU time) | P2 | "Is Cohere Rerank worth it vs bge on my own box?" |
| M10 | **Semantic cache simulation ★ NEW** — replay the query set, measure hit rate at a similarity threshold | P2 | Production reports claim up to 85% cost reduction. A cost model that ignores it is wrong by an order of magnitude |
| M11 | **Cache threshold sweep ★ NEW** — hit rate vs wrong-answer rate | P2 | The dangerous knob nobody publishes a curve for |
| M12 | Prompt-cache-aware context ordering | P3 | Stable prefixes cut generation cost; interacts with I1 |
| M13 | **Cost-constrained auto-tuner ★** — "best config under $X per 1k queries and 300 ms p95" | P2 | The actual question every engineer has. Allowed under revised N5 because it shows the frontier it searched |
| M14 | Batch vs realtime cost modelling | P3 | Batch embedding APIs are ~50% cheaper and never modelled |
| M15 | Carbon / energy estimate per config | P3 | Cheap to add, and some readers care |

---

## N. Failure analysis and diagnostics

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| N1 | **Per-query inspector** — retrieved chunks in rank order, gold spans highlighted, rank before/after rerank | P0 | The debugging surface |
| N2 | **Auto-classify every failure into the Seven Failure Points taxonomy ★ NEW** — FP1 missing content, FP2 missed top-ranked, FP3 not in context, FP4 not extracted, FP5 wrong format, FP6 wrong specificity, FP7 incomplete | P1 | Turns a leaderboard into a debugger, using a published vocabulary. Nobody has productised it |
| N3 | **Failure clustering ★** — group failing questions by shared cause | P2 | "These 12 failures are all tables split across pages" is worth more than 12 individual traces |
| N4 | Config diff — every parameter difference and every question where two configs disagree | P0 | How you actually learn what changed |
| N5 | **Axis-effect view ★** — marginal effect of each axis holding others fixed, as small multiples | P0 | The interpretable summary of a full-factorial sweep |
| N6 | **Interaction detection ★ NEW** — flag when two axes interact, i.e. when the staged/greedy shortcut would have been wrong | P2 | Honest about the limits of the mode we recommend as default |
| N7 | **Chunk boundaries drawn on the source page** | P0 | The strongest "oh, *that's* the problem" moment in the product |
| N8 | Gold-span-not-retrieved-at-any-k report | P1 | Questions no config can answer are an eval-set problem, not a retrieval problem |
| N9 | **Regression triage via run-manifest diff ★ NEW** | P1 | When a metric drops, diff the manifest against the last passing run. The changed line is the suspect. Widely described, never shipped |
| N10 | Query difficulty scoring — which questions are hard for every config | P2 | Separates "your retriever is bad" from "this question is unanswerable" |
| N11 | **Recommendation card ★** — one plain-English paragraph: what won, by how much, whether it's significant, what it costs, what it's worst at | P0 | Every tool outputs a table and leaves you to interpret it. Writing the conclusion in words is what a hiring manager reads |

---

## O. Security, safety and robustness

**NEW section — an entirely uncontested axis.**

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| O1 | Client-side BYOK — never persisted, never logged, stated in the UI | P0 | Table stakes for asking anyone to paste a key |
| O2 | Sandboxed parse workers — memory cap, wall-clock cap, no network | P0 | Parsing runs untrusted user files |
| O3 | Upload TTL and deletion policy, stated plainly | P0 | Makes the hosted mode defensible |
| O4 | **Corpus poisoning robustness score ★ NEW** — inject adversarial passages, measure how often they get retrieved | P2 | PoisonedRAG: five documents in millions, ~90% success. No comparison tool has ever scored a retrieval config for robustness |
| O5 | **Prompt-injection detection in retrieved chunks ★ NEW** | P2 | Document injection is the injection vector inside every RAG pipeline |
| O6 | Retrieved-content framing check — is external data marked as data, not instructions? | P2 | The cheapest real defence, and rarely verified |
| O7 | PII leak detection in retrieved context | P2 | The compliance question every enterprise asks first |
| O8 | Access-control-aware retrieval simulation (per-user filtering) | P3 | Real enterprise concern, out of scope here |
| O9 | **Robustness vs quality trade chart NEW** | P3 | The hardened config is usually slightly worse. Showing the price of safety is a good post |

---

## P. Reproducibility, sharing and distribution

| ID | Feature | P | Why it earns its place |
|---|---|---|---|
| P1 | Shareable permalink per run, public/private | P0 | How it spreads from a LinkedIn post |
| P2 | OG image rendering the headline chart | P0 | Determines whether the link gets clicked |
| P3 | Config export as YAML/JSON | P0 | The thing engineers actually take away |
| P4 | **Run manifest ★** — hash-pinned versions, params, corpus hash, eval-set version, seeds, library versions | P0 | Turns "reproducible" from a claim into a property, and makes N9 possible |
| P5 | Code export — plain Python (P0); LlamaIndex, LangChain, Haystack (P1) | P0/P1 | Non-goal N1 made concrete |
| P6 | One-page report export (PDF/Markdown) with methodology, matrix, leaderboard, significance | P1 | Engineers paste this into a team decision doc — the adoption path we want |
| P7 | Full run bundle — JSONL of every query, chunk and score | P1 | Lets a skeptic re-analyse your data. Confidence |
| P8 | **Methodology page ★** — how gold spans work, how tokenisation is normalised, what CIs mean, what the tool cannot tell you | P0 | The page a senior reviewer reads to decide whether you are serious |
| P9 | **Validation page ★ NEW** — our numbers on a LegalBench-RAG slice next to the published ones, with the delta explained | P1 | Converts "trust me" into "check me". Cheapest credibility feature in the document |
| P10 | Deterministic seeds recorded and replayable | P0 | Basic hygiene |
| P11 | CLI + pip package — `retrieval-lab sweep config.yaml` | P1 | Where serious usage happens |
| P12 | **CI regression gate with manifest diff ★↑** | P1 | Described everywhere in 2026, productised nowhere |
| P13 | GitHub Action wrapper | P2 | Distribution |
| P14 | Self-host via Docker Compose | P1 | The honest answer to "can I run this on my private corpus?" |
| P15 | Public community leaderboard by corpus type, opt-in | P2 | Network effect, and a data asset nobody else has |
| P16 | Run comparison across time — same corpus, months apart | P2 | "Did the new embedding model actually help?" |

---

## Q. Product surfaces

| ID | Surface | P | Note |
|---|---|---|---|
| Q1 | Landing — value prop plus the live parser-attribution chart above the fold | P0 | Lead with the one thing nobody else has |
| Q2 | New experiment — upload → fingerprint → suggested matrix → matrix builder → estimate → confirm | P0 | |
| Q3 | Eval set review — the keyboard queue with the quality score | P0 | |
| Q4 | Run monitor — progress, cost meter, cache savings, cancel/pause/resume | P0 | |
| Q5 | Results — recommendation card, leaderboard, Pareto charts, axis effects | P0 | |
| Q6 | Per-query inspector | P0 | |
| Q7 | Config diff | P0 | |
| Q8 | Parser diff (own screen) | P0 | |
| Q9 | Methodology | P0 | |
| Q10 | Validation | P1 | |
| Q11 | Shared run (read-only permalink) | P0 | |
| Q12 | Demo mode with precomputed runs, no keys, under 1.5 s cold start | P0 | The business case |
| Q13 | Failure-taxonomy dashboard | P2 | |
| Q14 | Cost explorer / what-if calculator | P2 | Move sliders, watch the bill |
| Q15 | Embedded widget — a single chart, iframe-able into a blog post | P2 | Every post becomes a distribution channel |

Design direction: dense, monospace-inflected, research-dashboard aesthetic. Every chart must be
legible in a screenshot at LinkedIn dimensions, because that is where most people will first see it.

---

## R. Deliberately out of scope

| Thing | Why not |
|---|---|
| Production serving, SLAs, multi-tenant hosting | N2. Different product, different obligations |
| Live tracing, dashboards, alerting | Phoenix and TruLens do this well. Competing is a losing fight |
| Full embedding fine-tuning (LoRA/DoRA) | N4. Adapters (D14) capture most of the gain at a fraction of the cost and risk |
| Being a RAG framework | N1. We emit configs; we do not execute production pipelines |
| Training or hosting LLMs | Not the project |
| General data labelling | N7. The review queue serves eval sets only |
| Web-scale crawling | Depth 0–1 only. Beyond that is a crawler product |

---

## Priority rollup

| Priority | Count | Shape |
|---|---|---|
| **P0** | ~75 | The launchable product: parser axis, chunking, embedding, index, rerank, eval set with span gold and review queue, `ranx` metrics with significance, cost/latency Pareto, inspector, demo mode, methodology, permalinks |
| **P1** | ~65 | Credibility and depth: character-level metrics, validation page, quantization, context assembly, generation panel, failure taxonomy, CLI and CI gate |
| **P2** | ~55 | The wide field: GraphRAG, agentic retrieval, adaptive routing, adapters, semantic caching, poisoning robustness, backend comparison, cost auto-tuner |
| **P3** | ~20 | Map completeness. Catalogued, probably never built |

## The twelve features that make this project unlike anything that exists

Ordered by how hard they are to copy.

1. **B4** — parser → retrieval attribution. The unclaimed result in the field.
2. **K6** — gold as character spans, resolved by IoU. Makes cross-parser and cross-chunker comparison valid at all.
3. **B12 + G13** — parser-free ColPali retrieval as a row on the parser axis.
4. **M1 + M7** — cost per 1k queries on the same Pareto chart as quality.
5. **K9** — a human review queue for auto-generated ground truth.
6. **L13 + L14** — paired significance testing, with "not distinguishable" as a visible result state.
7. **L8** — character-level precision, exposing configs that game Recall@k with huge chunks.
8. **G16** — naive vs advanced vs agentic vs graph vs visual, priced on one chart.
9. **N2** — every failure auto-classified into the Seven Failure Points taxonomy.
10. **E14** — the quantization cost/recall frontier.
11. **O4** — a poisoning-robustness score for a retrieval config.
12. **D14 + D15** — a query adapter trained from the user's own eval set, with hard negatives mined from the sweep itself.

Items 1, 2, 4, 5 and 6 are P0 and ship at launch. That is enough. The rest is the roadmap.

---

## Sources

**Landscape and prior art:** [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) · [RAGBuilder](https://github.com/kruxai/ragbuilder) · [Ragas](https://www.ragas.io/) · [open-rag-eval](https://github.com/vectara/open-rag-eval) · [reranker-eval](https://github.com/agentset-ai/reranker-eval) · [Vectorize RAG Evaluation](https://docs.vectorize.io/learn/getting-started/rag-evaluation-quick-start/) · [rag-chunk](https://github.com/messkan/rag-chunk) · [ChunkViz](https://testdev.tools/chunk-viz/)

**Libraries to build on:** [ranx](https://github.com/AmenRa/ranx) · [chonkie](https://docs.chonkie.ai/common/open-source) · [LLMLingua](https://github.com/microsoft/LLMLingua) · [Docling](https://arxiv.org/pdf/2501.17887)

**Papers and benchmarks:** [LegalBench-RAG](https://arxiv.org/abs/2408.10343) · [Seven Failure Points](https://arxiv.org/abs/2401.05856) · [UniDoc-Bench (multimodal RAG)](https://arxiv.org/pdf/2510.03663) · [Reasoning Agentic RAG survey](https://arxiv.org/pdf/2506.10408) · [RAGRouter-Bench](https://arxiv.org/pdf/2602.00296) · [Long-Context LLMs Meet RAG](https://arxiv.org/pdf/2410.05983) · [Securing RAG: taxonomy of attacks and defenses](https://arxiv.org/html/2604.08304v1) · [Semantic Chameleon (poisoning)](https://arxiv.org/pdf/2603.18034)

**Practice:** [Weaviate on rotational quantization](https://weaviate.io/blog/8-bit-rotational-quantization) · [Vector index tuning: HNSW, IVF, PQ](https://appscale.blog/en/blog/vector-index-tuning-hnsw-ivf-product-quantization-recall-latency-2026) · [Multimodal RAG in 2026](https://bigdataboutique.com/blog/multimodal-rag-retrieval-over-images-pdfs-and-text) · [Semantic caching for LLM inference](https://www.spheron.network/blog/semantic-cache-llm-inference-gpu-cloud/) · [Linear adapter embedding](https://github.com/ALucek/linear-adapter-embedding)

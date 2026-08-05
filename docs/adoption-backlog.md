# Adoption backlog — replacing hand-written plumbing with real libraries

**Date:** 2026-08-04 · Companion to [design.md](./design.md) and [roadmap.md](./roadmap.md)

## Why this exists

Everything in context-grid today is hand-written on top of numpy. That was one decision applied
to two very different kinds of code, and it was right for one of them and wrong for the other.

**Right for the scorer.** Span resolution, the coverage policy, the metrics — these are the
things that make the tool trustworthy, and they are verified: the metrics match `ranx` exactly
on a thousand random cases, and the span algebra is covered by property tests. Owning that code
is the point.

**Wrong for the plumbing.** Chunkers, embedder hosting, vector indexes, ingestion, LLM calls.
Writing my own means a worse implementation than a maintained library, a narrower set of
options to sweep, and — worst of all — the tool measures *my* chunker rather than the chunker
people actually deploy. A comparison of strategies nobody runs is a comparison of nothing.

So: adopt libraries for every dimension, keep the scorer.

## The rule for every adoption

Three conditions, all of them, or it does not land:

1. **It becomes a config value.** `chunker: chonkie:semantic` must work the same way
   `chunker: recursive:512` does today. No new API surface for the user.
2. **It is an optional extra.** `pip install context-grid` must stay one dependency. Every
   library goes behind `[chunk]`, `[embed]`, `[index]` and so on, registered lazily.
3. **It passes the conformance suite.** Offsets round-trip, no silent text loss, determinism.
   A library that cannot preserve character offsets declares `offsets_exact=False` rather than
   being trusted.

## Where this is going: one file runs everything

The end state the backlog serves. A user writes one YAML, and the whole thing runs.

```yaml
# contextgrid.yaml
corpus: ./documents
evalset: ./questions.jsonl

grid:
  parser:    [docling, marker, pymupdf]
  chunker:   [chonkie:recursive:512, chonkie:semantic, chonkie:late]
  embedder:  [tei:bge-base-en-v1.5, litellm:text-embedding-3-small]
  index:     [faiss:hnsw, faiss:flat, bm25, hybrid]
  transform: [none, hyde]
  reranker:  [none, tei:bge-reranker-v2-m3]
  candidates: [10, 50, 100]

run:
  mode: ofat            # factorial | ofat | staged
  k: 10
  headline: recall@5
  budget_usd: 2.00
  budget_seconds: 900

report:
  out: ./results
  formats: [markdown, json, html]
```

`contextgrid run contextgrid.yaml` → a bundle with the leaderboard, the significance test, the
failure diagnosis, the winning config and the manifest.

That file is the product. Everything below is in service of it.

---

## Phase 0 — the config file *(no research needed, do first)*

Every adoption below lands as new config values, so the config format should exist before they
start arriving rather than being retrofitted around six of them.

- A schema for the YAML above, validated with clear errors that name the offending key.
- `contextgrid run config.yaml` as the primary CLI entry point.
- `contextgrid init` to write a starter config with the installed plugins listed as comments.
- Config → run manifest, so the file itself is part of what reproduces a run.
- JSON accepted as well as YAML, same schema.

**Open question for the owner:** YAML needs a parser. `pyyaml` is the obvious one and would be
the *first* dependency outside numpy in the core. The alternative is keeping it in the `[config]`
extra and having `contextgrid run` say "pip install context-grid[config]". My inclination is to
put it in the core — a config-driven tool whose config format is optional is silly.

---

## Dimension 1 — Chunkers ✅ done

**Shipped.** chonkie (token, recursive, sentence, code) and langchain-text-splitters
(recursive, character, markdown) sit alongside the five hand-written ones, twelve arms on one
axis, all behind `pip install 'context-grid[chunk]'`. Nothing new to learn:
`chunker: [recursive:512, chonkie:recursive:512, langchain:recursive:512]` in the YAML.

Four things came out of doing it that were not obvious beforehand:

* **chonkie's offsets are exact.** Verified rather than assumed, and re-checked on every
  document at runtime. It was the open question in the table below and the answer is good.
* **LangChain's offsets are not.** It reports `start_index: -1` for roughly one chunk in eight
  on our fixtures — always tables — because it rebuilds each chunk by rejoining the pieces it
  split and then cannot find the result in the source. The content *is* still a literal slice,
  so the adapter locates and verifies the offset itself. Taking the -1 at face value would
  have dropped those chunks.
* **Both libraries measure size in the wrong unit by default** — chonkie in characters,
  LangChain in `len`. Left alone, `chonkie:recursive:512` would have meant 512 characters
  while `recursive:512` meant 512 tokens, and the axis would quietly have become a units
  comparison. Both adapters push our tokenizer down.
* **Plugin names needed namespacing**, so the registry now resolves the longest registered
  name rather than splitting on the first colon: `chonkie:recursive:512` is the plugin
  `chonkie:recursive` at size 512, not a plugin called `chonkie`.

All twelve pass the same conformance suite. First measurement, on the demo corpus at 192
tokens: chonkie's recursive 0.863 against our 0.849 and LangChain's 0.849 — and at 96 tokens
plain `fixed` beats all three. Which is exactly the sort of thing that is worth knowing and
nobody publishes.

**Not adopted:** semchunk (one strategy, covered) and llama-index node parsers (heavy, and its
`Node` type would need its own adapter for little gain over the two above).

**Originally.** Six hand-written: fixed, recursive, sentence, structural, semantic. All preserve
exact offsets. All are mine, which means the tool compares my chunkers rather than the ones
people deploy.

| Candidate | For | Against |
|---|---|---|
| **[chonkie](https://docs.chonkie.ai/oss/chunkers/overview)** | Purpose-built for exactly this. Nine strategies including late chunking, neural boundary detection and AST-based code chunking. Rust-backed, ~100× faster than pure Python. 505 KB installed against LangChain's ~50 MB. Used in production at scale | Young project. Offset preservation needs verifying against our conformance suite before it can be trusted |
| **[semchunk](https://pypi.org/project/semchunk/)** | Tiny, pure Python, no dependencies. Reported more semantically accurate than LangChain's recursive splitter and 90% faster than semantic-text-splitter | One strategy only. Complements chonkie rather than replacing it |
| **langchain-text-splitters** | Everybody's default, so "what LangChain does" is a meaningful arm on the grid | Six general-purpose splitters against chonkie's nine specialised ones. Pulls a heavier dependency tree |
| **llama-index node parsers** | Strong hierarchical and sentence-window parsers. The other default people actually run | Heavy. Coupled to LlamaIndex's `Node` type, which needs adapting to our `Chunk` |

**Recommendation:** chonkie as the primary, langchain-text-splitters as a second arm because
it is what most deployments actually use, and keeping our hand-written ones as the offset-exact
reference. Three sources on one axis is the comparison the tool exists to make.

---

## Dimension 2 — Embedders ✅ done

**Shipped.** Two backends, decided together: **litellm** for anything hosted and **TEI** for
anything local.

    embedder: [tei:bge-base-en-v1.5, litellm:text-embedding-3-small]

TEI needs no extra dependency at all — it is reached over plain `urllib`, so a running server
plus a bare `pip install context-grid` gives real embeddings with no key and no network.
litellm sits behind `[llm]` and reaches every hosted provider through one name.

Four things came out of building it:

* **Prefixes are now looked up from the model name.** E5 wants `query:` and `passage:`, BGE
  wants an instruction on the query and nothing on the document, OpenAI wants neither. Getting
  this wrong does not fail — it just costs several points, invisibly, and costs them *unevenly
  across the arms of a sweep*, which turns a model comparison into a comparison of one model
  against a handicapped version of another. An explicit prefix always wins, including an
  explicit empty one.
* **The cost model was pricing hosted models at zero.** It split the spec on the first colon,
  so `litellm:text-embedding-3-small` looked up the price of "litellm", found nothing, and
  charged zero. Now the price is looked up under the model, with the backend prefix stripped
  and provider routes (`cohere/…`) resolved.
* **`matrix()` silently accepted plugin instances**, which then blew up much later inside a
  report formatter with `expected str`. Axes take spec strings, because a configuration has to
  be writable into a leaderboard row, a cache key and a YAML file — a run nobody can write down
  is a run nobody can reproduce. The error now says so, and names the string to use.
* **There is a `transport` hook** on both backends: hand it a callable and a whole sweep runs
  with no server and no key. That is how these are tested, and it is a real need for anybody
  building on the package.

**Not adopted:** sentence-transformers (would drag ~2 GB of torch into an install whose whole
point is being small) and Infinity (TEI chosen instead; litellm can reach Infinity anyway).

**Originally.** TF-IDF, hashing and a length control. Useful as baselines, useless as real retrieval.
This is the biggest gap in the package: nobody can currently sweep a real embedding model.

| Candidate | For | Against |
|---|---|---|
| **[litellm](https://docs.litellm.ai/docs/embedding/supported_embedding)** | One interface to every hosted provider — OpenAI, Cohere, Voyage, Gemini, Bedrock, plus TEI and Infinity as backends. Solves the BYOK story in one dependency. Also gives us LLM calls (dimension 7) from the same library | Large. Its abstraction occasionally lags provider features. Needs keys, so untestable in CI |
| **[TEI](https://github.com/huggingface/text-embeddings-inference)** (HuggingFace) | The standard local server. Runs on Mac, Linux, Windows. Exposes `/embed`, `/rerank`, `/tokenize` — so it covers the reranker dimension too. Nearly any open model | A separate server process. Docker or a binary to manage |
| **[Infinity](https://github.com/michaelfeil/infinity)** | Higher throughput than TEI in most reports. ONNX/CTranslate2 backends, dynamic batching, CPU-friendly. Serves embeddings, rerankers, and ColPali — which would unlock the visual-retrieval axis. Already validated in your private research | Same server-process cost. Smaller community than TEI |
| **sentence-transformers** | In-process, no server, one pip install. Simplest possible path to a real model | Drags in torch (~2 GB). Slower than a dedicated server. Would make CI unusable |

**Recommendation:** litellm for hosted plus one local server. Between TEI and Infinity I lean
Infinity — you have already validated it, it is CPU-friendly, and it covers rerankers and
ColPali from the same process. Both are proxied by litellm, so one adapter can reach either.

---

## Dimension 3 — Indexing ✅ done

**Shipped.** All three, as decided:

    index: [dense, faiss:flat, faiss:hnsw, faiss:ivfpq, usearch:i8, pgvector:hnsw]

The chart this package most wanted to draw — *what did approximation actually cost you?* — had
nothing to plot before this, because every index was exact. Now `faiss` gives flat, IVF, HNSW
and IVFPQ from one wheel, `usearch` gives a second opinion on HNSW with f32/f16/i8/b1 storage,
and `pgvector` is the arm that is a database rather than a library.

On a 400-vector fixture, measured against exhaustive search: `faiss:hnsw` 0.98, `faiss:ivf` at
one probe 0.42 and at ten 1.00, `faiss:ivfpq` 0.175 for 4.4 KB against flat's 102 KB.
`usearch` f32 and f16 both 1.00, i8 0.95 for half the memory.

Four things worth recording:

* **A small corpus trains a bad codebook silently.** PQ learns 2^bits centroids per subspace
  and faiss wants ~39 points for each, so the default 8 bits needs ~10,000 vectors. Below that
  faiss prints a warning nobody reads and returns plausible, wrong neighbours. `nlist` and
  `pq_bits` are now reduced to fit and the reduction is recorded on `fitted_to_corpus`.
* **pgvector's session parameters were being ignored entirely.** `SET LOCAL` lasts until the
  end of the current transaction, and on an autocommit connection there is none — Postgres
  accepted it, did nothing, and every query ran at the default `probes = 1` while the sweep
  reported numbers for whatever was asked for. Only a live database caught this. With `SET`,
  recall goes 0.26 → 1.00 across the probe range; before, it was 0.26 at every setting.
* **usearch's own `memory_usage` is useless for this.** It reports the arena it allocated,
  which barely moves between f32 and i8 — so the dtype axis would have shown quantization
  saving nothing. Computed from the dtype instead.
* **`cosine` has to mean one thing.** Left to their defaults these libraries compare L2 on raw
  vectors, which ranks differently. Both normalise on both sides, so the index axis compares
  indexes rather than metrics in disguise.

pgvector's exact mode is cross-checked against the numpy reference, ranking for ranking — two
completely different implementations of the same thing, so a metric bug in either shows up.
Its tests skip without `PGVECTOR_DSN` and there is no in-process fake: a fake pgvector would
measure nothing, and passing in CI would be worse than the arm being absent.

**Originally.** Exact dense (numpy matmul), BM25, hybrid fusion, quantization. Correct, and exact
search only — there is no approximate index, so the ANN recall-loss chart has nothing to plot.

| Candidate | For | Against |
|---|---|---|
| **[faiss](https://github.com/facebookresearch/faiss)** | The reference. Flat, IVF, HNSW, PQ, OPQ — every index type on one axis, which is exactly the sweep we want. `faiss-cpu` is a reasonable wheel | Not a database: no CRUD, no persistence story worth having. Fine here, since we build indexes and throw them away |
| **[hnswlib](https://github.com/nmslib/hnswlib)** | Tiny, fast, `M`/`efConstruction`/`efSearch` exposed directly — the parameters the PRD wants swept | HNSW only. faiss covers it and more |
| **[usearch](https://github.com/unum-cloud/usearch)** | Smaller and faster than faiss in its own benchmarks. Built-in quantization | Smaller ecosystem. Its benchmarks are self-published |
| **pgvector** | What people actually deploy on. Tests the real thing rather than a library | Needs Postgres running. Slowest to set up, and hardest in CI |
| **qdrant / lancedb** | Real vector databases with filtering | Heavier than needed to compare index *types* |

**Recommendation:** faiss first — one dependency covering flat, IVF, HNSW and PQ, which turns
the ANN-versus-exact chart from impossible into free. pgvector second, because "what we actually
run in production" is a legitimate and different arm. usearch is a nice-to-have.

---

## Dimension 4 — Parsers ✅ done

**Shipped.** docling, marker and pymupdf4llm, as decided:

    parser: [pymupdf, pdfplumber, pymupdf4llm, docling, marker]

Five arms on the axis nothing else in the field measures, three of them real engines. All emit
Markdown, which is split into blocks with tables kept whole and page markers read and dropped.
Offsets stay exact — every block is a literal slice of the text that parser produced.

Measured on the contract fixture: `pymupdf` 7 ms and no table; `pymupdf4llm` 412 ms and finds
the table; `docling` 222 s on a cold start with model loading, 4 blocks, table found. That
spread is the finding, and it is the kind of thing usually left as folklore.

**pymupdf4llm is not deterministic in-process, and that made isolation non-negotiable.** Its
output for a document depends on which documents went through the same interpreter before it:
a prose PDF that parses to 1182 characters alone parses to 919 mangled ones — "notce perod s
trty", characters simply dropped — after a PDF with a table has gone through. The state is in
MuPDF's C layer, below Python: reloading the module, passing an explicit `hdr_info`, resetting
`small_glyph_heights` and emptying MuPDF's store all fail to clear it. Only a fresh process
does. So each document is parsed in its own subprocess, which costs ~100 ms against the ~400 ms
the conversion already takes. For a tool whose entire foundation is the parse, a corpus that
parses differently depending on file order is disqualifying.

docling and marker are behind `CG_SLOW_PARSERS=1` in the test suite — they load vision models
and take minutes cold. Both have been run for real.

**Not adopted:** MinerU (heavy, awkward PaddleOCR chain), extractous, and `unstructured`, which
stays registered-but-unimplemented.

**Originally.** Text, Markdown, PyMuPDF, pdfplumber. Docling and Unstructured are registered but not
implemented. This is the headline axis and it has four arms, two of which are trivial.

| Candidate | For | Against |
|---|---|---|
| **[docling](https://github.com/docling-project/docling)** (IBM) | PDF, DOCX, PPTX, XLSX, HTML, images, audio. Strong table extraction. Already registered | Heavy — layout models. Slow per page |
| **[marker](https://github.com/datalab-to/marker)** | Best-rated for structure fidelity. Surya OCR, 90+ languages | Slow. Reported ~100× slower than fast extractors, which is itself a finding worth charting |
| **[MinerU](https://github.com/opendatalab/MinerU)** | Excellent on complex layouts and CJK. Outputs Markdown *and* JSON | Heavy. PaddleOCR dependency chain is awkward |
| **unstructured** | `fast` and `hi_res` as two distinct arms from one library. Already registered | Heavier than needed. Local build reportedly weaker than their cloud |
| **pymupdf4llm** | Markdown output from PyMuPDF, no extra weight. Cheap third arm | Same engine as PyMuPDF, so it tests output format rather than extraction |
| **[extractous](https://github.com/yobix-ai/extractous)** | Rust, very fast, many formats | Newer. Less proven on tables |

**Recommendation:** docling and marker next — they are the two the field actually benchmarks,
and together with PyMuPDF they give fast/accurate/table-aware as three genuinely different arms.
MinerU after, for the CJK and complex-layout case.

---

## Dimension 5 — Rerankers ✅ done

**Shipped.** As predicted, it fell straight out of dimension 2 — the same two backends:

    reranker: [null, tei-rerank:bge-reranker-base, litellm-rerank:cohere/rerank-english-v3.0]
    candidates: [10, 50, 100]

TEI's `/rerank` needs no extra dependency, same as its `/embed`. One caveat worth knowing:
**a TEI process serves one model**, so reranking needs its own server on its own port. The
natural assumption is the opposite and the failure is an unhelpful 400, so the error says so.

Two things the adapter insists on:

* **Every candidate comes back or the run fails.** A backend that silently returns fewer
  results than it was given — a passage too long, a batch capped, `top_n` defaulting to five —
  drops documents from the ranking, and on a leaderboard that is indistinguishable from the
  reranker judging them irrelevant. Completely different claim. The litellm call explicitly
  asks for `top_n = len(passages)` for exactly this reason.
* **Ties keep the retriever's order.** Without a stable tie-break a rerun reshuffles
  equally-scored passages and a diff reports a change that did not happen.

`candidates` remains the axis that actually matters and the one every reranking blog post
omits: over the top 10 a reranker can only reorder what was already found; over the top 100 it
can rescue what ranked 47th.

**Not adopted:** sentence-transformers (torch again) and AnswerAI `rerankers` — worth
revisiting later for ColBERT and RankGPT, which no server exposes.

**Originally.** Identity, lexical overlap, MMR. All hand-written. The cross-encoders are registered
and unimplemented, which means the reranker axis currently compares three things nobody deploys.

| Candidate | For | Against |
|---|---|---|
| **TEI / Infinity `/rerank`** | Same server as the embedders. `bge-reranker-v2-m3`, `mxbai-rerank`, `jina-reranker` all served | Depends on dimension 2's decision |
| **sentence-transformers CrossEncoder** | In-process, simplest | torch again |
| **litellm rerank** | Cohere and Voyage rerank behind one interface. BYOK | Hosted only |
| **[rerankers](https://github.com/AnswerDotAI/rerankers)** (AnswerAI) | One tiny interface over cross-encoders, ColBERT, RankGPT, Cohere, FlashRank. Purpose-built for exactly this axis | Young. Another abstraction layer |

**Recommendation:** falls out of dimension 2 — whichever server we adopt gives rerankers for
free. `rerankers` is worth a look as a second arm since it covers ColBERT and RankGPT, which
the servers do not.

---

## Dimension 6 — Ingestion ✅ done (rebuilt)

**The first attempt was wrong and this replaces it.** It shipped `ingestion: [direct, agno]`,
which compares two *libraries* — a parser-shaped question wearing an ingestion label. agno is a
framework, not a strategy. agno's text extraction has moved to `parser: agno`, where it belongs
and where it can be compared against pymupdf, pdfplumber and docling fairly.

**The definition that should have come first:**

> A chunker produces units where the thing indexed and the thing returned are the same.
> **An ingestion strategy deliberately breaks that identity.**

Chunk size is a compromise nobody is happy with: small chunks embed precisely and arrive
stripped of context; large chunks keep their context and embed into mush. Every strategy here
refuses that compromise — index one thing, return another.

| Strategy | Indexes | Returns | Cost |
|---|---|---|---|
| `plain` | the chunk | the chunk | — the baseline |
| `parent-document:N` | small chunks | the passage they came from | free |
| `sentence-window:N` | one chunk | it plus N neighbours either side | free |
| `hierarchical:N` | leaves | the parent once enough siblings hit | free |
| `contextual` | chunk + LLM-written placing note | the chunk | 1 call/chunk |
| `hypothetical-questions:N` | the questions it answers | the chunk | 1 call/chunk |
| `propositions:N` | atomic facts | the chunk | 1 call/chunk |
| `summary` | an LLM summary | the whole document | 1 call/**doc** |

Measured on the demo corpus at 96-token chunks: **`parent-document` 0.863 against plain's
0.616** — a +0.247 gain for no model calls at all. `sentence-window` 0.825. `hierarchical` 0.575,
slightly *below* plain, because merging spends result slots on wider passages.

**One bug found, and it was a scoring bug rather than a strategy one.** `hierarchical` first
made both the leaves and their parents retrievable — so gold resolved to each of them, a
question with one answer acquired two things to find, and recall halved for a purely structural
reason. Measured: 1.86 relevant units per question against plain's 1.00. Parents are now
*presentation*, mapped back to the units they cover, so showing a generator more context never
changes what retrieval is credited with finding.

**Also fixed:** `recursive:64` was an error. The default overlap of 64 collided with a size of
64, and refusing a perfectly reasonable chunk size over a default the user never named is a bad
axis value. Overlap now defaults to an eighth of the size — exactly 64 at the default 512, so
nothing already written changes — and an overlap somebody *does* name is still checked.

**Not built:** RAPTOR and GraphRAG. Both construct whole tree and graph structures, which is a
larger piece of work than the other eight combined.

---

## Dimension 7 — LLM calls ✅ done

**Shipped.** litellm, and the hand-written OpenAI and Anthropic clients are gone:

    openai:gpt-4o-mini | anthropic:claude-sonnet-5 | litellm:gemini/gemini-2.0-flash
    litellm:ollama/llama3  (with api_base, for a local server)

Three clients became one adapter under three names, so every config already written stays
valid. A bare `openai` resolves too — an axis value that needs a parameter before it is usable
at all is a bad axis value.

**The cost bonus was real.** The price table here is ten hand-maintained entries against a
field of thousands, which meant nearly every real model was costed at zero — and a silent zero
reads as "free" rather than "unknown". litellm ships prices for 2985 models, so the local table
now wins where it has an opinion and litellm answers everything else. `claude-sonnet-5` prices
at $2/$10 per million in and out; `gpt-4o-mini` at $0.15/$0.60. Output tokens are priced
separately, which matters because every provider charges several times more for them and
collapsing the two makes a verbose model look as cheap as a terse one. An unknown model still
warns, so the fallback did not swallow the honest signal. Costing keeps working with litellm
absent.

**Not adopted:** instructor. Worth revisiting for question generation, where structured output
with validation and retries is exactly the need — `parse_json_reply` currently does that job by
being forgiving about wrappers and strict about content.

**Originally.** A one-method `LLM` protocol with hand-written OpenAI and Anthropic clients. Used by
generation, question generation, and the query transforms.

| Candidate | For | Against |
|---|---|---|
| **[litellm](https://docs.litellm.ai/)** | 100+ providers, one interface. Cost tracking built in, which the cost panel wants anyway. Retries, fallbacks, caching. Same dependency as dimension 2 | Large. Its own opinions about config |
| **Provider SDKs directly** | What we have. No abstraction to fight | One adapter per provider, forever |
| **[instructor](https://github.com/jxnl/instructor)** | Structured output with validation and retries — which is exactly what question generation needs | Narrow. Complements litellm rather than replacing it |

**Recommendation:** litellm, and delete the hand-written clients. Its built-in cost tracking is
a genuine bonus: the cost model currently carries a hand-maintained price table that will go
stale, and litellm maintains one.

---

## Dimension 8 — Transforms

**Today.** HyDE, multi-query, decomposition, step-back, acronym expansion. Hand-written prompts.

Honest finding from the research: **there is no good library for this.** Query transforms live
inside LangChain and LlamaIndex as pipeline components rather than as anything reusable.
[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) is the closest — a research toolkit with 17
RAG algorithms and 36 benchmark datasets — but it is built to reproduce papers rather than to be
embedded in another tool.

**Recommendation:** keep ours, and steal FlashRAG's *prompts* and algorithm list rather than its
code. Revisit if something purpose-built appears. This is the one dimension where hand-written
is still the right answer, and it is worth saying so rather than adopting a framework for the
sake of consistency.

---

## Dimension 9 — Metrics

**Today.** Hand-written, verified against `ranx` on random cases in CI.

| Candidate | For | Against |
|---|---|---|
| **Keep ours** | Zero-dependency core. Already proven correct against the reference. The verification is itself a selling point | It is code to maintain |
| **[ranx](https://github.com/AmenRa/ranx)** | Peer-reviewed. Fusion algorithms and statistical tests included. Already a dev dependency | Numba, and therefore LLVM, in the core install |
| **[ir_measures](https://ir-measur.es/)** | Unified interface over pytrec_eval, gdeval, trectools. Standardised measure names. The most *correct* option | Another layer. pytrec_eval needs compiling |
| **pytrec_eval** | Direct binding to trec_eval, the actual reference implementation | C extension. Awkward wheels |

**Recommendation:** keep ours as the default and add `ir_measures` as an optional *cross-check*
backend. "Our numbers match the reference implementation exactly" is a stronger claim than
"we imported the reference implementation", and it costs nothing to keep. This is the other
dimension where I would push back on adopting.

---

## Suggested order

Config first, then the dimensions that unlock the most, then the rest.

| Step | Dimension | Why here |
|---|---|---|
| 0 | **Config file** | Everything below lands as config values |
| 1 | **Embedders** | The biggest gap. Nobody can sweep a real model today, which limits every other axis |
| 2 | **Rerankers** | Falls out of step 1 for free |
| 3 | **Chunkers** | Cleanest adoption, immediate breadth, low risk |
| 4 | **Indexing** | faiss unlocks the ANN-versus-exact chart, which is currently impossible |
| 5 | **LLM calls** | litellm; deletes code and fixes the stale price table |
| 6 | **Parsers** | docling and marker. Heaviest installs, so last of the big ones |
| 7 | **Ingestion** | Connectors. Valuable and not blocking anything |
| 8 | **Transforms / Metrics** | Recommend keeping both. Revisit if that changes |

Each step: research, options with pros and cons, **owner decides**, then implement behind an
extra with conformance tests.

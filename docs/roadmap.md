# context-grid — Build Roadmap

**Date:** 2026-08-03 · Companion to [design.md](./design.md)

Feature IDs (`B4`, `K6`, `L13`…) refer to the catalogue in the blog repo at
`docs/prd/rag-retrieval-lab-features.md`.

---

## 0. The re-cut priorities

A sanity check of the catalogue's ~75 P0 features against an SDK-first release found six problems.
This roadmap is the response to them.

| # | Finding | Response |
|---|---|---|
| 1 | ~20 "P0" features are UI, not SDK — parser diff view (`B3`), boundary overlay (`C19`), recommendation card (`N11`), demo mode (`Q12`), permalinks (`P1`,`P2`), all 12 screens | Split into **SDK-P0** and **App-P0**. App work starts at M9, after the SDK is real |
| 2 | Even without UI, ~55 P0s is 2–3 months, not a first release | **v0.1 is 18 features.** Everything else is sequenced behind it |
| 3 | The critical path is offsets, not metrics. `B4` needs `K6` needs `B2` | **M0 is the offset spine.** Nothing ships before it is proven |
| 4 | `C14` (per-tokenizer sizing) and `M5` (content-hash cache) collide — a cache key without the tokenizer serves the wrong chunks | Tokenizer is part of the chunk cache key from M2. Designed in, not patched |
| 5 | `D6` (truncation warning) has no home in a library | **Structured warnings channel** in M0's core types. Every result object carries `.warnings` |
| 6 | Nothing in the catalogue covers "did this plugin behave correctly?" across ~40 plugins | **Conformance suites** are a first-class deliverable, introduced in M1 |

## 1. Release plan

| Version | Milestones | What it means |
|---|---|---|
| **v0.1** — private | M0–M3 | The scorer works and is proven. One real pipeline runs end to end |
| **v0.2** — private | M4–M5 | Grids actually sweep. Cost and caching are real |
| **v0.3** — **first public release** | M6–M8 | Eval sets, significance, exports, CLI, validation against LegalBench-RAG |
| **v0.4** | M9 | The blog UI at `/lab`, consuming the SDK |
| **v0.5+** | M10–M12 | Generation, the wide field, distribution |

---

## M0 — The offset spine ✅ *shipped*

**Why first.** Every number the SDK will ever print depends on the span algebra. A mistake here is
unrecoverable once results are published.

**Build**
- `core/types.py` — `Span` with the full overlap algebra, `Document`, `Chunk`, `GoldSpan`,
  `EvalItem`, `EvalSet`, `RelevanceLabel`
- `core/warnings.py` — structured warnings channel (`D6`, `B14`)
- `core/errors.py` — including `MissingExtraError`
- `score/resolve.py` — `SpanResolver` with `coverage` / `iou` / `containment` policies, per-chunk
  qrels and union coverage (`K6`)

**Test**
- Unit tests on the span algebra, including every boundary case: touching, nested, identical,
  zero-length, adjacent
- Hypothesis property tests: IoU symmetric and in [0,1], coverage in [0,1], resolution invariant to
  chunk ordering, union coverage monotone as the retrieved set grows, split gold reassembles
- `mypy --strict` clean, `ruff` clean

**Exit** — property tests pass on 1000+ generated cases; 100% coverage on `score/resolve.py`.

---

## M1 — Plugin architecture and conformance ✅ *shipped*

**Build**
- `core/protocols.py` — `Parser`, `Chunker`, `Embedder`, `Index`, `Reranker`, `Generator`
- `core/registry.py` — name-based registration, entry-point discovery, `MissingExtraError` on
  absent extras
- `tests/conformance/` — one parameterised suite per protocol

**Conformance rules, per family**

| Family | Must satisfy |
|---|---|
| Parser | Offsets round-trip: `doc.text[block.span]` equals `block.text` when `offsets_exact`. Deterministic. Empty input safe |
| Chunker | Same offset round-trip. Chunks cover the document without gaps unless declared. Overlap declared honestly. Deterministic |
| Embedder | Deterministic. Normalised when it claims to be. Correct query/document asymmetry (`D4`). Raises the truncation warning (`D6`) |
| Index | Exact search returns true nearest neighbours. ANN reports its recall against exact (`E5`) |
| Reranker | Order-invariant to input order. Deterministic |

**Exit** — a deliberately broken toy plugin per family fails its suite for the right reason. ✅

**What actually shipped.** Protocols for `Parser`, `Chunker` and `Tokenizer`; a registry with
lazy imports and spec strings (`recursive:512,overlap=64`); two zero-dependency parsers (text,
Markdown); four chunkers (fixed, recursive, sentence-window, structural); two tokenizers.
Conformance suites for parsers and chunkers, plus `test_conformance_catches_bugs.py`, which
builds six deliberately broken plugins and asserts each invariant catches its bug — an
off-by-one parser, a silently normalising parser, a table-losing parser, a gappy chunker, a
chunker that rewrites text while claiming exact offsets, and one with colliding ids.

**Two things the build changed.**

1. **Gold anchors (new).** Ground truth as character spans compares *chunkers* perfectly well,
   because they all cut up the same text. It cannot survive a change of *parser*, because two
   parsers produce different text — and the parser axis is the headline feature. So ground
   truth now has two forms: a portable `GoldAnchor` (a quoted passage plus a page hint) and a
   resolved `GoldSpan`. Anchors resolve to spans against each parse. The pleasing part is that
   a parser mangling a table so the quote no longer appears is itself the measurement — the
   anchor fails to resolve, and a parser that loses the evidence cannot retrieve it. The
   resolver lands in M2; the types are in place.
2. **A no-silent-loss invariant.** Every chunker must cover every non-whitespace character of
   the document unless it declares that it samples on purpose. Text in no chunk is evidence no
   retriever can ever return, and it shows up on a leaderboard as nothing more than slightly
   worse recall.

---

## M2 — Corpus, PDF parsers, the anchor resolver ✅ *shipped*

**Build** — `corpus/` loader and fingerprint (`A1`–`A5`); `parse/pymupdf.py` and
`parse/pdfplumber.py` (`B1`, `B2`); **the anchor resolver** — locating a quoted passage in a
parse by exact then whitespace-normalised match, and warning when a parser has lost it
entirely; `cache/` content-addressed store with the tokenizer in the chunk key.
Chunkers and tokenizer-normalised sizing arrived early, in M1.

**Test** — conformance suites pass; cache hit/miss behaviour asserted explicitly

**Exit** — a PDF becomes chunks with exact offsets, twice, the second time from cache. ✅ *(the
cache slips to M3; everything else shipped)*

**What actually shipped.** `Corpus` loading from a directory, a file list or a dict of texts, with
an order-independent content hash for the manifest. A corpus fingerprint that profiles the
documents and emits plain-English hints about which axes will matter — table share, code share,
heading density, document length, duplicate files, empty pages. `TextAssembler`, which builds
document text and block spans together so a PDF parser cannot produce drifting offsets. Two PDF
parsers, PyMuPDF and pdfplumber, both passing the full conformance suite. And the anchor resolver:
exact → whitespace-normalised → bounded matching, with the failure to find evidence reported as a
measurement of the parser rather than a bug in the eval set.

**Two bugs the build found.**

1. **Heading inference was voting by line, not by character.** A bordered table contributes a dozen
   two-word cells set smaller than the body; by line count they outvote the prose, the inferred
   body size drops to the cell size, and the actual body text gets promoted to a heading. Sizes are
   now weighted by characters, with a `min_ratio` so emphasis is not mistaken for structure.
2. **Table rendering changes what ground truth will match.** Markdown pipes give an embedder a
   better signal about cell boundaries and stop a row reading as "Premium 3400 500", so an anchor
   quoting the row verbatim no longer resolves. That is a real trade-off rather than a formatting
   preference, so it is a parameter (`table_format`) and it is documented rather than hidden.

---

## M3 — Cache, embed, index, retrieve, score ✅ *shipped*

**Build** — `cache/` content-addressed store with the tokenizer in the chunk key (carried over
from M2), `embed/` local CPU models via ONNX (`D1`, `D4`, `D6`), `index/` exact dense + BM25 +
RRF hybrid (`E1`, `E6`, `E8`), `retrieve/` single-shot dense/sparse/hybrid (`G1`–`G3`),
`score/metrics.py` wrapping `ranx` (`L1`–`L7`, `L11`)

**Test** — metrics cross-checked against hand-computed values on a tiny fixture; hybrid fusion
checked against `ranx` directly

**Exit** — **v0.1.** One corpus, one eval set, one config, a real Recall@5. The first honest number.

---

## M4 — The grid ✅ *shipped*

**Build** — `grid/matrix.py` (axis expansion, live config count), sweep modes full-factorial / OFAT
/ staged, `grid/runner.py` with prefix reuse, cancel and resume, `report/results.py` with the
leaderboard, `report/manifest.py` (`P4`)

**Test** — prefix reuse asserted: sweeping 20 rerankers embeds exactly once; manifest hash stable
across identical runs and different across changed params; resume produces identical results

**Exit** — a 20-config OFAT sweep completes under 10 minutes on 4 CPU cores.

---

## M5 — Cost and performance ✅ *shipped*

**Build** — `cost/` pricing tables and model (`M1`, `M2`), pre-run estimate and hard budget cap
(`M3`, `M4`), staged latency breakdown (`L18`), index build time and size (`L19`, `E20`),
cache-saving readout (`M6`), Pareto computation (`M7`)

**Test** — cost model against frozen pricing fixtures; budget cap actually halts a run; latency
attribution sums to wall-clock within tolerance

**Exit** — **v0.2.** Every config carries dollars and milliseconds beside its quality. ✅
*(shipped as v0.1.0 — the private v0.1/v0.2 split was not worth the ceremony)*

**What M3–M5 actually shipped.** Content-addressed cache (memory and disk) with prefix reuse.
Three zero-dependency embedders, including a chance-level control that proves the scoring chain
measures something. Exact dense, BM25, and hybrid search with both RRF and weighted fusion.
Retrieval metrics implemented natively and **verified against `ranx` on randomly generated
cases**, which keeps the core at numpy-and-nothing-else while making the correctness claim
checkable. The grid with all three sweep modes, per-stage timings, a cost model that prices
local compute by the hour so it lands on the same chart as a hosted API, and the results views:
leaderboard, Pareto frontier, axis effects, config diff and a plain-English summary.

**Three things the build found.**

1. **MAP has two conventions and they disagree by a lot.** Dividing by `min(relevant, k)` is the
   Kaggle convention; dividing by the total number of relevant chunks is trec_eval's, which
   `ranx` follows and which the IR literature means by MAP@k. The cross-check caught it
   immediately. We follow the standard and document the consequence.
2. **A sweep was running redundant configurations.** BM25 never looks at a vector, so
   `bm25 + tfidf` and `bm25 + hash` are the same run under two names. Worse than the wasted
   time: the embedder axis effect averaged three identical BM25 scores into the embedder's
   record as though it had earned them. Configurations are now canonicalised before expansion,
   which cut a 27-config factorial to 21 and moved the control embedder's apparent score from
   0.53 to its true 0.33.
3. **Sub-millisecond latency printed as "0 ms"**, which reads as a broken measurement rather
   than a fast one.

---

## M6 — Eval sets ✅ *shipped*

**Build** — LLM question generation (`K1`), the four filters plus the non-discriminating filter
(`K2`–`K5`), graded and multiple gold spans (`K7`, `K8`), question-type tagging (`K10`), import
including LegalBench-RAG format (`K11`), terminal review queue (`K9`), eval-set quality score
(`K12`), versioning (`K17`)

**Test** — each filter has a fixture that must be rejected and one that must survive; import
round-trips; review queue state machine unit-tested without a terminal

**Exit** — 100 questions generated, filtered and hand-reviewed in under 15 minutes. ✅

**What shipped.** Drafting from a corpus, by model or by keyword probe. Six filters, including
the non-discriminating one nothing else has. Heuristic question-type classification. A review
queue built as a plain state machine — accept, reject, edit, mark, undo — so it is testable
without a terminal and the UI on top can be swapped. Eval-set quality scoring, including the
minimum difference a set of that size could detect. Import from JSONL, CSV, BEIR and
LegalBench-RAG.

**One bug, found three times before it was understood.** `EvalItem.is_answerable` means "the
evidence has been *resolved* to character spans in this parse". A freshly drafted set has
quoted evidence and no spans, so every new question read as unanswerable: quality scoring
reported zero answerable questions, the classifier labelled everything `unanswerable`, and the
unresolved-evidence filter rejected the entire set. The fix is a second property,
`has_evidence`, and the discipline of asking which of the two every call site actually meant.

---

## M7 — Credibility ✅ *shipped*

**Build** — bootstrap confidence intervals (`L12`), paired significance testing via `ranx` (`L13`),
"not distinguishable" as a first-class result state (`L14`), character-level precision and recall
(`L8`), chunk attribution rate (`L9`), per-question-type slicing (`L15`), axis-effect view (`N5`),
config diff (`N4`), failure taxonomy classification (`N2`)

**Test** — significance testing validated against known-different and known-identical synthetic
runs; character metrics hand-verified on fixtures

**Exit** — the SDK can say "these two configs are not distinguishable (n=87)" and be right. ✅

**What shipped.** Bootstrap confidence intervals on every configuration. Paired significance
testing by randomisation, which is the right test here because both configurations always run
on identical questions -- each question acts as its own control, removing the large variance
that comes from some questions simply being harder. Character-level precision and recall wired
into the leaderboard. Per-question-type slicing. And the Seven Failure Points taxonomy, applied
automatically, so a score becomes a list of things to try.

`distinguishable` deliberately requires *both* a small p-value and a confidence interval that
stays on one side of zero. Two weak signals agreeing is a better basis for a decision than
either alone.

**One thing the build got wrong first.** The "not distinguishable" verdict stitched a phrase
into a shared template and produced "About many more than questions would be needed". There are
three genuinely different situations -- identical on every question, tied on average while
disagreeing, and a real gap too small for this many questions -- and each needs its own
sentence. Broken grammar in a statistical caveat is worse than no caveat: it makes a reader
stop trusting everything around it.

---

## M8 — Ship it ✅ *shipped*

**Build** — second and third parsers, pdfplumber and Docling (`B1`), the parser→retrieval
attribution report (`B4`), semantic and structural chunkers (`C4`, `C5`), a local cross-encoder
reranker with the candidate-depth sweep (`H2`, `H5`), config and code export (`P3`, `P5`), run
bundle (`P7`), CLI (`P11`), Docker Compose self-host (`P14`), LegalBench-RAG validation suite (`P9`)

**Test** — nightly validation run must reproduce published LegalBench-RAG numbers within tolerance;
CLI smoke tests; a clean-room `pip install context-grid` in a fresh venv

**Exit** — **v0.3, first public release.** README, docs site, PyPI, announcement post.

**Shipped so far (v0.4).** Rerankers as a real axis -- identity, lexical overlap and MMR --
plus `candidates`, the depth parameter most reranking advice omits and where most of the
effect lives. A run manifest that hashes everything which could change a number, and a diff
that names the suspect when one does. Config, Python, JSON and Markdown exports, and a bundle
a sceptic can re-derive every number from. A CLI with `profile`, `sweep`, `evalset`, `plugins`
and `diff`.

Two canonicalisation gaps the build found: `reranker="none"` is the identity and therefore the
same run as no reranker at all, and candidate depth means nothing without something to rerank.
Left alone, both would run identical configurations under different names and credit an axis
with differences it did not cause.

**Also shipped.** Semantic chunking, cutting on a percentile of *this document's own*
similarity drops rather than an absolute threshold -- 0.7 is a big drop for one embedding
model and noise for another, so a fixed value would silently mean something different on
every arm of a sweep. It comes with a similarity profile that says when a document was too
flat for the strategy to have anything to work with, which is worth knowing before believing
its score.

The LegalBench-RAG validation harness, which checks the corpus matches the annotations before
scoring anything, then compares our numbers with the published ones. And a Docker self-host
path, since the default install runs locally and talks to nothing.

**Still open.** Real embedding models and cross-encoders via ONNX -- registered lazily, so
asking for one today names the extra to install rather than failing obscurely.

---

## M9 — The blog UI

**Build** — FastAPI wrapper over the SDK; Next.js `/lab` on the existing site; demo mode with
precomputed runs (`Q12`); parser diff view (`B3`, `Q8`); chunk boundaries drawn on the page
(`C19`, `N7`); per-query inspector (`Q6`); permalinks with OG images (`P1`, `P2`); recommendation
card (`N11`); methodology and validation pages (`P8`, `P9`)

**Exit** — **v0.4.** A visitor reaches a real insight in under ten seconds, with no keys.

---

## M10 — Generation

Generator axis (`J1`), prompt template axis (`J2`), faithfulness, relevance and citation accuracy
(`J3`–`J5`), abstention (`J6`), the retrieval→generation lift chart (`J9`), context assembly:
ordering, budget, compression (`I1`–`I3`, `I8`).

**Exit** — the SDK answers "does better retrieval actually produce a better answer?"

---

## M11 — The wide field

Quantization axis and its cost/recall frontier (`E10`–`E14`) · hosted embedders, rerankers and
parsers via BYOK (`B6`, `D3`, `H3`) · query transforms (`F2`–`F6`) · advanced chunkers: contextual,
late, proposition, code/AST (`C7`, `C9`–`C11`) · GraphRAG as an index type (`E22`, `G11`, `G12`) ·
agentic and corrective retrieval (`G8`–`G10`) · ColPali page-image retrieval (`B12`, `G13`) ·
query-side adapters with mined hard negatives (`D14`, `D15`) · semantic cache simulation
(`M10`, `M11`) · poisoning robustness (`O4`, `O5`).

Each is independent. Order by what makes the best post.

---

## M12 — Distribution

CI regression gate with manifest diff (`P12`) · GitHub Action (`P13`) · community leaderboard
(`P15`) · spin out `span-eval` as a standalone package (`K19`).

---

## Cross-cutting standards

**Every PR:** `ruff`, `mypy --strict`, `pytest`, conformance suites, coverage gate (90% on `core/`
and `score/`, 75% elsewhere). No network in unit tests — anything calling a model or API is marked
`integration`.

**Every plugin added:** registered by name, declares its extra, passes its conformance suite,
declares `offsets_exact` honestly, reports its own cost and latency, and has a golden-file test.

**Every metric added:** a hand-computed fixture proving it, and a docstring stating what it does
*not* tell you.

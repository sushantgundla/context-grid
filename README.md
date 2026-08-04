# context-grid

A lab for grounding pipelines. Sweep **parser × chunker × embedder × index × reranker** on your
own documents and get back ranked, reproducible results scored on quality, latency and cost.

> **Status: alpha (v0.2).** Sweeps run end to end, and eval sets can be drafted, filtered and
> reviewed. Real embedding models, rerankers, significance testing and the CLI are next —
> see [docs/roadmap.md](docs/roadmap.md).

---

## Why

Most advice about retrieval is anecdote. *Semantic chunking is better. Use a reranker. 512 tokens
with 50 overlap is the sweet spot.* Almost nobody publishes the measurement behind it, and almost
nobody can reproduce someone else's setup on their own documents.

This turns those opinions into numbers on **your** corpus.

```python
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

## The idea that makes it work

Ground truth in retrieval evaluation is normally stored as a **chunk ID**. That works right up
until you change the chunker — which is the entire point of a tool like this. A gold chunk ID
recorded under a 512-token splitter means nothing under a semantic splitter, because the second
one produced different chunks. Comparisons built that way are invalid, and nothing warns you.

So here, ground truth is a **character span in the source document**:

```
Document text ─────────────────────────────────────────────────────
                        ╔═══════════════╗
gold span               ║ chars 840–1010 ║
                        ╚═══════════════╝
chunker A     [ 0–500 ][ 500–1000 ][ 1000–1500 ]   → gold straddles two chunks
chunker B     [ 0–800 ][ 800–1600 ]                → gold sits inside one
```

Gold is resolved to whichever chunks a configuration happened to produce, at scoring time. The
eval set is written once and stays correct across every configuration it ever scores.

### Why not IoU

The obvious way to resolve a span to a chunk is intersection-over-union. It builds a bias straight
into the measuring instrument.

Take a 170-character gold span. A 2000-character chunk containing every character of it scores an
IoU of **0.085** and is called a miss at any sensible threshold. A 250-character chunk containing
the same evidence scores **0.68** and passes. IoU systematically penalises large-chunk
configurations for being large — and chunk size is one of the axes under test.

The default policy is therefore **coverage**: what fraction of the gold span's characters does
this chunk hold? It asks the question that actually matters — *is the evidence there?* IoU stays
available for when punishing chunk bloat is the point.

## Install

```bash
pip install context-grid
```

The core is dependency-free. Everything heavy lives behind an extra, so nothing drags in CUDA:

```bash
pip install "context-grid[parse]"   # pymupdf, pdfplumber
pip install "context-grid[chunk]"   # chonkie
pip install "context-grid[embed]"   # onnx runtime, local models
pip install "context-grid[all]"     # everything
```

A missing extra raises an error naming the exact install command, never an `ImportError`
traceback.

## What's built

| | |
|---|---|
| ✅ | Span algebra, documents, chunks, gold spans, eval sets |
| ✅ | Structured warnings channel — truncation, approximate offsets, ANN recall loss |
| ✅ | Span → chunk resolution with coverage / IoU / containment policies |
| ✅ | Split-gold detection and union coverage |
| ✅ | Character-level precision, recall and F1 |
| ✅ | Portable ground truth: gold anchors that survive a change of parser |
| ✅ | Plugin protocols, registry, lazy extras, spec strings (`recursive:512,overlap=64`) |
| ✅ | Conformance suites, with proof they catch broken plugins |
| ✅ | Parsers: plain text, Markdown (no dependencies) |
| ✅ | Chunkers: fixed-token, recursive, sentence-window, structural |
| ✅ | Tokenizers: regex and character, each declaring whether it is exact |
| ✅ | PDF parsers: PyMuPDF and pdfplumber, with exact offsets and heading inference |
| ✅ | Corpus loading, and a fingerprint that suggests which axes will matter |
| ✅ | The anchor resolver — one eval set, re-resolved against every parser |
| ✅ | Embedders: TF-IDF, hashing, and a chance-level control. No downloads |
| ✅ | Indexes: exact dense, BM25, hybrid with RRF or weighted fusion |
| ✅ | Metrics — recall, precision, nDCG, MRR, MAP, hit rate — **verified against `ranx`** |
| ✅ | The grid: factorial, one-factor-at-a-time and staged sweeps |
| ✅ | Content-addressed caching with prefix reuse across configurations |
| ✅ | Cost and latency: per-stage timings, p50/p95/p99, dollars per 1k queries |
| ✅ | Leaderboard, Pareto frontier, axis effects, config diff, plain-English summary |
| ✅ | Eval sets: draft from a corpus, filter, classify, review, score their own power |
| ✅ | Import from JSONL, CSV, BEIR and LegalBench-RAG |
| ✅ | Bootstrap confidence intervals and paired significance testing |
| ✅ | Character-level precision, per-question-type slicing, failure diagnosis |
| ✅ | Rerankers, and the candidate-depth axis most advice omits |
| ✅ | Run manifest, config/code/report exports, result bundles |
| ✅ | A CLI: `contextgrid profile`, `sweep`, `evalset`, `plugins`, `diff` |
| ✅ | Semantic chunking, on a percentile of the document's own similarity drops |
| ✅ | LegalBench-RAG validation harness — `contextgrid validate` |
| ✅ | Docker self-host: documents never leave the machine |
| ✅ | Context assembly: ordering, token budget, deduplication |
| ✅ | Generation panel: groundedness, citation accuracy, abstention, the lift chart |
| ⬜ | Real embedding models and cross-encoders via ONNX |
| ✅ | Quantization: scalar, product and binary, with the rescoring pass that makes them work |
| ⬜ | GraphRAG, agentic retrieval, query transforms |

Full plan: [docs/roadmap.md](docs/roadmap.md) · Design: [docs/design.md](docs/design.md)

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                      # unit + property tests
pytest -m integration       # anything touching a model or the network
mypy && ruff check .

./scripts/check-oldest-numpy.sh   # type-check against the oldest toolchain CI uses
```

That last one exists because `python_version` in the mypy config does not change which numpy
is installed. A bare `np.ndarray` annotation type-checks fine under a recent numpy and fails
on the Python 3.10 runner under an older one — so the rule is to always write `Vectors` or
`npt.NDArray[...]`, never a bare `np.ndarray`, and the script catches it when the rule slips.

Unit tests never touch the network. Three things get more than ordinary care:

- **The span algebra and the resolver** are covered by Hypothesis property tests, because a bug
  there would be invisible in every number the tool prints.
- **Every plugin** must pass a conformance suite for its family — and the suites themselves are
  tested against six deliberately broken plugins, to prove they can fail.
- **Every metric** is checked against [`ranx`](https://github.com/AmenRa/ranx), the peer-reviewed
  reference implementation, on randomly generated judgements and runs. The core installs with
  numpy and nothing else, so the metrics are implemented here rather than delegated — and
  "our numbers match ranx on a thousand random cases" is a stronger claim than "we used ranx".

### Did the retrieval gain reach the answer?

Retrieval stays the default view, because generation noise swamps retrieval signal. But
retrieval is a means, and a tool that never checks whether its gains survive is asking to be
trusted about the one thing it did not measure.

```python
print(cg.lift(retrieval_score=0.80, answer_score=0.70, baseline_answer=0.70))
# Retrieval scored 0.800, and answer quality is unchanged against the baseline.
# The generator was finding the answer either way, so this retrieval gain bought nothing.
```

The generation panel scores three things without needing a second model to judge:
groundedness (is the answer in the context, or invented?), citation accuracy, and
**abstention** — when the evidence is absent, does the system say so instead of guessing?
That last one is almost never measured, and a system that confidently answers questions its
corpus cannot support is worse than one that scores lower and declines.

## Checking it rather than trusting it

Every number here depends on the span resolver, and the resolver is the one part with no
external reference to check against — nobody else stores ground truth as character spans.

Except [LegalBench-RAG](https://arxiv.org/abs/2408.10343), which does exactly that. So:

```bash
contextgrid validate ./legalbench-rag.json ./corpus --recall-at-10 0.72
```

It checks first that the gold spans point at real text in the documents as loaded — a
mismatch there is a loading problem and invalidates everything after it — then scores the
benchmark with a deliberately plain configuration and compares against the published number.
The point is to check the *scorer*, not to win the benchmark.

## Self-hosting

Everything in the default install runs locally and talks to nothing, which is the honest
answer to "can I run this on documents I cannot upload anywhere?".

```bash
docker compose run --rm contextgrid sweep /data/documents /data/evalset.jsonl --bundle /data/results
```

## Licence

MIT

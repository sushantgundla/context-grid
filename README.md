# context-grid

A lab for grounding pipelines. Sweep **parser × chunker × embedder × index × reranker** on your
own documents and get back ranked, reproducible results scored on quality, latency and cost.

> **Status: pre-alpha.** The scoring core is being built first. There is no working pipeline yet —
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
| ⬜ | PDF parsers, embedders, indexes, rerankers |
| ⬜ | The grid runner, caching, cost model |
| ⬜ | Eval-set generation and review |
| ⬜ | ranx metrics, significance testing |

Full plan: [docs/roadmap.md](docs/roadmap.md) · Design: [docs/design.md](docs/design.md)

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                      # unit + property tests
pytest -m integration       # anything touching a model or the network
mypy && ruff check .
```

Unit tests never touch the network. The span algebra and the resolver are covered by Hypothesis
property tests, because a bug there would be invisible in every number the tool prints.

## Licence

MIT

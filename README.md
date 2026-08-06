# context-grid

A lab for grounding pipelines. Write one YAML file, sweep **ingestion × parser × chunker ×
embedder × index × transform × retrieval × reranker × candidates** on your own documents, and
get back ranked, reproducible results scored on quality, latency and cost.

> **Status: v0.9.0, alpha.** All nine axes are shipped and real — see
> [what the numbers actually say](#what-the-numbers-actually-say) below. RAPTOR and GraphRAG
> are deliberately not built yet (see [docs/roadmap.md](docs/roadmap.md)). Not yet on PyPI —
> install from source, below.

---

## Why

Most advice about retrieval is anecdote. *Semantic chunking is better. Use a reranker. 512
tokens with 50 overlap is the sweet spot.* Almost nobody publishes the measurement behind it,
and almost nobody can reproduce someone else's setup on their own documents.

This turns those opinions into numbers on **your** corpus — one config file, not a script per
question.

## Quickstart

Not yet on PyPI. Install from source:

```bash
git clone https://github.com/sushantgundla/context-grid
cd context-grid
pip install -e .
```

The core installs with just `numpy` and `pyyaml` — nothing that drags in CUDA. Real parsers,
chunkers, indexes and hosted models live behind extras; see [Install extras](#install-extras)
below.

Point it at a folder of documents and a JSONL of questions, then let `contextgrid init` write
the config:

```bash
contextgrid init contextgrid.yaml --corpus ./documents --evalset ./questions.jsonl
contextgrid check contextgrid.yaml   # validate before running anything
contextgrid run contextgrid.yaml
```

```
contextgrid: 1 × 1 × 2 × 2 × 3 × 1 × 1 × 2 × 1 = 24 on paper, 5 to run in ofat mode (1 impossible combination(s) skipped), scored on recall@5

configuration                                         recall@5   p95 ms     $/1k
---------------------------------------------------------------------------------
markdown · recursive:512 · tfidf · dense                 1.000      0.4   0.0000
markdown · sentence:3 · tfidf · dense                     1.000      0.0   0.0000
markdown · recursive:512 · bm25                           1.000      0.0   0.0000
markdown · recursive:512 · tfidf · hybrid                 1.000      0.0   0.0000
markdown · recursive:512 · tfidf · dense · lexical@50     1.000      0.0   0.0000

markdown · recursive:512 · tfidf · dense scored best on recall@5 at 1.000, across 5 configurations on 3 questions. [...]

wrote 6 files to /you/are/here/results
```

That's the whole loop: `init` writes a config listing only the plugins your install can
actually run, `check` catches a typo'd axis or a missing path before anything expensive
starts, `run` sweeps and writes a leaderboard, a manifest, and a re-runnable copy of the
winning config. Full walkthrough: [docs/guide/getting-started.md](docs/guide/getting-started.md).

### The library, for people who want it in code

```python
import contextgrid as cg

lab = cg.Lab(corpus="./documents")
lab.grid(
    chunker=["recursive:256", "chonkie:recursive:256", "langchain:recursive:256"],
    index=["dense", "bm25", "hybrid"],
    reranker=[None, "lexical"],
)

evalset = cg.read_jsonl("questions.jsonl")
results = lab.run(evalset, headline="recall@3")
print(results.summary("recall@3"))
```

`lab.grid()` takes all ten axes, the same ones a config file does. The strategies that call a
model — `hyde`, `agentic` retrieval, `contextual` ingestion, an `llm` generator — need one
naming it, and `budget_usd` is worth setting whenever a sweep contains one:

```python no-run: needs an API key; the offline equivalent is in docs/recipes/
lab = cg.Lab(corpus="./documents", model="openai:gpt-4o-mini", seed=7)
lab.grid(
    ingestion=["plain", "parent-document:4", "contextual"],
    retrieval=["simple", "decomposed", "agentic"],
    transform=[None, "hyde"],
)
results = lab.run(evalset, budget_usd=5.00)
```

```
markdown · recursive:256 · tfidf · dense scored best on recall@3 at 1.000, across 6 configurations on 3 questions. [...] It runs locally at no cost per query, answering at under 1 ms p95.
```

Real embedding models and rerankers use the same shape, once a server or a key is available —
`embedder: tei:bge-base-en-v1.5` for a local
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) server,
`embedder: litellm:text-embedding-3-small` for any hosted provider through
[litellm](https://docs.litellm.ai/), `reranker: tei-rerank:bge-reranker-base` for a local
cross-encoder. Same call, real HTTP, no code change:

```python
lab.grid(embedder=["tei:bge-base-en-v1.5,api_base=http://127.0.0.1:8080"])
results = lab.run(evalset, headline="recall@3")
```

```
markdown · recursive:512 · tei:bge-base-en-v1.5,api_base=http://127.0.0.1:8080 · dense scored best on recall@3 at 1.000, across 1 configurations on 3 questions. It runs locally at no cost per query, answering at under 1 ms p95.
```

(Run against a stand-in server speaking TEI's real `/embed` wire protocol, so the code path —
prefixing, batching, the actual network call — is genuine; only the weights behind port 8080
are fake. Point `api_base` at a real server and nothing else changes.) More in
[docs/dimensions/embedders.md](docs/dimensions/embedders.md) and
[docs/recipes/local-only.md](docs/recipes/local-only.md).

## The nine axes

Every axis takes one value or a list — a list sweeps it, a single value holds it still —
and a combination that can't run (a dense index with no embedder, say) is counted as
impossible and skipped rather than attempted.

| Axis | Config key | What it decides | Docs |
|---|---|---|---|
| Ingestion | `ingestion` | What's indexed versus what's returned — `plain` makes them the same chunk; `parent-document`, `sentence-window` and five others deliberately don't | [dimensions/ingestion.md](docs/dimensions/ingestion.md) |
| Parser | `parser` | What reads the document — the axis nothing else in the field measures | [dimensions/parsers.md](docs/dimensions/parsers.md) |
| Chunker | `chunker` | How the text is cut up | [dimensions/chunkers.md](docs/dimensions/chunkers.md) |
| Embedder | `embedder` | What turns text into vectors, or `null` for none (BM25) | [dimensions/embedders.md](docs/dimensions/embedders.md) |
| Index | `index` | How the search is done — exact, approximate, sparse, hybrid | [dimensions/indexes.md](docs/dimensions/indexes.md) |
| Transform | `transform` | Rewriting the question before searching with it | [dimensions/transforms.md](docs/dimensions/transforms.md) |
| Retrieval | `retrieval` | How the index is used — one search, split, widened, or agentic | [dimensions/retrieval.md](docs/dimensions/retrieval.md) |
| Reranker | `reranker` | Reordering what came back | [dimensions/rerankers.md](docs/dimensions/rerankers.md) |
| Candidates | `candidates` | How deep the reranker looks before cutting to `k` — most of a reranker's effect lives here, not in which one you pick | [dimensions/rerankers.md](docs/dimensions/rerankers.md) |

Generation — scoring whether a retrieval gain actually reached the answer — sits downstream of
the grid rather than on it; see [dimensions/generation.md](docs/dimensions/generation.md).

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
eval set is written once and stays correct across every configuration it ever scores — including
across a change of *parser*, if the gold is written as a quoted anchor rather than a raw span.
See [docs/scoring/spans-and-anchors.md](docs/scoring/spans-and-anchors.md) for the two forms and
why both exist.

### Why not IoU

The obvious way to resolve a span to a chunk is intersection-over-union. It builds a bias
straight into the measuring instrument.

Take a 170-character gold span. A 2000-character chunk containing every character of it scores
an IoU of **0.085** and is called a miss at any sensible threshold. A 250-character chunk
containing the same evidence scores **0.68** and passes. IoU systematically penalises
large-chunk configurations for being large — and chunk size is one of the axes under test.

The default policy is therefore **coverage**: what fraction of the gold span's characters does
this chunk hold? It asks the question that actually matters — *is the evidence there?* IoU
stays available for when punishing chunk bloat is the point.

## What the numbers actually say

Real findings from building the nine axes out, recorded in full in
[docs/adoption-backlog.md](docs/adoption-backlog.md):

- **`parent-document` scored 0.863 against plain chunking's 0.616** on the demo corpus at
  96-token chunks — a +0.247 gain for zero model calls, just by indexing a small chunk and
  returning the passage it came from. `hierarchical`, the strategy that sounded most
  sophisticated, scored 0.575 — *below* plain — because merging spends result slots on wider
  passages.
- **LangChain's chunk offsets are not exact.** It reports `start_index: -1` for roughly one
  chunk in eight on our fixtures — always tables — because it rebuilds each chunk by rejoining
  the pieces it split and then can't find the result in the source text. The adapter locates
  and verifies the real offset itself rather than trusting the -1.
- **pgvector's session parameters were being silently ignored.** `SET LOCAL probes = N` lasts
  until the end of the current transaction, and on an autocommit connection there is none —
  Postgres accepted the setting, did nothing, and every query ran at the default `probes = 1`
  while the sweep reported numbers for whatever depth was actually asked for. Recall went
  0.26 → 1.00 across the probe range once the session setting actually took; before the fix, it
  was 0.26 at every setting. Only a live database caught this — there is no in-process fake for
  it, on purpose.

## Install extras

```bash
pip install "context-grid[parse]"      # pymupdf, pdfplumber, pymupdf4llm
pip install "context-grid[parse-ml]"   # docling, marker — layout models, heavy
pip install "context-grid[chunk]"      # chonkie, langchain-text-splitters
pip install "context-grid[llm]"        # litellm — hosted embedders, rerankers, transforms, judge
pip install "context-grid[index]"      # faiss-cpu, usearch
pip install "context-grid[pgvector]"   # psycopg — needs a running Postgres
pip install "context-grid[agent]"      # agno — the agno parser and agentic retrieval
pip install "context-grid[judge]"      # deepeval — faithfulness, relevancy, the generation metrics
```

There is no `[embed]` and no `[all]` extra, despite what the axis name and old habit might
suggest — the `litellm` embedder is under `[llm]`, and `tei`/`tei-rerank` need no extra at all
(they talk plain HTTP to a server you run separately). A missing extra raises an error naming
the exact install command, never a bare `ImportError`. Full matrix, sizes, and what each one
unlocks: [docs/reference/install.md](docs/reference/install.md).

## Checking it rather than trusting it

Every number here depends on the span resolver, and the resolver is the one part with no
external reference to check against — nobody else stores ground truth as character spans.

Except [LegalBench-RAG](https://arxiv.org/abs/2408.10343), which does exactly that. So:

```bash
contextgrid validate ./legalbench-rag.json ./corpus --recall-at-10 0.72
```

It checks first that the gold spans point at real text in the documents as loaded — a mismatch
there is a loading problem and invalidates everything after it — then scores the benchmark with
a deliberately plain configuration and compares against the published number. The point is to
check the *scorer*, not to win the benchmark.

## Self-hosting

Everything in the default install runs locally and talks to nothing, which is the honest
answer to "can I run this on documents I cannot upload anywhere?".

```bash
docker compose run --rm contextgrid sweep /data/documents /data/evalset.jsonl --bundle /data/results
```

## Documentation

- [docs/guide/](docs/guide/) — install, your first sweep, the full config reference, every CLI
  command, and how to write an eval set
- [docs/dimensions/](docs/dimensions/) — the nine axes, what each arm actually does, and how to
  write a spec string
- [docs/scoring/](docs/scoring/) — spans and anchors, metrics, the significance tests, the
  0–100 composite, failure diagnosis
- [docs/reference/](docs/reference/) — install sizes, plugin catalogue, caching, cost model,
  report formats
- [docs/internals/](docs/internals/) — architecture, plugin protocols, the registry, the
  conformance suites, how to add a plugin
- [docs/recipes/](docs/recipes/) — worked, real-output examples: choosing a chunker, choosing
  an embedder with or without an eval set, whether agentic retrieval is worth its cost,
  reproducing a run from its manifest
- [docs/roadmap.md](docs/roadmap.md) and [docs/adoption-backlog.md](docs/adoption-backlog.md) —
  where this is going, and the research behind every library adopted so far

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
```

```
Retrieval scored 0.800, and answer quality is unchanged against the baseline. The generator was finding the answer either way, so this retrieval gain bought nothing.
```

The generation panel scores three things without needing a second model to judge:
groundedness (is the answer in the context, or invented?), citation accuracy, and
**abstention** — when the evidence is absent, does the system say so instead of guessing?
That last one is almost never measured, and a system that confidently answers questions its
corpus cannot support is worse than one that scores lower and declines. With
[`deepeval`](docs/reference/install.md) installed and `run.model` set, faithfulness and
answer-relevancy join it and roll up into the 0–100 composite — see
[docs/scoring/composite.md](docs/scoring/composite.md).

## Licence

MIT

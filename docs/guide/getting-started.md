# Getting started

`context-grid` runs a sweep of retrieval pipeline configurations over your own documents and
tells you which combination actually works. This page gets you from a blank folder to a
leaderboard.

## Install

```bash
pip install context-grid
```

The core installs with just a few pure-Python dependencies — no CUDA, no downloads. Real
embedding models, PDF parsers, and some chunkers live behind extras. These are the ones you
reach for first — the full set is in [reference/install.md](../reference/install.md):

```bash
pip install "context-grid[parse]"    # pymupdf, pdfplumber, pymupdf4llm
pip install "context-grid[chunk]"    # chonkie, langchain-text-splitters
pip install "context-grid[embed]"    # tiktoken, for exact token counts
pip install "context-grid[llm]"      # litellm: hosted models, and their prices
pip install "context-grid[index]"    # faiss, usearch

pip install "context-grid[parse,chunk,index]"   # a useful working set
```

The rest, for when you need them: `parse-ml` (docling) and `parse-marker` (marker-pdf) for
layout-model parsing, `pgvector` (psycopg) for Postgres, `agent` (agno) for agentic retrieval,
`judge` (deepeval) for the generation metrics, and `dev` for the test suite. There is no
`[all]`.

You don't need any extra to follow this page — everything below runs on the bare install.

## What you need

Two things:

1. **A corpus** — a directory of documents (Markdown, text, or PDF).
2. **An eval set** — questions with known answers, so context-grid has something to score
   against. See [writing an eval set](evalsets.md) for the full format; this page uses a
   three-question one.

```
documents/
  refunds.md
  shipping.md
questions.jsonl
```

`questions.jsonl` looks like this — one line per question, each pointing at a quoted piece of
evidence in a source file (a `GoldAnchor`; see [evalsets.md](evalsets.md) for why quoting the
evidence, rather than recording a chunk ID, is what makes the parser and chunker axes
comparable at all):

```jsonl
{"_evalset": {"id": "policy-questions", "version": 1, "source": "manual", "meta": {}}}
{"id": "q1", "question": "How long do refunds take?", "gold": [], "anchors": [{"source_id": "refunds.md", "quote": "within 30 days of purchase", "grade": 2, "page_hint": null, "occurrence": 0}], "qtype": null, "answer": null, "meta": {}}
{"id": "q2", "question": "How fast is express shipping?", "gold": [], "anchors": [{"source_id": "shipping.md", "quote": "arrives the next business day", "grade": 2, "page_hint": null, "occurrence": 0}], "qtype": null, "answer": null, "meta": {}}
{"id": "q3", "question": "Can I return digital goods?", "gold": [], "anchors": [{"source_id": "refunds.md", "quote": "not refundable once downloaded", "grade": 2, "page_hint": null, "occurrence": 0}], "qtype": null, "answer": null, "meta": {}}
```

You'll usually generate this with `contextgrid.evalset.generate` or write it from a
spreadsheet as CSV — see [evalsets.md](evalsets.md).

## Write a config

`contextgrid init` writes a starter config you can run as it stands, and next to every axis it
lists what else you could put there — what your install can already run, and what a `pip
install` would add.

```bash no-run: narrative walkthrough -- ./documents and ./questions.jsonl aren't reconstructed in this snippet
contextgrid init contextgrid.yaml --corpus ./documents --evalset ./questions.jsonl
```

```
wrote contextgrid.yaml
edit it, then run:  contextgrid run contextgrid.yaml
```

Open the file. Every key under `grid:` is an axis of the pipeline — `parser`, `chunker`,
`embedder`, `index`, `reranker`, and so on. A key with a list sweeps that axis; a key with one
value holds it still. The full key reference, with every default, is in
[configuration.md](configuration.md).

Each axis comes with its options written underneath it:

```yaml
  parser: [markdown]
  # also available: agno, docling, pdfplumber, pymupdf, pymupdf4llm, text
  # needs `pip install "context-grid[parse-marker]"`: marker
```

The first line is what this config sweeps. `# also available:` is everything your install can
run today — move a name up onto that line and it just works. The `# needs pip install` line is
what you don't have yet, grouped so one command covers everything on it. **The starting sweep
is deliberately small.** Putting all seven installed parsers on that line would download two
models before you had read the file; widening an axis later is a one-line edit.

Some names need configuration rather than a package, and the file says so on a third comment
line:

```yaml
  generator: [null]
  # also available: extractive, llm
  # of those, llm needs `run.model` set.
```

The same line appears under `transform:` for `hyde` and its friends. Nothing is hidden: if you
can't run something yet, it's listed along with the reason.

## Check it before running it

`contextgrid check` parses the config, resolves its paths, and prints the shape of the sweep —
without running anything. Worth doing before a sweep that might take an hour.

```bash no-run: narrative walkthrough -- continues from the contextgrid init command above
contextgrid check contextgrid.yaml
```

```
contextgrid: 1 × 1 × 2 × 2 × 3 × 1 × 1 × 2 × 1 × 1 = 24 on paper, 5 to run in ofat mode (1 impossible combination(s) skipped), scored on recall@5
  ingestion   ['plain']
  parser      ['markdown']
  chunker     ['recursive:512', 'sentence:3']
  embedder    ['tfidf', None]
  index       ['dense', 'bm25', 'hybrid']
  transform   [None]
  retrieval   ['simple']
  reranker    [None, 'lexical']
  candidates  [50]
  generator   [None]

config is valid.
```

"24 on paper" is every combination the axes describe; "5 to run" is what the sweep mode
(`ofat`, one-factor-at-a-time, by default) actually executes — one baseline plus one change per
axis. "1 impossible combination skipped" is `index: dense` paired with `embedder: null`: a dense
index with no vectors can't run, so it's counted and dropped rather than silently attempted or
silently ignored.

## Run it

```bash no-run: narrative walkthrough -- continues from the contextgrid init command above
contextgrid run contextgrid.yaml
```

```
contextgrid: 1 × 1 × 2 × 2 × 3 × 1 × 1 × 2 × 1 × 1 = 24 on paper, 5 to run in ofat mode (1 impossible combination(s) skipped), scored on recall@5
  [1/5] markdown · recursive:512 · tfidf · dense
  [2/5] markdown · sentence:3 · tfidf · dense
  [3/5] markdown · recursive:512 · bm25
  [4/5] markdown · recursive:512 · tfidf · hybrid
  [5/5] markdown · recursive:512 · tfidf · dense · lexical@50

configuration                                         recall@5   p95 ms     $/1k
---------------------------------------------------------------------------------
markdown · recursive:512 · tfidf · dense                 1.000      0.4   0.0000
markdown · sentence:3 · tfidf · dense                    1.000      0.0   0.0000
markdown · recursive:512 · bm25                          1.000      0.0   0.0000
markdown · recursive:512 · tfidf · hybrid                 1.000      0.0   0.0000
markdown · recursive:512 · tfidf · dense · lexical@50     1.000      0.0   0.0000

markdown · recursive:512 · tfidf · dense scored best on recall@5 at 1.000, across 5 configurations, scored on 3 questions. [...]

cache: 17 of 24 lookups reused (71%), chunk 6/10, embed 3/4, parse 8/10

wrote 6 files to /you/are/here/results
```

(Paths in `report.out` always resolve to absolute — that's the last line's path, not literally
`./results`. The progress lines go to stderr, everything else to stdout; that is why they sit
between the shape and the leaderboard at a terminal, and why they will not once you redirect
one of the two streams. Pass `--quiet` to a `run` you're scripting to suppress them.)

## Reading the leaderboard

- **`configuration`** — one row per pipeline actually run, named by the axis values that make
  it up (parser · chunker · embedder · index · reranker@candidates, skipping any axis held at
  its default).
- **The headline metric column** (`recall@5` here — set by `run.headline` in the config) — what
  the leaderboard is sorted on. This eval set is small (3 questions), so a perfect score on all
  five rows is expected and not yet meaningful — see
  [evalsets.md](evalsets.md#eval-set-quality) for how big a set needs to be before differences
  are trustworthy.
- **`p95 ms`** and **`$/1k`** — latency and cost per 1,000 queries, so a metric win that costs
  10x isn't presented as a free lunch.
- **The prose paragraph below the table** — a plain-English summary, including whether the top
  two rows are statistically distinguishable on this eval set. With 3 questions they usually
  aren't; that's the eval set talking, not the tool being unsure.
- **`cache: ...`** — how much of the sweep reused work from an earlier stage (parsing, chunking,
  embedding) rather than redoing it. High reuse is expected: configurations that share a parser
  share its output, and so on down the pipeline.

## What got written

`report.out` in the config (`./results`, from the starter template — unset means nothing is
written) gets:

| File | What |
|---|---|
| `report.md` | The leaderboard and summary, as Markdown |
| `results.json` | Every run, every metric, machine-readable |
| `manifest.json` | The winning configuration's fingerprint — corpus hash, eval set hash, resolution policy, package version |
| `winning-config.yaml` | The winning configuration alone, as a runnable config |
| `use_winning_config.py` | The winning configuration as a Python snippet |
| `experiment.yaml` | A copy of the config that produced this — a results folder that can't be re-run is a screenshot |

Five of those six describe a winner. If a sweep runs nothing at all — a budget too small for a
single configuration — there is no winner to describe, so you get three files instead:
`experiment.yaml`, `report.md` and `results.json`, and the report says why nothing ran.

The full key-by-key reference for `grid:`, `run:` and `report:` is in
[configuration.md](configuration.md). Every `contextgrid` subcommand, with its flags, is in
[cli.md](cli.md). What makes a good eval set — and the difference between a `GoldAnchor` and a
`GoldSpan` — is in [evalsets.md](evalsets.md).

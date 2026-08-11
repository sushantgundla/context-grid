# context-grid docs

`context-grid` is a Python SDK that runs controlled experiments across a whole
retrieval-augmented-generation pipeline — parsing, chunking, embedding, indexing, retrieval,
reranking — on **your own documents**, and gives back ranked, reproducible, cost-aware results
instead of a blog post's opinion.

The one idea everything else depends on: **every chunk knows the exact character range of the
source document it came from.** Ground truth is stored as `(doc_id, start, end)`, not a chunk
ID, and it's re-resolved against whichever parse and whichever chunker a configuration actually
produced. That's what makes it possible to compare two chunkers, or two parsers, fairly — a
gold chunk ID from one strategy means nothing under another, and a span does not have that
problem. See [spans-and-anchors.md](scoring/spans-and-anchors.md) for the mechanics, or
[design.md](design.md) §4 for why it has to work this way.

## Find your way in

| I want to... | Go to |
|---|---|
| Try it for the first time | [guide/getting-started.md](guide/getting-started.md) |
| Know which chunker/embedder/index to pick on my corpus | [recipes/](recipes/README.md) |
| Look up what a config key or CLI flag does | [guide/configuration.md](guide/configuration.md), [guide/cli.md](guide/cli.md) |
| See every plugin a given axis has, and what extra it needs | [reference/plugins.md](reference/plugins.md), [reference/install.md](reference/install.md) |
| Understand what a metric or a warning actually means | [scoring/](scoring/spans-and-anchors.md) |
| Write and register a new plugin, or understand the internals | [internals/](internals/architecture.md) |
| Know why the tool is built the way it is | [design.md](design.md) |
| Know what's built vs. what's still planned | [roadmap.md](roadmap.md), [adoption-backlog.md](adoption-backlog.md) |

## Guide — using the tool

Task-oriented, in the order you'd actually do things.

| Page | What it covers |
|---|---|
| [guide/getting-started.md](guide/getting-started.md) | From a blank folder to a leaderboard: install, corpus, first sweep. |
| [guide/configuration.md](guide/configuration.md) | Every key the config file (YAML/JSON) accepts. If a key isn't listed there, context-grid rejects it. |
| [guide/cli.md](guide/cli.md) | The nine `contextgrid` subcommands — `run`, `sweep`, `check`, `profile`, `plugins`, `evalset`, `validate`, `diff`, `init`. |
| [guide/evalsets.md](guide/evalsets.md) | Writing ground truth: the two ways it's stored (anchors vs. spans), file formats, and how to build one. |

## Recipes — worked examples

Each one is a real question, run for real, with the config, command and actual pasted output —
not a hypothetical.

| Page | Question it answers |
|---|---|
| [recipes/README.md](recipes/README.md) | Index of every recipe. Start here. |
| [recipes/choose-a-chunker.md](recipes/choose-a-chunker.md) | Which chunker helps on *your* documents, and is the difference big enough to trust? |
| [recipes/choose-an-embedder.md](recipes/choose-an-embedder.md) | Which embedding model, weighed on quality and cost together. |
| [recipes/is-agentic-worth-it.md](recipes/is-agentic-worth-it.md) | Does agentic retrieval beat plain search on your corpus — and is it worth the model calls? |
| [recipes/local-only.md](recipes/local-only.md) | A full sweep — real embeddings, real reranking, real indexes — with no API key. |
| [recipes/reproducing-a-run.md](recipes/reproducing-a-run.md) | Turning "reproducible" from a claim into something checkable. |
| [recipes/without-an-evalset.md](recipes/without-an-evalset.md) | What you can measure before you've written a single eval question. |

## Dimensions — the axes you can sweep

One page per axis of the grid, in pipeline order. [dimensions/README.md](dimensions/README.md)
(**the axis model**) is the one to read first — it explains what an axis is, how a spec string
like `recursive:512` is written and parsed, and the three ways to walk the resulting matrix
(`factorial`, `ofat`, `staged`).

| Page | Axis |
|---|---|
| [dimensions/README.md](dimensions/README.md) | The axis model itself: all ten axes, spec strings, sweep modes, redundant-combination dropping. |
| [dimensions/ingestion.md](dimensions/ingestion.md) | What gets indexed vs. what a hit on it returns — parent-document, hierarchical, contextual, and friends. |
| [dimensions/parsers.md](dimensions/parsers.md) | How raw bytes become text and structure. |
| [dimensions/chunkers.md](dimensions/chunkers.md) | How that text is cut into retrievable units. |
| [dimensions/embedders.md](dimensions/embedders.md) | How a chunk becomes a vector — and the two ways getting it wrong stays invisible. |
| [dimensions/indexes.md](dimensions/indexes.md) | The store: where vectors or text live, and how one search runs. |
| [dimensions/transforms.md](dimensions/transforms.md) | Rewriting the question before it's searched with — HyDE, multi-query, decomposition. |
| [dimensions/retrieval.md](dimensions/retrieval.md) | What sits on top of the store: how many searches, who decides, whether one search's answer changes the next. |
| [dimensions/rerankers.md](dimensions/rerankers.md) | Reordering candidates using the query and passage together. |
| [dimensions/generation.md](dimensions/generation.md) | Whether the answer is any good, and whether a retrieval gain survived to it. |

## Scoring — what the numbers mean

| Page | What it covers |
|---|---|
| [scoring/spans-and-anchors.md](scoring/spans-and-anchors.md) | The character-span ground truth model this whole tool rests on: `Span`, `GoldAnchor` vs. `GoldSpan`, the three resolution policies. |
| [scoring/metrics.md](scoring/metrics.md) | The six retrieval metrics (recall, precision, nDCG, MRR, MAP, hit-rate) plus character-level recall/precision, cross-checked against `ranx`. |
| [scoring/composite.md](scoring/composite.md) | Rolling many metrics into one trustworthy 0–100 number — and why it's a harmonic mean, not an arithmetic one. |
| [scoring/significance.md](scoring/significance.md) | Why a leaderboard gap needs a significance test before anyone trusts it. |
| [scoring/diagnostics.md](scoring/diagnostics.md) | Warnings (what went wrong producing a result) and the Seven-Failure-Points taxonomy (what went wrong retrieving an answer). |

## Reference — look-up pages

| Page | What it covers |
|---|---|
| [reference/plugins.md](reference/plugins.md) | Every plugin registered on every axis, generated from the real registries. |
| [reference/install.md](reference/install.md) | The extras matrix — what `pip install "context-grid[...]"` you need for what. |
| [reference/caching.md](reference/caching.md) | Content-addressed caching and prefix reuse — why a sweep of 48 configurations doesn't mean 48 parses. |
| [reference/cost.md](reference/cost.md) | Where a dollar figure comes from, local vs. hosted, and the budget ceiling. |
| [reference/reports.md](reference/reports.md) | What a finished sweep produces: leaderboard, manifest, the four export formats, `contextgrid diff`. |

## Internals — extending and reviewing the codebase

For someone writing a new plugin or reviewing how the package is built, not using it day to day.

| Page | What it covers |
|---|---|
| [internals/architecture.md](internals/architecture.md) | How one configuration flows end to end, naming the real function at every stage. |
| [internals/protocols.md](internals/protocols.md) | The `Protocol` every plugin family implements, and what each method has to guarantee. |
| [internals/registry.md](internals/registry.md) | How plugins are named, registered lazily, and resolved from spec strings like `chonkie:recursive:512`. |
| [internals/extending.md](internals/extending.md) | A worked example: writing, registering and testing a new chunker and a new retrieval strategy, real output pasted in. |
| [internals/conformance.md](internals/conformance.md) | The conformance suites every parser and chunker is run through, and proof they actually catch bugs. |

## Where this came from, and what's next

| Page | What it is |
|---|---|
| [design.md](design.md) | The engineering design: the offset invariant, package layout, caching, dependency strategy, testing strategy. Read this for *why*, not *how*. |
| [roadmap.md](roadmap.md) | The build roadmap — what got re-prioritised out of the original feature catalogue and why. |
| [adoption-backlog.md](adoption-backlog.md) | The live record of which dimension is hand-written vs. backed by a real library, which library was adopted, and why the alternatives weren't. Check here before assuming a dimension is still hand-rolled. |
| [prd/README.md](prd/README.md) | Where the product idea came from — "RAG Retrieval Lab" was the working name. The original brief, the rewrite after a competitive survey, and the ~215-item feature catalogue (`A1`, `K6`, `L13`…) that roadmap and design docs point at by ID. |

## A note on how these were written

This documentation set was written by several agents working in parallel, each owning one
folder. Pages cross-link freely — if a link looks wrong, it's worth a second check against the
file it names before assuming the target text is right; filenames are what's settled, not the
prose describing them.

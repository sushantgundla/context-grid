# Docs-driven drive #3 — the Python library, the recipes, the reference pages

A third pass over `context-grid`, driven only by its public documentation. No source was read
(`src/` and `tests/` untouched), and the two earlier reports were not opened.

**Lane:** the Python library (`cg.Lab`, `Runner`/`matrix`, `cg.lift`, `contextgrid.embed`,
`contextgrid.score`, `contextgrid.report`), all seven `docs/recipes/` pages, the ten
`docs/dimensions/` pages, `docs/scoring/`, and `docs/reference/`. The CLI surface
(`init/check/run/profile/sweep/diff/validate`) was another agent's lane and was only touched
where a reference page made a claim that needed a real bundle on disk to check.

**Environment:** `.venv` at the repo root, Python 3.13.5, macOS. Corpus copied to `/tmp/drive3`
from `.claude/skills/docs-e2e-drive/data` (8 help-centre documents, 13 questions).

---

## How much was actually run

Every fenced `python` / `pycon` block in `docs/` was executed in document order, per file, with
globals carried forward the way a reader copy-pasting down the page would have them. Blocks
tagged `no-run:` were skipped, as the tag asks.

| | Count |
|---|---|
| `python` blocks executed | **149** |
| `python` blocks skipped (`no-run:`) | 20 |
| Blocks that raised, or whose `>>>` output did not match | **0** |
| `bash` blocks executed verbatim | 11 |

That is a genuinely good result and the headline of this report: **every runnable snippet in
`docs/` runs, and every `>>>` block prints what the page says it prints.** The findings below
are all in prose claims, pasted output, and one real reproducibility bug — not in the snippets.

---

## Findings

### 1. `hash` embedder is not reproducible across processes — `seed=` does not fix it

**Severity: wrong (high — it breaks the tool's headline promise)**

**What the docs say:**

- `README.md:5` — "get back ranked, **reproducible** results scored on quality, latency and cost."
- `docs/dimensions/embedders.md:56` — "Parameters: `dimensions: int = 256`, **`seed: int = 0`**."
- `docs/recipes/reproducing-a-run.md:209-213` — "A few axis values carry their *own* seed as a
  spec parameter — **`hash:512,seed=3` (the hashing salt)** ... and neither currently reads
  `run.seed` at all; **each defaults to `0` independently**."

Read together, those say the hashing embedder is deterministic for a given spec string.

**What actually happened.** Same corpus, same spec, three processes:

```
$ for s in 0 1 2; do PYTHONHASHSEED=$s .venv/bin/python -c "...matrix(embedder=['hash:512'], index='dense', k=5)..."; done
recall@5: 0.8904
recall@5: 0.8904
recall@5: 0.8562
```

The vectors themselves differ, and the explicitly-seeded form differs too:

```
$ for s in 0 1; do PYTHONHASHSEED=$s .venv/bin/python -c "...checksum of hash:512 and hash:512,seed=3 vectors..."; done
hash:512         498636eb36b098a7
hash:512,seed=3  ba12984e9ebbd0a2
hash:512         9f54d71b6f673029      <- different process, different vectors
hash:512,seed=3  1227d41fc65b4dbb      <- seed=3 does not pin it either
```

The embedder is reading Python's built-in `hash()`, which is salted per process. `seed=` changes
the output but does not stabilise it.

**Why this matters beyond the number.** `docs/recipes/choose-an-embedder.md` warns about exactly
this failure mode two sections later, at line 136: *"The seed has to come from a stable hash of
the text, not Python's built-in `hash()` — that's randomised per process, so a 'deterministic'
stand-in built on it would print a different number on every run."* The recipe's stand-in gets it
right; the shipped `hash` embedder does not.

**Knock-on:** `docs/recipes/choose-an-embedder.md:180` pastes
`hash:512 -> coherence +0.386, anisotropy 0.228, 13/512 effective dimensions`. Four runs on the
identical corpus gave `+0.386 / 0.228 / 13`, `+0.378 / 0.229 / 14`, `+0.382 / 0.221 / 14`,
`+0.385 / 0.220 / 14`. `tfidf` on the same line reproduces exactly every time. This is not
corpus drift — it is the bug above.

---

### 2. `contextgrid plugins` prints 6 of the 12 documented plugin families

**Severity: wrong**

**What the docs say:**

- `docs/reference/plugins.md:5` — "`contextgrid plugins` prints **the same list** for whatever's
  installed in your own environment."
- `docs/reference/plugins.md:196-199` — "`contextgrid.transform.available_transforms()` is the
  complete list of six, registry plus model-backed, and **is what `contextgrid plugins` and the
  config template actually print**."

**What actually happened:**

```
$ contextgrid plugins
parsers: ... chunkers: ... embedders: ... indexes: ... rerankers: ... tokenizers: ...
$ echo $?
0
```

Six families. `ingestion`, `transform`, `retrieval`, `generator`, `metric` and the LLM registry —
all of which have their own tables in `plugins.md` — are absent entirely.

```
$ contextgrid plugins --family transform
error: unknown plugin family 'transform'. Valid families: parser, chunker, embedder, index, reranker, tokenizer
$ echo $?
1
```

(Exit code checked bare, not through a pipe.)

The config template is fine — `contextgrid init` does write
`# also available: decompose, expand, hyde, multi-query, none, step-back` under `transform:`. It
is only the `plugins` command that disagrees with the page.

---

### 3. `report.md`'s title drops the config's `name:`

**Severity: wrong**

**What the doc says** — `docs/reference/reports.md:182-185`:

> The title carries the experiment's name — `name:` in an experiment config, passed as
> `results_to_markdown(..., name=...)` or left on `results.meta["name"]`. Without one it falls
> back to `# Retrieval configuration comparison`, which is fine for a single sweep and useless
> for a directory of them, where every file would otherwise have the same title.

**What actually happened.** Config with `name: my-experiment`:

```
$ head -1 results-name/report.md
# Retrieval configuration comparison
```

The name is not lost from the bundle — `winning-config.yaml` written by the same run carries
`name: "my-experiment"` on line 10. It just never reaches the report title. `results.json` has no
`name` or `meta` key either (`['mode', 'cache', 'warnings', 'runs', 'manifest']`).

This is the exact "useless for a directory of them" case the paragraph describes.

---

### 4. Two reference pages warn about a stale `marker` error message that is no longer stale

**Severity: wrong**

**What the docs say:**

- `docs/reference/plugins.md:85-89` — "**A stale error message from before that split:** ...
  `MarkerParser`'s own import failure, in `src/contextgrid/parse/layout.py` (line 273), **still
  raises `MissingExtraError("The marker parser", "parse-ml", package="marker-pdf")`** — the wrong
  extra ... If you see that message, install `context-grid[parse-marker]`, not `[parse-ml]`."
- `docs/reference/install.md:50-57` — the same warning again, ending "Worth fixing upstream".

**What actually happened:**

```
MissingExtraError | The marker parser requires the 'parse-marker' extra (needs marker-pdf). Install it with: pip install "context-grid[parse-marker]"
```

The message names `parse-marker` and the correct install command. Both pages now document a bug
that has been fixed, which sends a reader looking for a problem that isn't there.

---

### 5. `README.md` still puts `marker` inside the `[parse-ml]` extra

**Severity: wrong**

**What the doc says** — `README.md:214`:

```bash
pip install "context-grid[parse-ml]"   # docling, marker — layout models, heavy
```

`README.md` never mentions `parse-marker` at all.

**What actually happened.** `pyproject.toml`'s optional-dependency keys are:

```
['parse', 'parse-ml', 'parse-marker', 'embed', 'chunk', 'llm', 'index', 'pgvector', 'judge', 'agent', 'dev']
```

`parse-ml` is `docling` alone; `marker-pdf` lives in `parse-marker`. `docs/reference/install.md`
and `docs/reference/plugins.md` both say so at length, and both explain that installing the two
together breaks `docling` at runtime. Following the README's line gets you an environment with no
`marker` parser and no hint why.

---

### 6. `contextgrid plugins` advertises a `usearch` dtype the docs say was deliberately dropped

**Severity: wrong**

**What the doc says** — `docs/reference/plugins.md:171-176`:

> `usearch`'s valid `dtype`s are `f32`, `f16`, `i8` (`UsearchIndex.DTYPES` in
> `src/contextgrid/index/ann.py`) — **`b1` was tried and deliberately dropped.** usearch's binary
> mode wants bit-packed input and a Hamming metric ... registering `b1` as a dtype raised
> `ValueError: The number of vector dimensions doesn't match!` on the first build.

The page's own table at line 168 says `f32/f16/i8`, and `docs/reference/install.md:26` agrees
(`usearch (f32/f16/i8)`).

**What actually happened:**

```
$ contextgrid plugins
indexes:
  usearch                  usearch HNSW, with f32/f16/i8/b1 storage. A second opinion on the same idea.
```

The registry's own description string still offers `b1`. A user reading `contextgrid plugins` —
which `plugins.md:5` presents as the authoritative live list — will try a dtype the docs say
crashes on first build.

---

### 7. `relevance-feedback` is missing from `plugins.md`'s retrieval table

**Severity: cosmetic**

`docs/reference/plugins.md:206-212` lists four retrieval strategies: `simple`, `widened`,
`decomposed`, `agentic`. The registry has five, and so does
`docs/dimensions/retrieval.md:113` — which titles its own table "The five strategies" and gives
`relevance-feedback` a full worked section at line 179.

```
=== retrieve ===
agentic  decomposed  relevance-feedback  simple  widened
```

The page that calls itself "the flat reference ... generated by running the registries, not typed
by hand" is one row short on this axis. (Everything else in the page's twelve tables matched the
registry dump exactly.)

---

### 8. Every pasted `results.summary()` in the recipes is a sentence out of date

**Severity: cosmetic**

`docs/recipes/README.md:3` promises output "**pasted verbatim**". The summary sentence has since
changed shape in two places, so none of the pasted blocks match any more.

| Doc | Says | Tool now says |
|---|---|---|
| `choose-a-chunker.md:80` | `across 5 configurations on 73 questions.` | `across 5 configurations, scored on 73 questions. The eval set holds 74 questions in all; the other 1 was not scored, because no chunk in this index held its evidence.` |
| `choose-an-embedder.md:63` | `across 3 configurations on 73 questions.` | same change |
| `reproducing-a-run.md:145` | `across 2 configurations on 3 questions.` | same change |
| `reference/reports.md:51` | `across 2 configurations on 40 questions.` | `across 2 configurations, scored on 40 questions.` |
| `README.md:57` | `across 5 configurations on 3 questions.` | same change |

Real output now also appends a failure-taxonomy paragraph (`10 of 74 questions failed. 100% of
those are fp1_missing_content: ...`) that none of the pasted blocks show.

Separately, `choose-an-embedder.md:78` quotes a sentence as if it came off the screen —

> and the tool says plainly that gap is noise: **"about 20890 questions would be needed to settle
> a gap this size."**

— and repeats the figure at line 218 ("20,890 is a real number, not a hedge"). The tool now
prints:

```
Settling a gap this size would take roughly 21,000 questions -- on a two-sided test at alpha 0.05 with 80% power, assuming per-question scores vary as much as a 0-1 score possibly can. It is an order of magnitude, not a count.
```

The new wording is better; the quotation marks around the old one are the problem.

All the *numbers* in the recipes still reproduce exactly on the same corpus:
`0.877 / 0.877 / 0.877 / 0.491 / 0.452` (chunkers), `0.877 / 0.863 / 0.164` (embedders),
`0.918 / 0.918 / 0.918 / 0.890 / 0.877 / 0.877` (local-only tier 1), `recall@5: 0.1232876712328767`
(TEI stand-in), `chunks: 232` and `tfidf -> coherence +0.416, anisotropy 0.134, 19/510` (assess),
`model calls made: 74` (agentic), and the full `fingerprint()` block including
`33 files, 31,326 bytes, 31,286 chars via markdown, 8% tables`. One confidence interval moved
(`-0.027 to +0.068` in the doc, `-0.041 to +0.069` now).

---

### 9. `results_to_markdown()`'s example output is missing two sections the tool emits

**Severity: cosmetic**

`docs/reference/reports.md:151-180` shows a full report with the sections `What to use`,
`Leaderboard`, `Which decision mattered`, `Reproducing this`. Running the page's own regeneration
snippet (`reports.md:345`) produces:

```
# support-tickets — retrieval configuration comparison
## What to use
## Score            <- not in the doc
| Dimension | Score |
## Leaderboard
## Which decision mattered
> **These are averages over runs, not controlled comparisons.** ...   <- not in the doc
```

The `## Score` composite block and the ofat caveat blockquote are both real and both absent from
the pasted example. (`## Reproducing this` is correctly absent here only because the snippet
passes no manifest; a real `contextgrid run` bundle does emit it.)

---

### 10. `cost.md`'s pasted `CostBreakdown` and litellm count are stale

**Severity: cosmetic**

`docs/reference/cost.md:148-155` shows:

```
CostBreakdown(index_usd=..., query_usd_per_1k=..., index_tokens=..., query_tokens_per_query=..., compute_seconds=..., metered=...)
```

Real output carries four more fields: `generation_usd_per_1k=0.0, evaluation_usd=0.0,
generation_tokens=0, judge_tokens=0`.

`cost.md:44` says litellm's table covers "close to 3,000 models (**2,987 in this checkout** —
`len(litellm.model_cost)`)". The page's own regeneration command at line 202 prints `3002`.

Everything else on the page reproduced exactly, including all five `pricing_for` rows, all five
`price_key` rows, both `_litellm_pricing` outputs, and the local-vs-hosted `estimate()` totals
(`0.0033333333333333335` and `0.02053888888888889`).

---

### 11. Small internal inconsistencies

**Severity: cosmetic**

| Where | Says | Actual |
|---|---|---|
| `docs/scoring/spans-and-anchors.md:76` | "**Three** module-level functions in `contextgrid.core.span`" | The table under it lists four (`merge_spans`, `total_length`, `covered_length`, `coverage_fraction`), and the module also exports `intersection_length`, which `docs/scoring/metrics.md:258` refers to by name |
| `docs/scoring/composite.md:85` | "a 73 over **all four** dimensions" | The page's own section heading five lines earlier is "**The five** dimensions", and line 90 says "all five dimensions" |
| `docs/dimensions/retrieval.md:96-101` | Output block is labelled — `simple searches=1 queries=[...] notes={}` | The snippet above it is `print(strategy.name, trace.searches, trace.queries, trace.notes)`, which prints bare values: `simple 1 ['what is...'] {}` |
| `docs/reference/plugins.md:167` | `faiss` extra is "`index` (`faiss`)" | Registry reports the package as `faiss-cpu`; `install.md:26` gets it right |

---

## What was checked and found correct

Listed because a clean result with no evidence behind it is worth nothing.

**Library API from `README.md`.** `cg.Lab(corpus=...)` + `lab.grid(...)` + `cg.read_jsonl` +
`lab.run(evalset, headline=...)` + `results.summary(...)` runs as printed on the drive corpus.
`cg.lift(retrieval_score=0.80, answer_score=0.70, baseline_answer=0.70)` prints the README's
sentence word for word.

**All seven recipes.** Every runnable command block executed; numbers reproduce (see finding 8).
`reproducing-a-run.md`'s seed demonstration reproduces exactly, including the boundary case:
`seed=0 distinguishable=True p=0.0500` twice, then `seed=1 distinguishable=False p=0.0625`.

**Spec strings.** Every spec string named in `plugins.md`'s tables builds:
8 parsers, 8 ingestion strategies, 12 chunkers, 5 embedders, 7 indexes, 2 registry transforms,
5 retrieval strategies, 5 rerankers, 2 generators. `available_transforms()` returns the
documented six; `available_generators()` returns the documented two. `get_retriever(None)` returns
`SimpleRetrieval`. `get_retriever("agentic")` with no model raises the documented error naming the
model-free alternatives, rather than picking a provider.

**The matrix model.** `matrix(...).shape()`, `.count("ofat")`, `.count("factorial")`,
`deduplicate(...)`, `expand_with_report(...)` counts `(6, 3, 1, 2)` and `report.note()` all match
`docs/dimensions/README.md` exactly, including the `widened`-is-not-a-duplicate proof
(`(['a0','b0','a1'], ['both','a0','b0'])`).

**Spans, anchors, resolution.** `coverage_of` vs `iou` (`1.0`/`0.085`/`1.0`/`0.68`), all three
`ResolutionPolicy` values, `threshold=0` raising `ResolutionError`, split-gold detection
(`is_reachable=False, is_split=True, best 0.40, union 0.80`) and its warning string, the
`EXACT → NORMALISED → BOUNDED` anchor ladder including the OCR case, and the
`is_answerable`/`is_resolved` pair — all exactly as documented. `EvalSet.answerable`,
`.resolved`, `.is_portable` and the `with_evidence` alias all behave as
`spans-and-anchors.md:153-163` says.

**The corpus's two deliberate cases.** `nw12` (no evidence at all) is reported as
`is_answerable=False`, is excluded from scoring, and the summary names it by id: *"the other 1 was
not scored (nw12), because it has no ground truth at all -- no gold spans and no anchors -- so
there was nothing to score it against. That is a gap in the eval set, not a fault in this
pipeline."* `nw13` (quote across a soft line wrap) resolves through the `NORMALISED` strategy and
is scored. Both behave as `diagnostics.md`'s `q4`-vs-`q5` table promises.

**Metrics and significance.** All six metrics, the trec_eval MAP convention (`0.25` on 20 relevant
at k=5), `evaluate` / `per_query` / `mean_rank_of_first_relevant`, `available_metrics()`, the
unknown-metric `ValueError`, registering a custom `top1_only` metric and seeing it computed
per-`k`, and both `compare()` verdicts in `significance.md`.

**The composite.** Harmonic mean (`18.095238095238095`), zero-anywhere-is-zero, `missing`,
`sources`, the cut-off inference from `@1` metrics, `k=` meaning exactly that `k`, and
out-of-range values being ignored rather than clamped.

**Caching.** `cache_key`'s two hashes reproduce byte for byte. `run.cache: disk` really does
persist: first run `cache: 58 of 84 lookups reused (69%)`, second run of the same config
`cache: 84 of 84 lookups reused (100%)`. The on-disk layout is
`.contextgrid-cache/60/70/6070e775...pkl` — two levels of fan-out, as documented, written under
`report.out`.

**Resolution config.** `run.resolution_policy` accepts `coverage`, `iou` and `containment` and all
three change the numbers (`iou` collapsed this corpus to 0.000 and correctly said "5 warnings mark
this comparison as unsound"). Bad values are caught before the sweep:
`error: run.resolution_policy must be 'coverage', 'iou' or 'containment', got 'nonsense'` (exit 1)
and `error: run.resolution_threshold must be in (0, 1], got 0.0` (exit 1).

**The bundle.** All six files land as `getting-started.md:174-181` lists them. `winning-config.yaml`
matches the documented shape field for field, including absolute `corpus:`/`evalset:`, the quoted
`"recursive:512"`, `k` in `run:` while `candidates` is in `grid:`, and no `report:` section.
`use_winning_config.py` really executes and prints chunk ids. Its "only non-defaults appear" rule
holds — setting `k: 3` in the config produced `k=3,` in the snippet.

**Tokenizers.** `cl100k_base` and `o200k_base` both build, both report `exact=True`, and both
reconstruct `'Hello, world! 日本語のテスト 🎉'` from their spans, as `plugins.md:305-318` claims.

**`pyproject.toml`.** Eleven optional-dependency keys, ten real extras plus `dev`, as
`install.md:8` says.

---

## What was skipped, and why

Named plainly, because a skipped area presented as covered is worse than an admitted gap.

- **Anything needing an API key.** `litellm`/`openai`/`anthropic` models, `hyde`, `multi-query`,
  `decompose`, `step-back`, real `agentic` retrieval, `contextual`/`summary` ingestion, the `llm`
  generator, and the DeepEval-backed generation metrics (`faithfulness`, `answer_relevancy`).
  The offline stand-ins the recipes provide were run instead, and are reported above.
- **Anything needing Docker or a server.** Real `tei` embedding and `tei-rerank` reranking, and
  `pgvector` (needs a running Postgres with the extension). The `transport=` hook that stands in
  for TEI was run and works.
- **Model downloads.** `docling` and `marker` were never invoked on a real PDF; only their
  registration and their missing-extra errors were checked.
- **`contextgrid validate`** against LegalBench-RAG — the benchmark is not vendored and is not
  ours to fetch.
- **`docs/internals/`, `docs/prd/`, `docs/roadmap.md`, `docs/adoption-backlog.md`,
  `docs/COVERAGE.md`, `docs/design.md`.** Outside the lane. Their `python` blocks were included in
  the 149 executed and all passed, but their prose claims were not audited.
- **The CLI surface itself** (`init`, `check`, `run`, `profile`, `sweep`, `diff`, `validate`) —
  another agent's lane. `init`/`run`/`check`/`plugins` were used only where a reference page made
  a claim that needed a real bundle or a real registry dump to test, and those uses are reported
  above.
- **`script -q /dev/null`** does not work in this environment
  (`script: tcgetattr/ioctl: Operation not supported on socket`), so no claim is made anywhere in
  this report about stdout/stderr *ordering*. Exit codes were all checked on bare commands, never
  through a pipe.

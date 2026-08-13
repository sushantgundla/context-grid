# Docs-driven drive 5 — the new SDK site, before PyPI

**Four disagreements across 38 brand-new pages, and none of them a blocker.** Three Sonnet agents
installed the built wheel into their own clean virtual environments and drove `docs-site/` as
strangers — no `src/`, no `tests/`, no earlier report. Every documented path could be completed.
All four findings were fixed in this session.

This drive is the gate before publishing to PyPI. The pages themselves were written and checked by
a 28-agent workflow first (see commit `fbcf809`); this is the independent pass over the result.

| Lane | Pages | Findings |
|---|---|---|
| 1 | `index`, `quickstart`, `installation`, all `concepts/`, all `lab/` | 3 |
| 2 | all `pipeline/`, all ten `axes/` | 0 |
| 3 | all `evalsets/`, all `scoring/`, all `reference/` | 1 |

Every finding below was reproduced by hand before the page was touched. One was reproduced and the
reproduction itself turned up a separate bug in the tool, recorded at the end.

## The findings, worst first

### 1. `docs-site/reference/cli.mdx:190-199` — an example of "impossible" that is not impossible

**wrong** · doc's fault · found by lane 3

The page illustrated the exit-code-1 path — a sweep where every combination is skipped — with
`ingestion: [hypothetical-questions]` and `chunker: [structural:32]` and no `run.model`, and quoted:

```
error: every combination of ingestion=hypothetical-questions, chunker=structural:32 was skipped as impossible: hypothetical-questions needs run.model set
```

That line never appears. Reproduced on a two-file corpus:

```
$ contextgrid check t.yaml   ->  exit 0, "config is valid."
$ contextgrid run t.yaml     ->  exit 0
hypothetical-questions · markdown · structural:32 · tfidf · dense    1.000      0.3   0.0000
```

The configuration builds, runs, and scores 1.000. It degrades and warns per chunk instead:
`warning: budget_reached: the 'hypothetical-questions' plugin calls a model ... this sweep has no
budget_usd or budget_seconds`.

**Needing a model is not the same as being unbuildable**, and the page had conflated the two. Replaced
with a combination that really cannot be constructed — `embedder: [null]` with `index: [dense]` — and
its real error text:

```
error: no configurations were run, so nothing was measured
error: 1 combination(s) in this matrix cannot be built and were skipped -- a dense index with no embedder has nothing to search. The axes you wrote are almost certainly what you meant; this is just the product of them that is not
```

A `<Note>` now states the distinction outright, because it is the kind of thing a reader would
otherwise learn from a green CI run that measured nothing.

### 2. `docs-site/installation.mdx:74` — `[dev]` is not "everything above"

**wrong** · doc's fault · found by lane 1

The extras table described `dev` as "Everything above, plus `pytest`, `mypy`, `ruff` and the rest of
the toolchain". Checked against the wheel's own metadata rather than against `pyproject.toml`:

```
in an extra but NOT in dev: {'parse-marker': ['marker-pdf'], 'parse-ml': ['docling'], 'pgvector': ['psycopg']}
```

Nine of the eleven feature extras are in `dev`; three packages are not. A contributor following that
row and running `pip install -e ".[dev]"` cannot exercise the layout parser, the `marker` parser or
the `pgvector` index at all, and nothing tells them why.

`marker-pdf`'s absence follows from the conflict the same page already documents at length — it cannot
share an environment with `docling`, so no single extra can hold both. For `docling` and `psycopg`
there is no such reason on record, and the page now says exactly that rather than inventing one. A new
section, *What `[dev]` leaves out*, names the three and shows how to add them.

Everything else on the page held: all eleven extra names exist in `Provides-Extra`, none is missing
from the table, and the bare-install claim is exact — the only unconditional `Requires-Dist` lines are
`numpy>=1.24` and `pyyaml>=6.0`. `[embed]` and `[chunk]` were installed for real into fresh venvs and
pulled precisely what the page describes.

### 3. `docs-site/concepts/overview.mdx:120` — a token estimate from an older version

**wrong** · doc's fault · found by lane 1

The page's own worked example writes `mydocs/refund.md` and `mydocs/shipping.md` with the text given
inline, then prints `'approximate_index_tokens': 100`. Running that exact code gives **79**.
`lab/grid.mdx` states the rule — total bytes divided by four — and `wc -c` on the two files it creates
gives 319 bytes; `319 // 4 = 79`. No rounding rule produces 100.

### 4. `docs-site/lab/overview.mdx:35` — the same, smaller

**wrong** · doc's fault · found by lane 1

`'approximate_index_tokens': 8` against a 30-byte `./docs`; the real figure is **7**. The equivalent
number in `quickstart.mdx` (191) and the other two in `lab/grid.mdx` (3, 3, 121) were all correct, so
these two examples went stale after an edit rather than the formula being wrong.

## Found while reproducing, not by a lane — and it is the tool's fault

A JSONL eval set whose anchor uses `doc_id` instead of `source_id` fails like this:

```
$ contextgrid check t.yaml
error: 'source_id'
```

A bare `KeyError` printed as the entire message. It names no file, no line, no item id, and does not
say what was expected. `evalsets/loading.mdx` lists `doc_id` as an accepted **CSV** column alias, so
reaching for it in JSONL is a natural mistake — and the reader gets four characters in quotes.

Not fixed. It is a code change, not a documentation change, and it was not in scope for this drive.

## The two deliberate fixture traps

Both handled correctly and reported honestly, by two independent lanes.

- **`nw12`** — a question with no evidence at all. `assess()` reports `answerable=12 of 13` and excludes
  it; the run summary says *"the other 1 was not scored (nw12), because it has no ground truth at
  all... That is a gap in the eval set, not a fault in this pipeline."* The warning code is
  `GOLD_SPAN_UNREACHABLE`.
- **`nw13`** — a quote spanning a soft line wrap, so it is not verbatim in the file. Resolves correctly
  (`unresolved_gold: 0`) and logs `ANCHOR_NORMALISED`, naming `nw13` and saying it "was found only
  after collapsing whitespace." Not silent.

Neither warning code is named anywhere in the Get Started, Concepts or Lab pages, which is where a
reader first meets an eval set. `scoring/diagnostics.mdx` covers both. Left as it is: a reader who hits
one is given the code in the output and can search for it.

## What lane 2 found, which was nothing

Worth recording, because a zero is only meaningful if the lane was real. Lane 2 built every plugin on
all ten axes — 8 parsers, 12 chunkers, 5 embedders, 7 indexes, 5 rerankers, 6 transforms, 5 retrieval
strategies, 8 ingestion strategies, 2 generators — checked every documented default against the actual
repr, triggered every documented validation error, and confirmed the trap `pipeline/search.mdx` warns
about (passing a query string to `scored_ids` returns deduped single characters, silently). It also
went off-script: `hash` embedder determinism across four separate processes, a real four-configuration
sweep over the eight-file Northwind corpus, and a dot-prefixed corpus directory. Every printed float
and byte count matched.

It also judged each `no-run:` fence for honesty rather than trusting it, and reconstructed the inputs
for the narrative ones to check they would in fact run. All honest.

## What was skipped

- `marker`, `docling` and `agno` real `.parse()` calls — no PDF or DOCX fixtures, and multi-gigabyte
  model downloads.
- `parse-ml` and `parse-marker` installs — deliberately not attempted.
- Anything needing a live server or an API key: `tei`, `tei-rerank`, `litellm`, `litellm-rerank`,
  `pgvector`, `agentic` retrieval, `generator=llm`. Each page's own key-free substitute
  (`RecordingLLM`, fake transports) was run instead, so the logic underneath was still exercised.

## State after this drive

283 runnable code blocks compile, doctest-style blocks included; 71 are fenced `no-run:` with a
stated reason. `docs.json` lists all 38 pages and every page exists. Nothing is published to PyPI:
the packaging gaps — no `LICENSE` file, no `py.typed`, an sdist that ships `.claude/` and the reports,
no publish workflow, and README links that break off GitHub — are all still open.

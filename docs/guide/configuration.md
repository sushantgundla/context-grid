# Configuration reference

The config file — YAML or JSON, `contextgrid run config.yaml` — is the whole public interface.
This page lists every key. If a key isn't listed here, context-grid doesn't accept it: unknown
keys are always a hard error, not a warning, because a config that silently falls back to
defaults on a typo produces a leaderboard that answers a different question than the one you
asked.

For a walkthrough of writing your first one, see [getting-started.md](getting-started.md).

## Shape of the file

```yaml
name: my-experiment       # optional, defaults to the filename

corpus: ./documents       # required
evalset: ./questions.jsonl

grid:
  parser: [markdown, pymupdf]
  chunker: recursive:512
  # ... every other axis

run:
  mode: ofat
  k: 10
  headline: recall@5

report:
  out: ./results
  formats: [markdown, json]
```

Three rules apply everywhere in the file:

- **Any axis takes one value or a list.** `chunker: recursive:512` and
  `chunker: [recursive:512, sentence:3]` are both valid — a single value holds that axis still,
  a list sweeps it. You never have to wrap a single choice in brackets.
- **Paths resolve against the config file's own directory**, not your shell's working
  directory. `corpus: ./documents` in `experiments/exp1.yaml` means
  `experiments/documents`, no matter where you run `contextgrid` from. Absolute paths and
  `~/...` are left alone.
- **`${VAR}` is substituted from the environment** before the file is parsed, anywhere in the
  text — so a config can reference an API key without containing one. An unset variable is a
  clear error naming the variable, not a silent empty string.

The format is decided by content, not by file extension — a `.yaml` file holding valid JSON is
accepted as JSON. `contextgrid init` always writes YAML.

## Top level

| Key | Type | Default | What it does |
|---|---|---|---|
| `corpus` | path | *(required)* | A directory of documents, or a single file. The only required key. |
| `evalset` | path or `null` | `null` | A JSONL or CSV file of questions. Required to actually score a sweep — `check` works without it, `run` doesn't. |
| `name` | string | the filename, or `"experiment"` | Shows up in `describe()` / `check` output and in the report. |
| `grid` | mapping | *(see below)* | The axes and the values to try on each. |
| `run` | mapping | *(see below)* | How the sweep is executed. |
| `report` | mapping | *(see below)* | What to write out, and where. |

## `grid:` — the axes

Every axis left out is a single-valued axis at its default — which is what makes a minimal
config possible: naming one axis sweeps it and holds everything else still.

| Key | Type | Default | Accepts `null`? | What it does |
|---|---|---|---|---|
| `ingestion` | string or list | `null` | yes | The ingestion strategy — what goes into the index and what a hit on it returns. `null` means plain chunking, where those are the same thing. `parent-document`, `contextual`, `hierarchical` and friends deliberately break that equivalence. |
| `parser` | string or list | `markdown` | no | What reads the documents. The axis nothing else in the field measures — usually the one that matters most on documents with tables. |
| `chunker` | string or list | `recursive:512` | no | How the text is cut up. Values are spec strings, e.g. `recursive:512`, `sentence:3` — the number is chunker-specific (tokens for `recursive`/`fixed`, sentences for `sentence`). |
| `embedder` | string or list | `tfidf` | yes | What turns text into vectors. `null` means none — what a `bm25` index wants. |
| `index` | string or list | `dense` | no | How the search is done: `dense`, `bm25`, `hybrid`, and others behind extras. |
| `transform` | string or list | `null` | yes | Rewriting the question before searching with it (HyDE, multi-query, decompose, step-back). Each non-null value costs a model call on every query. |
| `retrieval` | string or list | `null` | yes | How the index is used, as opposed to what it is. `null`/`simple` is one search; others widen the net, split the question, or go agentic. |
| `reranker` | string or list | `null` | yes | Reordering what came back. `null` means no reranking. |
| `candidates` | int or list | `50` | no | How deep the reranker gets to look before cutting to `k`. Most of a reranker's effect lives here, not in which reranker you pick. |

`contextgrid plugins` lists every value actually installed for each axis; `contextgrid init`
writes a config listing only those.

**A combination that can't run is skipped, not attempted.** `index: dense` swept against
`embedder: null` (dense search needs vectors) is counted as an "impossible combination" in
`check`/`run` output and dropped, rather than erroring the whole sweep or running silently
with defaults.

## `run:` — how the sweep executes

| Key | Type | Default | What it does |
|---|---|---|---|
| `mode` | `ofat` \| `factorial` \| `staged` | `ofat` | `ofat` (one-factor-at-a-time) changes one axis from a baseline per run — cheap, good for a first pass. `factorial` runs every combination the axes describe — expensive, exhaustive. `staged` runs axes in a fixed order, carrying the best of each stage forward. |
| `k` | int (≥ 1) | `10` | How many chunks reach the generator / count as "retrieved" for top-level metrics. |
| `headline` | string, `metric@cutoff` | `recall@5` | What the leaderboard sorts on. Metric must be one of `recall`, `precision`, `ndcg`, `mrr`, `map`, `hit_rate`; the cutoff must be a number, e.g. `ndcg@10`. The cutoff is always included in the metrics reported, even if not in the default set. |
| `budget_seconds` | float or `null` | `null` (no limit) | Stop the sweep after this many wall-clock seconds. A sweep containing a strategy that decides its own number of model calls has no ceiling without this or `budget_usd`. |
| `budget_usd` | float or `null` | `null` (no limit) | Stop the sweep after this much estimated spend. `0.0` means "already spent" — nothing runs, and the report says why rather than showing an empty leaderboard as if the matrix had been covered. |
| `seed` | int | `0` | Random seed, recorded in the manifest for reproducibility. |
| `machine_usd_per_hour` | float | `0.0` | Prices local compute by the hour, so a CPU model and a hosted API land on the same cost chart. A local model is free per token and not free to run. |
| `resolution_policy` | `coverage` \| `iou` \| `containment` | `coverage` | How a gold span is resolved onto retrieved chunks. `coverage` (default): the chunk must hold at least `resolution_threshold` of the gold span's characters — the question that matters, "is the evidence there?". `iou`: intersection-over-union, which systematically penalises large chunks for being large. `containment`: the chunk must fully contain the gold span — strictest, useful for citation-accuracy work. |
| `resolution_threshold` | float, in `(0, 1]` | `0.5` | The threshold used by `resolution_policy`. |
| `cache` | `memory` \| `disk` \| `none` | `memory` | Where content-addressed cache entries (parse / chunk / embed results shared across configurations) live. `disk` persists across runs, at `report.out/.contextgrid-cache` (or beside the corpus if `report.out` is unset). `none` disables caching entirely. |
| `model` | string or `null` | `null` | The model every stage that needs one shares — query transforms, agentic retrieval, LLM-backed ingestion strategies, the generation judge. One key, one price, rather than four places to configure it. |

## `report:` — what gets written

| Key | Type | Default | What it does |
|---|---|---|---|
| `out` | path or `null` | `null` | Where to write the result bundle. `null` means nothing is written — the leaderboard still prints to the console. Directories are created if they don't exist. |
| `formats` | list of `markdown` \| `json` \| `yaml` \| `python` | `[markdown, json]` | Which files to write into `out`. `markdown` → `report.md` (leaderboard + summary). `json` → `results.json` (every run, every metric). `yaml` → `winning-config.yaml` (the winning configuration alone, re-runnable). `python` → `use_winning_config.py` (the winning configuration as a Python snippet). A `manifest.json` and a copy of the source config (`experiment.yaml`) are always written alongside, regardless of `formats`, whenever `out` is set and there's a winner. |
| `leaderboard_limit` | int | `20` | How many rows the Markdown leaderboard shows. |

## Errors you'll actually see

- **Unknown key**: `unknown key 'chunkers' in the 'grid' section. Did you mean 'chunker'? Known
  keys: ...` — a typo is guessed at with a Levenshtein-style match, and the full list of valid
  keys is always printed alongside.
- **Wrong section type**: `the 'grid' section must be a mapping, got list`.
- **Missing corpus**: `every config needs a corpus: a directory of documents, or a list of
  files.`
- **Bad headline**: `run.headline must name a cut-off, like 'recall@5'. Got 'recall'` — or, for
  a non-numeric cutoff, `run.headline has a non-numeric cut-off: 'recall@five'`.
- **Unset env var**: `the config refers to ${API_KEY} but that environment variable is not
  set`.

## Checking before running

`contextgrid check config.yaml` parses the file, resolves every path, and prints the shape of
the sweep without running anything — including which combinations will be skipped as
impossible. See [cli.md](cli.md#check) for real output.

## See also

- [getting-started.md](getting-started.md) — install and first sweep
- [cli.md](cli.md) — every subcommand
- [evalsets.md](evalsets.md) — the `evalset` this config points at

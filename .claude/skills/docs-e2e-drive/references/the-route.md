# The route

Two lanes. Drive one end to end, or split them across two sessions. Each step names the page
that promises the behaviour — read that page first, then run the command, then compare.

- [Lane A — the command line](#lane-a--the-command-line)
- [Lane B — the library, the recipes, the reference pages](#lane-b--the-library-the-recipes-the-reference-pages)
- [Surfaces that have regressed before](#surfaces-that-have-regressed-before)

Do not treat this as exhaustive. It is where to start, not where to stop. A drive that only
walks this list will find only what the last drive found.

## Lane A — the command line

### 1. Meet the tool

| Step | Page |
|---|---|
| `contextgrid --help` — subcommand list matches the usage block character for character | `docs/guide/cli.md` |
| `contextgrid plugins`, then `--family <each one it names>` | `docs/guide/cli.md`, `docs/reference/plugins.md` |
| Every family the reference pages document appears in the command's output | `docs/reference/plugins.md` |

### 2. The eval set, as a subject-matter expert hands it over

| Step | Page |
|---|---|
| `read_csv` on `data/questions.csv`, loose column names and all | `docs/guide/evalsets.md` |
| A row with a question and no quote loads, and is not answerable | `docs/guide/evalsets.md` |
| `write_jsonl` then `read_jsonl` — anchors, grades, page hints and qtypes survive | `docs/guide/evalsets.md` |
| `contextgrid evalset <file>` — size, % reviewed, smallest detectable difference | `docs/guide/cli.md` |
| `meta.reviewed: true` moves the reviewed percentage | `docs/guide/evalsets.md` |
| `is_answerable` versus `is_resolved` mean what the page says | `docs/scoring/spans-and-anchors.md` |

### 3. Look before you leap

| Step | Page |
|---|---|
| `contextgrid profile` on the corpus directory, and on a single file | `docs/guide/cli.md` |
| `contextgrid init`, then again without `--force` — must refuse, exit 1 | `docs/guide/cli.md` |
| The generated config's comments match what the page says init writes | `docs/guide/cli.md`, `docs/guide/getting-started.md` |
| `contextgrid check` the generated config | `docs/guide/cli.md` |

### 4. A sweep a real user would want

Sweep what matters for short factual articles — chunk size, and whether lexical beats dense:

```yaml
grid:
  chunker: [recursive:256, recursive:512, sentence:3]
  embedder: [tfidf, null]
  index: [dense, bm25, hybrid]
  reranker: [null, lexical]
  candidates: [20, 50]
```

Then read every file in `report.out` against the table in `docs/guide/getting-started.md`, and
check the leaderboard's prose against `docs/scoring/diagnostics.md`. Re-run the winning config
and the generated Python snippet — both are documented as runnable.

### 5. Second opinion, then diff

Run again in another mode into a second output folder and `contextgrid diff` the two manifests.
Check the message describes what actually differed, and does not claim more than it compared.

### 6. Make it angry

Each of these has a documented message and exit code. Check both, with a bare run for `$?`.

A typo'd plugin name · a plugin parameter out of range · a missing corpus · a missing eval set ·
an eval set that is not valid JSON · an eval set with no question column · a directory where a
file was expected · a misspelled config key · a headline metric with no cut-off · a non-numeric
cut-off · an unset `${VAR}` · `budget_usd: 0.0` · a config in a subfolder using relative paths ·
a plugin whose extra is not installed · a remote embedder with no server.

Then confirm the promise covering all of them: **nothing raises a raw traceback at the top
level** (`docs/guide/cli.md`). A raw `AttributeError` or `[Errno 21]` behind an `error:` prefix
is a finding — it satisfies the letter and fails the reader.

### 7. `validate`

`docs/guide/cli.md` documents the benchmark file's shape well enough to hand-write one. Test
that claim literally: build the file from the page alone and run it. Then feed it malformed
shapes — a bare array, a missing key, a wrong type, an empty file — and check each message names
the file and the problem.

## Lane B — the library, the recipes, the reference pages

| Surface | What to check |
|---|---|
| `README.md` snippets | Every one runs exactly as printed, including `cg.lift(...)`'s output sentence |
| Every `python` block in `docs/` | Execute in document order per file, globals carried forward, the way a reader copy-pasting down the page would have them. Skip only blocks tagged `no-run:` |
| `docs/recipes/` | All of them. The pages promise output "pasted verbatim" — hold them to it |
| `docs/dimensions/` | Every spec string named in the tables actually builds |
| `docs/scoring/` | Resolution policies, thresholds, metric names, the failure taxonomy |
| `docs/reference/` | Caching (does `disk` really persist?), the cost model, report formats, the plugin catalogue |

Skip anything needing an API key, Docker, a live database or a model download — and **say so in
the report**. A skipped area presented as covered is worse than an admitted gap.

## Surfaces that have regressed before

Check these even when nothing points at them. Each was broken once, and a fix in one has twice
broken another:

- Corpus loading, including a corpus path with a dot-prefixed component
- `check` rejecting what `run` would reject — the two must agree
- `is_answerable` versus `is_resolved`, and everything downstream of that distinction
- The failure taxonomy's prose, especially for a question with no ground truth
- The `warnings` array in `results.json` — duplicates, and per-configuration facts surviving dedup
- Anything claiming reproducibility: seeds, cache keys, hashing embedders
- Pages that claim to be generated from the registries. Check whether anything generates them

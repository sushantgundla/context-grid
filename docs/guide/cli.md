# The command line

`contextgrid` is a plain `argparse` CLI — no framework, so the core package installs with
nothing but numpy and pulls in nothing else just to give you a command line. Nine subcommands:

```
usage: contextgrid [-h] [--version]
                   {run,init,check,profile,sweep,plugins,evalset,validate,diff} ...

Sweep retrieval configurations on your own documents.

positional arguments:
  {run,init,check,profile,sweep,plugins,evalset,validate,diff}
    run                 Run everything a config file describes.
    init                Write a starter config for this installation.
    check               Validate a config and say what it would run.
    profile             Profile a corpus and say which axes will matter.
    sweep               Run a matrix and print the leaderboard.
    plugins             List everything registered.
    evalset             Inspect an eval set and what it can support.
    validate            Check the scorer against a published benchmark.
    diff                Say what changed between two run manifests.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

Every command exits non-zero and prints `error: ...` to stderr on failure — nothing raises a
raw traceback at the top level.

## `run`

Runs everything a config file describes and prints the leaderboard. The everyday command — see
[getting-started.md](getting-started.md) for the full walkthrough and what "reading the
leaderboard" means.

```bash no-run: usage synopsis -- config.yaml is a placeholder, not a real file
contextgrid run config.yaml [--quiet]
```

| Flag | What |
|---|---|
| `config` | A YAML or JSON experiment file (positional, required). |
| `--quiet` | Suppress the `[n/total] configuration` progress lines on stderr. Use this in CI or when piping output. |

Writes whatever `report:` in the config asks for (see [configuration.md](configuration.md#report--what-gets-written)).

Exits 1 if the sweep ran **no** configurations at all — a `budget_usd` or `budget_seconds` too
small for a single one, or a matrix whose every combination was impossible to build — and says
which of those it was on stderr. Nothing was measured, so a green build would be a lie.

A sweep that ran *some* of its configurations and was then stopped by its budget still exits 0:
the leaderboard it printed is real, and it is marked partial in the warnings.

## `init`

Writes a starter config listing every plugin *this installation* can actually run as a chosen
value, and everything else as a comment showing how to unlock it (which extra to install).

```bash no-run: usage synopsis, not a literal command
contextgrid init [path] [--corpus DIR] [--evalset FILE] [--force]
```

| Flag | Default | What |
|---|---|---|
| `path` | `contextgrid.yaml` | Where to write the file (positional). |
| `--corpus` | `./documents` | The `corpus:` value written into the file. |
| `--evalset` | `./questions.jsonl` | The `evalset:` value written into the file. |
| `--force` | off | Overwrite an existing file. Without it, `init` refuses and exits 1 — it will not silently replace a config you spent an afternoon tuning. |

```
$ contextgrid init contextgrid.yaml --corpus ./documents --evalset ./questions.jsonl
wrote contextgrid.yaml
edit it, then run:  contextgrid run contextgrid.yaml
```

## `check`

Parses a config, resolves its paths, and prints the shape of the sweep — without running or
scoring anything.

```bash no-run: usage synopsis -- config.yaml is a placeholder, not a real file
contextgrid check config.yaml
```

```
$ contextgrid check contextgrid.yaml
contextgrid: 1 × 1 × 2 × 2 × 3 × 1 × 1 × 2 × 1 = 24 on paper, 5 to run in ofat mode (1 impossible combination(s) skipped), scored on recall@5
  ingestion   ['plain']
  parser      ['markdown']
  chunker     ['recursive:512', 'sentence:3']
  embedder    ['tfidf', None]
  index       ['dense', 'bm25', 'hybrid']
  transform   [None]
  retrieval   ['simple']
  reranker    [None, 'lexical']
  candidates  [50]

config is valid.
```

It also builds one of every plugin the matrix names, with the parameters you gave it, and
reports whatever refuses to be built. Building a plugin reads no documents, embeds nothing,
indexes nothing and calls no model, so this stays fast and writes nothing.

Exits 1 on any of the following, reporting every one it found rather than stopping at the
first:

| Problem | Example message |
|---|---|
| The corpus path is missing | `corpus not found: /path/to/absent` |
| The corpus directory holds nothing readable | `no files under /path/to/docs matched ['*.txt', ...]` |
| The eval set is missing, or there is none | `no evalset, so there is nothing to score against` |
| An axis names a plugin that does not exist | `chunker 'banana:999': no chunker named 'banana'. Available: ...` |
| A plugin rejects its parameters | `chunker 'recursive:-5': chunk size must be positive, got -5` |

Each message is the one `run` would have printed; `check` only makes it arrive before the
expensive part. The matrix shape is still printed first — the config parsed fine, only its
contents are wrong — and then every problem, one `error: ...` line each, on stderr:

```
$ contextgrid check broken.yaml; echo "exit: $?"
broken: 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = 1 on paper, 1 to run in ofat mode, scored on recall@5
  ingestion   [None]
  parser      ['markdown']
  chunker     ['recursive:512']
  embedder    ['tfidf']
  index       ['dense']
  transform   [None]
  retrieval   [None]
  reranker    [None]
  candidates  [50]

error: corpus not found: /path/to/absent
error: no evalset, so there is nothing to score against
exit: 1
```

Worth running before any sweep that might take a while — it catches a missing path, an empty
corpus, a typo'd axis or a typo'd spec string before anything expensive starts.

## `profile`

Reads a corpus and suggests which axes are worth sweeping, based on the documents themselves —
no eval set needed.

```bash no-run: usage synopsis, not a literal command
contextgrid profile corpus [--parser NAME]
```

```
$ contextgrid profile ./documents
2 files, 381 bytes, 381 chars via markdown
  - The median document is 190 characters. Chunk sizes above that cannot differentiate, so sweep small sizes.
```

## `sweep`

A one-shot sweep from flags alone, no config file. Good for a quick check; `run` with a config
file is the one to script or keep in version control.

```bash no-run: usage synopsis, not a literal command
contextgrid sweep corpus evalset [options]
```

| Flag | Default | What |
|---|---|---|
| `corpus`, `evalset` | — | Positional: a documents directory and a JSONL eval set. |
| `--parser`, `--chunker`, `--embedder`, `--index`, `--reranker` | one built-in default each | Repeatable (`--chunker recursive:128 --chunker recursive:256` sweeps both). |
| `--mode` | `ofat` | `factorial`, `ofat`, or `staged`. |
| `--metric` | `recall@5` | The headline metric. |
| `--k` | `10` | Chunks reaching the generator. |
| `--budget-seconds` | none | Wall-clock stop. |
| `--bundle PATH` | none | Write a full result bundle (manifest, report, winning config) to this directory. |

```
$ contextgrid sweep ./documents ./questions.jsonl --chunker recursive:128 --chunker recursive:256 --index bm25 --embedder null --metric recall@3 --k 3 --bundle ./sweep-results
  [1/2] markdown · recursive:128 · bm25
  [2/2] markdown · recursive:256 · bm25
1 × 1 × 2 × 1 × 1 × 1 × 1 × 1 × 1 = 2 on paper, 2 to run (ofat)

configuration                   recall@3   p95 ms     $/1k
-----------------------------------------------------------
markdown · recursive:128 · bm25    1.000      0.0   0.0000
markdown · recursive:256 · bm25    1.000      0.0   0.0000

markdown · recursive:128 · bm25 scored best on recall@3 at 1.000, across 2 configurations on 3 questions. [...]

cache: 2 of 8 lookups reused (25%), chunk 0/4, parse 2/4

wrote 5 files to sweep-results
```

## `plugins`

Lists everything registered for each plugin family — parser, chunker, embedder, index,
reranker, tokenizer — with a one-line description each. Includes plugins that need an extra
you haven't installed (unlike the `init` template, which only offers what's runnable).

```bash no-run: usage synopsis, not a literal command
contextgrid plugins [--family NAME]
```

```
$ contextgrid plugins --family chunker
chunkers:
  chonkie:code             Splits on the syntax tree. Nothing hand-written comes close.
  chonkie:recursive        Chonkie's recursive splitter. The head-to-head against ours.
  chonkie:sentence         Whole sentences, chonkie's.
  chonkie:token            Fixed token windows, chonkie's.
  fixed                    Fixed-size token windows with overlap.
  langchain:character      One separator only. The naive baseline.
  langchain:markdown       Recursive, Markdown boundaries first.
  langchain:recursive      What most deployed systems are actually running.
  recursive                Split on the largest separator that fits. The default.
  semantic                 Cut where consecutive sentences change topic.
  sentence                 A sliding window of whole sentences.
  structural               One chunk per section, bounded by size.
```

## `evalset`

Inspects an eval set file and reports what it can support — size, how much has been human
reviewed, the smallest score difference it can reliably detect. See
[evalsets.md](evalsets.md#eval-set-quality) for what these numbers mean.

```bash no-run: usage synopsis -- path is a placeholder, not a real file
contextgrid evalset path
```

```
$ contextgrid evalset questions.jsonl
policy-questions v1 (manual)
3 questions (3 answerable), 0% reviewed, detects differences of 1.00 and above
types: {'unlabelled': 3}
  - 3 answerable questions can only detect differences of about 1.00 or larger. Anything smaller than that on a leaderboard built from this set is noise
  - only 0% of this set has been looked at by a human. Auto-generated ground truth is the weakest link in any retrieval comparison, and the review queue is the cheapest place to fix it
```

## `validate`

Checks the scoring chain itself against [LegalBench-RAG](https://arxiv.org/abs/2408.10343), the
one public benchmark that stores ground truth as character spans the way context-grid does.
Not vendored — point it at your own local copy.

```bash no-run: usage synopsis, not a literal command
contextgrid validate benchmark.json corpus [--limit N] [--recall-at-10 X]
```

| Flag | What |
|---|---|
| `benchmark` | A LegalBench-RAG JSON file (positional). |
| `corpus` | The directory the benchmark's spans point into (positional). |
| `--limit` | Only use the first N questions. |
| `--recall-at-10` | The published number to compare against. Without it, `validate` still runs and reports its own numbers, just with nothing to diff against. |

It checks first that the benchmark's spans point at real text in the corpus as loaded — a
mismatch there is a loading problem and invalidates everything that follows — then scores a
deliberately plain configuration and, if `--recall-at-10` was given, compares it:

```
$ contextgrid validate legalbench-sample.json ./documents --recall-at-10 0.72
1 of 1 spans point at real text in the documents as loaded, so the benchmark and the corpus agree and the scoring chain has something valid to run on.

# Validation against LegalBench-RAG

Resolved 1 of 1 questions (100%) to character spans in the corpus.

| Metric | Ours | Published | Delta |
|---|---:|---:|---:|
| recall@10 | 1.000 | 0.720 | +0.280 |

**recall@10 differs by +0.280, outside the 0.05 tolerance.** A difference in retrieval configuration explains some of this; anything left over is a problem with our scoring, not with the benchmark.
```

Exits 1 if fewer than 95% of spans resolve — below that, nothing downstream means anything, so
the command stops rather than reporting scores against broken ground truth.

## `diff`

Explains what changed between two run manifests (`manifest.json`, written by `run`, `sweep
--bundle`, or as part of any report bundle).

```bash no-run: usage synopsis -- before.json/after.json are placeholders, not real files
contextgrid diff before.json after.json
```

```
$ contextgrid diff results/manifest.json sweep-results/manifest.json
6 thing(s) changed between these runs:
  config.chunker: 'recursive:512' -> 'recursive:128'
  config.embedder: 'tfidf' -> None
  config.index: 'dense' -> 'bm25'
  config.k: 10 -> 3
  config.retrieval: 'simple' -> None
  seeds.run: 0 -> None
```

Useful for "why did the leaderboard change" after editing a config and re-running — the diff is
between the two *winning* configurations' manifests, not a diff of the config files themselves.

## See also

- [getting-started.md](getting-started.md) — the `init` → `check` → `run` cycle end to end
- [configuration.md](configuration.md) — every key `run` and `check` read from your config
- [evalsets.md](evalsets.md) — what `evalset` and `validate` are reading

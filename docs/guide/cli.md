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
    profile             Measure a corpus and flag settings its shape rules out.
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
| An axis names a plugin that does not exist | `grid.chunker: no chunker named 'banana'. Available: ...` |
| A plugin rejects its parameters | `chunker 'recursive:-5': chunk size must be positive, got -5` |

Each message is the one `run` would have printed; `check` only makes it arrive before the
expensive part.

Where they appear depends on when they are knowable. An unknown plugin *name* is caught while
the file is being parsed, before there is a matrix to print a shape for, so those arrive first,
as one `error:` block with the missing paths listed alongside them:

```
$ contextgrid check typo.yaml; echo "exit: $?"
error: 3 problems with this config:
  grid.chunker: no chunker named 'banana'. Available: fixed, recursive, semantic, sentence, ...
  grid.parser: no parser named 'nosuchparser'. Available: markdown, pdfplumber, pymupdf, text, ...
  corpus not found: /path/to/absent
exit: 1
```

Everything else needs the config to have parsed. The matrix shape goes to stdout — the file
itself was fine, only its contents are wrong — and every problem goes to stderr, one
`error: ...` line each. At a terminal they arrive in that order:

```
$ contextgrid check broken.yaml; echo "exit: $?"
broken: 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = 1 on paper, 1 to run in ofat mode, scored on recall@5
  ingestion   [None]
  parser      ['markdown']
  chunker     ['recursive:512']
  embedder    ['tfidf']
  index       ['dense']
  transform   [None]
  retrieval   [None]
  reranker    [None]
  candidates  [50]
  generator   [None]

error: corpus not found: /path/to/absent
error: no evalset, so there is nothing to score against
exit: 1
```

**Redirect either stream and that order flips.** The two are separate streams, and stdout is
buffered when it is not a terminal, so `contextgrid check broken.yaml > shape.txt` — or piping
into anything, or capturing in CI — makes the `error:` lines appear before the shape rather
than after it. Nothing is missing and nothing has changed; the interleaving you see at a
terminal is not a guarantee about which came first.

Worth running before any sweep that might take a while — it catches a missing path, an empty
corpus, a typo'd axis or a typo'd spec string before anything expensive starts.

## `profile`

Measures a corpus and flags sweep settings its shape rules out, from the documents alone — no
eval set needed.

It is narrower than it sounds, and the example below is representative rather than truncated.
The first line is the corpus as measured: how many files, how many bytes, how many characters
once the parser has read them, which parser did the reading, and what share of the text is
tables. Under it come the notes — one per thing the shape tells you, and only for the things it
does tell you. Three you will see often:

- **Tables.** A high table percentage says parser choice is likely to swamp the other axes, and
  that a chunker which cuts a table in half loses the answer with it.
- **Headings.** A high average heading count per document says structural chunking is worth
  putting on the grid.
- **Median document length.** Chunk sizes above the median cannot tell two configurations
  apart, so sweeping them wastes the sweep.

```bash no-run: usage synopsis, not a literal command
contextgrid profile corpus [--parser NAME]
```

```
$ contextgrid profile ./documents
8 files, 13,699 bytes, 13,635 chars via markdown, 17% tables
  - 17% of this corpus is tables. Parser choice will probably dominate every other axis here, and a chunker that splits a table in half will lose the answer outright.
  - Documents average 6 headings each. Structural chunking usually wins on corpora like this.
  - The median document is 1,773 characters. Chunk sizes above that cannot differentiate, so sweep small sizes.
```

Read those notes as pointers at which axes are worth sweeping, not as a result. `profile` runs
no configuration, scores nothing and produces no leaderboard — "parser choice will probably
dominate" is a statement about the shape of your documents, not a measurement of any parser.
Which axis actually matters on your corpus is what the sweep itself is for; `profile` exists to
stop you spending one on a setting the corpus already rules out.

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
1 × 1 × 2 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = 2 on paper, 2 to run (ofat)

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
  - only 0% of this set is marked as checked by a human. Ground truth nobody has read is the weakest link in any retrieval comparison. If you wrote these questions yourself, say so with `"meta": {"reviewed": true}` on each one; otherwise the review queue is the cheapest place to fix it
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

### The file it expects

Since nothing is vendored, here is the shape, so you can hand-write one and check the command
works before you go looking for the real benchmark. A JSON object with a `tests` array (a
bare array is also accepted):

| Field | Where | Required | What |
|---|---|---|---|
| `query` | on each test | yes | The question. A test with an empty or missing `query` is skipped. |
| `snippets` | on each test | yes, in practice | The evidence. A test with none loads, but has no gold, so it can never be scored. |
| `file_path` | on each snippet | yes | The document, named **relative to the `corpus` directory** you pass on the command line. `validate` reads `corpus/file_path` and skips any snippet whose file isn't there. |
| `span` | on each snippet | yes | `[start, end]` — character offsets into that file, half-open, counted from 0. Anything shorter than two entries is skipped. |
| `id` | on each test | no | Defaults to `lb0`, `lb1`, ... in file order. |
| `answer` | on each test | no | The expected answer text. Carried through, not scored by `validate`. |

The offsets are into the file **as it sits on disk** — `validate` reads the corpus verbatim
with the plain-text parser precisely so that nothing reflows the text and moves them. Count
characters, not lines, and not bytes.

A complete two-question file:

```json
{
  "tests": [
    {
      "id": "q1",
      "query": "Does the Northwind Cloud API have any unauthenticated endpoints?",
      "snippets": [
        {"file_path": "api-authentication.md", "span": [117, 202]}
      ]
    },
    {
      "id": "q2",
      "query": "When is a Northwind Cloud customer billed each month?",
      "snippets": [
        {"file_path": "billing.md", "span": [24, 108]}
      ]
    }
  ]
}
```

The quickest way to get the two numbers right is to let Python find them, rather than counting
by hand:

```python no-run: reads a corpus that isn't reconstructed in this snippet
from pathlib import Path

text = Path("documents/api-authentication.md").read_text()
quote = "There is no session, no cookie, and no unauthenticated endpoint apart from `/health`."
start = text.index(quote)
print([start, start + len(quote)])       # -> [117, 202]
```

### Running it

It checks first that the benchmark's spans point at real text in the corpus as loaded — a
mismatch there is a loading problem and invalidates everything that follows — then scores a
deliberately plain configuration (the text parser, `recursive:512` with 64 overlap, BM25, no
embedder, `k=10`) and, if `--recall-at-10` was given, compares it:

```
$ contextgrid validate legalbench-sample.json ./documents --recall-at-10 0.72
2 of 2 spans point at real text in the documents as loaded, so the benchmark and the corpus agree and the scoring chain has something valid to run on.

# Validation against LegalBench-RAG

Resolved 2 of 2 questions (100%) to character spans in the corpus.

| Metric | Ours | Published | Delta |
|---|---:|---:|---:|
| recall@10 | 1.000 | 0.720 | +0.280 |

**recall@10 differs by +0.280, outside the 0.05 tolerance.** A difference in retrieval configuration explains some of this; anything left over is a problem with our scoring, not with the benchmark.
```

That plain configuration is the point — it is there to check the *scorer*, not to win the
benchmark, so a `+0.280` delta on a two-question hand-written file says nothing except that the
chain ran. Leave `--recall-at-10` off and you get the same header followed by every metric the
run computed, one per line, with nothing to diff against.

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

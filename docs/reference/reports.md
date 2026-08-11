# The report bundle

What a sweep produces once it's done: a leaderboard, a manifest that pins down whether it's
reproducible, four export formats, and `contextgrid diff` for regression triage. Sources:
`src/contextgrid/report/results.py`, `manifest.py`, `export.py`. For the *composite score*
(one number across dimensions), see [the scoring docs](../scoring/) — this page covers the
layer above it: the leaderboard, the bundle, and reproducibility.

All examples below build `Results`/`RunResult`/`Manifest` directly (they're plain
dataclasses) with two made-up configurations, so the output is real, not hand-typed — it's
just not from an actual corpus run. See [configuration.md](../guide/configuration.md) for how
a real `contextgrid sweep --bundle ./out` produces the same files from an actual sweep.

## Leaderboard

`Results.leaderboard(metric)` sorts runs by one metric and attaches latency, cost and chunk
count to every row — never the metric alone:

```
$ .venv/bin/python -c "
from contextgrid.report.results import Results
# ... results built from two RunResults, cfg_a=recursive:512, cfg_b=structural:800 ...
for row in results.leaderboard('recall@5'):
    print(row)
"
{'config': 'markdown · structural:800 · tfidf · dense', 'recall@5': 0.76, 'p95_ms': 33,
 'cost_per_1k': 0.0, 'chunks': 290, 'ci_low': 0.76, 'ci_high': 0.76}
{'config': 'markdown · recursive:512 · tfidf · dense', 'recall@5': 0.71, 'p95_ms': 31,
 'cost_per_1k': 0.0, 'chunks': 340, 'ci_low': 0.7, 'ci_high': 0.7}
```

`format_leaderboard()` renders the same rows as a fixed-width table for a terminal —
this is the `cache: ...` line's neighbour in `contextgrid sweep` output
(see [caching.md](caching.md#reading-the-stats-back) for the cache summary line):

```
configuration                             recall@5   p95 ms     $/1k
---------------------------------------------------------------------
markdown · structural:800 · tfidf · dense    0.760     33.0   0.0000
markdown · recursive:512 · tfidf · dense     0.710     31.0   0.0000
```

Two other views sit next to `leaderboard()` on `Results` and matter more, per the module's
own docstring — `pareto()` (what quality costs — configurations nothing else beats on both
axes) and `axis_effect()` (mean score per value of one axis, e.g. "structural chunking
averaged 0.71 against recursive's 0.63" — an interpretable sentence, not a 48-row table).
`Results.summary()` turns the winner into one plain-English paragraph, including a real
significance test against the runner-up:

```
markdown · structural:800 · tfidf · dense scored best on recall@5 at 0.760, across 2
configurations on 40 questions. markdown · structural:800 · tfidf · dense beats markdown ·
recursive:512 · tfidf · dense by 0.060 on recall@5 (95% CI +0.060 to +0.060, p=0.000, n=40).
It wins on 40 questions, loses on 0 and ties on 0. It runs locally at no cost per query,
answering at 33 ms p95.
```

## Manifest

`Manifest` (`src/contextgrid/report/manifest.py`) is a hash-pinned record of everything that
could change a number: the config, the corpus's content hash, the eval set's id/version/hash,
the resolution policy, library versions, and seeds. It deliberately **excludes** timings,
costs and results — a manifest that changed every run couldn't be compared with another one.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Manifest:
    config: dict[str, Any]
    corpus_hash: str
    corpus_files: int
    evalset_id: str
    evalset_version: int
    evalset_hash: str
    resolution: dict[str, Any]
    versions: dict[str, str]
    seeds: dict[str, int] = field(default_factory=dict)
    created_at: str = ""  # recorded, but excluded from the hash
    notes: str = ""  # same
```

`.hash()` / `.short_hash` hash everything above except `created_at`/`notes`:

```
$ .venv/bin/python -c "print(manifest.hash()); print(manifest.short_hash)"
291deff3d9974dfdac295034598036712af86c234fe39e03eced68018c1744b1
291deff3d997
```

**Two runs with the same manifest hash must produce identical numbers.** If they don't,
something outside the manifest is affecting results, and that's a bug worth finding — the
manifest is the checklist of everything that's supposed to be a cause.

`build_manifest(config, corpus, evalset, ...)` is what actually constructs one during a real
run; `evalset_hash()` hashes every question's id, text and gold spans/anchors so an edited
eval set is detected even if nobody bumped its version number.

## `contextgrid diff` and regression triage

`diff(before, after)` compares two manifests field by field (`config.*`, `resolution.*`,
`versions.*`, `seeds.*`, plus `corpus_hash`/`evalset_*`) and returns only what changed:

```
$ .venv/bin/python -c "from contextgrid.report.manifest import diff; print(diff(manifest_a, manifest_b))"
{'config.chunker': ('recursive:512', 'structural:800')}
```

`explain_diff()` is the same thing in prose, and it's what `contextgrid diff before.json
after.json` prints (`_diff()` in `src/contextgrid/cli/__main__.py`, wrapping
`Manifest.load()` + `explain_diff()`):

```
1 thing(s) changed between these runs:
  config.chunker: 'recursive:512' -> 'structural:800'
```

Two manifests with nothing different:

```
Nothing in the manifest changed, so these two runs should have produced identical numbers. If
they did not, something outside the manifest is affecting results and that is worth finding.
```

**When a metric drops, diff the manifest against the last passing run — the changed line is
the suspect.** That's the whole design goal: turn regression triage from an investigation
into a comparison. A changed `corpus_hash` or `evalset_hash` gets a specific extra line,
because everything else in the diff is meaningless until that's accounted for.

## The four export formats

Set via `report.formats` in the config (`ReportConfig.VALID_FORMATS` in
`src/contextgrid/config/schema.py`): `markdown`, `json`, `yaml`, `python`. Default is
`["markdown", "json"]`.

| Format | Function | What it's for |
|---|---|---|
| `markdown` | `results_to_markdown()` | A one-page report to paste into a team decision doc. |
| `json` | `results_to_json()` | Every configuration, every metric, every per-question score — for a sceptic to re-run the statistics themselves. |
| `yaml` | `winning_config_to_yaml()` | The winning config as a complete experiment file you can hand straight back to `contextgrid run`. Hand-written YAML (the core has no YAML-writing dependency, on purpose — see the core's dependency-free design in [install.md](install.md)). |
| `python` | `config_to_python()` | The winning config as runnable Python — not a template, it actually executes. |

### `markdown` — `results_to_markdown()`

Ordered the way somebody reads it, not the way it was computed: conclusion first, then
evidence, then caveats. "A report that opens with a methodology section does not get read,"
per the module docstring.

```
# support-tickets — retrieval configuration comparison

## What to use

markdown · structural:800 · tfidf · dense scored best on recall@5 at 0.760, across 2
configurations on 40 questions. ... It runs locally at no cost per query, answering at 33 ms
p95.

## Leaderboard

| Configuration | recall@5 | p95 ms | $/1k queries | Chunks |
|---|---:|---:|---:|---:|
| `markdown · structural:800 · tfidf · dense` | 0.760 | 33.0 | 0.0000 | 290 |
| `markdown · recursive:512 · tfidf · dense` | 0.710 | 31.0 | 0.0000 | 340 |

## Which decision mattered

- **Chunker**: `structural:800` was best, +0.050 over the worst value tried.

## Reproducing this

- Manifest: `99d9c948541a`
- Corpus: `deadbeefdead` (12 files)
- Eval set: `support-tickets` v3 (`abc123abc123`)
- Resolution: coverage at 0.5
- context-grid 0.9.0 on Python 3.13.0

Two runs with the same manifest hash must produce identical numbers.
```

The title carries the experiment's name — `name:` in an experiment config, passed as
`results_to_markdown(..., name=...)` or left on `results.meta["name"]`. Without one it falls
back to `# Retrieval configuration comparison`, which is fine for a single sweep and useless
for a directory of them, where every file would otherwise have the same title.

If the top two configurations aren't statistically distinguishable, a callout goes in right
after "What to use": *"The top two are not statistically distinguishable. Either is a
defensible choice on this evidence; pick on cost or latency instead."* A "Warnings" section
appears too, whenever anything in the run marks the comparison unsound.

### `json` — `results_to_json()`

Same data, structured for re-analysis rather than reading — includes `per_query` (every
question's score, for re-running the statistics), `by_type`, `failures`, and the full
manifest if one is passed. `_run_payload()` in `export.py` is the per-run shape.

### `yaml` — `winning_config_to_yaml()`

`winning-config.yaml` is a complete, runnable experiment file, not a listing of fields.
`contextgrid run out/winning-config.yaml` re-runs the winner on its own.

```yaml
# context-grid configuration
#
# manifest: 99d9c948541a
# corpus:   deadbeefdead (12 files)
# evalset:  support-tickets v3
#
# Re-run this file directly:  contextgrid run winning-config.yaml
#

name: support-tickets

corpus: /home/you/support/documents
evalset: /home/you/support/questions.jsonl

# One value per axis: this file names a single configuration, not a sweep.
grid:
  ingestion: null
  parser: markdown
  chunker: "structural:800"
  embedder: tfidf
  index: dense
  transform: null
  retrieval: null
  reranker: null
  candidates: 50
  generator: null

run:
  mode: ofat
  k: 10
  headline: "recall@5"
  seed: 0
  resolution_policy: coverage
  resolution_threshold: 0.5
  machine_usd_per_hour: 0.0
  cache: memory
  model: null
```

Four things about that shape are deliberate:

- **`corpus:` and `evalset:` are absolute.** The file is written into `report.out/`, normally a
  subdirectory of wherever the original config lived, and paths resolve against the config
  file's own directory. A relative path copied across would quietly point somewhere else.
- **Every axis takes a single value, not a list.** This file names one configuration.
- **`k` is in `run:` while `candidates` is an axis in `grid:`.** They read like a pair and the
  schema does not treat them as one.
- **There is no `report:` section.** The file usually sits inside the previous run's report
  directory; inheriting `report.out` would have a re-run overwrite the report, the results and
  this very file. Budgets are left out for the same kind of reason — `budget_seconds` and
  `budget_usd` exist to cut a sweep short, and there is only one configuration here.

Hand-written (`_yaml_value()`), not via a YAML library — the core installs with only `numpy`
and `pyyaml` (and `pyyaml` is for *reading* configs, not for this). Quoting kicks in whenever
a value contains YAML-special characters (note `"structural:800"` gets quoted because of the
`:`, while plain `markdown` doesn't).

`config_to_yaml()` still exists beside it and writes the flat block of pipeline fields on its
own — a record of what a `Config` held, with no corpus and nothing to run. It is what
`write_bundle()` falls back to when nobody tells it where the documents are.

### `python` — `config_to_python()`

```python no-run: this is config_to_python()'s output, a script for your own ./documents corpus
"""The winning configuration, as context-grid found it."""

import contextgrid as cg

# parent-document:4 · markdown · recursive:96 · ~relevance-feedback:3 · bm25 · lexical@20
# Any field not named below is at its default; `winning-config.yaml` spells out all of them.
config = cg.Config(
    ingestion="parent-document:4",
    chunker="recursive:96",
    embedder=None,
    index="bm25",
    retrieval="relevance-feedback:3",
    reranker="lexical",
    k=3,
    candidates=20,
)

corpus = cg.Corpus.from_dir("./documents")
pipeline = cg.build(config, corpus)

for chunk_id in pipeline.search("your question here"):
    print(chunk_id)
```

**A field appears whenever it isn't at its default, and the field list comes from `Config`
itself** — so a `Config` that grows a new axis exports it without anybody remembering to
update `config_to_python()`. An earlier version listed six field names by hand, fell behind
the dataclass, and exported the winner above with no `ingestion=` and no `retrieval=` line:
the snippet built plain chunking and plain search while `winning-config.yaml` beside it
described the real pipeline. Two files from one run, two different answers.

Anything left out is at its default, so `cg.Config(...)` puts it back — the snippet
reconstructs the winner exactly. The label comment on top says which configuration it is,
matching the leaderboard row in `report.md`.

## `write_bundle()`: all of it, in one directory

```python
from pathlib import Path


def write_bundle(
    results,
    directory,
    *,
    metric="recall@5",
    manifest=None,
    name=None,
    corpus=None,
    evalset=None,
) -> list[Path]: ...
```

Writes `report.md`, `results.json`, and — if there's a winner — `winning-config.yaml` and
`use_winning_config.py`, plus `manifest.json` if a manifest was passed. This is what
`contextgrid sweep --bundle ./out` calls after a real sweep (`_sweep()` in
`src/contextgrid/cli/__main__.py`), printing `wrote N files to ./out` when it's done.

Pass `corpus` — the documents the sweep ran over — and `winning-config.yaml` comes out as the
runnable config described above; pass `evalset` too and the re-run can be scored as well as
executed. Without `corpus` there is no path to write, so the file falls back to
`config_to_yaml()`'s flat listing. `contextgrid run` never hits that fallback: it goes through
`write_report()` in `src/contextgrid/config/loader.py`, which always has the experiment.

**A sceptic should be able to re-derive every number in the report from the bundle without
asking for anything else** — that's the design constraint behind writing all of these
together rather than just the markdown report alone.

## Regenerating the examples on this page

```bash
.venv/bin/python -c "
from contextgrid.pipeline import Config, Timings
from contextgrid.cost.model import CostBreakdown
from contextgrid.report.results import Results, RunResult
from contextgrid.report.manifest import Manifest, diff, explain_diff
from contextgrid.report.export import results_to_markdown, config_to_yaml, config_to_python, format_leaderboard

cfg_a = Config(parser='markdown', chunker='recursive:512', embedder='tfidf', index='dense')
cfg_b = Config(parser='markdown', chunker='structural:800', embedder='tfidf', index='dense')
run_a = RunResult(config=cfg_a, metrics={'recall@5': 0.71}, timings=Timings(query_ms=[22,25,19,31]),
                   cost=CostBreakdown(), chunk_count=340, scored_queries=40,
                   per_query={f'q{i}': 0.7 for i in range(40)})
run_b = RunResult(config=cfg_b, metrics={'recall@5': 0.76}, timings=Timings(query_ms=[24,27,21,33]),
                   cost=CostBreakdown(), chunk_count=290, scored_queries=40,
                   per_query={f'q{i}': 0.76 for i in range(40)})
results = Results(runs=[run_a, run_b], mode='ofat', cache_summary='6 of 8 lookups reused (75%)')
# ... build a Manifest per run, then diff(), explain_diff(), results_to_markdown(), etc.
"
```

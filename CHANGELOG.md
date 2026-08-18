# Changelog

Notable changes to context-grid. This project follows [Semantic Versioning](https://semver.org),
with the usual pre-1.0 caveat: the public API can still change in a minor release, and will be
called out here when it does.

## [Unreleased]

Nothing yet.

## [0.9.4] — 2026-08-18

0.9.3 was installed from PyPI into containers and driven by three independent readers, each
using only the published documentation site and none of them allowed to see the source. They
covered install and the quickstart, the ten axes, metric arithmetic, eval-set generation,
statistical significance, cost and latency, runs that stop early, and deliberately hostile
input. Fifteen things were wrong. Every one was reproduced before anything changed.

The theme is the same as last time and cuts deeper: **the tool answered questions it had not
asked.** A significance test that reported a different metric's numbers under the name you
requested. A confidence interval printed beside a score it was not computed from. A sample-size
floor derived from the wrong test. Each one is a number that looks like an answer and is not.

The drive also found something no amount of reading the source would have shown: the
documentation site had never redeployed. It had been serving 0.9.0 pages for two releases, and
`/reference/reports` — written, listed in navigation, linked from five other pages — had never
been published at all. Four of the fifteen findings were readers hitting that, not bugs.

### Fixed — numbers that answered a different question

- **`Results.significance(metric=…)` ignored the metric.** Asking about `recall@1` tested the
  headline metric's per-question scores and stamped `recall@1` on the result. On one drive's
  corpus the leaderboard gap at `recall@1` was `+0.682` while `significance` reported
  `0.318 [0.091, 0.545]` — an interval that does not contain the true answer. On another it
  called a near-threefold difference "not distinguishable". `RunResult` now keeps per-question
  scores for every metric it computed, and a metric it cannot test raises instead of answering
  about another one.
- **Leaderboard confidence intervals came from the headline metric.** A row printed for
  `recall@1` carried an interval resampled from `recall@5`, so the interval beside a number need
  not have contained it. `interval()` and `row()` now take the metric.
- **`compare()`'s per-question fields were the headline's too.** The aggregate was always right;
  the disagreement counts were not.
- **The sample-size floor was derived from the wrong test.** `minimum_detectable_difference`
  uses the formula for two independent proportions, but `significance()` runs a paired test, so
  "detects differences of 0.45 and above" was not true as written — a real 0.45 gap at twenty
  questions came back undetectable. The same claim appeared in the "roughly 63 questions"
  sentence the documentation tells you to paste into a pull request. All three now say the
  estimate assumes an unpaired test and is a floor rather than a guarantee.

### Fixed — results that looked measured and were not

- **A metric no run computed printed as `0.000`.** With evidence that resolved nowhere, the
  scores were absent from `results.json` entirely, yet the terminal table and `report.md` filled
  the column with `0.000` and the summary named a winner "at 0.000". The Python `leaderboard()`
  already left it out, as documented; the two renderers did not. They now print `NOT_MEASURED`,
  and the summary says there is no ranking rather than inventing one.
- **An empty eval set was a green build.** Every configuration was built, indexed, scored
  nothing, and `run` exited `0` with a full leaderboard of zeros. It now exits `1` whenever
  nothing was scored. The warning also blamed the parser or the eval set's quotes for evidence
  it could not resolve; when the eval set has no questions, it says so and names the file.
- **The only reason a sweep failed was lost by the documented redirect.** `contextgrid run …
  > out.txt` with a missing extra kept `No configurations were run.` and discarded the
  `MissingExtraError` naming the `pip install` command, because that went to stderr alone. The
  reason is now on stdout under the empty table, as the budget case already did.
- **`contextgrid diff` hid a partial run.** Comparing a complete run against one stopped by
  `budget_seconds` printed "Nothing in these two manifests is different" and went on to say the
  two should have produced identical numbers, while the second manifest's `notes` recorded
  `PARTIAL RUN: 1 of 3 configurations ran`. `notes` is now compared, and an unfinished run is
  named before anything else.

### Fixed — differences that were not differences

- **A bundle written by `sweep` recorded no seed.** `contextgrid run` wrote `"seeds":
  {"run": 0}` and `contextgrid sweep --bundle` wrote `{}`, so `contextgrid diff` reported
  `seeds.run: 0 -> None` between two runs that used the same seed. The one command whose job is
  saying what changed named a change that never happened.
- **Two spellings of one configuration read as a change.** `diff` reported
  `config.retrieval: 'simple' -> None` although `get_retriever(None)` returns `SimpleRetrieval()`
  and every other surface — the leaderboard, the report, the bundle — treats them as one row.
  `diff` and `Manifest.hash()` now fold the same aliases the matrix folds, and still print what
  each manifest actually recorded.

### Fixed — messages that were wrong about themselves

- **An unreadable directory was reported as empty.** A directory the process could not list was
  described as holding "no files at all", with advice to rename files it had never seen. It now
  leads with the permission failure and drops the pattern list it could not have matched.
- **`contextgrid validate` failed its own tolerance and exited `0`.** It printed "recall@10
  differs by +0.100, outside the 0.05 tolerance" and "a problem with our scoring", then returned
  success — a green build for a failed validation, the exact trap the CLI page argues against.
  It now exits `1`, and the page documents its exit codes.
- **`diagnose()` accepted input `evaluate()` rejects.** A ranking containing the same chunk
  twice pushed the evidence down a rank and turned a success into "just outside the top 5"; a
  cut-off of `0` or below produced "retrieved at rank 1, just outside the top 0". Both now raise
  the same errors, in the same words, that `evaluate()` already did.
- **`generate(sample=0)` returned every chunk.** `sample=-1` returned none and `sample=0`
  returned all, from a check that could not tell zero from absent. Zero now means zero, and a
  negative sample raises.

### Added

- **A docs workflow, because nothing was watching the documentation site.** `.github/workflows/
  docs.yml` checks that every page in `docs.json` exists, that every internal link resolves, and
  that the version printed in the docs matches the package — then fetches the published site and
  fails when a page in navigation 404s or the published version is behind. The source checks
  passed the entire time the site was broken; only fetching the real site catches it, which is
  why that job also runs on a schedule.

## [0.9.3] — 2026-08-17

0.9.2 was installed from PyPI into containers and driven by seven independent readers, each
using only the published documentation site and none of them allowed to see the source. They
covered install and quickstart, the ten axes, eval sets and reports, statistical significance,
cost and latency, metric arithmetic, and deliberately hostile input. Thirty-three things were
wrong. Every one was reproduced before anything changed.

The theme is narrower than last time, and worse: **the tool kept telling you things it could not
know.** A score above 1.000. A failure diagnosis contradicting itself in the same object. A
leaderboard that stopped early and did not say so. Three separate warnings blaming the parser
for an eval set's mistake.

### Fixed — numbers that were wrong

- **`recall` and `map` could exceed 1.0.** A ranking containing the same chunk twice was scored
  as two hits, so `recall@3` came back `1.5` where the answer is `0.5`. Metrics now count a
  chunk once, and `evaluate()` rejects a repeated id outright rather than scoring it — a
  retriever returning duplicates has a bug, and quietly repairing it would hide that.
- **`diagnose()` sent you to fix the wrong stage.** Evidence that was in the index but never
  retrieved was labelled `fp1_missing_content`, whose remedy reads "the evidence is not in this
  index at all. No retriever can fix this" — while the `detail` field on the same object said
  the opposite. It is now `fp3_not_in_context`, and the two agree.
- **A question with no relevant chunk was scored two different ways.** `{}` excluded it from the
  average; `{"c1": 0}` counted it as a zero and dragged every score down. Both now mean the same
  thing. A cut-off below 1 is rejected instead of quietly using Python's negative slicing.
- **`Comparison.can_support()` disagreed with the number printed beside it.** The summary said
  "detects differences of 0.40 and above" while `can_support(0.40)` returned `False`, because
  the printed figure was rounded down from 0.404. The printed number is now rounded up, so the
  sentence is true as written.

### Fixed — results that looked complete and were not

- **A budget-stopped sweep now says so where you can see it.** A run that exhausted
  `budget_seconds` printed a header saying "18 to run", eleven rows, and exited 0, with the only
  warning on stderr — so the documented `contextgrid run … > leaderboard.txt` kept the
  misleading half and discarded the caveat. The note is now on stdout above the table, in
  `summary()`, and in the manifest.
- **One output directory could describe two different experiments.** Running again with fewer
  formats left the previous run's `report.md` and `winning-config.yaml` in place beside the new
  `manifest.json`. `use_winning_config.py` then handed you the earlier experiment's
  configuration. A bundle now clears the files it owns, and warns about any it removed and did
  not replace.
- **`write_bundle` never wrote the manifest** its own docstring promised, which left
  `contextgrid diff` — documented as reading "two `manifest.json` files from earlier bundled
  runs" — with no input at all. It now builds one when given the corpus and eval set, matching
  `contextgrid sweep --bundle` to the hash.
- **A folded `widened` arm read as a tie it never earned.** When `widened` provably returns what
  plain search returns, the matrix folds it — correctly — but said nothing, so the row looked
  like an independent arm that drew. It now carries `[~widened:factor=8 ran as plain search]`
  and an `arm_not_measured` warning.
- **`Lab(cache=DiskCache(...))` never wrote to disk.** An empty `DiskCache` is falsy and
  `cache or MemoryCache()` swapped it out, so every first run silently used memory and the
  directory never stopped being empty. Documented as a known defect for two releases; now fixed,
  and the warning block describing it is gone.

### Fixed — blame pointed at the wrong thing

- **Three warnings claimed a fact about the parser they could not know.** `anchor_not_found`,
  `gold_span_unreachable` and its per-configuration twin all asserted "a measurement of the
  parser, not the eval set" — for an invented quote, a wrong `source_id`, or an out-of-range
  `occurrence`, all of which are the eval set's doing. They now say what they can defend and
  point at the per-question warnings that can tell the causes apart.
- **`one_chunk_per_document` blamed the chunker for an ingestion strategy's work.**
  `parent-document` groups small chunks back into passages by design, which made the counts
  equal; the warning then told you to sweep smaller sizes, which cannot help. It now names
  whichever axis actually collapsed them.
- **An out-of-range `occurrence` was reported as the evidence being absent.** It now says the
  quote appears N times and the anchor asked for one past the end, numbered from 0.

### Fixed — input the tool accepted and should not have

- **`contextgrid check` passed a config the sweep could not finish.** A read-only output
  directory validated clean, then the whole matrix ran and died writing `manifest.json`. It is
  now caught before anything is spent.
- **`contextual` ingestion with no model** validated, ran, and scored as though it had enriched
  anything. It now fails like the transforms and generators already did.
- **`grid.candidates` below 1** validated and reached the leaderboard as `lexical@-3`.
- **A binary file with a `.md` extension** was indexed as ten chunks of replacement characters
  with no warning, and non-UTF-8 text was silently mangled — after which `anchor_not_found`
  blamed the parser. Both are now reported per file, naming the file and what is wrong with it.
- **A UTF-8 BOM deleted the first heading** of a Markdown document, silently, breaking every
  heading-aware chunker on files Windows editors write by default.

### Fixed — failures that took more with them than they should

- **One unbuildable configuration killed the whole sweep.** Every row already measured was
  discarded. A failing configuration is now dropped from the leaderboard with a
  `configuration_failed` warning naming it, and the sweep finishes. It is left out rather than
  scored zero: a zero is a measurement, and nothing measured it.
- **Two concurrent runs destroyed each other's cache.** On a cold cache the `.tmp` → `.pkl`
  rename raced and one process died, losing its sweep. Writes are now atomic per process.
- **`Ctrl-C` printed a fifty-one line traceback**, breaking the documented promise that no
  command ever shows one. It now prints one line and exits 130 — and says whether the work
  survives, which depends on `run.cache` and is only claimed when true.
- **`Corpus.from_dir` leaked a raw `PermissionError`** from the public API, outside the
  documented exception hierarchy, so the handler `/reference/errors` tells you to write never
  fired.

### Changed

- `BUDGET_REACHED` meant three different things — a sweep stopping, a model with no published
  price, and a plugin with no cost ceiling — while the documentation told you to filter on it to
  detect a stop. It now means only the stop; `MODEL_NOT_PRICED` and `NO_COST_CEILING` are their
  own codes.
- `budget_usd` was non-deterministic: it charged machine time, which falls with a warm cache, so
  the same budget bought a different number of configurations each run. It now charges metered
  token spend. The reported bill is unchanged; only the limit stopped moving.
- `machine_usd_per_hour` was accepted and appeared in no cost readout. It is now reported as
  `machine_usd`, and the summary stops saying "at no cost" when a rate is set. The `$/1k` column
  still excludes it on purpose — build cost and serving cost must not be summed.
- `contextgrid sweep` gained `--budget-usd`, which `--budget-seconds` already had.
- `contextgrid plugins` marks a plugin whose optional package is missing and prints the
  `pip install` line that fixes it, which is what the page always claimed it did.
- `--chunker` and its four siblings document that they are repeated once per value.

### Documentation

- `contextgrid evalset` stopped calling questions "answerable" in three more places. The word
  means *can be scored*; these counts only know an anchor is attached. 0.9.1 fixed one of four.
- `recall_against_exact`, `coverage_fraction`, `score_answer` and `AnswerScore` are reachable
  from `/scoring/metrics`. Each was already documented, on a page a reader looking for metrics
  would never open.
- A table on `/reference/cli` broke in 0.9.2 when a new section was inserted between two of its
  rows.

## [0.9.2] — 2026-08-17

Found the same way as 0.9.1: 0.9.1 was installed from PyPI into a container and driven using
only the documentation site, with no access to the source. Two of these change numbers you would
have acted on.

### Fixed

- **A leaderboard no longer reports a score for a chunker that never split anything.** If every
  document came out as a single chunk — the normal result of sweeping a chunk size larger than
  your documents — the run reported a clean `recall@5` of `1.000` and said nothing. That number
  was ranking documents, not passages, so the chunker axis was measuring a thing it never
  touched. The new `one_chunk_per_document` warning says so on the terminal, in the written
  report and in `results.warnings`. `contextgrid profile` already gave this advice; nothing on
  the path people actually run pointed at it.
- **`Comparison.verdict()` no longer contradicts its own numbers.** `distinguishable` requires
  two things — `p_value < alpha` **and** a confidence interval clear of zero — but the sentence
  only ever explained the second. A comparison with an interval of `+0.062 to +0.500`, nowhere
  near zero, was described as "consistent with no difference at all". It now names whichever
  condition actually failed. The wording for the genuinely-inconclusive case is unchanged.
- **A bad parameter in a spec string now gives an error rather than a Python internal.**
  `chunker: recursive:banana` reported `unsupported operand type(s) for //: 'str' and 'int'`, and
  so did `fixed:`, `sentence:`, `structural:`, `overlap=` and the embedder's `hash:`, each with a
  different and equally unhelpful message. A misspelled plugin *name* and a negative size were
  both reported properly; only a non-numeric value fell through. All of them now read
  `chunker 'recursive:banana': size must be a whole number, got 'banana'`. `contextgrid check`
  leaked the same internals and is fixed with it.
- **`contextgrid sweep` fails before it starts running, not part-way through.** A bad spec was
  reported only after the plan and the first `[1/1]` progress line had already been printed.

### Documentation

- The command-line flags say how to sweep more than one value. `--chunker` and its four siblings
  take the flag repeated once per value; nothing said so, and a comma-separated list gave an
  error about `key=value` that pointed nowhere near the real problem. Both `--help` and the CLI
  reference now spell it out, and the comma error suggests the right form.
- The quickstart shows `results.warnings`. Its worked example is exactly the shape that triggers
  `one_chunk_per_document`, and `summary()` does not carry warnings, so the page would otherwise
  have gone on showing a recall of `1.000` with no hint of what it meant.
- Stale sample output refreshed: version strings that still read `0.9.0` across five pages, and
  the extension example in `docs/internals/extending.md`, whose numbers no longer matched a real
  run.

## [0.9.1] — 2026-08-17

Everything here was found by installing 0.9.0 from PyPI into a container and using it with no
access to the source — the way anyone else meets it. The package worked; these are the places it
knew the right answer and did not hand it over.

### Fixed

- **A missing optional dependency now raises `MissingExtraError`.** `faiss`, `usearch` and
  `psycopg` all raised `IndexBuildError`, which inherits `ValueError` — so the
  `except MissingExtraError` the documentation hands out, and the `except ImportError` it says
  also works, both missed it. The messages were already right; the type was not. Only `faiss` was
  reported, but the same bug was in the other two and all three are fixed.
- **`anchor_normalised` now reaches the terminal.** When a quote matches only after whitespace is
  collapsed — the normal case for hard-wrapped Markdown — the CLI said nothing, because the
  warning was `INFO` and `INFO` is filtered out whenever a run produced results. The hard anchor
  failures printed loudly beside it, so silence read as "your evidence matched literally". It is
  now `CAUTION`, which is what it always was in substance.
- **`contextgrid evalset` no longer calls a question "answerable".** It never reads a corpus, so
  it cannot know whether an anchor resolves; it reported "14 questions (14 answerable)" for a set
  containing evidence that appears in no document. It now says "N with evidence, unchecked against
  a corpus". `contextgrid run` is where evidence meets documents, and it still says so there.
- **`contextgrid init` no longer cites a page that does not exist.** The generated config pointed
  at `extending.md`; the real page is `concepts/plugins`. A dead reference in a generated file is
  the worst kind, because it gets copied forward. The `map` metric's description had the same
  problem and lost its stale pointer too.

### Changed

- The `Documentation` URL in the package metadata points at
  [context-grid.mintlify.site](https://context-grid.mintlify.site) rather than the contributor
  docs on GitHub.

## [0.9.0] — 2026-08-17

First public release. Everything below is the state at the point the project went open source
and to PyPI, rather than a diff against a previous version — there isn't one.

### What it does

Sweeps **ingestion × parser × chunker × embedder × index × transform × retrieval × reranker ×
candidates × generator** over your own documents, and ranks the results on quality, latency and
cost. One YAML file, or the Python API.

All ten axes are shipped and measured. 58 plugins across them: 8 parsers, 12 chunkers, 5
embedders, 7 indexes, 5 rerankers, 6 query transforms, 5 retrieval strategies, 8 ingestion
strategies and 2 generators.

### The parts worth knowing about

- **The offsets guarantee.** A chunk always knows which characters of which source document it
  came from. That is what makes comparing two chunkers — or two parsers — a valid thing to do
  rather than a vibe.
- **Anchors, not chunk ids.** Ground truth is a quoted sentence, so it survives re-parsing. This
  is what makes the parser axis measurable at all, and it means a parser that mangles a table
  fails to resolve its own evidence — which is itself the measurement.
- **`is_the_winner_real()`.** Paired bootstrap and a randomisation test, so a leaderboard gap
  that is noise gets called noise.
- **Reproducibility, checked rather than claimed.** Two runs of one config produce byte-identical
  scores, confidence intervals and significance verdicts. Only timings and `created_at` differ.
- **A dependency-free core.** `pip install context-grid` brings `numpy` and `pyyaml`. Everything
  heavy — PDF engines, faiss, torch, hosted models — lives behind an extra.

### Not built yet

RAPTOR and GraphRAG. Deliberately, not accidentally — see `docs/roadmap.md`.

### Packaging

- MIT licensed, declared PEP 639 style.
- `py.typed` ships, so the annotations are usable downstream. The package is `mypy --strict`.
- Python 3.10 through 3.13.
- Eleven optional extras. `parse-marker` must be installed alone; `pyproject.toml` explains why
  at length.

### Documentation

- User documentation at `docs-site/`, 38 pages.
- Contributor documentation at `docs/`.
- `docs/drives/` records five end-to-end drives where the built package was installed clean and
  the documentation followed as a stranger would. 40-odd disagreements between the docs and the
  tool were found that way and fixed. It is the most honest thing in the repository.

[Unreleased]: https://github.com/sushantgundla/context-grid/compare/v0.9.3...HEAD
[0.9.0]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.0
[0.9.1]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.1
[0.9.2]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.2
[0.9.3]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.3

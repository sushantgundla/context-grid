# Changelog

Notable changes to context-grid. This project follows [Semantic Versioning](https://semver.org),
with the usual pre-1.0 caveat: the public API can still change in a minor release, and will be
called out here when it does.

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/sushantgundla/context-grid/compare/v0.9.2...HEAD
[0.9.0]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.0
[0.9.1]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.1
[0.9.2]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.2

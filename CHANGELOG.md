# Changelog

Notable changes to context-grid. This project follows [Semantic Versioning](https://semver.org),
with the usual pre-1.0 caveat: the public API can still change in a minor release, and will be
called out here when it does.

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/sushantgundla/context-grid/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/sushantgundla/context-grid/releases/tag/v0.9.0

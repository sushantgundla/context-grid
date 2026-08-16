# Contributing

Thanks for looking. This is a small project with a strong opinion about measurement, and the
bar for a change is the same as the bar for a number it prints: it has to be checkable.

## Setting up

```bash
git clone https://github.com/sushantgundla/context-grid
cd context-grid
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

`[dev]` installs the toolchain and most of the feature extras. Three packages are deliberately
not in it — `docling`, `marker-pdf` and `psycopg`. `marker-pdf` cannot share an environment with
`docling` (see the note in `pyproject.toml`), and the other two are heavy or need a running
server. Install them yourself if you are working on those code paths.

## Before you open a pull request

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest
```

All four run in CI on Python 3.10 through 3.13. There is one extra gate: the span algebra and
the resolver in `core/` and `score/` are held to 95% coverage, because a bug in either is
invisible in every number the tool prints.

Tests that touch the network, a model or an API are marked `integration` and stay out of the
default run.

## What a good change looks like

**A bug fix starts with a failing test.** Not because of ceremony — because a bug that was
never expressed as a test is a bug that comes back. Write the test, watch it fail, then fix it.

**A number in the documentation is a claim.** If you change behaviour that any documented
example prints, re-run the example and paste what it really says. Several of the findings in
`docs/drives/` are output that went stale after an edit, and each one cost a reader their trust
in the page.

**Say the awkward thing.** The comments in this codebase explain why something is the way it is,
including when the answer is "this is worse than it looks and here is what it cost". That is the
house style. A comment that only restates the code is noise; a comment that records a decision
someone would otherwise re-litigate is worth ten of them.

**Keep the core dependency-free.** `pip install context-grid` pulls `numpy` and `pyyaml` and
nothing else. Anything heavier belongs behind an extra, and reaching for it without the extra
installed must raise `MissingExtraError` naming what to install.

## Adding a plugin

Parsers, chunkers, embedders, indexes, rerankers, transforms and the rest are registry entries.
`docs/internals/extending.md` walks through adding one, and `docs/internals/conformance.md` lists
what a new plugin has to satisfy before it can be trusted in a comparison — chief among them the
offsets guarantee: a chunk always knows which characters of which source document it came from.
A plugin that breaks that makes every comparison it appears in meaningless.

## Documentation

There are two trees and they have different readers.

- `docs/` is for people working **on** context-grid. Internals, protocols, design notes.
- `docs-site/` is the published user documentation, in Mintlify MDX. It is for people who ran
  `pip install context-grid` and have never seen this repository. Never point them at a path
  under `src/`.

`docs/drives/` holds the reports from documentation drives — sessions where someone installs the
built package and follows the docs as a stranger, recording every place the tool and the docs
disagree. Reading one is the fastest way to understand what this project considers a defect.

## Reporting a bug

Include the config or the Python that reproduces it, the full error, and `contextgrid --version`.
If a documented example is wrong, say which page and quote what it printed for you — that is the
most useful bug report this project gets.

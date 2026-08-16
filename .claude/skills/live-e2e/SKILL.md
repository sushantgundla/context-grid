---
name: live-e2e
description: End-to-end test of the LIVE released package, not the working tree — installs context-grid from PyPI into a Docker container and drives it using only the published documentation site, with no access to this source. Verifies every finding before anything is changed, then sends the same agent back to fix what survives. Use when asked to test the live package, test what shipped, test the real release end to end, run a live e2e, check the published version, drive the package as a real user, verify a release, or test a subset of features against the docs.
---

# Live end-to-end: test what shipped

> **Sibling skill.** `docs-e2e-drive` tests the documentation against the **local** checkout.
> This one tests the **released** package from PyPI. Same discipline, different subject — and
> this is the only one that can catch a packaging bug, because it is the only one that installs.

Test what people actually install, not what is in the working tree. They are different things,
and the gap between them is where the embarrassing bugs live.

## What this is for

`pytest` proves the code does what the tests say. This proves the *published package* does what
the *published documentation* says, to someone who has neither. Those users cannot read `src/`,
cannot run the test suite, and will not guess an API name that is only in a docstring.

Every finding this skill has produced came from that gap. 0.9.0 shipped with a missing-extra
error that raised a type the docs told people to catch — the tests never caught it because they
import from the source tree, where the extras are installed.

## The three rules

**1. The container is the only place work happens.** Nothing runs against the local checkout or
`.venv`. The package comes from PyPI.

**2. The agent never reads this repository.** Not `src/`, not `tests/`, not `docs/`, not
`docs-site/`, not the README. Its only manual is the documentation *site*. If it wants to look
at the source, that impulse is the finding — the docs failed to say something.

**3. Nothing gets fixed until it is verified.** Agents report confidently and are sometimes
wrong. Reproduce every finding yourself, in this session, before a line changes. Two of the four
findings in the last drive were sharper than reported once reproduced, and one earlier drive
produced a "still broken" report about a bug that had already been fixed.

## Running it

### Step 1 — find out what is actually published

```bash
curl -s https://pypi.org/pypi/context-grid/json | python -c "
import json,sys; d=json.load(sys.stdin)['info']
print(d['version'], '|', d['requires_python'], '|', d['project_urls'])"
```

Use the real published version, not the version in `pyproject.toml`. They differ whenever a
release is pending, and testing a version nobody can install wastes the whole run.

Confirm Docker is up: `docker info >/dev/null 2>&1 && echo ok`.

### Step 2 — spin up one agent

One agent, not a fleet. It has to build a mental model of the tool as a new user and carry it
through the whole session; splitting that across agents loses exactly the continuity being
tested. It also has to be the one that fixes what it found, in step 5.

Give it, in the prompt:

- The PyPI URL — `https://pypi.org/project/context-grid/` — and `pip install context-grid`
- The documentation site — `https://context-grid.mintlify.site` — as its **only** manual
- The three rules above, stated as rules
- A scratch directory on the host, under the session scratchpad, never inside the repo
- Which features to cover (see below)
- The reporting style block from `~/.claude/CLAUDE.md`

Tell it to invent its own realistic test data rather than handing it a fixture. Building the
corpus and eval set *from the documentation's instructions* is itself part of the test: if the
docs never say what shape the questions file takes, that is a finding, and a supplied fixture
hides it.

### Step 3 — choose the subset

Full coverage is slow and mostly re-tests what has not changed. Pick by what moved:

| Ask | Cover |
|---|---|
| "test the release" | install, quickstart, one real sweep, the CLI, significance |
| "test everything" | all of the above plus every axis page, eval-set generation, exports, errors |
| after a change to one area | that area, plus quickstart — quickstart is the page everyone reads |

Anything needing an API key, a GPU or Postgres cannot run in a bare container. The agent must
say it skipped those, not invent credentials.

### Step 4 — verify before believing

For each finding, reproduce it yourself. Ask three questions:

- **Is it real?** Run the exact command. Some findings evaporate.
- **Is it worse than reported?** The last drive reported a warning as "not on the terminal". It
  was also absent from the written bundle, while the same warning was visible from the Python
  API — a sharper and more embarrassing bug than the one reported.
- **Is it wider than reported?** The agent found the missing-extra bug in `faiss` because that
  is what it tried. The same bug was in `usearch` and `psycopg`. Always grep for the pattern.

A finding is a *disagreement*, not automatically a code bug. Roughly a fifth get settled by
changing the documentation, because the documented behaviour was the better one. Decide which
side is wrong before touching either.

### Step 5 — send the same agent back to fix

Use `SendMessage` with the agent's name. It still has the container, the corpus and the context;
a fresh agent would rebuild all of it and know less.

Tell it explicitly:

- The findings that survived verification, and the ones that did not, and why
- That a failing test comes first for each fix, then the fix
- That it may now read the repository — this phase is source work, and the blindfold was only
  for the drive
- To run all five CI gates before reporting: `ruff check .`, `ruff format --check .`, `mypy`,
  `pytest`, and the coverage floor on `core/` and `score/`

### Step 6 — release, if the fixes warrant it

Bump the version in **both** `pyproject.toml` and `src/contextgrid/__init__.py` — the publish
workflow refuses a tag that disagrees with them. Add a CHANGELOG entry that says what a user
would have hit, not what the diff shows. Then tag, and let the workflow publish.

PyPI never allows a version to be reused, so the metadata in a released version is frozen
forever. 0.9.0 permanently carries a `Documentation` URL pointing at the wrong docs, because
that was noticed an hour too late.

## Traps that have produced false findings

These are earned, each from a real wrong report.

- **Piping reorders stdout against stderr.** Redirect to separate files when you care about both.
  A finding was raised and withdrawn over this.
- **`$?` after a pipeline reports the wrong command.** Capture the exit code directly.
- **A warm cache hides a fix.** Use a fresh directory when timing or testing repeat behaviour.
- **Determinism inside one process passes against a bug that only shows across processes.** Run
  two separate processes.
- **A doctest-style block is not a script.** Blocks beginning `>>>` do not compile with
  `compile()`; extract them with `doctest.DocTestParser` before judging them broken. This
  produced sixteen false "syntax error" findings in one sitting.
- **Nothing changed, so nothing deployed.** Before concluding a build or deploy step is broken,
  confirm there was actually something new for it to do.

## What a good report looks like

Findings worst first, each with the exact command, the real output, and what the page led the
reader to expect. Plus the things documentation never covers and every first user feels: how
long the install took, how big it is, whether the error messages made sense, and whether the
agent would trust the numbers it was shown.

That last question is the most valuable one in the whole exercise. The last drive answered "not
the top rows" — and was right, for a reason no test would have caught: the chunk size made one
chunk per document, so recall pinned at 1.000 and the leaderboard measured nothing.

---
name: docs-e2e-drive
description: Drives the context-grid tool end to end using only its public documentation, as a new user would, and reports every place the docs and the tool disagree. Never reads src/ or tests/. Use when asked to test the documentation, verify the SDK or CLI against its docs, run a docs-driven or docs-only end-to-end check, drive the tool as a real user, hunt for bugs without reading source code, or check whether a round of fixes introduced regressions. Bundles a fixture corpus and eval set, a fixture verifier, and the capture traps that have produced false findings before.
---

# Docs-driven end-to-end drive

Test context-grid the way a new user meets it: **read the documentation, do exactly what it
says, and record every place the tool disagrees with it.**

## The one rule

**Do not read `src/`. Do not read `tests/`.**

Reading the implementation tells you what the tool *does*, which is precisely the knowledge a
new user does not have. Every finding this skill has ever produced came from not having it.

If a command fails, read the error and the docs. Nothing else. Diagnosing the cause in source
is a separate job, in a separate session.

If you catch yourself thinking "let me just check the source to see what it expects" — that
thought is the finding. Write it down instead.

Do not read earlier reports (`docs/drives/`) either. Knowing where the last drive looked
is what stops you looking elsewhere.

## The scenario

You are the docs lead at **Northwind Cloud**. Support keeps escalating tickets the help centre
already answers, so you want retrieval over the help centre — and before building anything, you
want to know which pipeline configuration actually works on *these* documents.

The fixture is in `data/`, beside this file:

- `data/documents/` — eight help-centre articles: tables, code blocks, a pricing page, and two
  deliberately near-duplicate deletion policies that retrieval has to tell apart.
- `data/questions.csv` — thirteen questions in the loose column names the docs say are accepted,
  as a subject-matter expert would hand them over.
- `data/questions.jsonl` — the same set in the native format.

Small on purpose. You are testing the tool, not the corpus.

**Two entries are deliberate traps, not mistakes:**

| Item | What it is | Why |
|---|---|---|
| `nw12` | A question with no evidence at all | The docs allow it. Watch how the tool reports it — blaming the pipeline for missing ground truth is a bug |
| `nw13` | A quote spanning a soft line wrap, so it is not verbatim in the file | What every user gets copying a sentence out of wrapped Markdown. Watch whether the anchor resolves and whether the user is told |

## Before you start

Work outside the repository. Doc snippets create files, and a drive that litters the checkout
cannot tell its own mess from the tool's.

```bash
cp -r .claude/skills/docs-e2e-drive/data /tmp/drive && cd /tmp/drive
```

Then prove the fixture before trusting any result from it:

```bash
python .claude/skills/docs-e2e-drive/scripts/verify_fixtures.py
```

Exit 0 means every anchor resolves and any later mismatch belongs to the tool. Exit 1 means fix
the fixture first — otherwise you will report your own typos as tool bugs.

## How to look

**Read [references/measuring-honestly.md](references/measuring-honestly.md) before running
anything.** It is short, and it covers the traps that have produced false findings in real
drives: piped output reordering stdout against stderr, `$?` after a pipeline reporting the wrong
command, caches hiding a fix or inventing a bug, and determinism that must be tested across
processes to be tested at all.

It also covers the judgement that matters most: **a finding is a disagreement, not automatically
a code bug.** About a fifth of real findings were settled by changing the documentation, because
the documented behaviour was the worse behaviour. Report which half you think is wrong, and why.

## Where to look

**[references/the-route.md](references/the-route.md)** has the two lanes — the command line, and
the library plus recipes and reference pages — with the page that promises each behaviour, and a
list of surfaces that have regressed before.

Treat it as where to start, not where to stop. A drive that only walks the list finds only what
the last drive found. The most valuable findings have come from going off it.

## Recording findings

One row per disagreement:

| Field | Content |
|---|---|
| What the doc says | Quote it, with `file.md:line` |
| What happened | The real output, verbatim |
| Severity | **blocker** — a documented path cannot be completed · **wrong** — it runs, and contradicts the doc · **cosmetic** — harmless drift |

Also findings: behaviour that works but no page mentions, a flag in `--help` that nothing
documents, and two pages that contradict each other.

**Do not fix anything. Do not touch `src/`. Report only.**

## The report

Write to `docs/drives/drive-N.md`, taking the next free number when earlier ones
exists — never overwrite another drive's findings.

Structure:

1. **Lead with the count and the worst finding**, in the first two sentences.
2. **The findings**, worst first.
3. **What the documentation gets right** — with the evidence. A report listing only faults says
   nothing about coverage, and a clean area is a real result.
4. **What was skipped, and why.** Anything needing an API key, Docker, a live database or a
   model download. A skipped area presented as covered is worse than an admitted gap.

If you find nothing, say so plainly and show what you ran. A clean drive with evidence behind it
is a useful answer; a clean drive without it is not.

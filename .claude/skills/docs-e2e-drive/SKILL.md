---
name: docs-e2e-drive
description: Drive context-grid end to end using only its public documentation, as a real user would, and report every place the docs and the tool disagree. Use when asked to test the docs, verify the SDK against its documentation, run a docs-driven end-to-end check, or find bugs without reading source code.
---

# Docs-driven end-to-end drive

Test context-grid the way a new user meets it: **read the documentation, do exactly what it
says, and record every place the tool disagrees with it.**

## The one rule

**Do not read `src/`. Do not read `tests/`.**

The whole point is to find gaps a real user would hit. Reading the implementation tells you
what the tool *does*, which is precisely the knowledge a new user does not have. If a command
fails, you may read the traceback and the docs — nothing else. Diagnosing the cause in source
code is a separate job, done after this one, in a separate session.

If you catch yourself thinking "let me just check the source to see what it expects", that is
the bug. Write it down instead.

## The scenario

You are the docs lead at **Northwind Cloud**. Support keeps escalating tickets that the help
centre already answers, so you want to put retrieval over the help centre. Before building
anything you want to know which pipeline configuration actually works on *these* documents.

You have:

- `data/documents/` — eight help-centre articles, Markdown, with tables, code blocks,
  a pricing page, and two deliberately near-duplicate articles.
- `data/questions.csv` — twelve questions a support engineer wrote in a spreadsheet, in the
  loose column names `docs/guide/evalsets.md` says are accepted.
- `data/questions.jsonl` — the same set in the native format, with `meta.reviewed` set.

The corpus is small on purpose. You are testing the tool, not the corpus.

Two things in the data are deliberate traps, not mistakes:

- **`nw12` has a question and no evidence.** `evalsets.md` says this is allowed and makes the
  question unanswerable. Watch how it is reported.
- **`nw13`'s quote spans a soft line wrap** in `sso-setup.md`, so it does not appear verbatim
  in the raw file. A real user copying a sentence out of a wrapped Markdown file hits this
  constantly. Watch whether the anchor resolves and whether the user is told.

### Step 0 — copy the data out first

```bash
cp -r .claude/skills/docs-e2e-drive/data /tmp/nw && cd /tmp/nw
```

**This copy is itself finding #1.** A corpus under a dot-directory loads zero files and the
error says the directory is empty. Reproduce it once before copying, then work from the copy.

## The drive

Work through these in order. After **every** command, compare what happened against what the
documentation said would happen, and log it (see "Recording findings").

### 1. Meet the tool

| Step | Doc that promises it |
|---|---|
| `contextgrid --help` — nine subcommands, matching the usage block | `docs/guide/cli.md` |
| `contextgrid plugins` and `contextgrid plugins --family chunker` | `docs/guide/cli.md#plugins` |
| Check the four axes `plugins` admits it does **not** cover | `docs/guide/configuration.md` |

### 2. The eval set, as an SME hands it over

| Step | Doc |
|---|---|
| Read `data/questions.csv` with `read_csv`, loose column names and all | `evalsets.md#csv` |
| Confirm rows without a quote become unanswerable, not errors | `evalsets.md#csv` |
| `write_jsonl` it, `read_jsonl` it back, check the round-trip keeps anchors | `evalsets.md#jsonl` |
| `contextgrid evalset data/questions.jsonl` — size, % reviewed, detectable difference | `cli.md#evalset` |
| Confirm `meta.reviewed: true` moves the reviewed percentage | `evalsets.md#eval-set-quality` |

### 3. Look before you leap

| Step | Doc |
|---|---|
| `contextgrid profile data/documents` | `cli.md#profile` |
| `contextgrid init` a config pointed at the data | `cli.md#init` |
| `contextgrid init` again without `--force` — must refuse, exit 1 | `cli.md#init` |
| `contextgrid check` the generated config | `cli.md#check` |

### 4. The sweep a real user would want

Sweep the axes that matter for a help centre — chunker size, and whether lexical beats
dense on short factual articles:

```yaml
grid:
  chunker: [recursive:256, recursive:512, sentence:3]
  embedder: [tfidf, null]
  index: [dense, bm25, hybrid]
  reranker: [null, lexical]
  candidates: [20, 50]
```

Run it, then read every file in `report.out` against the table in
`getting-started.md#what-got-written`.

### 5. Second opinion, then diff

Run again with `run.mode: factorial` (or a changed chunker) into a second output folder, then
`contextgrid diff` the two manifests. Check the diff describes the change you actually made.

### 6. Make it angry

Every one of these has a documented message and a documented exit code. Check both.

| Break it | Documented behaviour | Doc |
|---|---|---|
| `chunker: banana` | `no chunker named 'banana'. Available: ...`, exit 1 | `cli.md#check` |
| `chunker: recursive:-5` | `chunk size must be positive, got -5`, exit 1 | `cli.md#check` |
| `corpus: ./absent` | `corpus not found: ...`, exit 1 | `cli.md#check` |
| no `evalset:` on `run` | `no evalset, so there is nothing to score against` | `cli.md#check` |
| `chunkers:` instead of `chunker:` | `Did you mean 'chunker'?` plus the full key list | `configuration.md#errors` |
| `headline: recall` | `must name a cut-off, like 'recall@5'` | `configuration.md#errors` |
| `headline: recall@five` | `non-numeric cut-off` | `configuration.md#errors` |
| `${NORTHWIND_KEY}` unset | names the variable, is not an empty string | `configuration.md` |
| `budget_usd: 0.0` | nothing runs, report says why, exit 1 | `configuration.md`, `cli.md#run` |
| a config in a subfolder with `corpus: ./documents` | paths resolve against the **config's** folder | `configuration.md` |
| `embedder: tei:...` with no server | error names the install/fix, never a bare `ImportError` | `README.md` |

Also confirm the promise that covers all of them:
**"nothing raises a raw traceback at the top level"** (`cli.md`).

### 7. The library, in code

`README.md` shows `cg.Lab(...)`, `lab.grid(...)`, `lab.run(...)`, `results.summary(...)`,
`cg.read_jsonl`, and `cg.lift(...)`. Run each one exactly as printed. A copy-pasteable snippet
that does not copy-paste is a finding.

## Recording findings

Keep a running table. One row per disagreement:

| Field | Meaning |
|---|---|
| What the doc says | Quote it, with `file.md:line` |
| What actually happened | The real output or error, verbatim |
| Severity | **blocker** (a documented path cannot be completed), **wrong** (works, output contradicts the doc), **cosmetic** (harmless drift) |

Rules for the table:

- A doc example whose *numbers* differ is usually **cosmetic** — the corpus is different.
  A doc example whose *shape*, *columns*, *file names* or *exit code* differ is **wrong**.
- "Works but the doc never mentions it" is a finding too. So is a flag in `--help` that no
  page documents.
- Do not fix anything. Do not touch `src/`. Report only.

Write the finished report to `docs-e2e-report.md` in the repo root.

## Finishing

Lead with the count and the worst finding. Then the table. Then one line on what the
documentation gets *right* — a report that only lists faults tells the reader nothing about
coverage.

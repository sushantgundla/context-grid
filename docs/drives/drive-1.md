# Docs-driven end-to-end drive — findings

**12 disagreements between the documentation and the tool. Six are real bugs; the worst is
that a corpus inside a dot-directory is invisible and the tool says the directory is empty
when it holds eight files.**

One further finding (#11, output ordering) was raised and then **withdrawn** — the docs were
right and the drive misread piped output. It is kept below, struck through, rather than
deleted.

---

## Status: all 12 fixed and verified

Fixed by five parallel agents on `fix/docs-promised-behaviour`, then re-driven from scratch
against the original repros. Nothing is committed.

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | Dot-directory corpus invisible; false "holds no files" | fixed | `corpus/loader.py` |
| 2 | `is_answerable` always False on anchors | fixed | `core/evalset.py` |
| 3 | `check` passes an unbuildable plugin | fixed | `cli/__main__.py`, `config/plugins.py` |
| 4 | No-ground-truth question blamed on the pipeline | fixed | `diagnose/taxonomy.py`, `report/` |
| 5 | `profile` rejects a single-file corpus | fixed | `cli/__main__.py` |
| 6 | `check` leaks `[Errno 21]` | fixed | `cli/__main__.py` |
| 7 | `configuration.md` wrong about `check` + evalset | fixed | `docs/guide/configuration.md` |
| 8 | `profile --help` contradicts `cli.md` | fixed | `cli/__main__.py` |
| 9 | `cli.md` `profile` description stale | fixed | `docs/guide/cli.md` |
| 10 | `experiment.yaml` write condition wrong | fixed | `docs/guide/configuration.md` |
| 11 | ~~`check` output order~~ | withdrawn — docs were right | — |
| 12 | `validate` had no documented input format | fixed | `docs/guide/cli.md` |
| 13 | Warnings duplicated per configuration | fixed | `report/results.py` |

### Two things the fixes turned up that the drive had not

**Finding 2 was not a simple bug.** `is_answerable` and its duplicate were a *deliberate*
split, recorded in `docs/roadmap.md:224-229` as the fix for an earlier bug — one property meant
"somebody wrote evidence for this question", the other "this parse located it". Collapsing them
would have silently destroyed the parser axis, which is the measurement this tool exists for.
The fix keeps both under honest names: `is_answerable` (`bool(gold or anchors)`) and the new
`is_resolved` (`bool(gold)`).

That rename then exposed four call sites that had been relying on the old ambiguous name, one
of which was a live regression: `evidence_resolvable` — the parse dimension's score — would
have read `1.0` even when the parser had lost the evidence. All four are fixed
(`grid/runner.py`, `score/resolve.py`, `generate/answer.py`, `diagnose/taxonomy.py`).

**Finding 11 was the drive's own error.** Every command in the original drive was captured
through a pipe, where stdout is block-buffered and stderr is not, so the output arrived in an
order the tool never produces at a terminal.

### Verification

```
pytest                3133 passed, 237 skipped, 0 failed
mypy                  no issues in 102 source files
ruff check .          all checks passed
scripts/check_docs.py 163 passed, 1 failed, 44 skipped
```

The single docs failure is pre-existing and unrelated: `docs/superpowers/specs/`
`2026-08-11-retrieval-cutoff-units-design.md:12` quotes a `self.` line out of `pipeline.py`
inside a spec marked *"Not yet implemented"*. No file involved in these fixes touches it.

Test count went 3089 → 3133: **44 new tests** across `tests/unit/test_corpus_hidden_root.py`,
`test_answerable.py`, `test_cli_extras_and_paths.py`, `test_unscored_and_warning_dedup.py`.

Driven from the public documentation only — `README.md`, `docs/guide/*`, `docs/dimensions/*`,
`docs/reference/*`. No source code was read. Scenario, corpus and eval set:
`.claude/skills/docs-e2e-drive/`.

Environment: `context-grid 0.9.0`, Python 3.13.5, darwin, numpy 2.4.6. Every extra installed
except `marker-pdf`.

---

## Bugs

### 1. A corpus inside a dot-directory is invisible, and the error message is false

**Severity: blocker.**

```
$ contextgrid profile .claude/skills/docs-e2e-drive/data/documents
error: no files under .claude/skills/docs-e2e-drive/data/documents matched
['*.txt', '*.md', ...]. The directory holds no files at all.

$ ls -1 .claude/skills/docs-e2e-drive/data/documents | wc -l
8
```

Copy the same folder to a path with no dot-prefixed component and it loads all 8 files. So
two things are wrong:

- **A corpus is silently skipped** if any component of its path starts with `.` — `.data/`,
  `.cache/`, `.claude/`, a checkout under `~/.local/share/...`. Nothing in the docs mentions
  this restriction.
- **The message states a falsehood.** "The directory holds no files at all" is checked and
  printed for a directory holding eight `.md` files. The same sentence is correct for a truly
  empty directory, so the tool is not distinguishing the two cases.

This is the only finding that stopped the drive dead. Everything below was run from a copy at
a non-hidden path.

### 2. `EvalItem.is_answerable` is always `False` for anchor-only eval sets

**Severity: wrong. Affects the format the docs tell you to use.**

`evalsets.md:114-116` says a CSV row with a question but no quote/document "becomes a question
with no evidence yet (`is_answerable` is `False` until someone fills that in)" — which says
plainly that filling it in makes it `True`. It never becomes `True`.

The docs' **own** two examples both return `False`:

```python
# evalsets.md:117-133, verbatim
read_csv("questions.csv")   # -> GoldAnchor(...) present, is_answerable: False

# getting-started.md:55-59, verbatim
read_jsonl("questions.jsonl")  # -> 3 items, 1 anchor each, is_answerable: False for all 3
```

Meanwhile the CLI reports the right answer on the same file:

```
$ contextgrid evalset questions.jsonl
3 questions (3 answerable), 0% reviewed, ...
```

So the CLI's "answerable" count and the `is_answerable` property disagree, and only the
property is exposed to library users. `evalsets.md:48` tells everyone to **"Write anchors, not
spans"** — meaning anyone who filters their eval set on `is_answerable` in code gets zero
questions back from a perfectly good set.

### 3. `check` passes a config whose plugin cannot be built

**Severity: wrong. This is the one thing `check` exists to prevent.**

`cli.md:104-106`: *"It also builds one of every plugin the matrix names, with the parameters
you gave it, and reports whatever refuses to be built."*

`marker-pdf` is not installed:

```
$ cat marker.yaml
corpus: ./documents
evalset: ./questions.jsonl
grid:
  parser: marker

$ contextgrid check marker.yaml
config is valid.
exit: 0
```

`init` already knows better — the config it generates lists `agno, docling, pdfplumber,
pymupdf, pymupdf4llm, text` under parser and deliberately omits `marker`. But `check` waves it
through, and `run` then fails on a real corpus with the correct message:

```
error: The marker parser requires the 'parse-marker' extra (needs marker-pdf).
Install it with: pip install "context-grid[parse-marker]"
```

The install-command promise in `README.md:227` holds. The `check` promise does not.

### 4. A question with no ground truth is blamed on the retrieval pipeline

**Severity: wrong. Sends the user hunting for a bug that does not exist.**

`nw12` in the eval set has a question and no anchor — the exact case `evalsets.md:114` says is
allowed. Every report says:

> ...scored on 12 questions. The eval set holds 13 questions in all; **the other 1 could not be
> scored, because no chunk in this index held their evidence.** [...] 1 of 13 questions failed.
> 100% of those are **fp1_missing_content: the evidence is not in this index at all. Either the
> parser lost it, the chunker dropped it, or the corpus does not contain it. No retriever can
> fix this.**

`results.json` knows the truth and records it correctly:

```json
{"code": "gold_span_unreachable",
 "message": "item 'nw12' has no gold spans and is excluded from ranking metrics"}
```

The real cause is a gap in the eval set. The console blames the parser, the chunker and the
corpus. It reproduces on the winning run too, which scored `recall@5 = 1.000` on all twelve
scorable questions and still reported a `fp1_missing_content` failure.

The mechanism is visible in the run record: `scored_queries: 12`, but `failures` sums to 13 —
the unscorable question is counted into the failure histogram.

### 5. `profile` rejects a single-file corpus that `check` and `run` accept

**Severity: wrong.**

`configuration.md:54` defines `corpus` as *"A directory of documents, **or a single file**."*

```
$ contextgrid check singlefile.yaml     # corpus: ./documents/billing.md
config is valid.
$ contextgrid run singlefile.yaml       # runs, scores 1.000
$ contextgrid profile ./documents/billing.md
error: documents/billing.md is not a directory
```

`cli.md:172` documents the argument as `contextgrid profile corpus`, so a reader has no way to
know "corpus" means something narrower here.

### 6. `check` on a directory leaks a raw OS error

**Severity: cosmetic.**

```
$ contextgrid check ./documents
error: [Errno 21] Is a directory: 'documents'
```

`cli.md:29-30` promises `error: ...` messages rather than raw tracebacks, and it keeps that —
this is not a traceback. But it is the only message found in the whole drive that is not plain
English, and `profile` handles the mirror-image case cleanly ("is not a directory").

---

## Documentation that contradicts itself or the tool

### 7. `configuration.md` says `check` works without an eval set. It doesn't

`configuration.md:55`, on the `evalset` key: *"Required to actually score a sweep — **`check`
works without it**, `run` doesn't."*

```
$ contextgrid check no-evalset.yaml
error: no evalset, so there is nothing to score against
exit: 1
```

`cli.md:117` lists exactly this as a `check` failure. The two pages disagree; `cli.md` matches
the tool.

### 8. `profile`'s `--help` contradicts `cli.md`

| Source | Text |
|---|---|
| `contextgrid --help` | `profile   Profile a corpus and say which axes will matter.` |
| `cli.md:17` | `profile   Measure a corpus and flag settings its shape rules out.` |
| `cli.md:170-171` | *"**It does not rank the axes for you.** Deciding which axis matters is what the sweep itself is for"* |

The help string promises the one thing the prose page insists the command does not do.

### 9. `cli.md`'s description of `profile` is stale in both directions

`cli.md:166-168`: *"the example below is representative rather than truncated: today it reports
size and encoding, and warns when a chunk size cannot discriminate."*

Actual output on an 8-file corpus:

```
8 files, 13,699 bytes, 13,635 chars via markdown, 17% tables
  - 17% of this corpus is tables. Parser choice will probably dominate every other axis here,
    and a chunker that splits a table in half will lose the answer outright.
  - Documents average 6 headings each. Structural chunking usually wins on corpora like this.
  - The median document is 1,773 characters. Chunk sizes above that cannot differentiate,
    so sweep small sizes.
```

- No encoding is reported anywhere.
- Table percentage and heading counts are reported, and neither is documented.
- "Parser choice will probably dominate every other axis here" **is** ranking the axes — the
  precise thing `cli.md:170` says it does not do.

The command is better than its documentation. That still counts.

### 10. `experiment.yaml` is written when the docs say it isn't

`configuration.md:150`: *"A `manifest.json` and a copy of the source config (`experiment.yaml`)
are always written alongside, regardless of `formats`, whenever `out` is set **and there's a
winner**."*

With `budget_usd: 0.0` there is no winner. `manifest.json` is correctly absent, but three files
are still written:

```
$ ls results-budget0/
experiment.yaml   report.md   results.json
```

Writing the config that produced a no-result run is the right call. The doc just says the
opposite.

### 11. ~~`cli.md` shows `check` output in the wrong order~~ — WITHDRAWN, not a defect

**This finding was wrong. The documentation was right and the original drive misread it.**

`cli.md:136-156` narrates *"The matrix shape is printed first ... and then every problem, one
`error: ...` line each"*. That is exactly what the tool does at a terminal:

```
$ script -q /dev/null contextgrid check both.yaml
both: 1 × 1 × ... = 1 on paper, 1 to run in ofat mode, scored on recall@5
  ingestion   [None]
  ...
error: corpus not found: /.../absent
error: no evalset, so there is nothing to score against
```

The reversal only appears when stdout is redirected to a pipe or a file, where stdout is
block-buffered and stderr is not. The original drive captured every command through a pipe, so
it saw the buffered order and blamed the docs.

The docs now carry a sentence explaining the buffering, which is a genuine improvement — a
reader piping one stream would otherwise be surprised. But there was no defect to fix.

**Lesson for the next drive: capture through `script -q /dev/null`, not a pipe, before claiming
an ordering bug.**

### 12. `validate` cannot be run from the documentation

`cli.md:265-302` and `README.md:236-245` document the flags, the exit code and the output
format of `contextgrid validate`, but never say what a LegalBench-RAG JSON file must contain,
and no sample is vendored (`cli.md:269`: *"Not vendored — point it at your own local copy."*).
`evalsets.md:152` names `read_legalbench_rag(path)` without a schema either.

A reader who wants to check the scorer — the section `README.md:231` calls *"Checking it rather
than trusting it"* — has no way to build an input.

### 13. Eval-set warnings are repeated once per configuration

`results.json` from a 7-configuration sweep carries the same two eval-set-level warnings six
times each:

```
anchor_normalised   nw13   × 6
gold_span_unreachable  nw12   × 6
```

Both are facts about the eval set, not about any configuration. Nothing in
`reference/reports.md` or `getting-started.md` describes the `warnings` array's shape, so a
consumer has no documented reason to expect duplicates.

---

## What the documentation gets right

Most of it, and precisely.

- **Every documented error message matched character for character** — the unknown-plugin
  block, `chunk size must be positive, got -5`, `corpus not found:`, the Levenshtein "Did you
  mean 'chunker'?" hint with the full key list, `run.headline must name a cut-off`, the
  non-numeric cut-off variant, and the unset-`${VAR}` message naming the variable. The
  three-problem example at `cli.md:126-132` reproduces exactly.
- **The `init` → `check` → `run` → `diff` loop works end to end** on a real corpus, with the
  matrix shape, the impossible-combination count and the ofat run count all as documented.
- **Paths resolve against the config file's directory**, exactly as `configuration.md:39-42`
  claims — verified from a subfolder.
- **`budget_usd: 0.0`** runs nothing, says why, writes no leaderboard and exits 1, as promised.
- **Both README library snippets run verbatim**, and `cg.lift(...)` prints the documented
  sentence word for word.
- **`use_winning_config.py` executes** and returns chunk IDs; **`winning-config.yaml` re-runs**
  and reproduces the winner. The generated snippet correctly names every non-default field and
  correctly omits every default one.
- **`plugins` covers exactly the six families** `configuration.md:117-119` says it does, and
  refuses `--family ingestion` with the right list.
- **No raw traceback appeared** across corrupt YAML, a garbage JSONL, a directory passed as a
  config, a non-manifest JSON, missing files, and an empty corpus directory.
- **The anchor machinery behaved better than advertised**: a quote that spans a soft line wrap
  in the source resolved anyway, with an honest `anchor_normalised` warning naming the parser
  that reflowed it.

---

## Reproducing

```bash
cp -r .claude/skills/docs-e2e-drive/data /tmp/nw     # finding 1 forces the copy
cd /tmp/nw
contextgrid profile ./documents
contextgrid init contextgrid.yaml --corpus ./documents --evalset ./questions.jsonl
contextgrid check contextgrid.yaml
contextgrid run contextgrid.yaml
```

The full scenario and the per-command checklist are in
`.claude/skills/docs-e2e-drive/SKILL.md`.

# Docs-driven end-to-end drive #2 — findings

**11 disagreements: 1 blocker, 5 wrong, 5 cosmetic.**

The worst is `contextgrid validate`. `cli.md:307` and `evalsets.md:164` both promise that the
benchmark file may be "a bare array" instead of an object with a `tests` key. Hand a bare array
to `validate` and it dies with `error: 'list' object has no attribute 'get'` — a Python
`AttributeError` message wearing an `error:` prefix. The documented path cannot be completed,
and the message tells a user nothing about how to fix it.

Second worst, and the one most likely to waste someone's afternoon: `contextgrid check` prints
`config is valid` and exits 0 on an eval set file that `contextgrid run` rejects a second later.
`check` exists to catch exactly that.

This drive was a regression check on `fix/docs-promised-behaviour`. **The previous drive's
finding #1 is fixed**: a corpus under a dot-directory
(`.claude/skills/docs-e2e-drive/data/documents`) now loads all 8 files and profiles correctly.

---

## Findings

| # | What the doc says | What actually happened | Severity |
|---|---|---|---|
| 1 | `cli.md:307` — "A JSON object with a `tests` array (**a bare array is also accepted**)". Repeated at `evalsets.md:164`: "a bare array works too". | `contextgrid validate bare-array.json ./documents` → `error: 'list' object has no attribute 'get'`, exit 1. `read_legalbench_rag("bare-array.json")` raises `AttributeError: 'list' object has no attribute 'get'`. The object-with-`tests` form works fine. | **blocker** |
| 2 | `cli.md:119` — "Each message is the one `run` would have printed; `check` only makes it arrive before the expensive part." `cli.md:164` — check "catches a missing path, an empty corpus, a typo'd axis or a typo'd spec string before anything expensive starts." | `check` resolves the eval set **path** but never opens the file. With `evalset: ./bad.jsonl` holding the text `garbage not jsonl`, `check` prints `config is valid` and exits **0**; `run` on the same config prints `error: .../bad.jsonl:1 is not valid JSON: Expecting value: line 1 column 1 (char 0)` and exits 1. Same with a CSV lacking a question column: `check` → `config is valid`, exit 0; `run` → `error: .../nocol.csv has no question column. Expected one of: question, query, q. Found: foo, bar`. | **wrong** |
| 3 | `cli.md:313` — "`validate` reads `corpus/file_path` and **skips any snippet whose file isn't there**." | It does not skip it — it counts it as an unresolved span. A four-test file where one snippet points at `not-here.md` gives `only 50% of the benchmark's spans fall inside the documents as loaded...`, exit 1. Delete that one test and the identical run reports `1 of 1 spans point at real text`, exit 0. So a benchmark that names one absent file fails the whole command instead of losing one snippet. | **wrong** |
| 4 | `reports.md:182` — "The title carries the experiment's name — `name:` in an experiment config... Without one it falls back to `# Retrieval configuration comparison`". | The H1 is *always* the fallback. With `name: northwind-sweep` in the config, `report.out/report.md` line 1 is `# Retrieval configuration comparison`. The name is being read — the console prints `northwind-sweep: 1 × 1 × 3 ...`, `experiment.yaml` line 1 is `name: northwind-sweep`, and `winning-config.yaml` line 10 is `name: "northwind-sweep"`. Only the Markdown title misses it. Same with `name: weak-run`. | **wrong** |
| 5 | `cli.md:58` — init "writes a starter config listing every plugin *this installation* can actually run **as a chosen value**, and everything else **as a comment showing how to unlock it (which extra to install)**". `getting-started.md:67-69` says the same. | Both halves are wrong in this install (where every extra except `marker-pdf` is present). (a) Runnable plugins are demoted to comments: the generated `contextgrid.yaml` has `parser: [markdown]` with `# also available: agno, docling, pdfplumber, pymupdf, pymupdf4llm, text` — `text` needs no extra at all, and `docling`/`pymupdf`/`pdfplumber`/`agno` are all installed and pass `check`. (b) `marker` — the one parser whose extra is *not* installed — appears nowhere in the file, not even as a comment. (c) No comment anywhere in the generated config names an extra or an install command. `check` gets this right where `init` does not: `parser: marker` → `error: The marker parser requires the 'parse-marker' extra (needs marker-pdf). Install it with: pip install "context-grid[parse-marker]"`, exit 1. | **wrong** |
| 6 | `evalsets.md:187-218` — a snippet that writes one file, `documents/policy.md`, then prints `2 questions drafted, 0 chunks skipped` followed by two items with ids `refunds.md:0-191#probe` and `shipping.md:0-188#probe`. | Run verbatim in a clean directory it prints `1 questions drafted, 0 chunks skipped` and one item, `policy.md:0-220#probe`. The shown output names two files the snippet never creates, and quotes text ("provided the item is unopened", "costs an additional $15") that is not in the text the snippet writes. This is not a different-corpus mismatch — the snippet builds its own corpus. | **wrong** |
| 7 | `cli.md:174` — the first `profile` line reports "how many files, how many bytes, how many characters once the parser has read them, which parser did the reading, and **what share of the text is tables**". | The tables share is dropped entirely when it is 0%, rather than printing `0% tables`. `contextgrid profile ./documents` → `8 files, 13,699 bytes, 13,635 chars via markdown, 17% tables`; `contextgrid profile ./documents --parser text` → `8 files, 13,699 bytes, 13,635 chars via text` (no tables field). | cosmetic |
| 8 | `cli.md:315` — test `id` "Defaults to `lb0`, `lb1`, ... in file order." | Ids are the test's index in the file, so dropped tests leave gaps. A five-test file whose second test has an empty `query` loads as `lb0, lb2, lb3, lb4` — there is no `lb1`. Arguably better behaviour than the doc describes, but not what it describes. | cosmetic |
| 9 | `cli.md:408` — the diff is "between the two *winning* configurations' manifests, not a diff of the config files themselves." | The caveat is documented, but the message itself overclaims. Diffing an `ofat` run against a `factorial` run of the same grid — 6 configurations versus 27, different leaderboards — prints `Nothing in the manifest changed, so these two runs should have produced identical numbers. If they did not, something outside the manifest is affecting results and that is worth finding.` Only the winner was identical. | cosmetic |
| 10 | `README.md:109-112` — a fenced ```python block with no `no-run:` marker, unlike other non-runnable blocks in the same file (e.g. `README.md:88`). | It needs a TEI server on `http://127.0.0.1:8080`, so it cannot be copy-pasted. Without one it fails with a good message (`error: tei failed embedding documents 0-7 with bge-base-en-v1.5: could not reach a TEI server at ... Start one with: docker run -p 8080:80 ...`), but the block is not marked as needing a server the way the repo marks its other unrunnable snippets. | cosmetic |
| 11 | Works, but no page mentions it. `configuration.md:54` says `corpus` is "A directory of documents, or a single file", and the empty-directory error lists the accepted extensions. | The extension list is enforced for a directory but not for a single file. `contextgrid profile x.log` (a `.log` file, not in the list) succeeds: `1 files, 3 bytes, 3 chars via markdown`, exit 0. The same file inside a directory would give `no files under ... matched ['*.txt', '*.md', ...]`, exit 1. | cosmetic |

---

## What the documentation gets right

Most of it, and the recently-changed areas in particular. Everything below was run and matched
the docs, so the report above is a list of exceptions, not a summary.

**The regression areas named for this drive — 5 of 6 clean.**

- Corpus under a dot-directory: **fixed**. `contextgrid profile .claude/skills/docs-e2e-drive/data/documents`
  loads all 8 files, exit 0. This was the previous drive's finding #1.
- `check` rejecting an uninstalled extra: correct. `parser: marker` → the exact install command,
  exit 1. Never a bare `ImportError`, as `README.md:227` promises.
- `profile` on a directory **and** on a single file: both correct, exit 0. The `./documents`
  output matches `cli.md:190-194` in shape, and the notes fire only when the shape warrants them
  (`--parser text` correctly swaps the headings note for "No headings were found...").
- `EvalItem.is_answerable` vs `is_resolved`: exact. A fresh CSV read gives `answerable=True,
  resolved=False` on every row with a quote; `nw12` (question, no evidence) gives
  `answerable=False` and is not an error. The `spans-and-anchors.md:134-146` doctest reproduces
  character for character, including `is_portable`, and the `has_evidence` / `with_evidence`
  aliases really are the identical property.
- Failure-diagnosis prose and the `warnings` array: correct, and the nw13 line-wrap trap is
  caught. `results.json` carries
  `{"code": "anchor_normalised", "message": "the evidence for 'nw13' was found only after collapsing whitespace, so 'markdown' reflowed it: ...", "severity": "info", "stage": "anchor", "subject": "nw13"}`
  alongside `gold_span_unreachable` for nw12 and `impossible_combination` for the dropped
  `dense`+`null` pair. Every run object carries `per_query`, `by_type` and `failures` as
  `reports.md:190-192` says. The summary prose says `11 of 12 questions failed. 100% of those
  are fp1_missing_content: ...` on a deliberately weak config, and does *not* repeat the
  no-ground-truth sentence twice — exactly the `include_unscored=False` behaviour documented at
  `diagnostics.md:212-214`.
- `validate` hand-written from `cli.md` alone: **the claim holds**. The two-question example at
  `cli.md:324-343` was copied into a file and run against the Northwind corpus with no other
  information. Its own offset-finding snippet (`cli.md:348-355`) returns `[117, 202]`, matching
  the number printed beside it, and the command's output is byte-for-byte the doc's:
  `2 of 2 spans point at real text ... | recall@10 | 1.000 | 0.720 | +0.280 |` and the
  "outside the 0.05 tolerance" paragraph. `--limit 1` and the no-`--recall-at-10` form (every
  metric, one per line) both behave as written, and the sub-95% stop fires with a clear message
  and exit 1.

**Errors — all 11 documented cases produce the documented message and exit code.** Checked with
`script -q /dev/null` so stream ordering is what a terminal sees, and with a bare run for `$?`:

| Broken thing | Result |
|---|---|
| `chunker: banana` | `error: grid.chunker: no chunker named 'banana'. Available: chonkie:code, ...`, exit 1 |
| `chunker: recursive:-5` | shape to stdout, then `error: chunker 'recursive:-5': chunk size must be positive, got -5`, exit 1 |
| `corpus: ./absent` | `error: corpus not found: /abs/path/absent`, exit 1 |
| no `evalset:` | `error: no evalset, so there is nothing to score against`, exit 1 |
| `chunkers:` for `chunker:` | `error: unknown key 'chunkers' in the 'grid' section. Did you mean 'chunker'? Known keys: candidates, chunker, ...`, exit 1 |
| `headline: recall` | `error: run.headline must name a cut-off, like 'recall@5'. Got 'recall'`, exit 1 |
| `headline: recall@five` | `error: run.headline has a non-numeric cut-off: 'recall@five'`, exit 1 |
| `${NORTHWIND_KEY}` unset | `error: the config refers to ${NORTHWIND_KEY} but that environment variable is not set`, exit 1 — and substitutes correctly once set |
| `budget_usd: 0.0` | `No configurations were run.` + `none of the 1 configurations ran: the $0.00 budget ran out ($0.0000 spent)...`, exit 1, and exactly the three documented files (`experiment.yaml`, `report.md`, `results.json`) — no `manifest.json`, no `winning-config.yaml`, no `use_winning_config.py` |
| config in a subfolder, `corpus: ./documents` | resolves against the config's folder from any working directory, including `$HOME` |
| `embedder: tei:...` with no server | names the `docker run` command to start one, exit 1 |
| three problems at once | one `error: 3 problems with this config:` block listing all three, exit 1 |

**Nothing raised a raw traceback at the top level** (`cli.md:29-30`) across everything thrown at
it: malformed YAML, `grid:` as a list, a missing config file, a directory passed where a file
was expected, an empty corpus directory, an unknown plugin family, no `corpus:` key. All gave an
`error: ...` line and exit 1. Finding #1's message is an internal Python string, but it is still
an `error:` line, not a traceback.

**Also verified, all matching:**

- `--help` — nine subcommands, usage block identical to `cli.md:6-27`. `--version` → `context-grid 0.9.0`.
- `plugins` — six families, and it does *not* cover `ingestion`/`transform`/`retrieval`/`generator`
  as `configuration.md:117-121` says; all four have their own sections in `reference/plugins.md`.
  `--family chunker` output matches `cli.md:251-265` exactly. `--family nosuch` → `error: unknown
  plugin family 'nosuch'. Valid families: parser, chunker, embedder, index, reranker, tokenizer`.
- `read_csv` on loose column names (`qid,q,file,evidence,rel,category`) — all 13 rows, ids and
  `qtype` picked up. `write_jsonl` → `read_jsonl` round-trips anchors, grades, page hints, qtypes
  and ids identically.
- `contextgrid evalset questions.jsonl` → `13 questions (12 answerable), 0% reviewed, detects
  differences of 0.57 and above` plus the type histogram — the `cli.md:278-283` shape. Setting
  `"meta": {"reviewed": true}` on 6 of 13 moves it to `46% reviewed` and drops the nag line, as
  `evalsets.md:235-242` promises.
- `init` → the two documented lines; a second `init` without `--force` → `error: contextgrid.yaml
  already exists. Pass --force to overwrite.`, exit 1.
- `check` on the generated config → `1 × 1 × 2 × 2 × 3 × 1 × 1 × 2 × 1 × 1 = 24 on paper, 5 to run
  in ofat mode (1 impossible combination(s) skipped), scored on recall@5` — identical to
  `cli.md:89` and `getting-started.md:95`, down to the impossible-combination count.
- Stream behaviour: at a terminal, shape (stdout) then `error:` (stderr); redirect stdout and
  they separate exactly as `cli.md:158-162` describes.
- `run` → 6 files, names matching `getting-started.md:174-181`. `formats: [yaml]` alone → the
  three files that implies. `leaderboard_limit: 3` → 3 rows. `cache: disk` → `.contextgrid-cache`
  inside `report.out`. `--quiet` → zero bytes on stderr. `mode: staged` and `mode: factorial` both
  run.
- `sweep` with the exact flags from `cli.md:223` → `wrote 5 files to sweep-results`, matching shape.
- `diff` → the documented `N thing(s) changed` list with `config.chunker`, `config.embedder`,
  `config.k` lines.
- Library: the `README.md:69-81` snippet (`cg.Lab`, `lab.grid`, `cg.read_jsonl`, `lab.run`,
  `results.summary`) runs verbatim. `cg.lift(retrieval_score=0.80, answer_score=0.70,
  baseline_answer=0.70)` returns `README.md:319` word for word. The `diagnose()` doctest at
  `diagnostics.md:150-208` reproduces exactly, including `report.no_ground_truth == ['q4']`,
  `(4, 5)`, the full `summary()` paragraph and the `cluster()` dict.

## What was not covered

- A missing extra other than `marker-pdf` — every other optional dependency is installed in this
  venv, so `README.md:227`'s "never a bare `ImportError`" promise was only exercised on one path.
- Anything needing an API key or a live model: `transform`, `generator: llm`, `retrieval: agentic`,
  `run.model`, the `deepeval` judge metrics, and the `budget_usd` path that stops a sweep
  *partway* (only the `0.0` "nothing ran" path was tested).
- PDF parsers, `pgvector`, and the Docker path.
- `read_beir`.

## One thing I did not do, and cannot explain

`git status` at the end of this drive shows three untracked paths in the repo root that were not
there when it started: `documents/policy.md`, `questions.csv` and `questions.jsonl`. They are the
exact artefacts of the `evalsets.md` snippets — the `read_csv` example's two-row CSV, the
`write_jsonl` example's `policy-questions` header, and the `generate` example's `policy.md`.

They are not mine. Every snippet in this drive was run from a scratchpad directory, and my own
copy of `documents/policy.md` sits there with a timestamp two minutes later (03:01) than the
repo-root one (02:59). Either another session was running doc snippets from the repo root at the
same time, or something runs the doc examples with the repo root as its working directory. I left
the files in place rather than delete work that may not be mine. Worth someone establishing which
it was — a docs harness that writes into the repo root would be a real problem.

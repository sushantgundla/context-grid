# Docs-driven drive 4 — the SDK lane

**Nine disagreements across 20+ documentation pages, and the worst is a worked example whose
transcript was pasted from a different example entirely** — `docs/guide/evalsets.md` showed two
questions drafted from two files, quoting sentences that appear nowhere in the code above it, when the
code writes one file and drafts one question. Nothing was a blocker: every documented path could be
completed, and the two deliberate fixture traps (`nw12`, `nw13`) were both handled correctly and
reported honestly.

All nine were fixed in this session (see [What was fixed](#what-was-fixed)). Eight were the
documentation's fault; one was the code's.

Three Sonnet agents drove the library surface in parallel, each in its own scratch directory, none
reading `src/`, `tests/`, or an earlier report:

| Lane | Pages |
|---|---|
| 1 | `README.md`, `docs/README.md`, `guide/getting-started.md`, `guide/evalsets.md`, `guide/configuration.md` |
| 2 | all seven `docs/recipes/` pages |
| 3 | all ten `docs/dimensions/` pages, all five `docs/scoring/` pages, all four `docs/reference/` pages |

Fixture verified sound before the drive (`verify_fixtures.py`, exit 0): 13 anchors, 11 exact, 1
whitespace-normalised, 1 with no evidence, 0 broken.

## The findings, worst first

### 1. `docs/guide/evalsets.md:212-219` — the drafting transcript belongs to a different example

**wrong** · doc's fault

The code block writes **one** file, `documents/policy.md`. The transcript below it claimed:

```
2 questions drafted, 0 chunks skipped
refunds.md:0-191#probe | issued item provided purchase refunds unopened | Refunds are issued within 30 days of purchase, provided the item is unopened.
shipping.md:0-188#probe | additional arrives business costs express next | Express shipping arrives the next business day and costs an additional $15.
```

Two files that the code never creates, and quotes — "provided the item is unopened", "costs an
additional $15" — that appear nowhere in the `f.write(...)` call. What actually runs:

```
1 questions drafted, 0 chunks skipped
policy.md:0-220#probe | digital downloaded goods once refundable | Digital goods are not refundable once downloaded.
```

Deterministic: identical across `PYTHONHASHSEED` 0/1/2/12345 in fresh processes.

### 2. `docs/guide/evalsets.md` — `min_chunk_words` is undocumented, and silently empties a draft

**wrong** · doc's fault (a gap, found while fixing #1)

`generate()` takes `min_chunk_words: int = 25` and skips every chunk below it. Chunk the page's own
corpus by sentence and you get nothing back:

```
sentence:1 -> 4 chunks
  count: 0 skipped: 4
```

No page mentions `min_chunk_words`, `sample`'s default of `50`, `seed`, or `evalset_id`. A reader
following the page prints only `draft.count` and `draft.chunks_skipped`, sees `0 questions drafted, 4
chunks skipped`, and has no way to learn why.

**The tool itself is blameless here** — `draft.warnings` says exactly what happened:

```
invalid | none of the 4 chunks has 25 words, so there is nothing to write questions about. Try a larger chunk size
```

The docs simply never point at it.

### 3. `docs/guide/configuration.md:191-207` — `budget_usd: 0.0` is shown as a clean run; it exits 1

**wrong** · doc's fault

The page frames this as a normal, deliberate outcome — "the report says why rather than showing an
empty leaderboard as if the matrix had been covered" — and its transcript uses `;` rather than `&&`,
so the exit code never appears. Reproduced with the streams split and the exit code taken from a bare
run:

```
EXIT=1
--- stderr:
error: no configurations were run, so nothing was measured
error: none of the 1 configurations ran: the $0.00 budget ran out ($0.0000 spent). Nothing was measured, so there is no leaderboard rather than an empty one
```

The three files are written correctly first. But a user scripting `contextgrid run ... && next-step`
on this documented scenario gets silently skipped.

**Which half is wrong: the doc.** Exiting non-zero when nothing was measured is the safer behaviour —
a script must not carry on as though it had a leaderboard. The page just never says so.

Note the same sentence is printed twice, once per stream. That is defensible (a user who redirects
stdout still learns it failed) but it does read as noise in a terminal.

### 4. `docs/dimensions/chunkers.md:155-157` — a documented refusal that never happens

**wrong** · doc's fault

> note `min_size` defaults to 64, so a `max_size` below 64 needs `min_size=` set explicitly too, or
> the constructor raises

No exception, and `min_size` is not 64:

```
max_size=   32 -> min_size=4    (max//8=4)
max_size=  100 -> min_size=12   (max//8=12)
max_size=  512 -> min_size=64   (max//8=64)
max_size= 1024 -> min_size=128  (max//8=128)
```

`min_size` defaults to `max_size // 8` throughout — 64 only at the default `max_size` of 512.

**Which half is wrong: the doc.** Scaling the floor down beats refusing to build, and it matches the
design principle the same page states two paragraphs earlier: "refusing a perfectly reasonable chunk
size because of a default nobody asked for is a bad axis value."

### 5. `docs/dimensions/indexes.md:15-32` — one of the seven indexes is missing from the exactness list

**wrong** · doc's fault

Line 8 promises the page "covers all of them". The `is_exact` section then classifies six of the
seven and never mentions `quantized` at all — despite `is_exact` being, in the page's own words, "the
one property that matters most for reading any of it".

```
none is_exact: False    scalar is_exact: False
product is_exact: False  binary is_exact: False
```

Even `scheme=none`, which applies no compression and returns the same top `k` as `dense`, reports
`is_exact=False`.

**Which half is wrong: the doc.** The code is deliberate and pinned:
`tests/unit/test_quantize.py::test_it_declares_itself_approximate` asserts
`QuantizedDenseIndex.is_exact is False` at the class level, with a stated reason, and
`test_no_quantization_is_exactly_exact` separately pins that `scheme=none` scores recall 1.0 against
`ExactDenseIndex`. `is_exact` is a property of the index *family*; a family whose purpose is throwing
information away should not claim exactness because one setting happens not to. The page owed the
reader that sentence and did not have it.

### 6. `docs/recipes/reproducing-a-run.md:35-39` — the manifest example is an axis behind

**wrong** · doc's fault

The pasted `config` block has 10 keys. The real manifest has 11:

```
['ingestion', 'parser', 'chunker', 'embedder', 'index', 'transform', 'retrieval', 'reranker', 'k', 'candidates', 'generator']
```

`generator` is a live axis — `contextgrid init` writes `generator: [null]` into the grid it
generates. The example simply was not refreshed.

### 7. `docs/recipes/reproducing-a-run.md:134-136` — "byte-identical" overclaims

**wrong** · doc's fault

The page says two independent runs of the same config at the same seed give a `diff` with "(no
output... byte-identical, down to the confidence interval)". Built it for real — two configs
identical but for their output folder, `seed: 7`, `--quiet`:

```
1c1
< config-x: 1 × 1 × 2 × ... = 2 on paper, 2 to run in factorial mode, scored on recall@5
> config-y: 1 × 1 × 2 × ... = 2 on paper, 2 to run in factorial mode, scored on recall@5
12c12
< wrote 4 files to .../results-x
> wrote 4 files to .../results-y
```

Two lines differ, and neither is nondeterminism: the first carries the run's **name**, which defaults
to the config's filename, and the last names the output folder. Both are inputs the reader chose
differently. Strip those two and the diff is genuinely empty.

The substantive claim holds. Comparing the two `results.json` field by field, only three things ever
differ — `runs[*].timings`, `runs[*].cost.compute_seconds` and `manifest.created_at`. Every score,
confidence interval, `per_query` entry, failure count and `config` value is identical.

The `p95 ms` column *is* real wall-clock timing, so it can wobble between runs — the seed promises
nothing about the clock, which the page never said either.

### 8. `docs/recipes/choose-an-embedder.md:231` — two different numbers for the same gap

**cosmetic** · doc's fault

The tool prints "roughly **2,300** questions" for the tfidf-vs-`hash:512` gap, and the page repeats
2,300 at lines 63, 74 and 78. Line 231 then says "**20,890** is a real number, not a hedge" about the
same comparison. 20,890 is not producible from any output.

### 9. `docs/reference/reports.md:305-314` — `config_to_python()` quotes strings the other way

**cosmetic** · **the code's fault**, and the only one

The page shows its output with double quotes; the function emitted single quotes, from `repr()`:

```python
    ingestion='parent-document:4',        # actual
    ingestion="parent-document:4",        # documented
```

**Which half is wrong: the code.** The generated file disagreed with *itself* — its own hardcoded
literal `pipeline.search("your question here")` is double-quoted, while every interpolated value was
single-quoted. It also disagreed with `ruff format`, whose default is double quotes and which most
projects will run over a file they just pasted in. The doc showed the better output.

## What was fixed

Eight documentation fixes, one code fix. Every new or edited snippet was run and its output pasted
from the real result.

| # | Change | File |
|---|---|---|
| 1 | Transcript regenerated from the real run, plus a note on why it is one question | `docs/guide/evalsets.md` |
| 2 | `min_chunk_words` documented, with a runnable `draft.warnings` example and all four keyword arguments named | `docs/guide/evalsets.md` |
| 3 | Exit code 1 and both stderr lines documented, with a `\|\| test -f` pattern for scripting round it | `docs/guide/configuration.md` |
| 3b | Backticks added to the missing-corpus error, matching the real text | `docs/guide/configuration.md` |
| 4 | The `max_size // 8` rule documented, the phantom raise removed, two doctests added | `docs/dimensions/chunkers.md` |
| 5 | `quantized` added to the exactness classification, with a paragraph on why `scheme=none` still declares itself approximate | `docs/dimensions/indexes.md` |
| 6 | `"generator": null` added to the manifest example | `docs/recipes/reproducing-a-run.md` |
| 7 | The "byte-identical" claim replaced with the two lines that really differ, plus a verified `results.json` comparison snippet | `docs/recipes/reproducing-a-run.md` |
| 8 | `20,890` → `2,300` | `docs/recipes/choose-an-embedder.md` |
| 9 | `_as_literal()` helper added; strings now emit double quotes, and the exported file is self-consistent | `src/contextgrid/report/export.py` |
| 9 | Two tests updated off single quotes, one new test added pinning the quoting rule | `tests/unit/test_rerank_export_cli.py`, `tests/unit/test_export_roundtrip.py` |

Verification after the fixes:

- `pytest tests/` — **3329 passed, 237 skipped, 0 failed**
- `scripts/check_docs.py` on all seven edited pages — **22 passed, 0 failed, 15 skipped**
- `scripts/check_docs.py` on all docs — 167 passed, 45 skipped, 1 failed:
  `docs/superpowers/specs/2026-08-11-retrieval-cutoff-units-design.md:12` raises `NameError: name
  'self' is not defined`. That file was not touched in this session and the failure is a bare `self`
  in a method snippet in a design spec — pre-existing, and left alone as out of scope.
- `ruff check`, `ruff format --check`, `mypy` on the changed source and test files — all clean

## What the documentation gets right

This is the substance of the drive, not a courtesy. Three agents ran every runnable block on 20+ pages
and almost everything matched.

**Both deliberate fixture traps were handled correctly, which is the result that matters most.**

- `nw12` (no evidence at all) is excluded from scoring and the blame is placed on the eval set, not
  the pipeline — verbatim: *"the other 1 was not scored (nw12), because it has no ground truth at all
  -- no gold spans and no anchors -- so there was nothing to score it against. That is a gap in the
  eval set, not a fault in this pipeline."* It is correctly kept out of `report.diagnoses`/FP1 and
  lands only in `.no_ground_truth`; `is_answerable=False`.
- `nw13` (a quote spanning a soft line wrap in `sso-setup.md`) resolves **and tells the user**, via a
  structured `anchor_normalised` warning naming the item and the affected text. Resolving quietly
  would have been the wrong behaviour; so would failing.

**Reproducibility claims hold.** `manifest_hash` is byte-identical across two independent runs of the
same config. `contextgrid diff` printed its documented sentences word for word, including the one
disqualifying a comparison when `corpus_hash` changed. The `hash` embedder is genuinely deterministic
across `PYTHONHASHSEED` 0/1/2/12345 in separate processes — real blake2b, not salted `hash()`.
`DiskCache` really does persist across processes, with the documented `key[:2]/key[2:4]/key.pkl`
fan-out.

**The plugin catalogue is accurate right now.** All 12 chunkers, 5 embedders, 7 indexes, 8 parsers and
8 ingestion strategies match their registries exactly. `docs/reference/plugins.md`'s two shell
snippets reproduce character for character, including the gaps the page itself explains as deliberate
(`TRANSFORMS` listing 2 of 6 names, `GENERATORS` 1 of 2, `LLMS` showing `extra=None`). Nothing
regenerates that page, as it claims.

**Error messages are actionable, and none raised a raw traceback.** The missing `marker` extra names
the exact `pip install "context-grid[parse-marker]"`. A TEI embedder with no server names the exact
`docker run`. `usearch:b1` raises `IndexBuildError` listing the valid dtypes. `overlap >= size`,
`parent-document:1`, `sentence-window:0`, `hierarchical:threshold=0`/`1.5` all raise with the
documented wording. A typo'd plugin gives `UnknownPluginError` with the full list.

**The recipes are honest about their own output.** `choose-a-chunker.md` (0.877/0.877/0.877/0.491/0.452),
`choose-an-embedder.md` Part 1 and Part 2, `local-only.md` Tiers 1 and 2, `is-agentic-worth-it.md`
(0.877×3, 74 model calls) and `without-an-evalset.md` all reproduced exactly. `is-agentic-worth-it.md`
even documents its own tool's shortcoming correctly — `cost_per_1k` reads 0.0 for the agentic row
despite 74 real calls, and the page's hand-priced math checks out.

**Scoring pages are exact.** Every doctest in `metrics.md`, `composite.md`, `significance.md`,
`spans-and-anchors.md` and `diagnostics.md` matched, including the harmonic-mean composite (18.10
against the arithmetic 52.5), `k=None` cut-off inference, and the seeded `compare()` verdict
sentences word for word.

**`check` catches more than it advertises.** The docs promise only "a typo'd axis or a missing path";
it also rejects a plugin whose extra is not installed, before any sweep starts, naming the install
command.

**stdout/stderr are split as documented** — progress on stderr, everything else on stdout, confirmed
by redirecting separately rather than piping. `--quiet` suppresses the progress lines as promised.

## What was skipped, and why

- **Anything needing a live Postgres with `pgvector`** — `indexes.md`'s `pgvector` family and
  `reports.md`'s DB-backed claims. No server.
- **Anything needing Docker** — the TEI embedder runs, `indexes.md:244`, `docker compose run`
  self-hosting. Only the failure paths were checked, and those are correct.
- **Anything needing an API key** — `README.md`'s `openai:gpt-4o-mini`/hyde/agentic block and every
  hosted-model example. All were already `no-run`-tagged. LLM-backed pages were driven through the
  docs' own `RecordingLLM`/`ScriptedLLM`/`ScriptedJudge` stand-ins, which is what those pages
  demonstrate. No network calls were made anywhere.
- **All PDF-fixture doctests in `docs/dimensions/parsers.md`** (`pdfplumber`, `pymupdf`,
  `pymupdf4llm`, `docling`) — each does `sys.path.insert(0, "tests"); from pdf_fixtures import
  contract_pdf`, which means importing from `tests/`. Off-limits under this skill's one rule, so all
  were skipped. **This is a real coverage gap, not a dependency gap** — those packages are installed
  in this checkout, and a drive allowed to touch `tests/` would have run them. Worth noting as a
  structural problem: a documentation page whose examples can only run by importing from `tests/`
  cannot be verified by a reader either.
- **`marker`** — not installed, and `docs/reference/install.md` says installing it breaks `docling` in
  the same environment. Not attempted, on the docs' own advice.
- **`contextgrid validate` against LegalBench-RAG** — no benchmark file available.
- **No ordering claims are made anywhere in this report.** Where stream ordering mattered, streams
  were redirected to separate files rather than piped, and exit codes were taken from bare runs.

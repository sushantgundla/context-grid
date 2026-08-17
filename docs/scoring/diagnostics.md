# Diagnostics: warnings and the failure taxonomy

Two systems here turn a score into something you can act on. `contextgrid.core.warnings`
catches problems in *how* a result was produced — the run-level things that can make a
number quietly unfair. `contextgrid.diagnose.taxonomy` explains *why* individual questions
failed, once a run has finished.

## The warning system

Retrieval comparisons go wrong quietly. A chunk gets truncated because it exceeded a
model's context, a parser reflows a column and loses its character offsets, an
approximate index returns 92% of what exact search would have found — and none of it
shows up in the final number by default. A user reads a leaderboard that looks fine and
draws a confident, wrong conclusion.

So warnings here are **data, not log lines**. Every result object carries a `WarningLog`,
it survives serialisation, and anything rendering results is expected to show it.

### `GridWarning` and `WarningLog`

```python
>>> from contextgrid.core.warnings import Severity, WarningCode, WarningLog
>>> log = WarningLog()
>>> log.add(
...     WarningCode.ANCHOR_NORMALISED, "evidence found after collapsing whitespace",
...     severity=Severity.INFO, stage="anchor", subject="q1",
... )
GridWarning(code=<WarningCode.ANCHOR_NORMALISED: 'anchor_normalised'>, message='evidence found after collapsing whitespace', severity=<Severity.INFO: 'info'>, stage='anchor', subject='q1', detail={})
>>> log.add(
...     WarningCode.SPLIT_GOLD_SPAN, "gold span split across two chunks",
...     severity=Severity.CAUTION, stage="resolve", subject="q2",
... )
GridWarning(code=<WarningCode.SPLIT_GOLD_SPAN: 'split_gold_span'>, message='gold span split across two chunks', severity=<Severity.CAUTION: 'caution'>, stage='resolve', subject='q2', detail={})
>>> log.summary()
'anchor_normalised x1, split_gold_span x1'
>>> log.is_sound
True
```

`WarningLog` is mutable by design — stages append to it as a run progresses, and a run's
final log is the concatenation of its stages' logs (`.extend()` or `.merge()`). Every
`GridWarning` carries a `code` (machine-readable, grouped by the stage that raised it),
a human-readable `message`, a `severity`, and optional `stage`/`subject` for filtering.

### Severity: how much a warning should change what you believe

| Severity | Meaning |
|---|---|
| `INFO` | Something happened worth knowing about; results stand as-is. |
| `CAUTION` | Results are usable, but a comparison built on them may be slightly unfair. |
| `INVALID` | A comparison built on this result is not sound. Do not publish it. |

`log.is_sound` is `not log.invalidating` — true only when nothing in the log is
`INVALID`. `log.at_least(Severity.CAUTION)` filters to warnings at or above a floor:

```python
>>> [str(w) for w in log.at_least(Severity.CAUTION)]
['CAUTION [resolve] (q2): gold span split across two chunks']
>>> log.add(WarningCode.APPROXIMATE_RESOLUTION, "an example INVALID-severity warning",
...         severity=Severity.INVALID, stage="resolve", subject="q3")
GridWarning(code=<WarningCode.APPROXIMATE_RESOLUTION: 'approximate_resolution'>, message='an example INVALID-severity warning', severity=<Severity.INVALID: 'invalid'>, stage='resolve', subject='q3', detail={})
>>> log.is_sound
False
>>> [str(w) for w in log.invalidating]
['INVALID [resolve] (q3): an example INVALID-severity warning']
```

(Severity is chosen per call site, not fixed per code — the example above raises
`APPROXIMATE_RESOLUTION` at `INVALID` for illustration; the real call site in
`SpanResolver.resolve_item` actually raises it at `CAUTION`. Check `.severity` on the
warning itself, not just its `.code`, when deciding whether to trust a result.)

### Codes you'll see from scoring

Every code below comes from [`spans-and-anchors.md`](spans-and-anchors.md) or
[`composite.md`](composite.md) — see those pages for the code paths that raise them.

| Code | Raised by | Severity | Meaning |
|---|---|---|---|
| `GOLD_SPAN_UNREACHABLE` | `SpanResolver.resolve_item` | `INFO` (no gold at all) / `CAUTION` (gold present, matches nothing) | A gold span matches no chunk — the message says whether it fell in a gap in the chunking, lies outside the offsets this parse produced, or belongs to a document that was never chunked — or the item has no gold spans to judge at all. |
| `SPLIT_GOLD_SPAN` | `SpanResolver.resolve_item` | `CAUTION` | No single chunk clears the threshold, but the chunk set together covers the evidence. See [split gold](spans-and-anchors.md#split-gold). |
| `APPROXIMATE_RESOLUTION` | `SpanResolver.resolve_item` | `CAUTION` | Some chunks report approximate (non-exact) offsets, so relevance judgements against them are estimates. |
| `ANCHOR_NOT_FOUND` | `AnchorResolver` | `CAUTION` | None of the three match strategies located the quoted evidence in this parse. See [the failure case is the measurement](spans-and-anchors.md#the-failure-case-is-the-measurement). |
| `ANCHOR_NORMALISED` | `AnchorResolver` | `INFO` | Found only after collapsing whitespace — the parser reflowed the text. |
| `ANCHOR_BOUNDED` | `AnchorResolver` | `CAUTION` | Matched on opening/closing words only — something in the middle was corrupted. |
| `ANCHOR_AMBIGUOUS` | `AnchorResolver` | `CAUTION` | The quote appears more than once; `occurrence` was left at its default of `0`. |
| `NO_PARSE_FOR_SOURCE` | `AnchorResolver.resolve_item` | `CAUTION` | An anchor's `source_id` has no matching parse, so its evidence can't be located. |
| `SMALL_EVAL_SET` | eval-set generation and quality checks | varies | The eval set (or a slice of it, e.g. one question type) is too small for its numbers to be trustworthy. |
| `EVALSET_AT_CEILING` | scoring, after a sweep | caution | Every configuration scored full marks on the headline metric, so the sweep ranked nothing. Usually means the questions are too easy for the corpus rather than that the configurations are equivalent — ask harder questions, or compare at a smaller cut-off where there is room to separate them. Only raised when two or more configurations tie at the top: one configuration at 1.000 is a result, not a ceiling. |
| `METRIC_FAILED` | `evaluate()` (`score/metrics.py`) | `CAUTION` | A metric raised while scoring this run -- built-in or a registered custom one -- and was left out of `RunResult.metrics` entirely rather than scored as `0.0`. Check `RunResult.has(name)` before reading a metric that could be missing. See [metrics.md](metrics.md#metrics-are-a-plugin-family). |

### Codes from the rest of the pipeline

The scoring codes above tell you the eval set or the resolution is shaky. These tell you the
*run* is — something was cut, skipped, approximated or given up on, and the number beside it is
smaller than it looks.

| Code | Stage | Severity | What happened, and what to do |
|---|---|---|---|
| `empty_text_layer` | parse | CAUTION | Pages with no extractable text — almost always scans. Nothing on them can be retrieved without OCR, so recall is capped by the corpus rather than by the config. |
| `parser_fallback` | parse | CAUTION | The chosen parser could not read a file and another one did. That row mixes two parsers, so the parser axis is not measuring what it claims for those documents. |
| `empty_chunk_set` | chunk | INVALID | A configuration produced no chunks at all, so every query scores zero. Either the parser found no text or the chunker rejected all of it. Not a bad configuration — a broken one. |
| `one_chunk_per_document` | chunk, or ingest | CAUTION | Every document came back as a single unit, so retrieval ranked documents rather than passages and configurations that tie here have not been compared — they have not been tried. Raised about whichever axis actually caused it. Usually the chunker: a chunk size larger than the documents leaves them whole, and the fix is smaller sizes or longer documents (`contextgrid profile` says the same thing from the corpus alone). But an ingestion strategy that returns the passage its small chunks came from — `parent-document`, `summary` — collapses the units on purpose, and then the warning names the strategy, reports how many units the chunker really cut, and says so: a smaller chunk size cannot help, because the strategy groups whatever it produces back together. Return finer units, or compare against `ingestion: plain`. |
| `chunk_exceeds_model_context` | assemble | CAUTION | A chunk is larger than the generator's context window, so part of what was retrieved never reaches the model. Reduce the chunk size or raise `k`'s budget. |
| `input_truncated` | embed | CAUTION | Text longer than the model's context was cut before embedding. If the answer was after the cut, it cannot be retrieved — and nothing else in the run will say so. |
| `missing_query_prefix` | embed | CAUTION | Nothing is known about whether this model wants `query:`/`passage:` prefixes. If it was trained with them, every score for that arm is several points low, and low *unevenly* against the other arms. Set them explicitly, or pass `query_prefix=""` to say none are needed. |
| `unnormalised_vectors` | embed | CAUTION | The model returned a different width than the config declared. Usually a wrong model name; always worth checking before trusting a cached run. |
| `ann_recall_loss` | index | CAUTION | Every index on the axis is approximate and none is exact, so nothing measures what the approximation cost. Add `dense` or `faiss:flat`. |
| `quantization_applied` | index | INFO | Vectors were compressed. Real memory saved, real recall lost — `recall_against_exact` is how much. |
| `impossible_combination` | run | INFO | Combinations that cannot be built were dropped, such as a dense index with no embedder. The axes are almost certainly what you meant; this is the product of them that is not. |
| `arm_not_measured` | retrieve | CAUTION | A row ran under a configuration other than the one written, so an arm somebody asked for was never measured. One case raises it: `widened` with no reranker, no transform, no ingestion and an exact index provably returns exactly what plain search returns, so the matrix folds it onto plain search rather than paying twice for one run — and the row is then labelled, manifested and read as plain search. Put a reranker, a transform or an ingestion strategy behind it, or use an approximate index, and it becomes a real arm again. |
| `budget_reached` | run | CAUTION | `budget_seconds` or `budget_usd` ran out, so the sweep stopped early or never started. The leaderboard is partial — `Results.is_partial` says so without reading warnings, and `Results.partial_note()` is the sentence. This code means only this; two facts that used to share it are the two rows below. |
| `configuration_failed` | run | CAUTION | One configuration could not be built, run or scored, so it is absent from the leaderboard rather than scored zero — a zero is a measurement, and nothing measured it. The rest of the sweep carried on, and `Results.is_partial` is true because fewer rows ran than were planned. The message names the configuration and the error. |
| `model_not_priced` | cost | CAUTION | A model has no published price, here or in litellm's table, so it is costed at zero. Any cost comparison involving it understates what it charges. |
| `no_cost_ceiling` | run, or cost | CAUTION | Nothing has been reached — there is no ceiling to reach. Raised when a plugin that calls a model per question or per chunk runs with no `budget_usd` and no `budget_seconds`, and when a sweep sets both `machine_usd_per_hour` and `budget_usd`, since the budget counts token spend and deliberately not machine time. |
| `generation_failed` | generate | CAUTION | A generator or judge failed on a question. The rest of the run stands; that question has no answer score. |
| `cache_miss_storm` | cache | INFO | Very little was reused. Usually means something in the cache key changed — a tokenizer, a parser version — and the sweep is slower than it needs to be. |
| `non_deterministic_stage` | varies | CAUTION | A stage did something a rerun may not repeat. Raised by the LLM-backed ingestion strategies when a model call fails and the chunk is indexed as written, so the row mixes two strategies. |

**Six codes were deleted rather than documented.** Each was declared and raised nowhere, which
is worse than a missing code: from the outside it reads as coverage. Two of them described
conditions worth reporting and are now real — `ann_recall_loss` and `missing_query_prefix`,
both in the table above. The other four described things already carried better elsewhere:
`offsets_exact` on a chunk or a parse says whether text is a literal slice, and a failed
`GoldAnchor.occurrence` comes back as `anchor_not_found` with the reason attached.

Two tests keep this page honest. One asserts every `WarningCode` is raised somewhere in `src/`;
the other asserts every one appears here. The second passed for a while by accident, because
`docs/COVERAGE.md` lists undocumented names and the search was finding them there — a coverage
report counting as its own coverage.


## The failure taxonomy: Seven Failure Points

A leaderboard says a configuration scored 0.62. It doesn't say *why* the other 0.38
failed, and "improve retrieval" isn't an action anyone can take.
`contextgrid.diagnose.taxonomy` sorts every failing question into one of the **Seven
Failure Points** (Barnett et al., 2024 — drawn from three real production RAG systems),
which turns a score into a list of things to actually try. FP4–FP7 need a generator to
observe; a retrieval-only run classifies FP1–FP3 and says plainly that it can't see the
rest.

| Failure point | Meaning | Retrieval-observable? |
|---|---|---|
| FP1 Missing content | Evidence was written down for this question, and it isn't in the index — not a retrieval failure. A question with no evidence written down at all is *not* FP1; see below. | Yes |
| FP2 Missed top-ranked | Evidence was retrieved, just not ranked high enough. Rerank. | Yes |
| FP3 Not in context | Evidence was found but fell outside `k`. Widen `k`, or consolidate. | Yes |
| FP4 Not extracted | Present in context; the generator missed it. Not a retrieval fix. | No |
| FP5 Wrong format | The answer was there, in a shape nothing could use. | No |
| FP6 Wrong specificity | Too general or too narrow to answer what was asked. | No |
| FP7 Incomplete | Partially answered — evidence spread wider than what was retrieved. | Yes (partial) |

### `diagnose()`

Five questions, covering a success, all three retrieval-observable failures, and the one
case that is **not** a failure of the pipeline at all:

```python
>>> from contextgrid.core.evalset import EvalItem, EvalSet, GoldSpan
>>> from contextgrid.core.span import Span
>>> from contextgrid.diagnose.taxonomy import diagnose
>>> items = [
...     EvalItem(id="q1", question="q1", gold=(GoldSpan(span=Span("d", 0, 10)),)),  # rank 1
...     EvalItem(id="q2", question="q2", gold=(GoldSpan(span=Span("d", 0, 10)),)),  # rank 8
...     EvalItem(id="q3", question="q3", gold=(GoldSpan(span=Span("d", 0, 10)),)),  # rank 150
...     EvalItem(id="q4", question="q4", gold=()),                        # no evidence at all
...     EvalItem(id="q5", question="q5", gold=(GoldSpan(span=Span("d", 0, 10)),)),  # never found
... ]
>>> evalset = EvalSet(id="e", items=tuple(items))
>>> qrels = {"q1": {"cA": 2}, "q2": {"cH": 2}, "q3": {"cZ": 2}}   # nothing matched q5's gold
>>> run = {
...     "q1": ["cA", "c2", "c3"],
...     "q2": [f"c{i}" for i in range(7)] + ["cH"],        # rank 8
...     "q3": [f"c{i}" for i in range(149)] + ["cZ"],       # rank 150
...     "q4": ["c1", "c2"],
...     "q5": ["c1", "c2"],
... }
>>> report = diagnose(evalset, qrels, run, k=5, deep_k=100)
>>> for d in report.diagnoses:
...     print(d)
q1: none -- evidence at rank 1
q2: fp2_missed_top_ranked -- the evidence was retrieved at rank 8, just outside the top 5
q3: fp3_not_in_context -- the evidence ranked 150, far below the top 5
q5: fp1_missing_content -- no chunk in this index holds the evidence for this question
```

**`q4` is not in that list, and that is the point.** It carries no gold spans and no
anchors — nobody ever wrote down what the right answer was — so there is nothing to be
right or wrong about and it is never diagnosed. It goes to `report.no_ground_truth`
instead, and `total_items` is the only place the two are added back together:

```python
>>> report.no_ground_truth
['q4']
>>> len(report.diagnoses), report.total_items
(4, 5)
```

`summary()` counts the failures against the four questions it actually scored, then says
separately what happened to the fifth:

```python
>>> print(report.summary())
3 of 4 questions failed. 33% of those are fp2_missed_top_ranked: the evidence was retrieved but ranked too low to be used. This is what a reranker is for, and it is the cheapest failure on this list to fix. This was a retrieval-only run, so failure points four to seven -- the ones about what the generator did with the context -- cannot be seen from here. A further 1 question (q4) has no ground truth -- no gold spans and no anchors -- so it was not scored at all. That is a gap in the eval set, not a fault in this pipeline.
>>> from contextgrid.diagnose.taxonomy import cluster
>>> cluster(report)
{'fp1_missing_content': ['q5'], 'fp2_missed_top_ranked': ['q2'], 'fp3_not_in_context': ['q3']}
```

Pass `summary(include_unscored=False)` to drop that last sentence, for a caller that has
already said it in its own words — `Results.summary` does exactly that, so the paragraph
under a leaderboard does not say it twice.

#### FP1 is about evidence that exists, not evidence nobody wrote

This distinction cost real debugging time before it was made, so it is worth stating
directly. Both cases end with a question that scored nothing. They have opposite fixes:

| | `q4` — eval-set gap | `q5` — FP1 missing content |
|---|---|---|
| What the item holds | No gold spans **and** no anchors | Gold or an anchor, written down properly |
| What went wrong | Nobody finished writing the question | The evidence is genuinely not in this index |
| Where it lands | `report.no_ground_truth` | `report.diagnoses`, as `fp1_missing_content` |
| In the histogram? | No — `counts()` and `cluster()` never see it | Yes |
| The fix | Write the evidence, or delete the question | Check the parser, the chunker, or the corpus |

Diagnosing `q4` as FP1 — which is what used to happen — told the reader their parser,
chunker or corpus had lost the evidence for a question that never had any. It also pushed
the failure histogram past the number of questions actually scored, which is how the bug
was eventually noticed.

FP1 itself still splits two ways, and the message says which:

- **`no chunk in this index holds the evidence for this question`** — the gold resolved,
  and no chunk matched it. A chunking or corpus problem.
- **`the quoted evidence for this question could not be found in this parse`** — the item
  has anchors and no gold, so this parser never located the quote. A parse problem, and
  the parser axis is exactly how you would find out which parser does better. See
  [spans-and-anchors.md](spans-and-anchors.md#is_answerable-vs-is_resolved) — this is an
  item that is answerable but not resolved.

`deep_k` (default 100 when called directly, must be passed explicitly as shown above — it
is what separates FP2 from FP3: found within `deep_k` but outside `k` is "ranked too low,
rerank it" (FP2); found beyond `deep_k` is "not in the index at this depth" (FP3). Those
have completely different fixes, which is the entire point of splitting them.

FP7 (`INCOMPLETE`) fires when some but not all relevant chunks for a question made it into
the top `k`:

```python
>>> item = EvalItem(
...     id="q1", question="q1",
...     gold=(GoldSpan(span=Span("d", 0, 10)), GoldSpan(span=Span("d", 50, 60))),
... )
>>> evalset = EvalSet(id="e", items=(item,))
>>> qrels = {"q1": {"cA": 2, "cB": 2}}
>>> run = {"q1": ["cA", "c2", "c3", "c4", "c5"]}   # only cA (of 2 relevant chunks) in top 5
>>> str(diagnose(evalset, qrels, run, k=5).diagnoses[0])
'q1: fp7_incomplete -- 1 of 2 relevant chunks made it into the top 5'
```

### Reading a `FailureReport`

- `.diagnoses` — one `Diagnosis` per question that had ground truth to score against.
- `.no_ground_truth` — the ids of questions that had none, so were never diagnosed. These
  appear in no count, no cluster and no histogram; they are a note about the eval set.
- `.total_items` — `len(.diagnoses) + len(.no_ground_truth)`, i.e. every question the eval
  set held. The only number that adds the two back together.
- `.counts()` — a `{failure_value: count}` dict, sorted, including successes (`"none"`).
  Over `.diagnoses` only, so it never exceeds the number of questions actually scored.
- `.failures()` — every non-`NONE` diagnosis.
- `.of(failure_point)` — diagnoses matching one `FailurePoint`.
- `.dominant` — the `FailurePoint` accounting for the most failures, or `None` if
  everything succeeded.
- `.summary()` — the report as a sentence: what dominates, and — because
  `Diagnosis.remedy` looks up `REMEDIES[failure]` — what to actually try next. Adds a
  closing sentence naming any `no_ground_truth` questions; `summary(include_unscored=False)`
  drops it.
- `cluster(report)` — failing question ids grouped by failure point. "These twelve
  failures are all evidence ranked just outside k" is worth more than twelve separate
  traces, because it's one fix rather than twelve investigations.

Every `Diagnosis` also carries `.gold_rank` (where the evidence actually landed, or
`None`) and `.retrieved` (how many chunks came back) — useful when `.detail` alone doesn't
say enough.

See [the composite score](composite.md) for how these same runs roll up into one number,
and [spans and anchors](spans-and-anchors.md) for where the `gold`/`qrels` that
`diagnose()` consumes come from in the first place.

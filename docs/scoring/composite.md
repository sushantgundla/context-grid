# The composite score

A leaderboard with fourteen columns is a leaderboard nobody reads to the end of. A single
0–100 number is what people actually want — and it is also the easiest thing in this
package to make dishonest, so `contextgrid.report.composite` follows three rules, stated
plainly here because they're what makes the number worth trusting.

## Getting one from a real run

```python no-run: needs a Results object from a sweep; see docs/recipes/
score = results.composite()  # the leading configuration
score = results.runs[0].composite()  # or any particular one
print(score.summary())
```

**Do not build the input by hand from `RunResult.metric()`.** That method takes a default,
because a leaderboard sorting on a metric some runs lack needs *some* number to sort by. Feed
that default into a composite and it stops being a placeholder and becomes a measurement --
and since the mean is harmonic, one fabricated zero takes the whole score to nought. A
configuration with perfect recall reporting 0/100 is not a hypothetical; it is what this API
did until somebody tried to use it from the documentation alone.

`RunResult.has(name)` answers the question `metric()` cannot: whether the run measured it at
all. `leaderboard(extra=[...])` now omits a column no run computed, and says so, rather than
filling it with zeroes.


## Rule 1: harmonic, not arithmetic

A configuration retrieving at 0.95 and generating faithfully at 0.10 averages
*arithmetically* to 0.53 — which reads as middling. It is not middling. It's a system that
confidently invents answers, and 0.53 hides that behind a good retriever. The harmonic
mean puts it at 18, because a chain is worth what its weakest link is worth. Every
composite score built on an arithmetic mean is a way of not noticing the worst thing about
a system.

```python
>>> from contextgrid.report.composite import composite
>>> metrics = {
...     "recall@5": 0.95,
...     "ndcg@5": 0.95,
...     "faithfulness": 0.10,
...     "answer_relevancy": 0.10,
... }
>>> result = composite(metrics)
>>> result.score
18.095238095238095
>>> result.parts
{'retrieval': 0.95, 'generation': 0.1}
>>> 100 * (0.95 + 0.10) / 2   # what an arithmetic mean would have said instead
52.5
```

`harmonic_mean(values)` — `len(values) / sum(1/v for v in values)` — is the piece doing
this. **Zero anywhere gives zero**, deliberately: a system that generates nothing faithful
has no score worth reporting, however well it retrieves.

```python
>>> from contextgrid.report.composite import harmonic_mean
>>> harmonic_mean({"retrieval": 0.95, "generation": 0.0})
0.0
```

## Rule 2: only what ran

Somebody sweeping ingestion and retrieval alone has no generator in the run. Scoring the
missing generation dimension as zero would punish them for a question they never asked.
`composite` scores the dimensions that produced numbers, and it names which ones those
were — so a 73 over three dimensions is never mistaken for a 73 over six.

```python
>>> partial_metrics = {"recall@5": 0.8, "ndcg@5": 0.7, "char_recall@5": 0.6}
>>> partial = composite(partial_metrics)
>>> partial.summary()
'67/100 over 2 dimension(s): chunk, retrieval (not measured: embed, generation, parse)'
>>> partial.parts
{'chunk': 0.6, 'retrieval': 0.75}
>>> partial.missing
{'parse': 'no value for evidence_resolvable', 'embed': 'no value for embedding_quality', 'generation': 'no value for faithfulness or answer_relevancy'}
```

`CompositeScore.summary()` is the right thing to print or log — never the bare number.
`.dimensions` (`tuple(sorted(self.parts))`) and `.missing` travel with `.score` for
exactly this reason: a 73 over `retrieval, generation` is a different claim from a 73 over
all five dimensions, and printing them identically invites the comparison that's wrong.

## Rule 3: comparable only within a run

Two scores computed over different dimension sets are not comparable — a 67 that measured
`chunk, retrieval` and an 82 that measured all five dimensions are not "82 beats 67". This
is why `.dimensions` and `.missing` are attached to every `CompositeScore` rather than
left as a footnote the reader has to go find.

## The five dimensions

```python
DIMENSION_METRICS: dict[str, tuple[str, ...]] = {
    "parse": ("evidence_resolvable",),
    "chunk": ("char_recall",),
    "embed": ("embedding_quality",),
    "retrieval": ("recall", "ndcg"),
    "generation": ("faithfulness", "answer_relevancy"),
}
```

| Dimension | Metric name(s) | What it asks |
|---|---|---|
| `parse` | `evidence_resolvable` | Did the parse make the evidence findable at all? Comes out of [anchor resolution](spans-and-anchors.md#the-failure-case-is-the-measurement) — the only dimension whose failure makes every later number meaningless. |
| `chunk` | `char_recall` | Did the chunking return the evidence intact? See [character recall](metrics.md#character-level-precision-recall-and-f1). **Not** `char_precision`: that is bounded above by roughly `gold_chars / (k × chunk_size)` — about 0.005 for 512-token chunks — so putting it in a harmonic mean makes it the only dimension that counts, and rewards tiny chunks over good ones. Precision is still computed and still reported as a column; it measures wasted context, which is a real cost but not a 0–1 quality score. |
| `embed` | `embedding_quality` | Can this embedder discriminate on this corpus at all? Measurable with no eval set, which makes it the only dimension you can score before writing a single question. |
| `retrieval` | `recall`, `ndcg` | Did the right passages come back? See [metrics](metrics.md). |
| `generation` | `faithfulness`, `answer_relevancy` | Is the answer supported by the retrieved evidence, and does it address the question? |

**One metric per dimension is deliberate.** Averaging four retrieval metrics into a
retrieval score and then averaging *that* with generation would give retrieval four votes
and generation one — an opinion about what matters, dressed up as arithmetic. When a
dimension does list more than one metric name (`retrieval`, `generation`), those are two
views of the *same* thing and average arithmetically with each other before the harmonic
mean runs across dimensions — one being low is not the same kind of failure as a whole
dimension being low.

`composite(metrics, k=None, dimensions=None)` looks up each metric by name, trying the bare
name first and then `f"{name}@{k}"` — so `recall` finds `recall@5` from a normal
[`evaluate()`](metrics.md#aggregating-over-a-run) run without the caller spelling out the
cut-off. Pass a custom `dimensions` mapping to score a different set — the same shape as
`DIMENSION_METRICS` above.

### Which cut-off `k` means

`k` used to default to **5**, which is a guess about a run rather than a fact about it. A
sweep with `headline: recall@1` emits every metric at `@1` — `recall@1`, `ndcg@1`,
`char_recall@1` — so the lookup went hunting for `char_recall@5`, found nothing, and printed
`88/100 over 3 dimension(s): embed, parse, retrieval (not measured: chunk, generation)` for a
run whose `char_recall@1` was **0.8824**. Rule 2 is about not scoring a dimension that did not
run. Reporting a dimension that *did* run as unmeasured is the same dishonesty pointing the
other way, and it is worse, because the score still gets printed beside it.

So `k=None` — the default — reads the cut-off off the metrics themselves: the one most of this
run's keys carry, which is the headline's, because that is what the runner scores at. A metric
absent at that cut-off falls back to the nearest cut-off it does have, since a value measured
at *some* k is a measurement. Nothing here ever falls back to `0.0`.

```python
>>> blind_run = {"recall@1": 0.8824, "ndcg@1": 0.8824, "char_recall@1": 0.8824}
>>> composite(blind_run).dimensions
('chunk', 'retrieval')
>>> composite(blind_run).sources["chunk"]
('char_recall@1',)
```

**Passing `k` explicitly still means exactly that `k`.** A caller asking about the top 5 is
asking a question, and answering with a number measured over the top 3 would be answering a
different one.

```python
>>> composite({"recall@3": 1.0}, k=5).missing["retrieval"]
'no value for recall or ndcg'
>>> composite({"recall@3": 1.0}, k=3).parts["retrieval"]
1.0
```

`.sources` records the exact key behind every dimension, and `summary()` names them whenever
the cut-offs disagree — a score that averaged character recall over the top 2 with recall over
the top 5 is a fair thing to print, but not silently.

```python
>>> composite({"recall@5": 0.6, "ndcg@5": 0.6, "char_recall@2": 0.9}).summary()
'72/100 over 2 dimension(s): chunk, retrieval (cut-offs differ: chunk from char_recall@2; retrieval from recall@5, ndcg@5) (not measured: embed, generation, parse)'
```

### Values outside 0–1 are ignored, not clamped

```python
>>> composite({"recall@5": 3.2}).missing["retrieval"]   # not a 0-1 value; something is wrong upstream
'no value for recall or ndcg'
```

A composite is a comparison of like-scaled things. `3.2` isn't on that scale, and silently
squashing it to `1.0` would put a number into the score that nothing actually measured —
`_lookup` treats it as absent instead, and it shows up in `.missing` like any other
unmeasured dimension (here, every dimension is missing, since the input supplied only one
unusable value).

Next: what `.missing` and low dimension scores mean question-by-question, not just in
aggregate — see [diagnostics](diagnostics.md).

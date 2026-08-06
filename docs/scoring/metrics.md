# Metrics

`contextgrid.score.metrics` implements six retrieval metrics, plus the character-level
measures in `contextgrid.score.resolve`. All six ranking metrics are cross-checked
against [`ranx`](https://github.com/AmenRa/ranx) — the peer-reviewed reference
implementation (ECIR 2022) — on randomly generated qrels and runs, in CI
(`tests/unit/test_metrics_vs_ranx.py`). They're implemented from scratch rather than
delegated, because the core package installs with numpy and nothing else — a metrics
library that drags in a JIT compiler would make `pip install context-grid` a minute-long
event. "Our numbers agree exactly with ranx on ten thousand random cases" is a stronger
claim than "we used ranx", and it's one anyone can re-run.

Every metric takes graded relevance judgements (`{chunk_id: grade}`, the shape produced
by [`SpanResolver.qrels`](spans-and-anchors.md#spanresolver-and-its-three-policies)) and
a ranked list of chunk ids. Anything with `grade > 0` is relevant; `ndcg_at_k` uses the
grades themselves.

## The six metrics

| Function | Question it answers |
|---|---|
| `recall_at_k(judgements, ranked, k)` | Of everything relevant, what fraction is in the top k? |
| `precision_at_k(judgements, ranked, k)` | Of the top k, what fraction is relevant? |
| `hit_rate_at_k(judgements, ranked, k)` | Is anything relevant in the top k at all? (1.0 or 0.0) |
| `reciprocal_rank(judgements, ranked, k)` | `1 / position` of the first relevant hit, 0 if none in top k. |
| `average_precision(judgements, ranked, k)` | Precision at each relevant hit, averaged — MAP. |
| `ndcg_at_k(judgements, ranked, k)` | Graded ranking quality against the best possible ordering. |

```python
>>> from contextgrid.score.metrics import (
...     hit_rate_at_k, ndcg_at_k, precision_at_k, rank_of_first_relevant,
...     recall_at_k, reciprocal_rank,
... )
>>> judgements = {"a": 2, "b": 1, "z": 0}   # a: fully relevant, b: partially, z: irrelevant
>>> ranked = ["x", "a", "y", "b", "w"]
>>> recall_at_k(judgements, ranked, 5)
1.0
>>> precision_at_k(judgements, ranked, 5)
0.4
>>> hit_rate_at_k(judgements, ranked, 5)
1.0
>>> reciprocal_rank(judgements, ranked, 5)
0.5
>>> rank_of_first_relevant(judgements, ranked)
2
>>> round(ndcg_at_k(judgements, ranked, 5), 3)
0.643
```

Notes worth knowing before you read a leaderboard:

- **`recall_at_k` is the headline metric for RAG.** A generator doesn't need the evidence
  ranked first, it needs the evidence *present* — so recall at the `k` you actually put in
  the prompt is what predicts whether the answer can be right at all.
- **`precision_at_k` divides by `k`, not by how many chunks were returned.** Dividing by
  the returned count would let a configuration that finds only 3 chunks score 1.0 for
  precision@10 — flattering exactly the configurations that are failing to fill k.
- **Ranks are one-based.** `rank_of_first_relevant` returns `None` — never a sentinel like
  `0` or `-1` — when nothing relevant appears in the searched window. `0` would be
  indistinguishable from "found at position 0" under any careless `if rank:` check, so the
  type signature is `int | None` on purpose.
- **`iou`/`coverage_of` don't enter here at all.** These are chunk-*ranking* metrics; the
  relevance judgements they consume already came out of
  [`SpanResolver`](spans-and-anchors.md#spanresolver-and-its-three-policies) upstream.

## The MAP convention: `len(relevant)`, not `min(relevant, k)`

`average_precision`'s denominator is the **total number of relevant chunks for that
query**, not `min(len(relevant), k)`. Both conventions exist in the wild and they
disagree substantially. This package follows trec_eval's convention, which `ranx` follows
and which the IR literature means by "MAP@k" — but it is easy to assume the Kaggle-style
`min(relevant, k)` instead, and the two give very different numbers.

The consequence is worth seeing directly: a query with 20 relevant chunks cannot score
above 0.25 at k=5, however perfect the ranking, because only 5 of the 20 relevant chunks
can possibly appear in the top 5.

```python
>>> from contextgrid.score.metrics import average_precision
>>> judgements = {f"c{i}": 2 for i in range(20)}   # 20 relevant chunks
>>> ranked = [f"c{i}" for i in range(20)]           # perfect ranking: all 20, in order
>>> round(average_precision(judgements, ranked, 5), 3)
0.25
```

That is a property of the metric, not a fact about the retriever — which is why MAP is
reported alongside `recall_at_k` rather than instead of it in
[`evaluate`](#aggregating-over-a-run) below.

## Aggregating over a run

```python
>>> from contextgrid.score.metrics import evaluate, per_query
>>> qrels = {"q1": {"a": 2, "b": 1, "z": 0}}
>>> run = {"q1": ["x", "a", "y", "b", "w"]}
>>> evaluate(qrels, run, ks=(1, 5), metrics=("recall", "ndcg"))
{'recall@1': 0.0, 'recall@5': 1.0, 'ndcg@1': 0.0, 'ndcg@5': 0.6433224083306327}
>>> per_query(qrels, run, "recall", 5)
{'q1': 1.0}
```

`evaluate(qrels, run, ks=DEFAULT_KS, metrics=all six)` scores a whole run, averaged over
queries. `DEFAULT_KS = (1, 3, 5, 10, 20)` — small values show precision, large ones show
whether the evidence is present at all, which for RAG is usually the question that
matters.

Two exclusion rules that change what a mean actually means:

- **A query with no relevant chunks (`qrels[qid]` empty) is left out of the average
  entirely**, not scored as zero. It can't be got right or wrong, and including it as a
  zero would drag the mean down for a reason that has nothing to do with the retriever.
- **A query present in `qrels` but missing from `run` scores zero.** That's a real
  failure — the configuration never answered it — so it counts against the mean.

`evaluate` raises `ValueError` for an unknown metric name, listing the available ones
(`available_metrics()` returns `("hit_rate", "map", "mrr", "ndcg", "precision", "recall")` by
default — sorted, and it grows if you register a custom metric; see below).

`per_query(qrels, run, metric, k)` returns one score per query instead of a mean. This is
the input [significance testing](significance.md) needs — a paired test operates on
per-question scores, not on the aggregate — and it's also how you'd find the specific
questions where two configurations actually disagree.

`mean_rank_of_first_relevant(qrels, run, k=None)` returns `(mean_rank, unanswered_count)`
as a pair, deliberately — a mean rank of 2.1 computed over 90 of 100 queries means
something very different from the same number over 40 of them, and reporting the mean
alone would hide which one you're looking at.

## Metrics are a plugin family

Every other axis in this package — chunker, retriever, reranker, ingestion strategy — is a
plugin behind a `Registry`. A metric is one too: `name`, `version`, and one method,
`evaluate(judgements, ranked, k) -> float` — the same shape `recall_at_k` and the other five
functions above already have, just wrapped so `METRICS` (`contextgrid.score.METRICS`) can
resolve one by name the way `CHUNKERS` resolves `"recursive"`.

```python
>>> from contextgrid.score import METRICS, Metric, get_metric
>>> {"recall", "precision", "hit_rate", "mrr", "map", "ndcg"} <= set(METRICS.names())
True
>>> built = get_metric("recall")
>>> built.name, built.version
('recall', '1')
>>> isinstance(built, Metric)
True
```

`evaluate()` and `per_query()` (above) resolve every metric name through `METRICS` — not a
private, six-entry table — so a registered custom metric is computed for real, per question,
alongside the built-ins. Registering one is exactly what registering a `Chunker` or a
`RetrievalStrategy` looks like — see [registry.md](../internals/registry.md) — except a
metric's `evaluate()` takes relevance judgements and a ranked list instead of a document or a
searcher:

```python
>>> from collections.abc import Mapping, Sequence
>>> from dataclasses import dataclass
>>> from typing import ClassVar
>>>
>>> @dataclass(frozen=True, slots=True)
... class Top1Only:
...     '''1.0 only when the very first result is relevant -- ignores everything below rank 1.'''
...     name: ClassVar[str] = "top1_only"
...     version: ClassVar[str] = "1"
...
...     def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
...         if not ranked:
...             return 0.0
...         return 1.0 if judgements.get(ranked[0], 0) > 0 else 0.0
...
>>> if "top1_only" not in METRICS:
...     _ = METRICS.register("top1_only", doc="1.0 only when the very first result is relevant.")(
...         Top1Only
...     )
```

`recall_at_k` can't tell these two runs apart — the same relevant chunk turns up in the top 5
either way — but `top1_only` is exactly sensitive to *where*:

```python
>>> qrels = {"q1": {"a": 2, "b": 1, "z": 0}}
>>> run_a_first = {"q1": ["a", "y", "b", "w"]}
>>> run_a_second = {"q1": ["y", "a", "b", "w"]}
>>> evaluate(qrels, run_a_first, ks=(5,), metrics=("recall", "top1_only"))
{'recall@5': 1.0, 'top1_only@5': 1.0}
>>> evaluate(qrels, run_a_second, ks=(5,), metrics=("recall", "top1_only"))
{'recall@5': 1.0, 'top1_only@5': 0.0}
```

Cut-offs stay part of the *name*, not the metric — `evaluate()` calls `top1_only.evaluate(...)`
once per `k` in `ks` and builds `top1_only@5` itself, the same as every built-in. A metric that
needs its own `k` on top of that (an unusual case) takes it as an ordinary constructor
parameter, resolved through the registry's spec-string parsing like any other plugin's
parameters — see [registry.md](../internals/registry.md#spec-strings-create-and-parse_spec).

Two things a registered metric gets for free, with no further wiring:

- **`run.headline` accepts any registered name**, not a hard-coded list — `run.headline:
  top1_only@5` works exactly like `run.headline: recall@5`. The config validator
  (`RunConfig.validate` in `config/schema.py`) asks `available_metrics()`, so a typo is still
  caught before a sweep starts, and the error names what's actually registered.
- **`run.metrics` names extra metrics to compute** alongside the built-ins and the headline's
  own — `run.metrics: [top1_only]`, or a list of several. See
  [configuration.md](../guide/configuration.md#run--how-the-sweep-executes).

A metric that fails does not take the run down, and does not silently score zero.
`evaluate()` catches an exception from a metric's `evaluate()` and leaves that metric's keys
out of the result rather than raising or reporting `0.0` — `RunResult.has(name)` is how a
result says "this was never measured", the same guarantee `row()` and `leaderboard()` already
make for a metric nobody computed. Pass a `WarningLog` (`evaluate(..., warnings=log)`) to have
the failure recorded with a `METRIC_FAILED` warning; `Runner.run_one` does this for every
sweep — see [diagnostics.md](diagnostics.md#codes-youll-see-from-scoring).

A full metric written, registered, swept and shown on a leaderboard —
[extending.md](../internals/extending.md) has the worked example end to end.

## Character-level precision, recall and F1

Chunk-level metrics above answer "was the evidence in the top k retrieved chunks?" — they
say nothing about how much *other* text came along with it. A configuration returning
enormous chunks can score `recall_at_k` of 1.0 while filling the context window with text
that has nothing to do with the question; chunk recall applauds it, character precision
shows what it costs.

These live in `contextgrid.score.resolve` (not `metrics`) because they work directly on
`Span` arithmetic and don't depend on a resolution policy at all — the honest, policy-free
check on the chunk-level numbers above them.

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.evalset import EvalItem, GoldSpan
>>> from contextgrid.core.span import Span
>>> from contextgrid.score.resolve import character_f1, character_precision, character_recall
>>> gold = GoldSpan(span=Span("doc1", 100, 120), grade=2)   # 20 characters of evidence
>>> item = EvalItem(id="q1", question="...", gold=(gold,))
>>> huge_chunk = Chunk(id="c1", span=Span("doc1", 0, 520), text="x" * 520)  # 520 chars total
>>> round(character_recall(item, [huge_chunk]), 3)     # all the evidence is in there
1.0
>>> round(character_precision(item, [huge_chunk]), 3)  # ...buried in 25x its weight in noise
0.038
>>> round(character_f1(item, [huge_chunk]), 3)
0.074
```

Chunk recall@5 of 1.0 alongside character precision of 0.04 is the sentence this metric
exists to let you write: the right evidence arrived, buried in 25x its weight in
irrelevant text, and every generation call downstream is paying for it in tokens.

| Function | Meaning |
|---|---|
| `character_recall(item, retrieved)` | Fraction of gold characters present anywhere in the retrieved chunks. Union-based: gold split across two chunks counts as fully retrieved when both come back. |
| `character_precision(item, retrieved)` | Fraction of retrieved characters that are gold. The metric that exposes context waste. |
| `character_f1(item, retrieved)` | Harmonic mean of the two above. |
| `retrieved_character_count(retrieved)` | Total characters sent downstream, overlapping chunks counted once — the real driver of the generation bill. |
| `gold_coverage_by_chunk(item, chunks)` | Per chunk, the fraction of this question's gold characters it holds. Feeds the per-query inspector — "this chunk holds 60% of the evidence" rather than a flat relevant/not mark. |

Both `character_recall` and `character_precision` are built on
`intersection_length`/`total_length` from `contextgrid.core.span`, which merge overlapping
spans first — see [spans and anchors](spans-and-anchors.md#interval-algebra-over-sets-of-spans)
— so overlapping chunks on either side never double-count a character.

Next: how these per-query numbers turn into a claim you can defend — see
[significance testing](significance.md) — and how they roll up into one number — see
[the composite score](composite.md).

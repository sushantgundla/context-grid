# Significance testing

A leaderboard reporting 0.71 against 0.68 invites exactly one conclusion, and on 40
questions that conclusion is usually wrong — the difference is well inside what you'd get
from simply asking 40 different questions, and nothing on the screen says so. A
leaderboard without a significance test invites reading noise as a finding.
`contextgrid.score.significance` exists to say, plainly, whether two configurations are
actually distinguishable on the eval set you ran them on.

## Why paired, not independent

Two configurations here are always run on **identical questions**. That's a gift: pairing
uses each question as its own control and removes the variance that comes from some
questions simply being harder than others. It's far more sensitive than comparing two
independent means, and it's the standard test in information retrieval for exactly this
situation.

Both the confidence interval and the p-value below are paired for that reason.

## `bootstrap_interval`: a confidence interval for one configuration

```python
>>> from contextgrid.score.significance import bootstrap_interval
>>> scores = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]   # per-question recall@5
>>> bootstrap_interval(scores)
Interval(estimate=0.75, low=0.5, high=1.0, confidence=0.95)
```

Bootstrapping — resampling the questions with replacement, many times, and taking the
spread of resample means — rather than a normal approximation, because per-question
retrieval scores are nothing like normal. `recall_at_k` on a single question is often just
0 or 1; there's no bell curve to approximate. Resampling makes no assumption about the
shape of the underlying distribution.

`Interval` has `.estimate`, `.low`, `.high`, `.width`, and `.excludes_zero` (true when the
whole interval sits on one side of zero — the property a *difference* interval needs).

## `paired_bootstrap` and `randomisation_test`: comparing two configurations

`paired_bootstrap(left, right)` builds a confidence interval for the **difference**
between two configurations, resampling matched question-pairs together — so question
difficulty stays constant across the resample and only the part of the variance that's
actually about the configurations comes through.

`randomisation_test(left, right)` is the paired p-value. The null hypothesis is that the
two configurations are interchangeable — so on any given question, their two scores could
equally well have been swapped. Swapping them at random many times builds the
distribution of differences you'd see under that null, and the p-value is how often
chance alone beats what was actually observed. No distributional assumption, exact in the
limit.

Both live behind `compare`, below — you'll rarely call either directly.

## `compare`: the function you actually use

```python
>>> from contextgrid.score.significance import compare
>>> import random
>>> rng, rng2 = random.Random(7), random.Random(11)
>>> qids = [f"q{i}" for i in range(40)]
>>> config_a = {qid: (1.0 if rng.random() < 0.71 else 0.0) for qid in qids}
>>> config_b = {qid: (1.0 if rng2.random() < 0.68 else 0.0) for qid in qids}
>>> result = compare(config_a, config_b, left="chunk_512", right="chunk_256", metric="recall@5")
>>> print(result.verdict())
chunk_512 and chunk_256 are not distinguishable on this eval set (n=40). The gap of +0.150 on recall@5 sits inside the confidence interval +0.000 to +0.300, so it is consistent with no difference at all. Settling a gap this size would take roughly 180 questions -- on a two-sided test at alpha 0.05 with 80% power, assuming per-question scores vary as much as a 0-1 score possibly can. It is an order of magnitude, not a count.
>>> result.distinguishable
False
```

`compare` takes `Mapping[query_id, score]` for each side — the output of
[`per_query`](metrics.md#aggregating-over-a-run) — and only uses questions **both sides
answered**. Comparing on different question sets would confound the difference between
the configurations with the difference between the questions, which is the whole reason
pairing exists.

Now a clearer case, where the gap is real and consistent:

```python
>>> qids = [f"q{i}" for i in range(40)]
>>> config_a = {qid: 1.0 for qid in qids[:34]} | {qid: 0.0 for qid in qids[34:]}   # 0.85
>>> config_b = {qid: 1.0 for qid in qids[:20]} | {qid: 0.0 for qid in qids[20:]}   # 0.50
>>> result = compare(config_a, config_b, left="rerank_on", right="rerank_off", metric="recall@5")
>>> print(result.verdict())
rerank_on beats rerank_off by 0.350 on recall@5 (95% CI +0.200 to +0.500, p=0.000, n=40). It wins on 14 questions, loses on 0 and ties on 26.
>>> result.distinguishable, result.winner
(True, 'rerank_on')
```

### Reading `Comparison`

`compare` returns a `Comparison` with `left_mean`, `right_mean`, `difference: Interval`,
`p_value`, `wins`/`losses`/`ties`, and:

- **`.distinguishable`** — the field to actually read. Deliberately conservative: it
  requires **both** `p_value < alpha` **and** the confidence interval excluding zero,
  because two weak signals agreeing is a better basis for a decision than either alone.
- **`.winner`** — `left` or `right` if distinguishable, else `None`. Never guess a winner
  from a non-significant gap.
- **`.verdict()`** — the result as a sentence, not a table of numbers. The negative case
  gets the longer explanation on purpose: a reader who sees "not distinguishable" needs to
  know what would change that, or they'll ignore the sentence and read the leaderboard
  order instead — which is precisely the wrong instinct this whole module exists to
  correct.

### The sample-size note

When two configurations aren't distinguishable, `.verdict()` ends with a sentence about
what it would take to settle it — computed from the same worst-case power calculation the
eval-set quality score uses, so the two numbers in a report agree with each other. Three
genuinely different cases, not one templated phrase, because stitching a single template
across all three produced sentences like "About many more than questions would be needed"
— the kind of thing that makes a reader stop trusting everything else on the page:

| Situation | What the sentence says |
|---|---|
| Identical score on every question | "they are behaving the same way", not a close call |
| Same mean, but disagreeing per-question | "no number of questions like these would separate them" |
| A real gap, just not enough questions yet | "Settling a gap this size would take roughly N questions" — rounded to two significant figures, with the test, alpha and power named. It read "About 4532 questions" from an eval set of seventeen, which is four significant figures of power calculation from n=17. |

## Defaults

`DEFAULT_RESAMPLES = 2000` — enough for a stable interval without making a sweep feel
slow; override with `resamples=` on any of the functions above. `compare`'s default
`alpha=0.05`, `confidence=0.95`, `seed=0` (all resampling is seeded, so results are
reproducible run to run).

Next: how a `Comparison` and a set of per-dimension metrics turn into
[the composite score](composite.md).

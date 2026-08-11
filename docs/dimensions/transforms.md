# Query transforms

A transform rewrites the question before it is searched with. Retrieval fails most often
because the user's words and the document's words do not match — someone asks "how long
before I can walk away?" and the contract says "termination for convenience on thirty days
written notice." They share almost no vocabulary. Every transform here is an attempt to close
that gap, one way or another.

Set it with `grid.transform` in the config, or pass a spec string to
`contextgrid.transform.get_transform`. See [retrieval](retrieval.md) for how the rewritten
queries are actually searched with, and [rerankers](rerankers.md) for what happens to the
results afterward.

## The cost that never goes away

Four of the six transforms need a model call. That cost is not like chunking or embedding,
which you pay once when you build the index. It is paid **on every query, forever**. So the
question worth asking is never "does HyDE help?" — it's "does HyDE help *enough to justify a
model call on every query, forever*?" Often the answer is no, and that is a real finding, not
a disappointment.

`none` — searching with the question exactly as asked — is the arm every other transform has
to beat. It is not a placeholder value. The whole point of this axis is that most transforms
do not clear their own cost, and you cannot show that without `none` sitting on the same chart
with the same cost column.

Every transform reports what it spent, so the sweep can attribute the cost to the
configuration that caused it:

```python
>>> from contextgrid.evalset.llm import RecordingLLM
>>> from contextgrid.transform import get_transform, describe_cost
>>> llm = RecordingLLM(default="Either party may terminate on thirty days written notice.")
>>> t = get_transform("hyde", llm=llm)
>>> result = t.transform("How much notice is needed to terminate for convenience?")
>>> result.queries
('How much notice is needed to terminate for convenience?', 'Either party may terminate on thirty days written notice.')
>>> result.fan_out, result.llm_calls
(2, 1)
>>> describe_cost([result])
'1.0 model calls and 2.0 searches per question, on every query forever'
```

`fan_out` is how many searches this one question now costs — each query in `TransformedQuery.queries`
gets searched separately and the results are fused, which is a real cost in latency as well as
in tokens.

## Why the model-backed ones are not in the registry

`none` and `expand` build from a plain spec string, no model required, so they live in the
ordinary plugin registry (`contextgrid.transform.TRANSFORMS`). `hyde`, `multi-query`,
`decompose` and `step-back` do not — you cannot build them from a string alone, because a
transform built with no model would silently fall back to doing nothing, and a config that
*looks* like it's testing HyDE while testing nothing is worse than an error.

```python
>>> from contextgrid.transform import get_transform, available_transforms, MODEL_BACKED
>>> available_transforms()
('decompose', 'expand', 'hyde', 'multi-query', 'none', 'step-back')
>>> MODEL_BACKED
('hyde', 'multi-query', 'decompose', 'step-back')
>>> get_transform("hyde")
Traceback (most recent call last):
    ...
contextgrid.evalset.llm.LLMError: the 'hyde' transform needs a model. Set `run.model` in your config, or use one of the model-free transforms: expand, none
```

In a config file, the model comes from `run.model` — one name, so the model-backed transforms,
the LLM-backed ingestion strategies, and the generation judge all share one key and one price:

```yaml
grid:
  transform: [null, hyde, multi-query]

run:
  model: openai:gpt-4o-mini
```

`available_transforms()` and `MODEL_BACKED` exist so `contextgrid plugins` and the config
template (`contextgrid init`) can *say these transforms exist* even though they are not in the
registry. Before this, they were reachable only if you already knew the name — the axis looked
like it had two arms when it actually has six.

## The six transforms

| name | spec | needs a model | what it does |
|---|---|---|---|
| `none` | `none` | no | Search with the question as asked. The arm to beat. |
| `expand` | `expand` | no | Spell out configured acronyms before searching. |
| `hyde` | `hyde` | yes | Invent a hypothetical answer and search with that instead of the question. |
| `multi-query` | `multi-query` or `multi-query:5` | yes | Paraphrase the question several ways and fuse the results. |
| `decompose` | `decompose` or `decompose:2` | yes | Split a compound question into sub-questions. |
| `step-back` | `step-back` | yes | Add a more general question alongside the specific one. |

### `none` — `contextgrid.transform.NoTransform`

Returns the question unchanged: `TransformedQuery(original=query, queries=(query,))`. Zero
model calls, zero fan-out. This is the baseline every comparison needs.

### `expand` — `contextgrid.transform.ExpandAcronyms`

No model, and free. Unglamorous, but it moves BM25 scores more than most of the clever
transforms below — a corpus that says "recovery point objective" cannot be found by a query
that says "RPO," and no embedding fixes a term the model has never seen.

| parameter | default | meaning |
|---|---|---|
| `expansions` | `{}` | a dict of short form → long form, e.g. `{"RPO": "recovery point objective"}` |

It appends the expansion next to the abbreviation rather than replacing it, so the acronym is
still searchable too. With no expansions configured, or nothing in the query to expand, it is
the identity — `is_identity` is `True` and no extra query is produced.

```python
>>> from contextgrid.transform import ExpandAcronyms
>>> ExpandAcronyms(expansions={"RPO": "recovery point objective"}).transform("What is our RPO?").queries
('What is our RPO recovery point objective?',)
```

### `hyde` — `contextgrid.transform.HyDE`

Searches with a hypothetical *answer* rather than the question. A question and its real answer
share little vocabulary; a plausible fake answer and the real one usually share a lot. So the
model invents a short passage that would answer the question, and that passage — not the
question — gets embedded and searched.

| parameter | default | meaning |
|---|---|---|
| `llm` | — | required |
| `include_question` | `True` | keep the original question alongside the invention |
| `max_tokens` | `200` | cap on the invented passage |

It works best where the model already knows the domain, and worst where it does not — on a
corpus of internal jargon it invents confident nonsense that matches nothing in the corpus.
That's exactly the case this tool exists to catch, and you cannot predict it from outside the
sweep. Keeping `include_question=True` (the default) hedges that failure mode: when the
invention is nonsense, the real question is still in the fused results.

If the model call fails or comes back empty, `HyDE` falls back to searching with the original
question — `is_identity` becomes `True` — rather than returning nothing.

### `multi-query` — `contextgrid.transform.MultiQuery`

Asks the same question several different ways and fuses the results. The most reliable of
these transforms and the least clever: it doesn't need the model to know anything about the
domain, only to paraphrase. The gain is usually small and usually real, at a cost of one model
call plus `variants` extra searches per question.

| parameter | default | meaning |
|---|---|---|
| `llm` | — | required |
| `variants` | `3` | how many paraphrases to generate |
| `max_tokens` | `250` | cap on the model's reply |

```python
>>> import json
>>> from contextgrid.evalset.llm import RecordingLLM
>>> from contextgrid.transform import MultiQuery
>>> llm = RecordingLLM(replies=[json.dumps(["how long is notice", "what is the notice period"])])
>>> result = MultiQuery(llm=llm, variants=2).transform("How much notice is needed to terminate for convenience?")
>>> result.fan_out
3
```

Spec string form: `multi-query:5` sets `variants=5`.

### `decompose` — `contextgrid.transform.Decompose`

The only one of these that fixes a *structural* failure rather than a vocabulary one. "Which
vendor has the shortest notice period and what is their monthly fee?" cannot be answered by
any single passage, however well it's embedded — the question itself has to become two. On a
plain factoid, decomposing is pure overhead, which is why it's an axis to sweep rather than a
default to switch on.

| parameter | default | meaning |
|---|---|---|
| `llm` | — | required |
| `max_parts` | `3` | cap on how many sub-questions come back |
| `max_tokens` | `250` | cap on the model's reply |

Duplicate sub-questions are dropped. Spec string form: `decompose:2` sets `max_parts=2`.

### `step-back` — `contextgrid.transform.StepBack`

Asks the more general question alongside the specific one — "what is Northwind's notice
period?" becomes "what do the termination clauses say?" as well. It helps when the specific
answer sits inside a passage about the general topic, and it hurts when the general query
drags in every document that mentions termination, which on a corpus of near-duplicates is
most of them.

| parameter | default | meaning |
|---|---|---|
| `llm` | — | required |
| `max_tokens` | `120` | cap on the model's reply |

If the model just repeats the question back, `StepBack` treats that as a no-op and returns the
identity.

## Failure handling, uniformly

Every model-backed transform degrades to `NoTransform`'s behavior — search with the original
question — when the model call errors, returns nothing usable, or returns something that
doesn't parse as expected. Searching with an empty query would score zero and *look like* a
retrieval failure; falling back to the original question means a broken transform costs you a
wasted model call, not a corrupted result.

## What ends up in the config

```yaml
grid:
  transform: [null, expand, hyde, "multi-query:5", decompose, step-back]

run:
  model: openai:gpt-4o-mini   # supplies the model to hyde, multi-query, decompose, step-back
```

`null` in YAML is `None` in Python, which `contextgrid.transform.get_transform` treats the
same as `"none"` — no rewriting. In a sweep, `transform: none` and `transform: null` collapse
to the same run (`contextgrid.grid.matrix.canonicalise` normalizes `"none"` to `None` so the
axis doesn't waste a slot counting the identity twice under two names).

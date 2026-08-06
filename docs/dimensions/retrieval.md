# Retrieval strategies

An [index](indexes.md) is a **store** — where the vectors or text live, and how one search runs
against them. A retrieval strategy is what sits on top of a store: how many searches happen, who
decides what to search for, and whether the answer to one search changes the next.

That split is the entire point. `dense`, `bm25`, `faiss:hnsw`, `pgvector` — those are places to
put your documents. `simple`, `widened`, `decomposed`, `agentic` — those are what you do with the
place once it exists. Keeping them apart turns "does agentic retrieval beat plain search on my
pgvector index, and is it worth the model calls?" from a rewrite into a cell in a grid: pick any
index, sweep any strategy over it, nothing about the strategy has to know which index it got.

Source: `src/contextgrid/retrieve/`. Tests: `tests/unit/test_retrieve.py`,
`tests/unit/test_retrieve_agentic.py`.

## The index versus the strategy

A strategy never sees the index. It's handed a `Searcher` — a plain function, `(text, k) ->
list[Scored]` — that runs one search against whatever index the config picked:

```python
from collections.abc import Callable, Sequence
from contextgrid.index.base import Scored

Searcher = Callable[[str, int], Sequence[Scored]]
```

That's the whole seam. It's why a new store (say, a future vector database) never has to touch
any strategy, and why every strategy below works identically whether the searcher underneath is
`bm25` or `pgvector:hnsw`.

## `RetrievalTrace`: what a strategy actually did

A recall number alone can't tell two strategies apart if they tie. `RetrievalTrace` is what
carries the difference:

```python
from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalTrace:
    searches: int = 0
    model_calls: int = 0
    queries: list[str] = field(default_factory=list)
    notes: dict[str, object] = field(default_factory=dict)
```

Every strategy that costs a model call reports it via `trace.record_model_call()`. Two
strategies with the same recall and a different `model_calls` count are a decision, not a tie —
and the runner refuses to start a sweep containing a strategy where `uses_model` is `True`
without a spending limit, because a strategy that decides its own number of calls has no upper
bound anybody can eyeball in advance.

Concretely, on the question *"what is the refund window and are digital goods refundable?"*,
`k=5`:

```python
from contextgrid.retrieve import (
    RetrievalTrace,
    SimpleRetrieval,
    WidenedRetrieval,
    DecomposedRetrieval,
)

question = "what is the refund window and are digital goods refundable?"
searcher = lambda text, k: []  # a real Searcher would query an index; the trace doesn't care

for strategy in [SimpleRetrieval(), WidenedRetrieval(factor=4), DecomposedRetrieval()]:
    trace = RetrievalTrace()
    strategy.retrieve(question, [question], searcher, 5, trace)
    print(strategy.name, trace.searches, trace.queries, trace.notes)
```

```
simple       searches=1  queries=['what is the refund window and are digital goods refundable?']  notes={}
widened      searches=1  queries=['what is the refund window and are digital goods refundable?']  notes={'depth': 20}
decomposed   searches=3  queries=['what is the refund window and are digital goods refundable?',
                                   'what is the refund window', 'are digital goods refundable']    notes={'parts': 3}
```

`widened` made one search too, but asked the index for 20 results instead of 5 (`depth` in the
notes); `decomposed` made three, splitting the question. Neither made a model call.

## The four strategies

| Spec | Class | `uses_model` | What it does |
|---|---|---|---|
| `simple` | `SimpleRetrieval` | `False` | One search per query, fused if the transform produced several. |
| `widened` | `WidenedRetrieval` | `False` | Search deeper than asked, cut back to `k`. |
| `decomposed` | `DecomposedRetrieval` | `False` | Split a multi-part question mechanically, search each part. |
| `agentic` | `AgenticRetrieval` | `True` | A model plans the searches, over one or more rounds. |

`get_retriever(None)` returns `SimpleRetrieval()`, so a config that has never heard of this axis
keeps behaving exactly as it did before the axis existed.

### `simple` — the arm every other strategy has to beat

```python
from contextgrid.retrieve import get_retriever

retrieval = get_retriever("simple")
```

No parameters. Exactly what this package did before retrieval became an axis, extracted
unchanged. It wins on a great many corpora, which is itself worth publishing, because the
field's default advice usually assumes otherwise.

### `widened` — free recall, sometimes

```python
retrieval = get_retriever("widened:8")  # factor=8 (shorthand)
```

Parameters: `factor: int = 4`. Asks the index for `k * factor` results, then cuts back to `k`
after fusion. On its own this changes nothing — the same top-`k` comes back — but it changes a
great deal once a reranker sits downstream, and it's the cheapest way to find out whether a
configuration is limited by the retriever's *ordering* or its *reach*. Costs a little index time,
zero model calls: the first thing to try before reaching for anything that bills per query.

### `decomposed` — split, search, fuse, mechanically

```python
retrieval = get_retriever("decomposed:3")  # max_parts=3 (shorthand)
```

Parameters: `min_words: int = 2`, `max_parts: int = 4`. A question like *"what is the refund
window and does it cover digital goods?"* has two answers, usually in two different chunks —
one search ranks whichever half the embedding favoured and the other half is simply lost.
`DecomposedRetrieval.parts()` splits on conjunctions and clause punctuation
(`and`, `or`, `also`, `;`, `?`, and a comma before a question word), with a `min_words` floor so
fragments like "and by when" don't become searches of their own. The whole question always
leads — decomposition adds recall, it doesn't replace the search that was already working:

```python
>>> import pprint
>>> pprint.pprint(DecomposedRetrieval().parts("what is the refund window and are digital goods refundable?"))
['what is the refund window and are digital goods refundable?',
 'what is the refund window',
 'are digital goods refundable']
```

Splitting is deliberately mechanical rather than model-driven. A model would split better and
would cost a call per query; this arm exists to show how much of that gain is free, which is the
comparison `agentic` has to be judged against.

### `agentic` — a model decides what to search for, and when to stop

```python
retrieval = get_retriever("agentic:gpt-4o-mini,rounds=2")
```

Parameters: `model: str = "openai:gpt-4o-mini"`, `rounds: int = 1`, `max_queries: int = 4`,
`backend: str = "auto"` (`auto` / `agno` / `llm`).

Every other strategy above decides its searches in advance. `agentic` reads what came back and
decides what to look for next — the number of searches isn't knowable before the run, which is
exactly why it belongs on a grid next to the free strategies rather than being taken on faith.
"Agentic RAG improves retrieval" is a claim about a trade: recall against latency and dollars,
and this axis is what lets you check it on your own corpus instead of somebody else's.

**The ranking comes from what the agent searched for, not from what it says.** A model asked to
name its chosen chunk IDs invents them — an invented ID is either a crash or, worse, a silent
mismatch that scores as a miss. So the model only ever decides *queries*; the index still decides
what matches, and results across rounds are fused by rank (never by raw score, for the same
reason two BM25 and cosine scores can't be averaged — see [indexes](indexes.md#hybrid--dense-and-sparse-fused)).

`rounds=1` is one planning call: read the question, write the queries. `rounds >= 2` lets the
model see what the first round found and search again for what's missing — the real behaviour,
and where the cost doubles. It stops early if a later round returns an empty plan, because a
model saying "I have enough" is a real signal and cheaper to trust than one more round of
nothing. `max_queries` caps how many searches one round can produce; a model asked for "the
queries" will happily write nine.

Two backends, one behaviour: `agno` when it's installed (`pip install 'context-grid[agent]'`),
otherwise a plain loop over this package's own `LLM` protocol — the fallback exists because a
strategy that can't run without a heavy optional dependency is a strategy most people never
measure.

**A planner failure never fails the sweep.** If the model errors, times out, or writes prose
instead of JSON, the strategy falls back to searching the question as asked and notes
`trace.notes["fell_back"] = True` plus the error text — returning nothing would score as a
retrieval failure when what actually failed was the planner. The model call is still counted as
spent even when it fails; a cost column that omits failed calls understates what the run cost.

`_parse_queries` (`src/contextgrid/retrieve/agentic.py`) is deliberately forgiving about the
wrapper and strict about the result: it handles fenced JSON, a leading "Here are the queries:",
and numbered or bulleted lists, but a refusal like *"I'm sorry, I can't help with that."* is
recognised as prose, not a plan, and falls back rather than becoming a search for the apology
itself.

### Measured: does the model's plan actually beat the free strategies?

Same two-question, six-document corpus as above (each question answered by two different
documents), scored with `index="bm25"`, `k=2`, `headline="recall@2"`, and a **scripted** planner
standing in for a real model — no key, no network, same technique the test suite uses:

```python
plans = {
    "what is the refund window and are digital goods refundable?": (
        '["refund window 30 days purchase", "digital goods not refundable downloaded"]'
    ),
    "how long is standard shipping and when does express arrive?": (
        '["standard shipping business days", "express shipping next business day"]'
    ),
}


class ScriptedPlanner:
    def complete(self, prompt, *, max_tokens=256):
        return next((plan for q, plan in plans.items() if q in prompt), "[]")
```

```
simple             recall@2=0.750
scripted-agentic   recall@2=1.000
```

Plain search ranks whichever half BM25 favoured and misses the other; a planner that writes one
query per part finds both. Scripting the planner isolates the *mechanism* from a particular
model's mood on a particular day — see `test_a_good_plan_beats_plain_search_on_two_part_questions`
in `tests/unit/test_retrieve_agentic.py` for the full setup, including how to register a scripted
strategy under its own name (`RETRIEVERS.register(...)`) so it's reachable from `matrix()` like
any built-in.

## Fusion: ranks, not scores

Every strategy above that runs more than one search combines the results with
`contextgrid.retrieve.base.fuse`, which is reciprocal rank fusion, not score averaging — the
same reasoning as `hybrid` on the index axis: a cosine similarity from one query and a cosine
similarity from another aren't on the same scale, however similar the raw numbers look, and
averaging them lets whichever query happened to produce larger magnitudes win a result it didn't
earn. Ranks have no such problem.

## Config reachability

Every strategy is a spec string, so a sweep over the whole axis is one line:

```
retrieval: [simple, widened:8, decomposed:3, agentic:gpt-4o-mini,rounds=2]
```

A config's label only names the strategy when it isn't `simple` — `Config(retrieval="simple").label`
carries no extra text, `Config(retrieval="decomposed").label` contains `~decomposed`, so a
leaderboard row for the default arm doesn't carry a word that adds nothing, and a row for a
strategy that costs money is never silently unlabelled.

## See also

- [indexes](indexes.md) — the stores a `Searcher` is built from.
- [embedders](embedders.md) — what a dense `Searcher` embeds the query with.

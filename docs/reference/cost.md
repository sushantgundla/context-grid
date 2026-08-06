# Cost model

`context-grid` prices every configuration in a sweep, not just its quality. Source:
`src/contextgrid/cost/model.py`. This page covers where a price comes from, how local and
hosted models get compared on one chart, and how `budget_usd`/`budget_seconds` actually stop
a sweep.

## Two kinds of cost

- **Token cost** — hosted models, charged in dollars per million tokens. Paid once at index
  time (embedding the corpus) and again on every query.
- **Compute cost** — local models. Free per token, **not** free per second. `CostModel`
  turns wall-clock seconds into dollars via `machine_usd_per_hour`, so a local CPU model
  lands on the same chart as a hosted API instead of reading as "free."

## Where a price comes from: `PRICES` first, litellm second

```python
# src/contextgrid/cost/model.py
PRICES: dict[str, Pricing] = {
    "hash": Pricing(metered=False),
    "tfidf": Pricing(metered=False),
    "length": Pricing(metered=False),
    "bge-base-en-v1.5": Pricing(metered=False),
    "e5-base-v2": Pricing(metered=False),
    "all-MiniLM-L6-v2": Pricing(metered=False),
    "text-embedding-3-small": Pricing(embed_per_million=0.02),
    "text-embedding-3-large": Pricing(embed_per_million=0.13),
    "embed-v3": Pricing(embed_per_million=0.10),
    "embed-english-v3.0": Pricing(embed_per_million=0.10),
    "voyage-3": Pricing(embed_per_million=0.06),
    "text-embedding-ada-002": Pricing(embed_per_million=0.10),
}
```

That's it — twelve hand-written entries, "last checked: August 2026" per the module
docstring. It's deliberately a plain literal, not a live lookup: a cost comparison has to be
reproducible, and a number that changes under you between runs is worse than one that's three
months stale and labelled as such.

**`PRICES` wins where it has an opinion. `litellm.model_cost` answers everything else.** That
table covers close to 3,000 models (2,987 in this checkout — `len(litellm.model_cost)`).
`CostModel.pricing_for()` checks `PRICES` first, falls back to `litellm.model_cost`, and only
warns "no published price" if neither has an entry:

```
$ .venv/bin/python -c "
from contextgrid.cost.model import CostModel
cm = CostModel()
for spec in ['tfidf', 'litellm:text-embedding-3-small', 'tei:bge-base-en-v1.5',
             'litellm:voyage-3', 'litellm:some-made-up-model-xyz']:
    p = cm.pricing_for(spec)
    print(f'{spec:35} -> embed=\${p.embed_per_million}/M metered={p.metered}')
"
tfidf                               -> embed=$0.0/M metered=False
litellm:text-embedding-3-small      -> embed=$0.02/M metered=True
tei:bge-base-en-v1.5                -> embed=$0.0/M metered=False
litellm:voyage-3                    -> embed=$0.06/M metered=True
litellm:some-made-up-model-xyz      -> embed=$0.0/M metered=False
```

The last one triggers a real warning, not a silent zero:

```
{'code': 'budget_reached', 'message': "no published price for 'some-made-up-model-xyz', so it
is costed at zero. Any cost comparison involving it understates what it charges", 'severity':
'caution', 'stage': 'cost', 'subject': 'some-made-up-model-xyz', 'detail': {}}
```

`tei:bge-base-en-v1.5` costs `$0`/M not because nobody priced it, but because `tei` is a
`_LOCAL_BACKEND` — `_is_local()` catches it before the price tables are even consulted, and
`Pricing(metered=False)` routes it to the compute-cost path below instead.

## `price_key`: turning a spec string into a model name

A spec string can be a bare name (`tfidf`), a backend-prefixed one
(`litellm:text-embedding-3-small,dimensions=256`), or a provider-qualified one
(`cohere/embed-english-v3.0`). `price_key()` normalizes all three to the name a price is
actually looked up under:

```
$ .venv/bin/python -c "
from contextgrid.cost.model import price_key
for spec in ['tfidf', 'tfidf:5000', 'litellm:text-embedding-3-small,dimensions=256',
             'tei:bge-base-en-v1.5', 'cohere/embed-english-v3.0']:
    print(f'{spec!r:50} -> {price_key(spec)!r}')
"
'tfidf'                                            -> 'tfidf'
'tfidf:5000'                                        -> 'tfidf'
'litellm:text-embedding-3-small,dimensions=256'    -> 'text-embedding-3-small'
'tei:bge-base-en-v1.5'                              -> 'bge-base-en-v1.5'
'cohere/embed-english-v3.0'                        -> 'embed-english-v3.0'
```

Keyword parameters come off (`,dimensions=256`), the `litellm:`/`tei:` backend prefix comes
off when there's a model name after it, and a provider path (`cohere/...`) keeps only the
last segment — the price belongs to the model, not to the route taken to reach it.

## Input and output tokens are priced separately

`litellm.model_cost` entries carry `input_cost_per_token` and `output_cost_per_token`
independently — generation is never one blended rate. `_litellm_pricing()` maps that onto
`Pricing.generate_input_per_million` and `Pricing.generate_output_per_million`:

```
$ .venv/bin/python -c "
from contextgrid.cost.model import _litellm_pricing
print('gpt-4o-mini:', _litellm_pricing('gpt-4o-mini'))
print('claude-3-5-sonnet-20241022:', _litellm_pricing('claude-3-5-sonnet-20241022'))
"
gpt-4o-mini: Pricing(embed_per_million=0.0, rerank_per_million=0.0,
  generate_input_per_million=0.15, generate_output_per_million=0.6, metered=True)
claude-3-5-sonnet-20241022: Pricing(embed_per_million=0.0, rerank_per_million=0.0,
  generate_input_per_million=3.0, generate_output_per_million=15.0, metered=True)
```

The Anthropic model isn't a direct key in `litellm.model_cost` — it's found by the
provider-qualified fallback (`vercel_ai_gateway/anthropic/claude-3-5-sonnet-20241022` in this
checkout's litellm version), which is exactly the "keep the last segment" rule `price_key`
also applies. `mode` in the litellm entry decides which fields get filled: `embed_per_million`
only for `mode: embedding`, `rerank_per_million` only for `mode: rerank`, generation fields
for everything else. An embedding model never carries a generation price and vice versa —
that's what "priced separately" means here, not just input vs. output.

## `machine_usd_per_hour`: why a local model isn't free

`CostModel.estimate()` takes `compute_seconds` and turns it into `machine_usd_per_hour *
(compute_seconds / 3600)`. For a non-metered (local) model, **all** of the cost is that
machine time — there's no per-token rate at all:

```
$ .venv/bin/python -c "
from contextgrid.cost.model import CostModel
cm = CostModel(machine_usd_per_hour=0.10)

local = cm.estimate(embedder='tei:bge-base-en-v1.5', index_tokens=1_000_000,
                     query_tokens_per_query=20, compute_seconds=120)
print('local (tei), 120s at \$0.10/hr:', local)
print('total at 1000 queries:', local.total_at(1000))

hosted = cm.estimate(embedder='litellm:text-embedding-3-small', index_tokens=1_000_000,
                      query_tokens_per_query=20, compute_seconds=5)
print('hosted (openai small), 5s:', hosted)
print('total at 1000 queries:', hosted.total_at(1000))
"
local (tei), 120s at $0.10/hr: CostBreakdown(index_usd=0.0033333333333333335,
  query_usd_per_1k=0.0, index_tokens=1000000, query_tokens_per_query=20,
  compute_seconds=120, metered=False)
total at 1000 queries: 0.0033333333333333335
hosted (openai small), 5s: CostBreakdown(index_usd=0.020138888888888888,
  query_usd_per_1k=0.0004, index_tokens=1000000, query_tokens_per_query=20,
  compute_seconds=5, metered=True)
total at 1000 queries: 0.02053888888888889
```

`machine_usd_per_hour` defaults to `0.0` — leave it there and a local model reports as free,
which is true per token and false in every other sense. The module docstring's rule of thumb:
a commodity 4-core cloud box is roughly $0.10/hour, and that's the value to put in if you want
the chart to tell the truth about self-hosting.

**One nuance:** `compute_seconds` passed to `estimate()` in the runner is the whole
configuration's wall-clock time — building the index *and* answering the eval set — charged
once into `index_usd`. `query_usd_per_1k` stays `0.0` for a non-metered model no matter what.
So `CostBreakdown.total_at(queries)` for a local model reflects what this one run actually
cost on the clock, not a projected per-query rate at arbitrary scale the way a hosted
model's `query_usd_per_1k` does.

## `budget_usd` and `budget_seconds`

Both are `run:` config keys (`RunConfig.budget_seconds`, `RunConfig.budget_usd` in
`src/contextgrid/config/schema.py`), consumed by `grid.runner.Budget`:

```python
@dataclass(slots=True)
class Budget:
    seconds: float | None = None
    usd: float | None = None
    spent_usd: float = 0.0
```

- `budget_seconds` — checked against wall-clock elapsed since the sweep started.
- `budget_usd` — checked against `spent_usd`, which accumulates
  `CostBreakdown.total_at(len(evalset.items))` after **every** completed configuration.

Both are checked *before* each configuration in the matrix, not during one — so a budget is
"honoured to within one configuration," per the code's own docstring, never predicted in
advance. Cost can't be known before a model is actually called, especially for something
like `agentic` retrieval that decides its own number of calls. When either ceiling is hit,
the sweep stops, the leaderboard is marked partial, and a `BUDGET_REACHED` warning names how
many of the planned configurations actually ran.

`machine_usd_per_hour` lives next to these two in `run:` — all three are cost-control knobs
on the same config block, see [configuration.md](../guide/configuration.md).

## Regenerating the numbers on this page

```bash
.venv/bin/python -c "
from contextgrid.cost.model import CostModel, price_key, PRICES
import litellm
print(len(PRICES), 'hand-written prices')
print(len(litellm.model_cost), 'litellm model entries')
"
```

# Is agentic retrieval worth it?

## The question

`retrieval` — not `index`, not `chunker` — is the axis that answers this: **how** the index gets
searched, as opposed to what it contains. Three arms sit on it:

| Strategy | What it does | Cost |
|---|---|---|
| `simple` | one search, the question as asked | free |
| `decomposed` | splits a multi-part question and searches each part | free — mechanical, no model |
| `agentic` | a model plans the searches, and can look again | a model call per query (per round) |

"Agentic RAG improves retrieval" is a claim about a trade — recall against latency and dollars —
and almost nobody checks it on their own corpus, because checking it means building both sides.
This recipe builds both sides.

## The config

```python no-run: abbreviated -- shown in full in "The command" below
grid = matrix(
    parser="markdown",
    chunker="recursive:256,overlap=32",
    embedder="tfidf",
    index="dense",
    retrieval=["simple", "decomposed", "agentic:openai:gpt-4o-mini"],
    k=5,
)
```

`agentic`'s shorthand parameter is the model: `agentic:openai:gpt-4o-mini` sets
`AgenticRetrieval(model="openai:gpt-4o-mini")`. See
[configuration](../guide/configuration.md) for `run.model`, the other way to supply it.

## The command, and why it needs a stand-in

`agentic` needs a real model — no key is checked into this repo, and a doc recipe that only
works with `OPENAI_API_KEY` set is a doc recipe most readers can't run. What follows uses the
exact same pattern `tests/unit/test_llm_litellm.py` uses to test the real adapter: swap
`sys.modules["litellm"]` for a fake module before importing, so `contextgrid.evalset.llm.get_llm`
finds *something* named `litellm` and never touches the network. This is not a `transport=`
hook — the retrieval axis doesn't expose one on the string-spec path — it's a lower-level
substitution, one level down.

**With a real key**, delete the fake-litellm block below and set `OPENAI_API_KEY`; everything
else is unchanged.

```bash
.venv/bin/python - <<'PY'
import sys, types, json, re
sys.path.insert(0, "examples")

# Stand in for litellm -- no key, no network. The agent's "plan" is scripted: echo the
# question back as the one search query. Deterministic, and the point is to see what that
# costs, not to build a clever agent.
calls = {"n": 0}
def completion(**kwargs):
    calls["n"] += 1
    prompt = kwargs["messages"][0]["content"]
    question = re.search(r"Question: (.+)", prompt).group(1).strip()
    class Message: content = json.dumps([question])
    class Choice: message = Message()
    class Response: choices = [Choice()]
    return Response()

fake = types.ModuleType("litellm")
fake.completion = completion
fake.model_cost = {}
sys.modules["litellm"] = fake

import lab_demo as d
from contextgrid.grid import Runner, matrix

evalset, corpus = d.build_evalset(), d.markdown_corpus()
grid = matrix(parser="markdown", chunker="recursive:256,overlap=32", embedder="tfidf",
              index="dense", retrieval=["simple", "decomposed", "agentic:openai:gpt-4o-mini"], k=5)
results = Runner(corpus=corpus, headline="recall@5").run(grid, evalset, mode="factorial")

for row in results.leaderboard("recall@5"):
    print(f"{row['config']:60} {row['recall@5']:6.3f}")
print()
print("model calls made:", calls["n"])
PY
```

## The real output

```
markdown · recursive:256,overlap=32 · tfidf · dense                              0.877
markdown · recursive:256,overlap=32 · tfidf · ~decomposed · dense                0.877
markdown · recursive:256,overlap=32 · tfidf · ~agentic:openai:gpt-4o-mini · dense 0.877

model calls made: 74
```

## How to read it

**All three tie at 0.877, and here that's the honest result, not a bug.** `decomposed` costs
nothing and ties `simple` because none of these 74 questions are actually multi-part — there's
nothing for it to split. `agentic` also ties `simple`, for a plainer reason: the scripted stand-in
plans exactly one search, and it's the question verbatim — functionally identical to `simple`.
**And it still cost 74 model calls to get there.**

That's the number this recipe exists to surface. The leaderboard's `$/1k` column read `$0.0000`
for the agentic row — not because the calls were free, but because `contextgrid/grid/runner.py`
says so directly: *"the cost model cannot know for an agentic strategy that decides its own
number of calls."* Retrieval-time model calls aren't priced into the built-in cost column the
way embedding and generation calls are. The leaderboard will not warn you that agentic retrieval
cost anything — you have to count the calls yourself, which is exactly what `calls["n"]` is
doing above.

**Pricing it by hand**, using the real `gpt-4o-mini` numbers `docs/adoption-backlog.md` records
($0.15 / $0.60 per million tokens, in / out) and a rough 80-token prompt, 10-token completion per
call: 74 calls × 80 tokens ≈ 5,900 prompt tokens, 74 × 10 ≈ 740 completion tokens →
`5900/1e6 * 0.15 + 740/1e6 * 0.60 ≈ $0.0013` for these 73 answerable questions — about **$0.018
per 1,000 queries**. Small in absolute terms on this corpus, and the point isn't the dollar
figure — it's that nothing in the tool computed it for you.

## What would change the answer

- **A real model, not a one-line stand-in.** The honest version of this experiment gives the
  agent a real reasoning pass and measures whether its actual query plan — not an echo — finds
  evidence `simple` misses. That requires a key: `OPENAI_API_KEY` set, and the fake-litellm block
  removed.
- **Multi-part questions.** `decomposed` and `agentic` both exist for questions like "what's the
  notice period *and* the late fee" — a single vector search often retrieves one half well and
  the other poorly. None of this corpus's 74 questions are built that way; a question set that
  is would be where this axis has something to show.
- **`rounds > 1` on `agentic`.** The default is one planning call; a second round lets the agent
  see what came back and search again for what's missing — where the interesting behaviour, and
  the doubled cost, actually shows up.
- **A `budget_usd` or `budget_seconds` cap** in `run:` — since the cost model can't price
  `agentic` in advance, the budget guard is the safety net instead. See
  [configuration](../guide/configuration.md).

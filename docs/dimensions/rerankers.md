# Rerankers

A reranker reorders a candidate list using the query and each passage *together*, rather than
comparing the query to a vector that was computed in isolation. That's why it can judge whether
a passage answers *this* question instead of merely sitting nearby in embedding space — and
why it costs more per candidate than the retriever did.

Set it with `grid.reranker`, and set how deep it looks with `grid.candidates`. See
[retrieval](retrieval.md) for what builds the candidate list in the first place, and
[generation](generation.md) for what happens to the reranked list next.

## `candidates` is the parameter every blog post skips

Almost all "use a reranker" advice stops at naming a model. It doesn't say how many candidates
to hand it, and that second number is where most of the effect actually lives:

- Over the top 10, a reranker can only reorder what the retriever already found.
- Over the top 100, it can rescue a passage the retriever ranked 47th.

Cost scales with depth; benefit does not scale the same way. Somewhere on that curve is the
right answer for your corpus, and it's specific to the corpus — this is exactly the kind of
curve nobody publishes and this tool is built to produce, because `candidates` is just another
axis on the grid:

```yaml
grid:
  reranker: [null, lexical, "tei-rerank:bge-reranker-base"]
  candidates: [10, 50, 100]
```

`contextgrid.pipeline.BuiltPipeline.search` is where this plays out: with a reranker set, the
retriever is asked for `candidates` results and the reranker cuts that down to `k`; with no
reranker, the retriever is asked for `k` directly, so the no-reranker arm never pays for depth
it would only throw away. And `candidates` is meaningless without a reranker to use it —
`contextgrid.grid.matrix.canonicalise` resets `candidates` back to its default (`50`) whenever
`reranker` is `None`, so a sweep doesn't waste runs on identical configurations that differ
only in an unused number:

```python
>>> from contextgrid.grid.matrix import canonicalise
>>> from contextgrid.pipeline import Config
>>> canonicalise(Config(reranker=None, candidates=100)).candidates
50
>>> canonicalise(Config(reranker="lexical", candidates=100)).candidates
100
```

## The five rerankers

| name | spec | extra needed | when to use it |
|---|---|---|---|
| `none` | `none` | — | the arm every reranker has to beat |
| `lexical` | `lexical` or `lexical:0.4` | — | free floor: query-term coverage |
| `mmr` | `mmr` or `mmr:0.5` | — | fixes a top-k that's near-duplicate passages |
| `tei-rerank` | `tei-rerank:bge-reranker-base` | none (plain `urllib`) | a real cross-encoder, self-hosted |
| `litellm-rerank` | `litellm-rerank:cohere/rerank-english-v3.0` | `pip install "context-grid[llm]"` | a hosted cross-encoder: Cohere, Jina, Voyage, AWS |

### `none` — `contextgrid.rerank.NoReranker`

Keeps the retriever's order. Not a placeholder: half of "use a reranker" advice is untested,
and an honest comparison needs this baseline on the same chart, with the same latency and cost
columns, as everything else.

### `lexical` — `contextgrid.rerank.LexicalOverlapReranker`

Scores a passage by how much of the query it actually contains — a cross-encoder without the
encoder. It's the coverage of query terms in the passage, plus a length-penalized density term
so a long passage can't win purely by containing everything.

| parameter | default | meaning |
|---|---|---|
| `length_penalty` | `0.25` | how much long passages are punished for their length |

Weak, free, and genuinely useful as a floor: a neural reranker that costs real money and beats
this by 0.01 has told you something important about whether it's worth deploying at all.

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.span import Span
>>> from contextgrid.rerank import get_reranker
>>> texts = [
...     "Shipping takes five to seven business days.",
...     "Refunds are issued within thirty days of purchase.",
...     "The office is closed on public holidays.",
...     "Digital goods are not refundable once downloaded.",
... ]
>>> chunks = [
...     Chunk(id=f"doc:{i}", span=Span("doc", i * 100, i * 100 + len(t)), text=t)
...     for i, t in enumerate(texts)
... ]
>>> top = get_reranker("lexical").rerank("do I get a refund on digital goods?", chunks, k=2)
>>> [scored.chunk_id for scored in top]
['doc:3', 'doc:2']
```

### `mmr` — `contextgrid.rerank.MMRReranker`

Maximal marginal relevance: relevance minus similarity to what's already been picked. The fix
for a top-5 that is five near-copies of the same paragraph — overlapping chunks make a
leaderboard look fine, since the evidence really was retrieved (five times), while the
generator sees one fact repeated across the whole context window instead of five distinct
ones.

| parameter | default | meaning |
|---|---|---|
| `diversity` | `0.3` | `0` keeps the retriever's order; `1` always picks the most different remaining passage, regardless of relevance |

Spec string form: `mmr:0.5` sets `diversity=0.5`.

### `tei-rerank` — `contextgrid.rerank.TEIReranker`

A cross-encoder served by [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
(TEI). No API key, no extra Python dependency — it's reached over plain `urllib`.

```bash
docker run -p 8081:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
    --model-id BAAI/bge-reranker-base
```

```yaml
grid:
  reranker: ["tei-rerank:bge-reranker-base,api_base=http://localhost:8081"]
```

**One TEI process serves one model.** A TEI server started for embeddings is not also a
reranking server — reranking needs its own process, on its own port, separate from the
embedding one (`8081` above, versus whatever port the embedder's TEI server uses). This is
worth saying because the natural assumption is that one server does both, and the failure mode
when it doesn't is a confusing HTTP 400, not an error that names the real problem.

| parameter | default | meaning |
|---|---|---|
| `model` | — | required, e.g. `bge-reranker-base` |
| `api_base` | `http://localhost:8081` | the TEI server's own `/rerank` endpoint |
| `batch_size` | `64` | candidates per request |
| `max_chars` | `8000` | passage text is trimmed to this before sending |
| `timeout` | `60.0` | seconds |
| `retries` | `2` | retried only for transient errors — timeouts, rate limits, 5xx |
| `transport` | `None` | replaces the network call entirely; used in tests and in the examples below |

### `litellm-rerank` — `contextgrid.rerank.LiteLLMReranker`

A hosted reranker through [litellm](https://docs.litellm.ai/): Cohere, Jina, Voyage, AWS, one
name each. Needs `pip install "context-grid[llm]"`.

```yaml
grid:
  reranker: ["litellm-rerank:cohere/rerank-english-v3.0"]
```

The key comes from the environment (`COHERE_API_KEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, ...),
the same as every other model access in this project — never from the config file.

| parameter | default | meaning |
|---|---|---|
| `model` | — | required, `provider/model`, e.g. `cohere/rerank-english-v3.0` |
| `api_base` | `None` | override the provider's default endpoint |
| `api_key_env` | `None` | which environment variable holds the key |
| `batch_size`, `max_chars`, `timeout`, `retries`, `transport` | same as `tei-rerank` | shared with `TEIReranker` via `_RemoteReranker` |

## Every candidate must come back, or the run fails

Both remote rerankers insist on one thing: the backend has to return a score for *every*
candidate it was sent, or `contextgrid.rerank.remote.RerankerError` is raised. A backend that
quietly returns fewer results than it was given — a passage too long, a batch silently
capped — has dropped documents from the ranking. On a leaderboard, that looks exactly like the
reranker judging those documents irrelevant, and it is a completely different claim. Failing
loudly here is what keeps `candidates` meaning what it says.

```python
>>> from contextgrid.rerank.remote import TEIReranker
>>> def dropping_transport(query, passages):
...     return [(i, 1.0) for i, _ in enumerate(passages)][:-1]  # drops the last one
...
>>> bad = TEIReranker(model="bge-reranker-base", transport=dropping_transport)
>>> bad.rerank("refund", chunks, k=2)
Traceback (most recent call last):
    ...
contextgrid.rerank.remote.RerankerError: tei-rerank scored 3 of 4 candidates in the batch starting at 0. Every candidate must come back, or the ones that did not look like the model rejected them.
```

`transport` is a `Callable[[query, passages], list[tuple[position, score]]]` that stands in for
the real HTTP call — set it to run a whole sweep with no server and no key, exactly the way
`tests/unit/test_rerank_remote.py` does:

```python
>>> def by_keyword(word):
...     def transport(query, passages):
...         return [(i, 1.0 if word in p.lower() else 0.0) for i, p in enumerate(passages)]
...     return transport
...
>>> tei = TEIReranker(model="bge-reranker-base", transport=by_keyword("refund"))
>>> top = tei.rerank("refund", chunks, k=2)
>>> [(scored.chunk_id, scored.score) for scored in top]
[('doc:1', 1.0), ('doc:3', 1.0)]
```

Ties are broken by the incoming rank — `(-score, position)` — so two passages scored
identically keep the retriever's order rather than an arbitrary one. Without that, rerunning
the same sweep can reorder tied passages and a diff shows a change that never actually
happened.

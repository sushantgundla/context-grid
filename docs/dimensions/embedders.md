# Embedders

An embedder turns text into vectors. That sounds like one job. It is actually two, and getting
the difference wrong is the quietest way a comparison goes wrong: the numbers still look
reasonable, they are just several points lower than they should be, and nothing on the chart
says why.

This page covers the five registered embedders, the prefix system that stops that silent
mistake, the `transport` hook that lets you test any of them with no server and no key, and the
`quality` diagnostics that score an embedder against your own corpus with no eval set at all.

Source: `src/contextgrid/embed/`. Tests: `tests/unit/test_embed_index.py`,
`tests/unit/test_embed_remote.py`, `tests/unit/test_embed_quality.py`.

## Why queries and documents are embedded differently

Some models were trained with a prefix in front of the text. E5 wants `query: ` on the question
and `passage: ` on the document. BGE wants an instruction on the query and nothing on the
document. Most OpenAI models want neither. Embed both sides the same way and the model still
runs — nothing errors — the scores just come out lower, uniformly, with nothing to say why.

Worse for a tool built to compare models: the effect is not even *across arms*. Get the prefix
wrong for one model in a sweep and right for another, and you are no longer comparing two
models — you are comparing one model against a handicapped version of the other.

That is why every `Embedder` (`src/contextgrid/embed/base.py`) has two methods instead of one,
and no default that quietly makes them the same:

```python
from collections.abc import Sequence
from contextgrid.embed.base import EmbeddingResult


def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...
def embed_queries(self, texts: Sequence[str]) -> EmbeddingResult: ...
```

## The five embedders

| Spec | Class | Extra | What it is |
|---|---|---|---|
| `hash` | `HashEmbedder` | none | Hashed bag of words. No model, no training. |
| `tfidf` | `TfidfEmbedder` | none | Classical TF-IDF over the corpus vocabulary. |
| `length` | `TokenCountEmbedder` | none | One dimension: text length. A chance-level control. |
| `litellm` | `LiteLLMEmbedder` | `[llm]` | Any hosted model — OpenAI, Cohere, Voyage, Gemini, Bedrock, Azure — through one interface. |
| `tei` | `TEIEmbedder` | none | HuggingFace text-embeddings-inference, a local server. |

### `hash` — no model needed

```python
from contextgrid.embed import get_embedder

hash_emb = get_embedder("hash:512")  # dimensions=512
```

Parameters: `dimensions: int = 256`, `seed: int = 0`. Two documents are close when they share
words; word order is invisible. Its job is to be the floor — a dense model that cannot beat this
on your corpus is not earning its cost.

`dimensions` and `seed` are the whole of its state, and the digest behind them is
`hashlib.blake2b`, so **the same spec string on the same text gives the same vectors on any
machine and in any process.** `seed` genuinely pins the output: `hash:512,seed=3` and
`hash:512,seed=4` give different vectors, and each is stable across runs. That is worth saying
because it was not true before version `2` of this embedder, which hashed with Python's
built-in `hash()` — salted per process, so the same corpus scored differently on every run. The
`version` is part of the embed cache key, so bumping it retired the vectors the old one wrote
rather than serving them back. If you have a cached run from before, its `hash` numbers were
never reproducible and should not be compared against new ones.

### `tfidf` — the classical baseline that keeps winning

```python
tfidf = get_embedder("tfidf:5000")  # max_features=5000
```

Parameters: `max_features: int = 4096`, `min_document_frequency: int = 1`,
`sublinear_tf: bool = True`.

Not a toy. On corpora with distinctive vocabulary — legal, medical, code — TF-IDF is genuinely
competitive with dense retrieval. It needs `prepare(documents)` before it can embed anything,
because it learns its vocabulary and IDF from the corpus:

```python
from contextgrid.embed import TfidfEmbedder

DOCS = [
    "Refunds are issued within thirty days of purchase.",
    "Shipping takes five to seven business days.",
]
model = TfidfEmbedder()
model.prepare(DOCS)  # learns the vocabulary
model.embed_documents(DOCS)  # now this works
```

Calling `embed_documents` before `prepare` raises `RuntimeError`, on purpose — falling back to
zero vectors would score zero and read as "this embedder is bad" rather than "it was never
fitted". Queries are embedded against the *document* IDF, never their own: a query's own
statistics over one sentence are meaningless.

### `length` — deliberately useless

```python
length = get_embedder("length")  # dimensions == 1
```

One dimension: how many tokens the text has. Included so the scoring pipeline has something
that should score near chance — if it doesn't, the scoring is broken, not the model.

### `litellm` — any hosted model

```
embedder: litellm:text-embedding-3-small
embedder: litellm:cohere/embed-english-v3.0,dimensions=1024
```

Needs `pip install 'context-grid[llm]'`. Parameters (shared with `tei`, below):
`model: str`, `dimensions: int = 0`, `batch_size: int = 32`, `max_tokens: int | None = 512`,
`api_base: str | None = None`, `api_key_env: str | None = None`, `timeout: float = 60.0`,
`retries: int = 2`, `query_prefix: str | None = None`, `document_prefix: str | None = None`,
`normalise_vectors: bool = True`.

The API key is read from the environment (`OPENAI_API_KEY`, `COHERE_API_KEY`, and so on),
never from the config file — a config with a secret in it ends up in version control and then
in a screenshot.

### `tei` — a local server, no key, no network

```
embedder: tei:bge-base-en-v1.5,api_base=http://localhost:8080
```

Start the server once:

```bash no-run: needs a running Docker daemon and pulls an image from ghcr.io
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
    --model-id BAAI/bge-base-en-v1.5
```

Same parameters as `litellm`, plus `api_base` defaults to `http://localhost:8080`. `tei` needs
no extra at all — it is reached over plain `urllib` from the standard library, so a running
server plus a bare `pip install context-grid` is enough. `model` here is a label the server does
not enforce, so call `TEIEmbedder(...).info()` before a sweep to confirm the server is actually
running the weights you think it is.

## The prefix lookup

`src/contextgrid/embed/prefixes.py` matches a model name (case-insensitive substring, longest
pattern first) against known families and returns the query/document prefixes it was trained
with. Both `litellm` and `tei` apply this automatically:

```python
>>> from contextgrid.embed.prefixes import for_model
>>> for_model("intfloat/e5-base-v2")
Prefixes(query='query: ', document='passage: ')
>>> for_model("BAAI/bge-base-en-v1.5")
Prefixes(query='Represent this sentence for searching relevant passages: ', document='')
>>> for_model("nomic-embed-text-v1.5")
Prefixes(query='search_query: ', document='search_document: ')
>>> for_model("text-embedding-3-small")
Prefixes(query='', document='')
>>> for_model("some-mystery-model")
Prefixes(query='', document='')
```

Known families (`known_families()`): `multilingual-e5`, `e5-mistral`, `e5-`, `bge-large-en-v1.5`,
`bge-base-en-v1.5`, `bge-small-en-v1.5`, `nomic-embed`, `gte-multilingual`. An unrecognised model
name gets no prefix — adding one a model was not trained with is exactly as wrong as omitting
one it was, and silence is the safer default for a name the lookup does not know.

An explicit `query_prefix=` or `document_prefix=` always wins over the lookup, **including an
explicit empty string** — writing `query_prefix=""` means "this model needs none," and guessing
over the top of that would be worse than guessing in the first place.

## The `transport` hook: real embedders, no server, no key

Both `litellm` and `tei` accept a `transport` callable: given one batch of texts, it returns
`(vectors, token_count)` and replaces the network call entirely. This is how the whole embed
pipeline gets exercised in CI with nothing installed, and it is the way to test anything you
build on top of this package without paying for a real API call every run.

```python
from contextgrid.embed.remote import TEIEmbedder

seen = []


def fake_server(batch):
    """Stands in for a running TEI server: no docker, no network, no key."""
    seen.extend(batch)
    return [[float(len(t)), 1.0, 0.0] for t in batch], 0


embedder = TEIEmbedder(model="e5-base-v2", dimensions=3, transport=fake_server)

embedder.embed_queries(["how long do refunds take"])
embedder.embed_documents(["refunds take thirty days"])

print(seen)
```

Output:

```
['query: how long do refunds take', 'passage: refunds take thirty days']
```

Notice the prefixes were applied *before* the transport ever saw the text — `transport`
replaces the HTTP call, not the logic in front of it, so batching, truncation, prefixing and
normalising are all still exercised. `dataclasses.replace(embedder, transport=...)` is the usual
way to wire it onto an embedder built from a spec string; see `wire()` in
`tests/unit/test_embed_remote.py`.

## `quality`: scoring an embedder against your corpus, with no questions at all

MTEB reports a model's average score over 56 public datasets. None of them are your documents.
Recall answers the retrieval question properly, but it needs an eval set — and most people have
a corpus and no questions at the moment they are choosing an embedder, which is the worst
possible time to have no signal.

`contextgrid.embed.quality.assess(chunks, vectors)` works from the vectors alone and reports:

| Metric | What it means | Read it as |
|---|---|---|
| `anisotropy` | Mean cosine between chunks from *different* documents. | Lower is better. Transformer embeddings crowd into a narrow cone; a model reporting 0.85 between a refund policy and a shipping schedule has almost no room left to say one matches a query and the other doesn't. `crowded` is true at ≥0.6, `degenerate` at ≥0.95. |
| `separation` | Mean cosine of *consecutive* chunks minus `anisotropy`. | Higher is better. Adjacent text really is related on any corpus, which makes it a safer signal than "shares a document" — a contract's fees clause and its termination clause share a file and little else. |
| `redundancy` | `-separation`, kept under its own name. | Positive means unrelated documents look *more* alike than consecutive paragraphs do — the corpus is full of near-copies (templated contracts, boilerplate policies) and no embedder will separate them cleanly. This is a fact about the documents, not the model. |
| `effective_dimensions` | Participation ratio of the vector covariance: `(Σλ)² / Σλ²`. | How many of the paid-for dimensions actually carry variance. A 1536-dimension model living in a 40-dimension subspace charges for 1536 and thinks in 40. |
| `collapsed` | Fraction of chunks whose nearest neighbour is a near-identical vector (cosine ≥ 0.99) but genuinely different text (word-overlap < 0.75). | Distinct passages the model cannot tell apart. Near-identical *text* landing on near-identical vectors is fine — that's boilerplate working correctly. |

Two derived flags: `templated` (a fact about the corpus — `redundancy > 0` and not degenerate)
and `degenerate` (a fact about the model — nothing else is worth reading once anisotropy hits
0.95, because a model reporting every pair as identical produces a corpus diagnosis that is
entirely about itself).

```python no-run: chunks/vectors stand in for your own corpus and its embeddings, not real data
from contextgrid.embed.quality import assess

result = assess(chunks, vectors)  # vectors: one row per chunk, same order
print(result.summary())
print(result.score)  # 0-1 blend, for the composite
```

Run against three synthetic corpora — a well-behaved one, a collapsed one, and a templated one
(same generators as `tests/unit/test_embed_quality.py`):

```
clustered (healthy):
  coherence +0.915, anisotropy -0.017, 4/64 effective dimensions, 0.0% collapsed, over 20 chunks
  score=0.844
collapsed:
  degenerate: anisotropy 1.000, 100% of chunks collapsed onto a neighbour, over 20 chunks
  score=0.000
templated:
  coherence -0.168 (templated corpus), anisotropy 0.153, 4/64 effective dimensions, 100.0% collapsed, over 20 chunks
  score=0.218
```

**These are diagnostics, not a verdict.** They say whether a model *can* discriminate on this
corpus, never whether it retrieves the right thing — that's recall's job, and recall needs
questions. A model that scores well here and badly on recall is a real and useful finding; a
model that scores badly here will not be rescued by anything later in the pipeline.

`assess()` samples rather than computing the full pairwise matrix (`sample=2000`,
`pairs=20_000` by default) — a 50,000-chunk corpus is 1.25 billion pairs to answer a question a
few thousand random pairs settle to more precision than the conclusions need. It refuses a
corpus under 4 chunks (`EmbeddingQualityError`, "too small to say anything") and a chunk/vector
count mismatch (`"out of step"`).

## Query-side adapters

`src/contextgrid/embed/adapter.py` adds one more thing you can do once you have embeddings: a
small matrix, `LinearAdapter`, that nudges *query* vectors towards where their answers actually
sit, without touching document vectors or rebuilding the index. It is fitted from the same
(question, evidence) pairs an eval set already provides, plus the near-miss chunks a sweep
surfaces — training data that already exists, for free. Measured effect: `strength=0.1` gained
+0.081 recall@5 on a dense embedder on a held-out split; `strength=1.0` lost 0.216. It hurt at
every setting tried on TF-IDF (-0.027 to -0.230) — it belongs on learned dense vectors, not
sparse ones. See the module docstring and `AdaptedEmbedder` for the wrapper that composes it
with any embedder above.

## See also

- [indexes](indexes.md) — what holds the vectors this page produces.
- [retrieval](retrieval.md) — how those vectors get searched.

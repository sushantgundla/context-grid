# Choose an embedder

## The question

Two parts, because they need different evidence:

1. With an eval set, which embedder actually retrieves better on your questions?
2. **Without one** — the state most people are actually in when they're picking a model — what
   can you tell about an embedder from the corpus alone?

MTEB reports a model scoring 63.5 averaged over 56 public datasets, none of which are your
documents. Neither of these questions asks about that leaderboard. Both ask about *this*
corpus.

## Part 1 — with an eval set: sweep and score

### The config

`embedder` as an axis, everything else fixed, same corpus and eval set as
[choose-a-chunker.md](choose-a-chunker.md) (33 documents, 74 questions):

```python
grid = matrix(
    parser="markdown",
    chunker="recursive:256,overlap=32",
    embedder=["tfidf", "hash:512", "length"],
    index="dense",
    k=5,
)
```

`length` is included on purpose — it embeds each chunk into one dimension (its token count) and
is a chance-level control. If `length` scores anywhere near `tfidf`, the eval set isn't testing
retrieval.

### The command

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "examples")
import lab_demo as d
from contextgrid.grid import Runner, matrix

evalset, corpus = d.build_evalset(), d.markdown_corpus()
grid = matrix(parser="markdown", chunker="recursive:256,overlap=32",
              embedder=["tfidf", "hash:512", "length"], index="dense", k=5)
results = Runner(corpus=corpus, headline="recall@5").run(grid, evalset, mode="factorial")
for row in results.leaderboard("recall@5"):
    print(f"{row['config']:52} {row['recall@5']:6.3f}")
print()
print(results.summary("recall@5"))
PY
```

### The real output

```
markdown · recursive:256,overlap=32 · tfidf · dense       0.877
markdown · recursive:256,overlap=32 · hash:512 · dense    0.863
markdown · recursive:256,overlap=32 · length · dense      0.164

markdown · recursive:256,overlap=32 · tfidf · dense scored best on recall@5 at 0.877, across 3
configurations on 73 questions. markdown · recursive:256,overlap=32 · tfidf · dense and markdown
· recursive:256,overlap=32 · hash:512 · dense are not distinguishable on this eval set (n=73).
The gap of +0.014 on recall@5 sits inside the confidence interval -0.027 to +0.068, so it is
consistent with no difference at all. About 20890 questions would be needed to settle a gap
this size.
```

### How to read it

`length` collapses to 0.164 — well above the 0 you'd get from returning random chunks (which
would land near k/chunks ≈ 5/35 ≈ 0.14, close enough that this is basically the floor), and far
below `tfidf`'s 0.877. That's the sanity check passing: the eval set can tell a real signal from
none.

`tfidf` beats `hash:512` by 0.014 — and the tool says plainly that gap is noise: **"about 20890
questions would be needed to settle a gap this size."** That number is doing real work. It isn't
"maybe with more data" hand-waving — it's the eval set's own quality assessment telling you
exactly how far you are from a trustworthy answer, and 20,890 is far enough that nobody should
act on this gap. Report the leaderboard order if you like; don't report a winner.

### Real models: what it takes, and the offline stand-in

Real embedding models need a key or a server — bare `tfidf`/`hash` can't answer "does a real
model help here." Two backends, both documented in
[embedders](../dimensions/embedders.md):

```yaml
# hosted, needs an API key in the environment (OPENAI_API_KEY, COHERE_API_KEY, ...)
embedder: [litellm:text-embedding-3-small]

# local, needs a server running:
#   docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
#       --model-id BAAI/bge-base-en-v1.5
embedder: [tei:bge-base-en-v1.5,api_base=http://localhost:8080]
```

Neither runs in this recipe — no key is checked into this repo, and standing up a server just to
generate documentation is the wrong trade. What *does* run, with no network at all, is the
`transport` hook every remote embedder takes — the same pattern
`tests/unit/test_llm_litellm.py` uses to test the real adapter without calling out:

```python
from contextgrid.pipeline import Config, build
from contextgrid.grid.runner import Runner
from contextgrid.embed.remote import TEIEmbedder
import numpy as np


def fake_embed(batch):
    """Stands in for a running TEI server -- deterministic, no docker, no download."""
    vecs = [np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(64).tolist() for t in batch]
    return vecs, 0  # (vectors, token_count)


embedder = TEIEmbedder(model="bge-base-en-v1.5,dev-stand-in", dimensions=64, transport=fake_embed)
cfg = Config(
    parser="markdown", chunker="recursive:256,overlap=32", embedder=embedder, index="dense"
)
result = Runner(corpus=corpus, headline="recall@5").run_one(cfg, evalset)
print(result.metric("recall@5"))
```

Run for real: **`recall@5: 0.096`** — near the random-chunk floor, exactly as it should be,
because `fake_embed` returns pure noise with no relationship to the text at all. That's not a
finding about embedders; it's proof the wiring works. Swap `fake_embed` for a real batch call and
the same script measures a real model — see [local-only.md](local-only.md) for a fuller offline
sweep on the same idea. Note the API: `run_one(config, evalset)` takes a single already-built
`Config`, not the string-based `matrix()` — `matrix()` deliberately rejects plugin instances (see
[configuration](../guide/configuration.md)), so a live `transport` callable can only be swept one
configuration at a time, not across a whole grid in one call.

## Part 2 — no eval set at all: `contextgrid.embed.assess`

Most people choosing an embedder have a corpus and no questions yet — the worst moment to have
no signal. `contextgrid.embed.assess` works from the vectors alone: embed the corpus, look at
the geometry, and ask whether the model can tell chunks apart on *these* documents. It never
sees a query and never claims to predict recall — see
[without-an-evalset.md](without-an-evalset.md) for the full boundary of what that can and can't
tell you.

### The command

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "examples")
import lab_demo as d
from contextgrid.pipeline import Config, build
from contextgrid.embed import assess as embed_assess, get_embedder

corpus = d.markdown_corpus()
chunks = build(Config(parser="markdown", chunker="sentence:2"), corpus).chunks
print("chunks:", len(chunks))

for name in ["tfidf", "hash:512"]:
    emb = get_embedder(name)
    emb.prepare([c.text for c in chunks])
    vectors = emb.embed_documents([c.text for c in chunks]).vectors
    quality = embed_assess(chunks, vectors, seed=0)
    print(name, "->", quality.summary())
PY
```

### The real output

```
chunks: 232
tfidf -> coherence +0.416, anisotropy 0.134, 19/510 effective dimensions, 0.0% collapsed, over 232 chunks
hash:512 -> coherence +0.386, anisotropy 0.228, 13/512 effective dimensions, 0.0% collapsed, over 232 chunks
```

### How to read it

- **Coherence is positive for both (+0.416, +0.386).** Consecutive chunks land closer together
  in vector space than random pairs do — both models see *some* structure in the text. Neither
  is degenerate.
- **`tfidf` uses more of its space:** 19 of 510 dimensions actually carry variance, against
  hash's 13 of 512. Both numbers are small in absolute terms (this is a 33-document corpus — of
  course the model isn't using hundreds of dimensions), but the *ratio* is the comparable part,
  and `tfidf` is ahead.
- **Neither is flagged `templated`, and that's worth a second look, not a shrug.** This corpus
  was deliberately built with twenty near-duplicate vendor contracts — same clauses, different
  company names and numbers — specifically to make retrieval hard. You might expect that to trip
  the "templated corpus" flag (`redundancy > 0`: documents look more alike than they cohere
  internally). It doesn't, under either embedder, because `redundancy` came back negative
  (-0.416 and -0.386): each contract's *vendor name* is a token TF-IDF and hashing both weight
  heavily, so the near-duplicates aren't actually close in these vector spaces. That's the honest
  limit of a lexical embedder's geometry: it separates documents by their unique tokens, not by
  their structure, and "templated" here would need a model that reads the *shape* of the text
  rather than counting its words.
- **These diagnostics roughly tracked the recall gap in Part 1** (tfidf ahead of hash both times,
  by a similarly small margin), but they're not certified to. Read
  `contextgrid.embed.quality`'s own docstring on this: "these are diagnostics, not a verdict...
  never whether it retrieves the right thing." Treat agreement as a nice confirmation, not
  something to expect every time.

## What would change the answer

- **A larger corpus.** Effective-dimension counts here are small because there are only 232
  chunks to estimate a covariance matrix from. On a corpus with thousands of chunks, the same
  metric becomes far more informative about whether a high-dimensional model is actually earning
  its width.
- **A real hosted or local model**, run for real rather than through the offline stand-in. The
  geometry of a genuine sentence embedding is a different shape entirely from TF-IDF's sparse
  bag-of-words space, and `anisotropy` in particular is where transformer embeddings are known to
  struggle (the "crowded cone" problem `contextgrid/embed/quality.py` describes at length).
- **More questions**, to make Part 1's comparison trustworthy — 20,890 is a real number, not a
  hedge. See [evalsets](../guide/evalsets.md) for eval set sizing.

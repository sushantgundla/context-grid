# A complete sweep, no API key

## The question

Can you run a real sweep — real embeddings, real reranking, real indexes — with no key, no
hosted account and no bill? Yes, in two tiers: what's genuinely free with a bare
`pip install context-grid`, and what needs one local server (still no key, still no network
beyond your own machine).

## Tier 1 — actually free, right now: `tfidf` + local indexes

`tfidf` and `hash` need no model, no download, no server — they're computed from the corpus
itself. Paired with the three local index types, this is a full, real sweep with nothing to
install beyond the base package.

### The config

```python no-run: abbreviated -- shown in full in "The command" below
grid = matrix(
    parser="markdown",
    chunker="recursive:256,overlap=32",
    embedder="tfidf",
    index=["dense", "bm25", "hybrid"],
    reranker=[None, "lexical"],
    k=5,
)
```

### The command

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "examples")
import lab_demo as d
from contextgrid.grid import Runner, matrix

evalset, corpus = d.build_evalset(), d.markdown_corpus()
grid = matrix(parser="markdown", chunker="recursive:256,overlap=32", embedder="tfidf",
              index=["dense", "bm25", "hybrid"], reranker=[None, "lexical"], k=5)
results = Runner(corpus=corpus, headline="recall@5").run(grid, evalset, mode="factorial")
for row in results.leaderboard("recall@5"):
    print(f"{row['config']:60} {row['recall@5']:6.3f} {row['p95_ms']:8.3f} {row['cost_per_1k']:8.4f}")
PY
```

### The real output

```
markdown · recursive:256,overlap=32 · tfidf · dense · lexical@50   0.918    0.767   0.0000
markdown · recursive:256,overlap=32 · bm25 · lexical@50            0.918    0.799   0.0000
markdown · recursive:256,overlap=32 · tfidf · hybrid · lexical@50  0.918    0.826   0.0000
markdown · recursive:256,overlap=32 · bm25                         0.890    0.037   0.0000
markdown · recursive:256,overlap=32 · tfidf · dense                0.877    0.032   0.0000
markdown · recursive:256,overlap=32 · tfidf · hybrid                0.877    0.083   0.0000
```

Every single column reads `0.0000` on cost, and it's genuinely zero — `lexical` is query-term
overlap, not a model call, and `dense`/`bm25`/`hybrid` are numpy matrix math over vectors already
computed. This is not a "free tier" in the SaaS sense; there's no meter running at all.

### How to read it

- **The `lexical` reranker earns +0.041 over the best un-reranked row** (0.918 vs 0.877) at the
  cost of about 0.7ms of latency — for free. It's the arm every real reranker has to beat,
  precisely because it's this cheap; a paid reranker that doesn't clear this bar by a wide margin
  isn't worth its latency and dollar cost.
- **`bm25` alone (0.890) beats `tfidf`+`dense` alone (0.877) on this corpus** — lexical overlap
  is doing real work here, which tracks: the eval set's questions quote exact phrases ("thirty
  days written notice", "$3,400") that literal term matching finds directly.
  `hybrid` (dense + bm25 fused) doesn't beat plain `bm25` here, which is worth knowing before
  assuming fusion is strictly additive.

## Tier 2 — a real local model: TEI, no key

For a real embedding model without a hosted account, TEI (HuggingFace's
text-embeddings-inference server) is the local option — reached over plain `urllib`, so it's
zero extra dependency once the server's running. One process serves one model, so embedding and
reranking need **separate ports**:

```bash no-run: needs a running Docker daemon and pulls images from ghcr.io
# embeddings
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
    --model-id BAAI/bge-base-en-v1.5

# reranking -- a different port; one TEI process serves one model
docker run -p 8081:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
    --model-id BAAI/bge-reranker-base
```

With both running, the config is:

```yaml
grid:
  chunker: [recursive:256]
  embedder: [tei:bge-base-en-v1.5,api_base=http://localhost:8080]
  index: [dense, bm25, hybrid]
  reranker: [null, tei-rerank:bge-reranker-base,api_base=http://localhost:8081]
  candidates: [50]
```

That doesn't run in this recipe — pulling and running a model server is the reader's call to
make, not something to do silently while generating a doc page. What runs here instead, with no
docker and no download, is the same `transport` hook `local-only` production code uses to test
these two backends (`tests/unit/` and the classes' own docstrings both point at it):

### The command

```bash
.venv/bin/python - <<'PY'
import sys, hashlib
sys.path.insert(0, "examples")
import numpy as np
import lab_demo as d
from contextgrid.pipeline import Config
from contextgrid.grid.runner import Runner
from contextgrid.embed.remote import TEIEmbedder

evalset, corpus = d.build_evalset(), d.markdown_corpus()

# Deterministic stand-in for a running TEI server -- hashes each text into a fixed vector.
# It is not a model. It exists to prove the wiring works with no server, exactly the way
# tests/unit/test_llm_litellm.py stands in for litellm. The seed has to come from a stable
# hash of the text (sha256), not Python's built-in hash() -- that's randomised per process,
# so a "deterministic" stand-in built on it would print a different number on every run.
def fake_embed(batch):
    vecs = []
    for text in batch:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vecs.append(rng.standard_normal(64).tolist())
    return vecs, 0

embedder = TEIEmbedder(model="bge-base-en-v1.5,dev-stand-in", dimensions=64, transport=fake_embed)
cfg = Config(parser="markdown", chunker="recursive:256,overlap=32", embedder=embedder, index="dense")
result = Runner(corpus=corpus, headline="recall@5").run_one(cfg, evalset)
print("recall@5:", result.metric("recall@5"), "| chunks:", result.chunk_count)
PY
```

### The real output

```
recall@5: 0.1232876712328767 | chunks: 35
```

### How to read it

0.123 is near the random-chunk floor (k=5 over 35 chunks with no useful signal lands around
5/35 ≈ 0.14) — which is exactly what it should be, since `fake_embed` returns pure Gaussian
noise with zero relationship to the text. **This number says nothing about whether a real BGE
model is good.** It says the plumbing — batching, prefixing, dimension checks, the index build —
runs correctly with no server present. Swap `fake_embed` for a real one and the identical script
measures a real model, which is the whole reason the hook exists: it's how you'd write a test for
code you build on top of this package, not a substitute for running the real thing.

**One real limitation to know before relying on this for a sweep**: `Runner.run_one` takes a
single already-built `Config`; the string-based `matrix()` used for a full grid deliberately
*rejects* plugin instances (see `configuration.md`), so a live `transport` callable can only
drive one configuration at a time, not a swept axis of them in one call.

## What would change the answer

- **Actually running TEI.** The commands above are real and copy-pasteable; running them turns
  Tier 2 from "the wiring is correct" into "here's what this model retrieves." That's the
  difference between this recipe and [choose-an-embedder.md](choose-an-embedder.md)'s Part 1.
- **`usearch` or `faiss` in place of the exact `dense` index** — both are also local, no key, no
  network, and both are already installed in this environment (`pip install
  'context-grid[index]'`). Worth a look for the "what does approximation cost you" question,
  which Tier 1 doesn't touch at all since `dense` here is exact search.
- **A GPU-backed TEI image**, if the CPU one is too slow on a large corpus — same API, same
  `docker run`, different image tag.

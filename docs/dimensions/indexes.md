# Indexes

An index is a **store**: where the vectors (or the text, for BM25) live, and how one search is
executed against them. That's a different question from *how many* searches happen and *who*
decides what to search for — that's the [retrieval](retrieval.md) axis, which sits on top of
whatever index you pick.

Seven indexes are registered: `dense`, `bm25`, `hybrid`, `quantized`, `faiss`, `usearch`,
`pgvector`. This page covers all of them, every faiss index type, every usearch dtype, and the
one property that matters most for reading any of it: `is_exact`.

Source: `src/contextgrid/index/`. Tests: `tests/unit/test_embed_index.py`,
`tests/unit/test_index_ann.py`, `tests/unit/test_index_pgvector.py`.

## `is_exact`: the property that makes the other numbers mean something

```python
@property
def is_exact(self) -> bool:
    """False for approximate search.

    An approximate index must be compared against its exact twin before its numbers mean
    anything. Tuning `efSearch` without knowing what recall it cost is guessing, and it
    is guessing in the direction that looks good.
    """
```

`dense`, `bm25`, `hybrid` and `pgvector:exact` are exact — every candidate is actually scored.
`faiss` (except `kind="flat"`), `usearch` and the ANN pgvector kinds are approximate: they trade
recall for speed and memory, and the size of that trade is corpus-specific. `is_exact` is what
lets `recall_against_exact()` (`src/contextgrid/index/quantize.py`) turn "I tuned `efSearch`
until it felt fast" into a number:

```python
from collections.abc import Sequence
from contextgrid.index.base import Scored


def recall_against_exact(approximate: Sequence[Scored], exact: Sequence[Scored], k: int) -> float:
    """What fraction of exact search's top k the approximate index also found."""
```

Every approximate index in this codebase is judged against an exact one below, with real
numbers, on the same corpus, in the same run.

## The exact stores: `dense`, `bm25`, `hybrid`

### `dense` — cosine or dot product, no approximation

```python no-run: chunks/vectors/query_vector stand in for your own corpus, embedded
from contextgrid.index import get_index

dense = get_index("dense:cosine")  # metric="cosine" (shorthand) — or "dot"
dense.build(chunks, vectors)
dense.search("which header carries the api key?", query_vector, k=2)
# [Scored(chunk_id='c2', score=0.691), Scored(chunk_id='c3', score=0.126)]
```

Parameters: `metric: str = "cosine"`. `cosine` normalises both sides, so it is unaffected by
vector magnitude; `dot` does not, which is correct for models trained that way (some are) and
wrong for models that were not — another quiet way a comparison becomes unfair if picked
carelessly. This is the reference every other dense-ish index below is judged against.

### `bm25` — the baseline that keeps embarrassing people

```python no-run: chunks stands in for your own corpus
bm25 = get_index("bm25:1.5")  # k1=1.5 (shorthand)
bm25.build(chunks)  # no vectors, no embedder needed
bm25.search("X-Api-Key header", None, k=2)
# [Scored(chunk_id='c2', score=5.424)]
```

Parameters: `k1: float = 1.5`, `b: float = 0.75`. `needs_vectors = False`. On keyword-heavy
corpora — error codes, product names, statute numbers — BM25 regularly beats a dense model that
costs real money to run. `k1` and `b` are swept rather than assumed here, because the usual
defaults were tuned on TREC news articles, not on 200-token chunks.

### `hybrid` — dense and sparse, fused

```python no-run: chunks/vectors/query_vector stand in for your own corpus, embedded
hybrid = get_index("hybrid:rrf")  # fusion="rrf" (shorthand) — or "weighted"
hybrid.build(chunks, vectors)  # builds both a dense and a BM25 index internally
hybrid.search("api key header", query_vector, k=2)
```

Parameters: `fusion: str = "rrf"`, `rrf_k: int = 60`, `alpha: float = 0.5` (weighted-fusion
weight on the dense side), `candidates: int = 100` (how deep each side is read before fusing).

Two fusion methods that behave differently:

- **`rrf`** (reciprocal rank fusion) ignores score magnitude and uses only rank —
  `1 / (k + rank)` per side. Robust, because a cosine of 0.31 and a BM25 score of 14.2 are not
  on the same scale and no normalisation makes them mean the same thing.
- **`weighted`** min-max normalises each side to `[0, 1]` then blends by `alpha`. It can express
  "the dense side is usually right, lean on it," which `rrf` cannot, at the cost of being more
  sensitive to one side producing an outlier.

Which wins is corpus-dependent — that's the whole premise of the tool — so neither is assumed
to be the default. Configured via `INDEXES.register("hybrid", ...)` as `dense=ExactDenseIndex`
+ `sparse=BM25Index`, both swept from the same spec: `hybrid:weighted,alpha=0.8,k1=2.0`
configures both sides in one string.

## `quantized` — trading recall for memory, and measuring the trade

```python
from contextgrid.index import get_index

quantized = get_index("quantized")  # scheme="scalar" (shorthand default)
```

Parameters: `scheme: str = "scalar"` (`none` / `scalar` / `product` / `binary`),
`subspaces: int = 8` (for `product`), `rescore: int = 0`, `metric: str = "cosine"`.

Four schemes, increasing in how much they throw away. Measured on a 400-vector, 64-dimension
corpus against exact search (`ExactDenseIndex`), same seed, same query:

| Scheme | recall@10 | size | vs. exact (102,400 bytes) |
|---|---|---|---|
| `scalar` | 1.000 | 25,600 bytes | 4× smaller, free |
| `product` (8 subspaces) | 0.900 | 3,200 bytes | 32× smaller, real cost |
| `binary`, `rescore=0` | 0.300 | 3,200 bytes | 32× smaller, badly hurt |
| `binary`, `rescore=100` | 0.800 | 105,600 bytes | rescoring restores most of it, at the cost of keeping the originals |

**`scalar`** maps each dimension linearly to one byte; on normalised embeddings the recall cost
is usually near nothing. **`product`** replaces subspaces of the vector with the nearest of 256
k-means centroids learned from the corpus itself. **`binary`** keeps one bit per dimension —
above/below the corpus mean — and ranks on Hamming distance.

Binary alone is crude, which the table shows. `rescore=N` pulls `N` candidates from the
compressed shortlist and re-ranks them against the kept originals, and that's what makes
aggressive quantization usable: without it, `QuantizedDenseIndex` emits a warning —

> "binary quantization without a rescoring pass ... ranks on Hamming distance alone. It will
> score far below its potential, and concluding from that that binary quantization does not
> work is the most common mistake made with it. Set rescore to 50 or more."

`size_bytes()` for a rescored index includes the kept originals, on purpose — reporting the
compressed size alone would flatter every configuration that rescores, which is most of the
good ones.

## `faiss` — flat, HNSW, IVF, IVFPQ, from one wheel

Needs `pip install 'context-grid[index]'`. `kind` is the shorthand: `faiss:hnsw`.

```python
faiss_index = get_index("faiss:hnsw")
```

Parameters: `kind: str = "hnsw"` (`flat` / `hnsw` / `ivf` / `ivfpq`), `metric: str = "cosine"`,
`m: int = 32`, `ef_construction: int = 200`, `ef_search: int = 64`, `nlist: int = 100`,
`nprobe: int = 8`, `pq_subquantizers: int = 8`, `pq_bits: int = 8`.

Measured on 2,000 vectors, 64 dimensions, against `ExactDenseIndex`:

| `kind` | `is_exact` | recall@10 | size |
|---|---|---|---|
| `flat` | `True` | 1.000 | 512,000 bytes |
| `hnsw` | `False` | 1.000 | 1,024,000 bytes |
| `ivf`, `nprobe=1` | `False` | 0.200 | 528,000 bytes |
| `ivf`, `nprobe=10` | `False` | 0.600 | 528,000 bytes |
| `ivfpq`, `pq_bits=8` | `False` | 0.200 | 26,000 bytes |

`flat` is exhaustive — the reference the other three are judged against, computed by the same
library so implementation differences don't confound the comparison. `nprobe` is the
recall/latency knob for `ivf`: how many clusters get searched, one integer, and the table above
shows exactly what raising it from 1 to 10 buys.

**A small corpus trains a bad codebook silently.** faiss wants roughly 39 training points per
cluster for `ivf` and per *codebook entry* for `ivfpq` — at the default 8 bits that's 256
entries, ~10,000 vectors, before `ivfpq` trains properly at all. Below that, faiss prints a
warning on stderr nobody reads and trains anyway, returning plausible, wrong neighbours. Rather
than let that happen quietly, `FaissIndex` reduces `nlist` and `pq_bits` to fit the corpus and
records the reduction:

```python no-run: chunks/vectors stand in for your own corpus, embedded; needs faiss (context-grid[index])
index = get_index("faiss:kind=ivfpq,pq_bits=8")
index.build(chunks, vectors)  # 2,000 vectors
print(index.fitted_to_corpus)
# {'nlist': (100, 51), 'pq_bits': (8, 5)}
```

That's what actually ran, not what was asked for — the run can say so instead of quietly
reporting the recall of parameters nobody chose.

## `usearch` — a second opinion on HNSW

Needs `pip install 'context-grid[index]'`. `dtype` is the shorthand: `usearch:i8`.

```python
usearch_index = get_index("usearch:i8")
```

Parameters: `connectivity: int = 16`, `expansion_add: int = 128`, `expansion_search: int = 64`,
`dtype: str = "f32"` (`f32` / `f16` / `i8`).

Worth having as a second implementation of HNSW for exactly one reason: when two libraries
disagree about the recall of nominally identical settings, that disagreement is itself the
finding. Measured on the same 2,000×64 corpus:

| `dtype` | recall@10 | size |
|---|---|---|
| `f32` | 1.000 | 768,000 bytes |
| `f16` | 1.000 | 512,000 bytes |
| `i8` | 0.900 | 384,000 bytes |

`size_bytes()` is computed from the dtype rather than read from usearch's own
`Index.memory_usage`, on purpose: that call reports the arena usearch allocated, which barely
moves between `f32` and `i8` on a small index — using it would show quantization saving
nothing, the opposite of true and the entire reason `dtype` is on this axis.

**`b1` is not offered.** It was registered as a valid `dtype`, but usearch's binary mode wants
bit-packed input and a Hamming metric, not the raw `float32` matrix every other arm on this axis
takes — the first real `build()` call raised `ValueError: The number of vector dimensions
doesn't match!` inside usearch's own `add_many`. Rather than ship a dtype that fails on first
use, `USearchIndex.DTYPES` no longer includes it. `faiss` has no binary index type either;
`quantized:binary` (above) is where this package does binary vectors, with a rescoring pass to
recover the recall it costs.

## `pgvector` — what people actually deploy on

Needs `pip install 'context-grid[pgvector]'` (installs `psycopg`) and a running Postgres with
the `vector` extension:

```bash no-run: needs a running Docker daemon and pulls an image from Docker Hub
docker run -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg17
```

```
index: pgvector:hnsw,dsn=${PGVECTOR_DSN}
```

Parameters: `kind: str = "hnsw"` (`exact` / `hnsw` / `ivfflat`), `metric: str = "cosine"`,
`dsn: str | None = None` (or `PGVECTOR_DSN` / `DATABASE_URL` from the environment),
`m: int = 16`, `ef_construction: int = 64`, `ef_search: int = 40`, `lists: int = 100`,
`probes: int = 8`, `table_prefix: str = "contextgrid"`.

Every other index on this page is a library holding vectors in memory. `pgvector` is a
database, which is exactly why it earns a place on the axis: "the HNSW settings my Postgres is
using" is a genuinely different question from "the HNSW settings faiss would use," and measuring
the library while shipping the database is how the two quietly diverge. It differs from its
neighbours in ways that are inherent, not incidental:

- **It needs a server.** There is no in-process fallback — a fake pgvector would measure
  nothing, and passing in CI would be worse than the arm being absent. `tests/unit/test_index_pgvector.py`
  skips outright without `PGVECTOR_DSN` set.
- **`ivfflat.probes` and `hnsw.ef_search` are session settings, not index settings.** They're
  applied per query, because that's the only way to sweep them — worth knowing, because a
  production system that never sets them runs on whatever the server defaults to.
- **Building an index writes to a real database.** Every run creates a table named
  `{table_prefix}_{uuid}` and drops it on `close()`, so two sweeps against the same database
  never collide.

Measured against a real Postgres 17 + pgvector container, 500 vectors, 32 dimensions:

| `kind` | `is_exact` | recall@10 | size |
|---|---|---|---|
| `exact` | `True` | 1.000 | 196,608 bytes |
| `hnsw` | `False` | 1.000 | 425,984 bytes |
| `ivfflat`, `probes=1` | `False` | 0.000 | 1,040,384 bytes |
| `ivfflat`, `probes=8` | `False` | 0.500 | 1,040,384 bytes |

`size_bytes()` here calls `pg_total_relation_size` — the one index on this page that can answer
"how much memory does this use" honestly instead of estimating it.

There's a real bug this arm caught that nothing else could have: `SET LOCAL ivfflat.probes = …`
only lasts until the end of the current transaction, and on an autocommit connection there is no
transaction to last for. Postgres accepted the statement, did nothing, and every query silently
ran at the default `probes = 1` while the sweep reported numbers for whatever probe count was
asked for. Fixed by switching to plain `SET`. Only a live database could have caught this — the
recall numbers above are what the fixed code actually measures, not what was assumed.

## See also

- Recall/size numbers for `faiss`, `usearch` and `pgvector` are also recorded in
  [`../adoption-backlog.md`](../adoption-backlog.md) under **Dimension 3 — Indexing**, alongside
  the library-selection reasoning.
- [embedders](embedders.md) — what produces the vectors these indexes hold.
- [retrieval](retrieval.md) — how many searches run against an index, and who decides.

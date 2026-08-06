# What you can measure without an eval set

## The question

You have a corpus. You don't have questions yet — writing a good eval set takes real work (see
[evalsets](../guide/evalsets.md)), and most decisions about parser, chunker and embedder get
made before that work is done. What can `context-grid` tell you before a single question exists,
and — just as important — what can it *not* tell you, so you don't mistake a corpus diagnostic
for a retrieval verdict?

## What you can measure: the corpus's own shape

### The command

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "examples")
import lab_demo as d
import contextgrid as cg
from contextgrid.pipeline import Config, build

corpus = d.markdown_corpus()
built = build(Config(parser="markdown", chunker="recursive:256,overlap=32"), corpus)
profile = cg.fingerprint(corpus, built.parses)
print(profile.summary())
for hint in profile.hints():
    print(" -", hint)
PY
```

### The real output

```
33 files, 31,326 bytes, 31,286 chars via markdown, 8% tables
 - Documents average 7 headings each. Structural chunking usually wins on corpora like this.
 - The median document is 1,007 characters. Chunk sizes above that cannot differentiate, so sweep small sizes.
```

### How to read it

This is a **prior**, not a result — it says what's likely to matter, before you've spent a
single query on finding out. Both hints here are concrete enough to act on directly: try
`structural` chunking, and don't bother sweeping chunk sizes above ~1,000 characters, because
above the median document length every size behaves the same (one chunk per document). Table
ratio (8% here) is the same kind of signal for the parser axis — a corpus that's mostly tables
is exactly where [parsers](../dimensions/parsers.md) documents the biggest spread between a fast
text extractor and a table-aware one.

**Worth knowing: this hint was wrong on this corpus.**
[choose-a-chunker.md](choose-a-chunker.md) ran the real sweep and structural chunking scored
0.452 against recursive chunking's 0.877 — it lost, decisively, once real questions were
involved. That's not a bug in the fingerprint; it's the honest limit of a prior built from
document shape alone. It doesn't know what your questions ask for. Use it to decide what to
*try first* in a sweep, never as a substitute for running one.

## What you can measure: whether an embedder can even discriminate

`contextgrid.embed.assess` works from a corpus's embedded vectors alone — no query, no gold
answer. It answers "can this model tell chunks apart on these documents," never "does it
retrieve the right one." See [choose-an-embedder.md](choose-an-embedder.md) for the full worked
example (`tfidf` vs `hash:512`, real output, real geometry); the short version:

```python no-run: abbreviated -- chunks stands in for your own corpus; full worked example in choose-an-embedder.md
from contextgrid.embed import assess as embed_assess, get_embedder

emb = get_embedder("tfidf")
emb.prepare([c.text for c in chunks])
vectors = emb.embed_documents([c.text for c in chunks]).vectors
quality = embed_assess(chunks, vectors)
print(quality.summary())
# coherence +0.416, anisotropy 0.134, 19/510 effective dimensions, 0.0% collapsed, over 232 chunks
```

Five things this can catch with zero questions: **anisotropy** (is the model too crowded to
discriminate at all — `crowded` at ≥0.6, `degenerate` at ≥0.95), **coherence** (do consecutive
chunks actually land closer together than random ones — the honest floor is "is there any
structure here at all"), **redundancy** (does the corpus look templated — near-duplicate
documents that no embedder will cleanly separate), **effective dimensions** (is a 1536-d model
actually using 1536 dimensions, or paying for width it doesn't use), and **collapse** (chunks the
model literally cannot tell apart).

## What you cannot measure without questions

Be precise about the boundary, because it's easy to over-read a good-looking diagnostic:

| Question | Needs questions? | What answers it |
|---|---|---|
| "Does this corpus have a lot of tables?" | No | `fingerprint().table_ratio` |
| "Is structural chunking likely to help?" | No | `fingerprint().hints()` — a prior |
| "Can this embedder tell chunks apart at all?" | No | `embed.assess` |
| "Does this embedder retrieve the *right* chunk?" | **Yes** | `recall@k`, needs an eval set |
| "Is chunker A actually better than chunker B here?" | **Yes** | a scored sweep + significance test |
| "Does the generator's answer stay faithful to what was retrieved?" | **Yes** — and even the retrieval eval set isn't enough; see `DeepEval`-backed generation metrics in [scoring](../scoring/metrics.md) |

The load-bearing sentence is in `contextgrid/embed/quality.py` itself: *"These are diagnostics,
not a verdict. They say whether a model can discriminate on this corpus, never whether it
retrieves the right thing — that is recall's job, and recall needs questions."* A model that
scores badly on the geometry diagnostics will not be rescued by anything downstream — that part
is a real, actionable negative signal. A model that scores *well* on them is necessary, not
sufficient: it still might not retrieve what a real question needs, and the only way to know is
to ask it real questions.

## What would change the answer

- **A real embedding model instead of `tfidf`.** These diagnostics are most informative on the
  kind of embedding they were written to catch trouble in — a dense transformer model, where
  anisotropy ("the crowded cone problem") is a known, well-documented failure mode. TF-IDF's
  sparse space doesn't suffer it the same way, so `tfidf`'s numbers here are a weaker test of the
  diagnostic than a real model's would be.
- **A larger corpus**, for the effective-dimensions number specifically — 232 chunks is a small
  sample to estimate a covariance spectrum from, and the ratio (dimensions used / dimensions
  paid for) gets more trustworthy as the chunk count grows.
- **Once you do have an eval set**, run `contextgrid.evalset.assess(evalset)` too — it scores the
  *eval set's* quality the same way this page scores the corpus and embedder: size, review
  status, type balance, and the smallest difference it can detect. See
  [evalsets](../guide/evalsets.md).

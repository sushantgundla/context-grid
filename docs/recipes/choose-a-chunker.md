# Choose a chunker

## The question

Which chunker actually helps on *your* documents — and is the difference big enough to trust,
or just noise from a small eval set?

`chunker` is one axis on the grid, so answering this is one sweep: hold everything else still,
vary the chunker, read the leaderboard, and check whether the winner is real before you believe
it. This recipe runs that sweep for real, on the corpus and eval set shipped in
`examples/lab_demo.py` — three real documents (a contract, an API reference, a security policy)
plus thirty near-duplicate distractor documents, with 74 questions. It's the same corpus behind
`sushantgundla.com/lab`, so nothing here is staged for the doc.

## The config

Five chunkers, everything else fixed: `tfidf` embedder, `dense` index, `k=5`.

```python no-run: abbreviated -- corpus/evalset come from examples/lab_demo.py, built in full in "The command" below
from contextgrid.grid import Runner, matrix

grid = matrix(
    parser="markdown",
    chunker=[
        "recursive:256,overlap=32",
        "chonkie:recursive:256",
        "langchain:recursive:256",
        "sentence:3",
        "structural:200,min_size=24",
    ],
    embedder="tfidf",
    index="dense",
    k=5,
)
results = Runner(corpus=corpus, headline="recall@5").run(grid, evalset, mode="factorial")
```

`recursive:256` is context-grid's own splitter; `chonkie:recursive:256` and
`langchain:recursive:256` are the same idea from two libraries other tools actually ship (see
[chunkers](../dimensions/chunkers.md) for what "the same idea" means in each library's own
terms). `sentence:3` and `structural:200,min_size=24` cut on different boundaries entirely —
whole sentences, and whole sections.

## The command

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "examples")
import lab_demo as d
from contextgrid.grid import Runner, matrix

evalset = d.build_evalset()
corpus = d.markdown_corpus()

grid = matrix(
    parser="markdown",
    chunker=["recursive:256,overlap=32", "chonkie:recursive:256", "langchain:recursive:256",
             "sentence:3", "structural:200,min_size=24"],
    embedder="tfidf", index="dense", k=5,
)
results = Runner(corpus=corpus, headline="recall@5").run(grid, evalset, mode="factorial")

for row in results.leaderboard("recall@5", extra=["ndcg@5"]):
    print(f"{row['config']:56} {row['recall@5']:6.3f} {row['ndcg@5']:7.3f} {row['chunks']:7}")
print()
print(results.summary("recall@5"))
PY
```

## The real output

```
markdown · recursive:256,overlap=32 · tfidf · dense       0.877   0.831      35
markdown · chonkie:recursive:256 · tfidf · dense          0.877   0.831      35
markdown · langchain:recursive:256 · tfidf · dense        0.877   0.831      35
markdown · sentence:3 · tfidf · dense                     0.491   0.478     199
markdown · structural:200,min_size=24 · tfidf · dense     0.452   0.382     182

markdown · recursive:256,overlap=32 · tfidf · dense scored best on recall@5 at 0.877, across 5
configurations on 73 questions. markdown · recursive:256,overlap=32 · tfidf · dense and markdown
· chonkie:recursive:256 · tfidf · dense are not distinguishable on this eval set (n=73). The gap
of +0.000 on recall@5 sits inside the confidence interval +0.000 to +0.000, so it is consistent
with no difference at all. They scored identically on every single question, so this is not a
close call between two different configurations -- they are behaving the same way.
```

## How to read it

**The top three are the same chunker wearing three names.** `recursive:256`, `chonkie:recursive:256`
and `langchain:recursive:256` produce 35 chunks each and score *exactly* 0.877 — not close,
identical on every question. That's not a coincidence: [chunkers](../dimensions/chunkers.md)
documents that both adapters get their size unit corrected to match context-grid's own
tokenizer, so "256" means the same 256 tokens everywhere. When three independently-written
splitters agree exactly, the finding isn't "chonkie is as good as ours" — it's "this corpus's
document boundaries are simple enough that recursive splitting on any implementation lands in
the same place." That's worth knowing before you credit a library for a win it didn't need to
earn.

**Sentence and structural chunking lose, and lose for a legible reason.** They produce far more
chunks (199 and 182 against 35) — smaller units that scatter each answer's evidence across more
candidates competing for the same top-5 slots. Structural chunking, in particular, is often the
*hint* the tool itself would give you first — run `contextgrid profile` on a heading-heavy
corpus and it usually recommends structural chunking. Here it loses, which is the honest
counter-example: a hint based on document shape is a prior, not a verdict, and this sweep is
what overrides it.

**The winner-vs-runner-up test came back "not distinguishable" — and here it's telling you
something different than usual.** Normally that sentence means "your eval set is too small to
tell." Here it means the opposite: the two configurations produced *identical* per-question
scores, because they're the same chunker under the hood. Read the wins/losses breakdown (or
just the chunk count) before assuming a tie means "eval set too small" — sometimes it means "not
actually two different things."

## What would change the answer

- **A corpus with denser or more irregular structure.** This corpus's documents average 7
  headings each (`contextgrid profile` says so directly — see
  [without-an-evalset.md](without-an-evalset.md)), which is exactly what structural chunking is
  built for, and it still loses here because the eval set's answers are short factual spans that
  a small recursive window catches easily. A corpus of long narrative sections with the answer
  spread across a whole section would likely flip this.
- **A bigger `k`.** At k=5, sentence and structural chunking are competing many small chunks for
  five slots. Widen `k` to 10 or 20 and the gap should close, because it stops being a
  slots-scarcity problem.
- **Fewer, larger documents.** With one chunk per document (this corpus, at 256 tokens, mostly
  produces that), the chunker axis mostly can't matter — there's nothing to cut differently.
  [without-an-evalset.md](without-an-evalset.md) shows how to check chunks-per-document before
  running the sweep at all.
- **More questions.** `contextgrid.evalset.assess` puts this eval set's detectable difference at
  0.23 — the smallest gap it can tell from noise, at 74 questions. The 0.877 vs 0.491 gap here is
  nearly double that, so it's trustworthy; a gap that size on a 12-question eval set would not
  be. [significance](../scoring/significance.md) and [evalsets](../guide/evalsets.md) cover
  eval-set sizing directly.

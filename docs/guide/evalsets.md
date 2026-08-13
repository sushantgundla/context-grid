# Writing an eval set

An eval set is a list of questions with known answers — the thing every sweep is scored
against. This page covers the two ways ground truth is stored, the file formats, and how to
write one.

## Why ground truth is a quote, not a chunk ID

Most retrieval tools store ground truth as a **chunk ID**: "the answer to this question is
chunk 17." That works exactly as long as you never change how the document is chunked. The
moment you sweep the chunker — the entire point of context-grid — chunk 17 under a 512-token
splitter is different text than chunk 17 under a semantic splitter. A gold chunk ID from one
strategy means nothing under another, and nothing warns you the comparison has become invalid.

context-grid stores ground truth as a **character span** in the source document instead:
`(document, start, end)`. Chunks carry the same kind of span. At scoring time, the gold span is
resolved onto whichever chunks a configuration happened to produce — by measuring how much of
the gold span's characters each chunk covers. The eval set is written once and stays correct
across every chunker you ever sweep.

That's `GoldSpan`. There's a second problem it doesn't solve.

## `GoldAnchor` vs `GoldSpan`

| | `GoldAnchor` | `GoldSpan` |
|---|---|---|
| What it stores | The evidence, quoted — plus which document, and optionally a page hint | `(document, start_char, end_char)` |
| Survives a change of... | Parser *and* chunker | Chunker only |
| Where it lives | `EvalItem.anchors` | `EvalItem.gold` |
| When you'd write one | Almost always — this is what you author by hand | Rarely by hand — it's what an anchor *resolves to* |

The reason both exist: a `GoldSpan`'s offsets are only meaningful against the one parse that
produced them. Two parsers read the same PDF into different text — different whitespace,
different table linearisation, different heading placement — so a character offset from one
parser's output is meaningless against another's. Sweeping the **parser** axis, which is the
axis "nothing else in the field measures" (see [getting-started.md](getting-started.md)),
requires ground truth that doesn't depend on any particular parse. A `GoldAnchor` is that:
it says "the evidence is this quoted sentence, on page 4 of this file" — true regardless of
which parser read the file.

At scoring time, `GoldAnchor`s are resolved into `GoldSpan`s against whichever parse a
configuration used. There's a useful side effect: if a parser mangles a table badly enough that
the quoted evidence no longer appears anywhere in its output, the anchor **fails to resolve** —
and that failure is itself a measurement. A parser that loses the evidence cannot retrieve it,
and the eval set catches that automatically rather than silently scoring it as a miss with no
explanation.

**Write anchors, not spans, unless you have a specific reason not to.** An anchor-only eval set
is portable across every axis; a span-only one is locked to the parse it was authored against
and can't sweep `parser` at all.

### `GoldAnchor` fields

| Field | Type | Default | What it does |
|---|---|---|---|
| `source_id` | string | — | Which document, matching the corpus's file naming (e.g. `refunds.md`). |
| `quote` | string | — | The evidence, copied verbatim from the source text. Must actually appear in the document. |
| `grade` | int | `2` | Relevance grade: `2` fully answers, `1` partially relevant, `0` irrelevant. Graded relevance is what makes nDCG mean anything. |
| `page_hint` | int or `null` | `null` | Narrows the search to one page — useful when the same sentence (boilerplate, a repeated header) appears more than once across a document. |
| `occurrence` | int | `0` | Disambiguates a quote that appears more than once in the document, counting from zero in reading order. |

## File formats

### JSONL — the native format

One JSON object per line. The first line is a header carrying the eval set's identity; every
line after is one question.

```jsonl
{"_evalset": {"id": "policy-questions", "version": 1, "source": "manual", "meta": {}}}
{"id": "q1", "question": "How long do refunds take?", "gold": [], "anchors": [{"source_id": "refunds.md", "quote": "within 30 days of purchase", "grade": 2, "page_hint": null, "occurrence": 0}], "qtype": null, "answer": null, "meta": {}}
```

Written and read with `write_jsonl` / `read_jsonl`:

```python
from contextgrid.core.evalset import EvalSet, EvalItem, GoldAnchor
from contextgrid.evalset.io import write_jsonl, read_jsonl

evalset = EvalSet(
    id="policy-questions",
    items=(
        EvalItem(
            id="q1",
            question="How long do refunds take?",
            anchors=(GoldAnchor(source_id="refunds.md", quote="within 30 days of purchase"),),
        ),
    ),
)
write_jsonl(evalset, "questions.jsonl")
read_jsonl("questions.jsonl")  # round-trips everything, including anchors
```

JSONL is the only format that round-trips anchors, grades, page hints, and question types
without loss — use it as the format you keep in version control.

### CSV — what a subject-matter expert will actually hand you

Because they wrote the questions in a spreadsheet, and telling them their spreadsheet is wrong
is a worse experience than just reading it. Column names are matched loosely, case-insensitive:

| Field | Accepted column names |
|---|---|
| `id` | `id`, `question_id`, `qid` |
| `question` | `question`, `query`, `q` |
| `source_id` | `source_id`, `document`, `doc`, `doc_id`, `file`, `filename` |
| `quote` | `quote`, `evidence`, `answer_span`, `context`, `passage` |
| `grade` | `grade`, `relevance`, `rel` |
| `page` | `page`, `page_hint`, `page_number` |
| `qtype` | `qtype`, `type`, `question_type`, `category` |
| `answer` | `answer`, `expected_answer`, `gold_answer` |

Only `question` is required — a row with a question but no quote/document becomes a question
with no evidence at all, and `is_answerable` stays `False` until someone fills one of them in.
A `GoldAnchor` is built automatically whenever both a quote and a document are present, and an
anchor is enough: `is_answerable` is `bool(gold or anchors)`, so a row with a quote and a
document is answerable the moment it is read, before anything has been scored or even parsed.

The stricter question — "has that evidence been *located* in this parser's output?" — is
`is_resolved`, which is `bool(gold)` and is `False` on a freshly read CSV. See
[spans-and-anchors.md](../scoring/spans-and-anchors.md#is_answerable-vs-is_resolved) for why
the two are separate and what the gap between them measures.

```csv
question,document,evidence
How long do refunds take?,refunds.md,within 30 days of purchase
How fast is express shipping?,shipping.md,arrives the next business day
```

```python
from pathlib import Path
from contextgrid.evalset.io import read_csv

Path("questions.csv").write_text(
    "question,document,evidence\n"
    "How long do refunds take?,refunds.md,within 30 days of purchase\n"
    "How fast is express shipping?,shipping.md,arrives the next business day\n"
)
evalset = read_csv("questions.csv", evalset_id="from-csv")
```

```
>>> for item in evalset:
...     print(item.id, "|", item.question, "|", item.anchors)
q1 | How long do refunds take? | (GoldAnchor(source_id='refunds.md', quote='within 30 days of purchase', grade=2, page_hint=None, occurrence=0),)
q2 | How fast is express shipping? | (GoldAnchor(source_id='shipping.md', quote='arrives the next business day', grade=2, page_hint=None, occurrence=0),)
```

An `id` column is optional — rows without one get `q1`, `q2`, ... in file order.
`write_csv(evalset, path)` writes the same shape back out for hand editing.

### BEIR and LegalBench-RAG — importing published benchmarks

`read_beir(queries_path, qrels_path)` imports the standard IR layout. Its gold is
**document-level**, not span-level — good enough to compare retrievers, not good enough to
compare chunkers fairly, since every chunk of a gold document counts as relevant whether or not
it actually holds the evidence. The imported set's `meta` says so.

`read_legalbench_rag(path)` imports [LegalBench-RAG](https://arxiv.org/abs/2408.10343), the one
public benchmark that stores ground truth as character spans — the same decision context-grid
makes. It's what `contextgrid validate` uses to check the scorer itself against a published
number; see [cli.md](cli.md#validate).

It reads a JSON object with a `tests` array (a bare array works too). Each test needs a `query`
and a list of `snippets`, and each snippet needs a `file_path` — the document, named relative
to the corpus directory — and a `span` of `[start, end]` character offsets into that file
counted from 0. `id` and `answer` are optional; ids default to `lb0`, `lb1`, ... in file order.
Snippets missing a `file_path` or a two-element `span` are dropped rather than erroring, and a
test with no `query` is skipped entirely, so a file that imports cleanly can still be short of
what you put in it — check the length of what comes back.
[cli.md](cli.md#the-file-it-expects) has a complete example file.

## Drafting one instead of writing it by hand

Writing a hundred questions by hand is a day nobody has. `contextgrid.evalset.generate` drafts
one from a corpus's chunks, using an LLM to write questions and quote their own evidence — so
an invented question (one that doesn't quote real text) is detectable, not just suspected.
A drafted set is explicitly **not** ground truth yet: it needs the filters
(`contextgrid.evalset.default_filters`) and a human pass through the review queue
(`contextgrid.evalset.ReviewQueue`) before it's trustworthy.

There's also `KeywordProbeGenerator` — no model, no cost, and not a real question generator:
it turns a passage's rarest words into a query. Good for smoke-testing that a pipeline is wired
up end to end; the scores it produces are far higher than anything a real user's phrasing would
get, so don't publish a leaderboard built on it.

```python
import os

os.makedirs("documents", exist_ok=True)
with open("documents/policy.md", "w") as f:
    f.write(
        "# Refund Policy\n\nRefunds are issued within 30 days of purchase. Digital goods "
        "are not refundable once downloaded.\n\n# Shipping\n\nStandard shipping takes 5 to "
        "7 business days. Express shipping arrives the next business day.\n"
    )

from contextgrid import Corpus, get_parser, get_chunker
from contextgrid.evalset.generate import generate, KeywordProbeGenerator

corpus = Corpus.from_dir("./documents")
parser = get_parser("markdown")
chunker = get_chunker("recursive:256")

chunks = []
for source in corpus:
    chunks.extend(chunker.chunk(parser.parse(source)))

draft = generate(chunks, KeywordProbeGenerator(), sample=None)
```

```
>>> print(draft.count, "questions drafted,", draft.chunks_skipped, "chunks skipped")
1 questions drafted, 0 chunks skipped
>>> for item in draft.evalset:
...     print(item.id, "|", item.question, "|", item.anchors[0].quote)
policy.md:0-220#probe | digital downloaded goods once refundable | Digital goods are not refundable once downloaded.
```

One question, not two: the whole file is one chunk at `recursive:256`, and the generator writes at
most one probe per chunk.

`generate` skips any chunk shorter than `min_chunk_words` (default 25) — there isn't enough text in
a short chunk to build a question about. That is why chunking *smaller* can hand you fewer questions,
or none at all. When it does, `draft.warnings` says so in as many words, so read it before concluding
the generator is broken:

```python
chunker = get_chunker("sentence:1")   # four chunks, all under 25 words
chunks = [c for source in corpus for c in chunker.chunk(parser.parse(source))]
draft = generate(chunks, KeywordProbeGenerator(), sample=None)
print(draft.count, "questions drafted,", draft.chunks_skipped, "chunks skipped")
for w in draft.warnings:
    print(w.severity.value, "|", w.message)
```

```
0 questions drafted, 4 chunks skipped
invalid | none of the 4 chunks has 25 words, so there is nothing to write questions about. Try a larger chunk size
```

`generate`'s other keyword arguments are `sample` (default `50` — how many chunks to draw, spread
across documents rather than uniformly; `None` means all of them), `seed` (default `0`, so the
sample is reproducible), `min_chunk_words` (above), and `evalset_id` (default `"generated"`).

## Eval set quality

Before trusting a leaderboard, know what the eval set underneath it can actually detect.
`contextgrid evalset questions.jsonl` (or `contextgrid.evalset.assess` in Python) reports:

```
$ contextgrid evalset questions.jsonl
policy-questions v1 (manual)
3 questions (3 answerable), 0% reviewed, detects differences of 1.00 and above
types: {'unlabelled': 3}
  - 3 answerable questions can only detect differences of about 1.00 or larger. Anything smaller than that on a leaderboard built from this set is noise
  - only 0% of this set is marked as checked by a human. Ground truth nobody has read is the weakest link in any retrieval comparison. If you wrote these questions yourself, say so with `"meta": {"reviewed": true}` on each one; otherwise the review queue is the cheapest place to fix it
```

**If you wrote the questions yourself, say so.** "Reviewed" is `meta.reviewed` on each
question, and until this was pointed out only the review queue ever set it — so a hand-written
eval set reported 0% forever and was told off on every run for work that had already been done.
It reads straight from the file:

```json
{"id": "q1", "question": "How much notice to terminate?", "anchors": [...], "meta": {"reviewed": true}}
```

The number that matters most: **detects differences of X and above**. A 3-question set can
only detect a near-total reversal (here, 1.00 — meaning nothing smaller than a complete flip is
distinguishable from noise). Below 30 answerable questions, treat every leaderboard gap as
noise unless it's large; 100+ starts to detect differences around 0.1. This is what the
"not distinguishable on this eval set" line in a report's summary is checking — see
[getting-started.md](getting-started.md#reading-the-leaderboard).

## See also

- [getting-started.md](getting-started.md) — pointing a config at an eval set and reading the results
- [configuration.md](configuration.md) — the `evalset:` key and `run.resolution_policy` / `run.resolution_threshold`, which control how anchors and spans resolve onto chunks
- [cli.md](cli.md#evalset) — `contextgrid evalset` and `contextgrid validate`

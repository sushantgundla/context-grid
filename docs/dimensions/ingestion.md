# Ingestion

Set it with `grid.ingestion` in the config, or pass a spec string to
`contextgrid.ingest.get_ingester`. See the [axis model](README.md) for how spec strings work in
general, and [chunkers](chunkers.md) for the axis this one sits on top of.

## The one idea this axis rests on

> **A chunker produces units where the thing indexed and the thing returned are the same.
> An ingestion strategy deliberately breaks that identity.**

Chunk size is a compromise nobody is happy with. Small chunks embed precisely — a 128-token
passage about one thing has a vector that means one thing — and they arrive at the generator
stripped of the context that made them make sense. Large chunks keep their context and embed
into mush, because a vector averaging six topics is close to nothing in particular.

Plain chunking just accepts that compromise: whatever unit got embedded is exactly what a hit
hands back. Every strategy on this page is a way of refusing it instead — **index one thing,
return another**:

| strategy | what is indexed | what is returned | model calls |
|---|---|---|---|
| `plain` | the chunk | the same chunk | none |
| `parent-document` | small chunks | the passage they came from | none |
| `sentence-window` | one chunk | that chunk plus its neighbours | none |
| `hierarchical` | leaf chunks | the leaf, or the parent once enough siblings hit | none |
| `contextual` | the chunk + an LLM-written note on where it sits | the original chunk | one per chunk |
| `hypothetical-questions` | the questions a chunk answers | the chunk | one per chunk |
| `propositions` | atomic facts pulled from a chunk | the chunk they came from | one per chunk |
| `summary` | a summary of the whole document | the whole document | one per document |

The distinction is unmissable in the return type: `IngestionStrategy.ingest` doesn't return a
list of chunks, it returns an `Ingested`
(`contextgrid.ingest.base.Ingested`) — two lists and the map between them.

```python
from dataclasses import dataclass
from contextgrid.core.documents import Chunk


@dataclass(slots=True)
class Ingested:
    indexed: list[Chunk]  # embedded and searched
    retrievable: list[Chunk]  # what a hit turns into
    parent_of: dict[str, str]  # indexed chunk id -> the retrievable chunk it stands for
    ...
```

For plain chunking `indexed` and `retrievable` are the same list and `parent_of` is empty.
Everywhere else, a hit on the indexed side is resolved through `Ingested.resolve(indexed_id)`
to find out what actually comes back.

**Both sides stay spans into the same parse.** Gold evidence resolves against the *retrievable*
units — the things a generator would actually be handed — so a strategy that returns bigger
passages is scored on whether the answer is in what came back, exactly like every other arm. A
strategy that rewrites text for the index (the four paid ones below) says so with
`offsets_exact=False` on the indexed side; the retrievable side always keeps its real, exact
offsets. That is what stops a strategy being credited for a model's paraphrase rather than for
finding the document.

## `expansion`: the cost this axis doesn't put on a recall chart

```python
@property
def expansion(self) -> float:
    """Indexed units per retrievable unit."""
    return len(self.indexed) / len(self.retrievable) if self.retrievable else 0.0
```

Above 1 means several vectors point at the same passage — `hypothetical-questions` indexes
`count` questions for one chunk — which multiplies embedding cost and index size without
multiplying what can be returned. It is worth having on the chart beside the recall it bought,
because a strategy that wins on recall by indexing 4× the vectors is not free even when it
costs no model calls at query time.

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.span import Span
>>> from contextgrid.ingest import get_ingester, IngestionContext
>>> chunks = [
...     Chunk(id="policy.md:0-25", span=Span("policy.md", 0, 25), text="Refunds are issued with"),
...     Chunk(id="policy.md:25-50", span=Span("policy.md", 25, 50), text="in 30 days of purchase."),
...     Chunk(id="policy.md:50-75", span=Span("policy.md", 50, 75), text="Digital goods are not r"),
...     Chunk(id="policy.md:75-99", span=Span("policy.md", 75, 99), text="efundable once downlded"),
... ]
>>> plain = get_ingester("plain").ingest(chunks, IngestionContext())
>>> len(plain.indexed), len(plain.retrievable), plain.expansion
(4, 4, 1.0)
>>> parent = get_ingester("parent-document:3").ingest(chunks, IngestionContext())
>>> len(parent.indexed), len(parent.retrievable), parent.expansion
(4, 2, 2.0)
```

`parent-document:3` groups every 3 chunks into one parent, so 4 indexed chunks resolve to 2
retrievable passages — an expansion of 2.0, meaning twice as many vectors per thing that can
actually be handed back.

## Free: structure only

These four cost nothing but arithmetic — no model, no tokens, no bill. That makes them the arms
the paid strategies have to beat before anyone should pay for one, and on a great many corpora
they are not beaten. All four work by grouping the chunker's output: the chunker still decides
where the cuts fall, these decide which cut is embedded and which is handed back
(`contextgrid.ingest.structural`).

### `plain` — `contextgrid.ingest.PlainIngestion`

Index the chunk, return the chunk. The baseline every other strategy is judged against — not a
placeholder. The entire premise of this axis is that the small-versus-large compromise is worth
escaping, and that premise is a claim that has to be checked against the arm that simply accepts
the compromise, on the same corpus, with the same questions and the same cost columns.

No parameters. Spec: `plain`.

### `parent-document` — `contextgrid.ingest.ParentDocumentIngestion`

Index small chunks, return the passage they came from. The oldest answer to the compromise and
still the strongest free one: a 128-token chunk embeds precisely because it is about one thing;
the 512-token passage around it is what the generator actually needs to answer from.

| parameter | default | meaning |
|---|---|---|
| `group` | `4` | how many consecutive chunks form one parent |

`group` chunks are gathered into each parent, never crossing a document boundary. The chunker
decides the small size, so `chunker: recursive:128` with `parent-document:4` indexes 128-token
chunks and returns roughly 512 tokens of context. `group` must be at least 2 — a group of one is
plain chunking under a different name, and the constructor raises `IngestionError` if you ask
for it. Spec: `parent-document`, `parent-document:4` (shorthand for `group=4`).

### `sentence-window` — `contextgrid.ingest.SentenceWindowIngestion`

Index one chunk, return it with `window` chunks on either side. The sharpest form of the idea:
the embedded unit is as small as the chunker will make it, and what comes back centers on the
match rather than stopping dead at a fixed boundary. Where `parent-document` always returns the
same fixed passage whatever matched inside it, this one returns different context depending on
*where in the passage* the hit landed — a hit at the end of a passage brings back what follows
it.

| parameter | default | meaning |
|---|---|---|
| `window` | `2` | how many chunks on each side come back with a hit |

Windows overlap by design — two adjacent hits return overlapping context, and the result
assembler deduplicates. `window` must be at least 1. Spec: `sentence-window`,
`sentence-window:2` (shorthand for `window=2`).

### `hierarchical` — `contextgrid.ingest.HierarchicalIngestion`

Index leaves, and return the parent once enough of its children have hit. The one free strategy
that decides at **query time** rather than at index time. `parent-document` always returns the
parent, which wastes context when a single leaf held the whole answer; `hierarchical` returns
the leaf when one leaf matched and the parent when several siblings did, on the reasoning that
several hits in one passage mean the passage is the answer, not any one line of it.

| parameter | default | meaning |
|---|---|---|
| `group` | `4` | how many consecutive chunks form one parent |
| `threshold` | `0.5` | fraction of a parent's children that must hit before it merges |

`group` must be at least 2. `threshold` must be in `(0, 1]` — at 1, every child must hit before
the parent is returned. Both leaves and their parent are `retrievable` here, but the parent is
kept as *presentation*, not scored as its own unit
(`Ingested.presentation` / `Ingested.presented_chunks`): scoring stays on the leaves, so merging
shows a generator more context without changing what retrieval is credited with having found.
Spec: `hierarchical`, `hierarchical:4` (shorthand for `group=4`).

**Why that presentation/scored split exists — a real bug this axis found.** An earlier version
made both the leaves *and* their parents directly retrievable and scored. Gold evidence then
resolved against each of them, so a question with one correct answer acquired two things to
find. Measured on this package's demo corpus: **1.86 relevant units per question against plain
chunking's 1.00** — recall halved for a purely structural reason that had nothing to do with
retrieval quality. `Ingested.scored_ids` is what fixed it: a returned passage counts as the
units it covers, never as an extra unit of its own.

## Paid: one model call at index time, never again

A genuinely different bargain from anything on the [transform](transforms.md) axis. These call
a model **once per chunk (or once per document, for `summary`) while building the index** and
never again — so on a corpus answering a thousand questions a day the cost is amortised to
nothing, and on one answering three it is the dominant expense. That trade is invisible on a
recall chart, which is the reason `uses_model` and `model_calls` exist as their own columns
(`contextgrid.ingest.generated`).

**The written text is indexed and never returned.** All four write text that is not in the
document and index *that* — the retrievable side is always the original chunk, offsets intact,
so gold evidence resolves exactly as it does for every other arm. A strategy that returned the
LLM-written text would be scoring the model's paraphrase against the document, which measures
nothing. That's `_rewritten`'s whole job: build a chunk whose `text` is not a slice of its own
`span`, and mark it `offsets_exact=False` so anything scoring against character offsets knows
the difference.

**A failed model call never loses the chunk.** Half an index is worse than a slow one, so a
provider hiccup two thousand chunks into a build must not throw the first nineteen hundred away.
On failure the chunk falls back to being indexed as itself — still findable, just not enriched —
and a `NON_DETERMINISTIC_STAGE` warning records how many. (`_GeneratedIngestion._ask`)

Pass the model with `IngestionContext(llm=...)`, or set `run.model` in the config — the same key
supplies the transform axis's model-backed strategies and the generation judge.

### `contextual` — `contextgrid.ingest.ContextualIngestion`

Prepend an LLM-written explanation of where the chunk sits, then index that. This is
Anthropic's contextual retrieval, and the problem it solves is specific and common: a chunk
reading "the notice period is thirty days" is a perfect answer that no search for "termination
notice under the services agreement" will ever find, because the words that would connect the
two are in a heading four chunks earlier. Anthropic's published result: 67% fewer retrieval
failures — the strongest published number on this axis.

| parameter | default | meaning |
|---|---|---|
| `model` | `"openai:gpt-4o-mini"` | used when `context.llm` isn't supplied |
| `max_document_chars` | `12_000` | how much of the document is put in the prompt |

One call per chunk. The context is indexed and thrown away; the chunk is what comes back. Spec:
`contextual`, `contextual:model=openai:gpt-4o-mini` (shorthand for `model=`).

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.span import Span
>>> from contextgrid.ingest import get_ingester, IngestionContext
>>> class ScriptedLLM:
...     def __init__(self, *replies): self.replies = list(replies)
...     def complete(self, prompt, max_tokens=256):
...         return self.replies.pop(0) if self.replies else ""
>>> chunks = [Chunk(id="policy.md:0-50", span=Span("policy.md", 0, 50), text="Refunds are issued within 30 days of purchase.")]
>>> ctx = IngestionContext(llm=ScriptedLLM("This chunk is from the refund policy, about the 30-day window."))
>>> result = get_ingester("contextual").ingest(chunks, ctx)
>>> result.model_calls
1
>>> result.indexed[0].text
'This chunk is from the refund policy, about the 30-day window.\n\nRefunds are issued within 30 days of purchase.'
>>> result.indexed[0].offsets_exact
False
```

### `hypothetical-questions` — `contextgrid.ingest.HypotheticalQuestionsIngestion`

Index the questions a chunk answers, and return the chunk. A question embeds closer to a
question than a statement does, which is the asymmetry every other arm on this axis fights.
Instead of rewriting the *query* to look like a document — what HyDE does at query time, per
query, forever — this rewrites the *document* to look like a query, once.

| parameter | default | meaning |
|---|---|---|
| `count` | `3` | how many questions to generate per chunk (must be ≥ 1) |
| `model` | `"openai:gpt-4o-mini"` | used when `context.llm` isn't supplied |

Several vectors per chunk, so the index grows by `count` and so does the embedding bill — this
is the strategy where `expansion` matters most. Spec: `hypothetical-questions`,
`hypothetical-questions:5` (shorthand for `count=5`).

```python
>>> ctx = IngestionContext(llm=ScriptedLLM('["How long is the refund window?", "Are digital goods refundable?"]'))
>>> result = get_ingester("hypothetical-questions:count=2").ingest(chunks, ctx)
>>> len(result.indexed), len(result.retrievable), result.expansion
(2, 1, 2.0)
```

### `propositions` — `contextgrid.ingest.PropositionsIngestion`

Index atomic facts, and return the chunk they came from. A chunk covering six topics has a
vector meaning roughly none of them; splitting it into standalone facts — pronouns resolved,
each one true on its own — gives six vectors that each mean one thing, all pointing back at the
passage that supports them.

| parameter | default | meaning |
|---|---|---|
| `count` | `6` | cap on how many propositions per chunk (must be ≥ 1) |
| `model` | `"openai:gpt-4o-mini"` | used when `context.llm` isn't supplied |

The most expensive of the four in index size, and the one that most changes what "a chunk"
means for retrieval. Spec: `propositions`, `propositions:4` (shorthand for `count=4`).

### `summary` — `contextgrid.ingest.SummaryIngestion`

Index a summary of the whole document, and return the whole document. The coarsest strategy on
the axis and the cheapest of the paid ones — one call per **document** rather than per chunk. It
answers a different question from the rest: not "which passage answers this?" but "which
document is this about?", which is what a corpus of many short documents actually needs.

No parameters beyond `model` (`"openai:gpt-4o-mini"`) and `max_document_chars` (`12_000`),
inherited from the shared base. What comes back is the entire document, so it costs context
window rather than model calls — there is no `count` to tune. A failed summary falls back to
indexing the document itself: worse at matching, still findable. Spec: `summary`.

## The measured numbers, and the bug behind them

Measured on this package's demo corpus at 96-token chunks:

| ingestion | recall |
|---|---|
| `parent-document` | **0.863** |
| `sentence-window` | 0.825 |
| `plain` | 0.616 |
| `hierarchical` | 0.575 |

`parent-document` beats `plain` by **+0.247** for zero model calls — the clearest result on this
axis, and the reason it is worth reaching for before any paid strategy. `hierarchical` lands
*below* plain, which looks like a defect and isn't one: merging spends some of its result slots
on wider passages instead of more distinct hits, a real property of the strategy rather than a
bug in it. (See the presentation/scored split above for the bug that *was* real, and the recall
numbers it was hiding before it was fixed.)

## Not built

RAPTOR and GraphRAG — both construct a whole tree or graph structure over the corpus rather than
transform its chunks — are not on this axis. Building either one properly is a larger piece of
work than the other eight strategies combined, and reimplementing a partial version would
misrepresent what each is capable of.

## What ends up in the config

```yaml
grid:
  ingestion: [null, "parent-document:4", "sentence-window:2", contextual]

run:
  model: openai:gpt-4o-mini   # supplies the LLM to contextual, hypothetical-questions, propositions, summary
```

`null` and `plain` mean the same thing (`contextgrid.grid.matrix.canonicalise` rewrites
`ingestion: plain` to `None`), so writing both in one sweep wastes a slot rather than adding a
second baseline. See [chunkers](chunkers.md) for the axis this one groups, and the
[axis model](README.md) for how spec strings and sweep modes work in general.

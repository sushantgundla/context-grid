# Spans and anchors

Everything in `context-grid`'s scoring rests on one idea: a piece of text always knows
exactly which characters of which document it came from. This page covers the value
objects that make that true, and the two forms ground truth takes because of it.

## `Span`: a half-open character range

A [`Span`](../../src/contextgrid/core/span.py) (`contextgrid.core.span.Span`) is
`(doc_id, start, end)` — the characters `[start, end)` of one document. Half-open on
purpose: `text[start:end]` is the literal Python slice, adjacent spans share a boundary
without overlapping, and lengths just subtract.

Nothing here is identified by a chunk ID. Chunk IDs change the moment you change the
chunker, which is exactly the thing this tool exists to compare — see
[chunkers](../dimensions/chunkers.md). Spans don't move when the chunker does.

```python
>>> from contextgrid.core.span import Span
>>> a = Span("doc1", 100, 120)
>>> b = Span("doc1", 110, 130)
>>> a.length
20
>>> a.overlaps(b)
True
>>> a.intersection(b)
Span('doc1', 110, 120)
```

Two spans in different documents never overlap, and an `intersection` of non-overlapping
spans is `None`.

### `coverage_of` vs `iou`: why coverage is the default

`Span` has two similarity measures, and picking the wrong one quietly biases every
chunk-size comparison you run.

- **`iou`** (intersection over union) is symmetric. It punishes size differences in both
  directions.
- **`coverage_of`** is asymmetric. `chunk.coverage_of(gold)` asks "how much of the
  evidence does this chunk hold?" — nothing about the chunk's own size counts against it.

Take a 170-character gold span, held in full by a 2000-character chunk and, separately,
by a 250-character chunk:

```python
>>> from contextgrid.core.span import Span
>>> gold = Span("doc1", 1000, 1170)          # 170 characters of evidence
>>> big_chunk = Span("doc1", 500, 2500)      # 2000 characters, contains all of gold
>>> small_chunk = Span("doc1", 950, 1200)    # 250 characters, contains all of gold
>>> round(big_chunk.coverage_of(gold), 3)
1.0
>>> round(big_chunk.iou(gold), 3)
0.085
>>> round(small_chunk.coverage_of(gold), 3)
1.0
>>> round(small_chunk.iou(gold), 3)
0.68
```

Both chunks ground the answer perfectly — the evidence is entirely present in either one.
`coverage_of` says so for both. `iou` would call the big chunk a miss at any sensible
threshold (0.085 against a 0.5 threshold) purely because it is large, and chunk size is
one of the axes this tool exists to test. Scoring evidence retrieval with IoU builds a
bias into the measuring instrument before a single question is asked.

`iou` isn't wrong, it answers a different question — "how much of this chunk is wasted?"
— which is what [`character_precision`](metrics.md) measures instead, directly, on
retrieved text rather than on the resolution step. See
[`SpanResolver`](#spanresolver-and-its-three-policies) below for where coverage is used
by default.

### Interval algebra over sets of spans

Three module-level functions in `contextgrid.core.span` handle sets of spans, which is
what "how much of the gold did the retrieved set cover, in total?" needs:

| Function | What it does |
|---|---|
| `merge_spans(spans)` | Collapses overlapping and touching spans into a minimal disjoint set, per document, in reading order. |
| `total_length(spans)` | Characters covered, counting shared characters once (merges first). |
| `covered_length(target, others)` | Characters of `target` that appear anywhere in `others` — the core of union recall. |
| `coverage_fraction(target, others)` | `covered_length` as a fraction of `target`. |

`covered_length` is what makes split evidence count correctly: a gold span held half by
one chunk and half by another is fully covered when both come back, even though neither
chunk alone would clear a per-chunk threshold. See [`is_split`](#split-gold) in
`SpanResolver` below.

## `GoldAnchor` vs `GoldSpan`

Ground truth exists in two forms in `contextgrid.core.evalset`, and which one an eval
item carries decides whether it survives a change of parser.

| | `GoldSpan` | `GoldAnchor` |
|---|---|---|
| What it says | "characters 840–1010 of this document" | "the evidence is this quoted sentence" |
| Meaningful against | one specific parse | any parse, once resolved |
| Fields | `span: Span`, `grade: int` | `source_id`, `quote`, `grade`, `page_hint`, `occurrence` |
| Survives a new parser | No | Yes |

Chunkers all cut up the **same** parsed text, so span-level gold compares chunkers fairly
with no anchor needed — the text never changes underneath it. Parsers produce
**different** text from the same source file, so span-level gold cannot survive a change
of parser. Anchors are the layer that makes the parser axis possible at all — see
[parsers](../dimensions/parsers.md) — and they resolve down to spans once a parse exists.

Grades follow the usual IR convention: 2 fully answers, 1 partially relevant, 0
irrelevant. Graded relevance is what makes nDCG mean anything (see [metrics](metrics.md));
binary gold turns it into a noisier hit rate.

### `is_answerable` vs `has_evidence`

An `EvalItem` carries both `gold: tuple[GoldSpan, ...]` (resolved) and
`anchors: tuple[GoldAnchor, ...]` (portable), and it exposes two properties that look
similar and mean different things:

- **`is_answerable`** — `bool(self.gold)`. True only once evidence has been **resolved**
  to character spans in a particular parse.
- **`has_evidence`** — `bool(self.gold or self.anchors)`. True when the item points at
  evidence in *either* form, resolved or not.

A freshly authored item carries anchors and no spans yet. It **has evidence** and is
**not answerable** — nothing has located that evidence in a parse yet. Conflating the two
properties was a real bug here three times: code that checked `is_answerable` to decide
"does this question have ground truth" read every freshly drafted eval set as entirely
unanswerable, before anchor resolution ever ran.

```python
>>> from contextgrid.core.evalset import EvalItem, GoldAnchor
>>> item = EvalItem(
...     id="q1",
...     question="How much notice must a tenant give before leaving?",
...     anchors=(GoldAnchor(source_id="doc1", quote="the tenant must give 30 days written notice"),),
... )
>>> item.is_answerable   # no gold spans yet -- nothing has been resolved
False
>>> item.has_evidence    # but there is evidence, in portable form
True
>>> item.is_portable      # bool(self.anchors)
True
```

`EvalSet` mirrors the same distinction: `.answerable` (resolved) vs `.with_evidence`
(either form) vs `.is_portable` (every item with evidence can be re-resolved). Use
`.with_evidence` to sanity-check a freshly generated eval set before it has been resolved
against any parse; use `.answerable` once it has, to know how many questions ranking
metrics will actually score.

## `SpanResolver` and its three policies

[`SpanResolver`](../../src/contextgrid/score/resolve.py)
(`contextgrid.score.resolve.SpanResolver`) turns span-level ground truth into chunk-level
relevance judgements — deciding, for one gold span and one chunk set, which chunks count
as relevant. This is the step that makes chunker comparisons valid: ground truth is
character offsets, not a chunk ID, so it is re-resolved against whatever chunks each
configuration happened to produce.

```python
from enum import Enum


class ResolutionPolicy(str, Enum):
    COVERAGE = "coverage"  # chunk holds >= threshold of the gold span's characters
    IOU = "iou"  # intersection over union of chunk and gold >= threshold
    CONTAINMENT = "containment"  # chunk holds the gold span entirely; threshold ignored
```

The default is `COVERAGE` at `threshold=0.5` — a chunk counts once it holds at least half
the gold span's characters. Same underlying arithmetic as `Span.coverage_of` /
`Span.iou` above:

```python
>>> from contextgrid.core.span import Span
>>> from contextgrid.score.resolve import ResolutionPolicy, SpanResolver
>>> gold_span = Span("doc1", 1000, 1170)     # 170 characters
>>> chunk_span = Span("doc1", 500, 2500)     # 2000 characters, contains all of gold
>>> for policy in ResolutionPolicy:
...     r = SpanResolver(policy=policy, threshold=0.5)
...     print(policy.value, round(r.score(chunk_span, gold_span), 3), r.is_relevant(chunk_span, gold_span))
coverage 1.0 True
iou 0.085 False
containment 1.0 True
```

`threshold` must be in `(0, 1]` — a threshold of 0 would mark every merely-touching chunk
relevant, and `SpanResolver.__post_init__` raises `ResolutionError` rather than allow it.

### Resolving one question

`resolver.resolve_item(item, chunks)` returns a `Resolution` holding one `GoldResolution`
per gold span:

- `chunk_ids` — chunks that individually cleared the threshold.
- `best_score` — the highest score any single chunk reached.
- `union_coverage` — `coverage_fraction(gold, all_chunk_spans)`, ignoring the threshold.
- `is_reachable` — `bool(chunk_ids)`.

### Split gold {#split-gold}

`is_split` is true when **no single chunk** clears the threshold but the chunk set
**together** covers the evidence. This is a real, under-reported situation: a gold
sentence straddles a chunk boundary, every individual chunk scores below the threshold,
and a per-chunk-only scorer calls it a miss — even though retrieving both chunks would
ground the answer perfectly.

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.evalset import EvalItem, GoldSpan
>>> from contextgrid.core.span import Span
>>> from contextgrid.score.resolve import SpanResolver
>>> gold = GoldSpan(span=Span("doc1", 100, 200), grade=2)   # 100 characters
>>> item = EvalItem(id="q1", question="What is the penalty clause?", gold=(gold,))
>>> chunk_a = Chunk(id="c1", span=Span("doc1", 60, 140), text="x" * 80)   # covers 40 chars
>>> chunk_b = Chunk(id="c2", span=Span("doc1", 160, 240), text="x" * 80)  # covers 40 chars
>>> resolution = SpanResolver().resolve_item(item, [chunk_a, chunk_b])
>>> g = resolution.per_gold[0]
>>> g.is_reachable, g.is_split
(False, True)
>>> round(g.best_score, 2), round(g.union_coverage, 2)
(0.4, 0.8)
>>> [str(w) for w in resolution.warnings]
["CAUTION [resolve] (q1): gold span 100-200 in 'doc1' is split across chunks; no single chunk reaches the coverage threshold of 0.5 (best 0.40), but the chunk set covers 80% of it"]
```

`resolver.resolve(evalset, chunks)` runs `resolve_item` over every item and returns
`(dict[item_id, Resolution], WarningLog)`. `resolver.qrels(evalset, chunks)` turns that
into the `{query_id: {chunk_id: grade}}` shape [metrics](metrics.md) and `ranx` both
consume — questions with no resolvable gold are **left out**, not included as an empty
judgement set, because an empty set would be scored as a legitimate zero for a reason
that has nothing to do with the retriever.

Items with no gold spans at all (`item.is_answerable is False`) are skipped and logged at
`Severity.INFO` — they can't contribute judgements, but they're still useful eval items
for testing whether a system correctly declines to answer.

## `AnchorResolver`: exact → normalised → bounded

[`AnchorResolver`](../../src/contextgrid/score/anchor.py)
(`contextgrid.score.anchor.AnchorResolver`) finds a `GoldAnchor`'s quoted text inside one
parse, and turns it into a `GoldSpan`. Run it once per parser — the same eval set,
re-resolved against each parser's output, is what makes the parser axis a fair comparison
instead of a re-annotation exercise.

Three strategies are tried in order; the first that succeeds is recorded on the
`AnchorMatch`:

| Strategy | When it matches | Typical cause |
|---|---|---|
| `EXACT` | the quote appears verbatim | born-digital text |
| `NORMALISED` | matches once runs of whitespace are collapsed on both sides | PDF extraction inserting line breaks mid-sentence |
| `BOUNDED` (off by default) | the quote's first and last `boundary_words` words both appear, close enough together (within `bounded_slack`) | a parser corrupted something in the *middle* — a table cell, a ligature, a footnote marker |

`NORMALISED` only relaxes the **comparison** — the span it returns still points at real
characters in the original, un-collapsed text:

```python
>>> from contextgrid.core.documents import Document, ParsedDocument
>>> from contextgrid.core.evalset import GoldAnchor
>>> from contextgrid.score.anchor import AnchorResolver
>>> anchor = GoldAnchor(source_id="doc1", quote="the tenant must give 30 days written notice")
>>> clean_text = "Section 4. Termination. the tenant must give 30 days written notice before leaving."
>>> parsed_a = ParsedDocument(document=Document(id="doc1", text=clean_text), parser="pdfminer")
>>> messy_text = "Section 4. Termination.\nthe tenant must  give 30\ndays written  notice before leaving."
>>> parsed_b = ParsedDocument(document=Document(id="doc1", text=messy_text), parser="pypdf")
>>> resolver = AnchorResolver()   # allow_normalised=True, allow_bounded=False by default
>>> resolver.locate(anchor, parsed_a).strategy
<MatchStrategy.EXACT: 'exact'>
>>> match_b = resolver.locate(anchor, parsed_b)
>>> match_b.strategy
<MatchStrategy.NORMALISED: 'normalised'>
>>> messy_text[match_b.span.start:match_b.span.end]
'the tenant must  give 30\ndays written  notice'
```

`BOUNDED` is off by default, and that default is deliberate: **a wrong span is worse than
a missing one.** A missing anchor is visible in the warnings; a wrongly bounded one
silently scores the wrong text as correct.

```python
>>> anchor = GoldAnchor(source_id="doc1", quote="the tenant must give 30 days written notice")
>>> # OCR garbled the middle two words ("give 30"); the opening/closing three words survive.
>>> ocr_text = "Section 4. the tenant must g1v3 3O days written notice before leaving."
>>> parsed_c = ParsedDocument(document=Document(id="doc1", text=ocr_text), parser="tesseract")
>>> AnchorResolver().locate(anchor, parsed_c).strategy               # bounded off (default)
<MatchStrategy.NOT_FOUND: 'not_found'>
>>> match = AnchorResolver(allow_bounded=True).locate(anchor, parsed_c)
>>> match.strategy
<MatchStrategy.BOUNDED: 'bounded'>
>>> ocr_text[match.span.start:match.span.end]
'the tenant must g1v3 3O days written notice'
```

### The failure case is the measurement

When a parser mangles evidence badly enough that none of the three strategies find it,
`resolve()` doesn't treat that as a bug in the eval set — it's the measurement. A parser
that loses the evidence cannot retrieve it, whatever the rest of the pipeline does:

```python
>>> from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
>>> item = EvalItem(
...     id="q1",
...     question="How much notice must a tenant give before leaving?",
...     anchors=(GoldAnchor(source_id="doc1", quote="the tenant must give 30 days written notice"),),
... )
>>> evalset = EvalSet(id="lease-questions", items=(item,))
>>> broken_text = "Section 4. [TABLE DATA CORRUPTED]"
>>> parsed_d = ParsedDocument(document=Document(id="doc1", text=broken_text), parser="broken-ocr")
>>> resolved_evalset, log = AnchorResolver().resolve(evalset, {"doc1": parsed_d})
>>> resolved_item = resolved_evalset.get("q1")
>>> resolved_item.is_answerable, resolved_item.has_evidence
(False, True)
>>> [str(w) for w in log]
["CAUTION [anchor] (q1): the evidence for 'q1' does not appear in 'broken-ocr''s reading of 'doc1': 'the tenant must give 30 days written notice'", "CAUTION [anchor] (broken-ocr): parser 'broken-ocr' lost 1 of 1 pieces of evidence entirely. Those questions cannot be answered under this parse, whatever the retriever does -- which is a fact about the parser, not the eval set"]
```

That's `is_answerable` vs `has_evidence` again, one layer up: the item still has evidence
(the anchor is still there, quote intact) but is not answerable under this parse.

### Ambiguous quotes: `occurrence` and `page_hint`

If a quote appears more than once in a document, `AnchorResolver` picks
`spans[anchor.occurrence]` (0-indexed, in reading order) and logs `ANCHOR_AMBIGUOUS` at
`CAUTION` when `occurrence` was left at its default of `0`. `page_hint` narrows the
candidate list first, when the parser reports pages — useful for catching boilerplate
that repeats on every page.

See [diagnostics](diagnostics.md) for the full list of warning codes this resolver can
raise, and their severities.

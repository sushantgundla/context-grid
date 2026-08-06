# Chunkers

Set it with `grid.chunker` in the config, or pass a spec string to
`contextgrid.chunk.get_chunker`. See the [axis model](README.md) for how spec strings and
extras work in general, [parsers](parsers.md) for the text a chunker cuts up, and
[ingestion](ingestion.md) for what happens to a chunker's output afterward.

## Why chunkers are comparable at all

Chunkers all cut up the same text, which is what makes comparing twelve of them fair without
any re-annotation of ground truth: gold evidence is stored as character spans and resolved
against whichever chunks each strategy happened to produce. A chunker's whole job reduces to
deciding character ranges; `contextgrid.chunk.base.ChunkBuilder` turns those ranges into
`Chunk` objects, which is where the shared invariants — exact offsets, per-tokenizer sizes,
heading and page provenance, stable ids — are guaranteed once rather than re-implemented
correctly in nine places and incorrectly in a tenth.

**Chunk sizes are always in tokens, and always in *this package's* tokenizer**, not characters
and not whichever tokenizer a library defaults to. `size=512` under a byte-pair encoder and
`size=512` under whitespace splitting describe different amounts of text, and an axis that
sweeps both under one name is not measuring what it claims to. Every chunker below takes a
`tokenizer` parameter (`contextgrid.tokens.get_tokenizer`) for exactly this reason, and the
library adapters bridge their own tokenizer to ours rather than using their default.

## The twelve

Five are this package's own, offset-exact by construction. Seven come from the two libraries
people actually deploy — chonkie and LangChain — because a comparison that shows this package's
own recursive chunker beating chonkie's is interesting, but a comparison that shows either
beating the recursive splitter someone already has running in production is the one that tells
them whether switching is worth it. None of the library strategies is reimplemented; each is
adapted onto this package's `Chunk` instead.

| name | shorthand | needs | what it does |
|---|---|---|---|
| `fixed` | `size` | nothing | Fixed-size token windows with overlap |
| `recursive` | `size` | nothing | Split on the largest separator that fits. The default |
| `sentence` | `window` | nothing | A sliding window of whole sentences |
| `structural` | `max_size` | nothing | One chunk per section, bounded by size |
| `semantic` | `percentile` | nothing | Cut where consecutive sentences change topic |
| `chonkie:token` | `size` | `chunk` | Fixed token windows, chonkie's |
| `chonkie:recursive` | `size` | `chunk` | Chonkie's recursive splitter. The head-to-head against ours |
| `chonkie:sentence` | `size` | `chunk` | Whole sentences, chonkie's |
| `chonkie:code` | `size` | `chunk` | Splits on the syntax tree. Nothing hand-written comes close |
| `langchain:recursive` | `size` | `chunk` | What most deployed systems are actually running |
| `langchain:character` | `size` | `chunk` | One separator only. The naive baseline |
| `langchain:markdown` | `size` | `chunk` | Recursive, Markdown boundaries first |

All seven library chunkers share one extra: `chunk`
(`pip install "context-grid[chunk]"`, installs `chonkie`, `langchain-text-splitters` and
`litellm`), and are registered lazily like the heavier parsers — `import contextgrid` never
pulls in either library, and asking for one without the extra raises `MissingExtraError`.

## The five built in

### `fixed` — `contextgrid.chunk.FixedTokenChunker`

The baseline everyone claims to have tuned. Cuts every `size` tokens with `overlap` tokens
carried backwards from the previous chunk, ignoring sentence, paragraph and table boundaries
entirely. Worth having precisely *because* it is naive: it is the arm every cleverer chunker has
to beat, and on a surprising number of corpora it does not lose by much.

| parameter | default | meaning |
|---|---|---|
| `size` | `512` | tokens per chunk |
| `overlap` | `None` → `size // 8` | tokens carried back from the previous chunk |
| `tokenizer` | `None` (the package default) | which tokenizer measures `size` |

`overlap` defaults to an eighth of `size` — 64 at the default 512 — rather than a fixed 64,
deliberately: a fixed default made `fixed:64` an error, because the inherited overlap of 64
collided with a chunk size of 64 the user *did* name. Refusing a perfectly reasonable chunk size
because of a default nobody asked for is a bad axis value; an overlap you do name explicitly is
still checked (`overlap` must be smaller than `size`, or it never advances through the
document). Spec: `fixed`, `fixed:512` (shorthand for `size=512`).

### `recursive` — `contextgrid.chunk.RecursiveChunker`

The de-facto default in every RAG framework, and the arm most real systems are actually running.
Splits on the largest natural boundary that fits — paragraphs, then lines, then sentence
punctuation, then words — and only falls back to cutting mid-word when nothing else works.
Implemented over character ranges rather than by concatenating strings, so every chunk is a
literal slice of the document and offsets are exact by construction.

| parameter | default | meaning |
|---|---|---|
| `size` | `512` | tokens per chunk |
| `overlap` | `None` → `size // 8` | tokens carried back from the previous chunk |
| `separators` | `("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")` | tried in order, largest first |
| `tokenizer` | `None` | which tokenizer measures `size` |

Each piece too large for the current separator is split again with the next one down; adjacent
pieces are then packed back together greedily up to `size` — without that packing step, a
document of short paragraphs produces a chunk per paragraph and `size` does nothing at all, a
failure mode that looks like a working chunker until you plot recall against chunk size and get
a flat line. Same `overlap` default and reasoning as `fixed`. Spec: `recursive`,
`recursive:512` (shorthand for `size=512`).

```python
>>> from contextgrid.core.documents import SourceFile, MediaType
>>> from contextgrid.parse import get_parser
>>> from contextgrid.chunk import get_chunker, CHUNKERS
>>> CHUNKERS.parse_spec("recursive:64,overlap=8")
('recursive', {'size': 64, 'overlap': 8})
>>> md = b"# Refund Policy\n\n## Digital goods\n\nRefunds are issued within 30 days of purchase. Digital goods are not refundable once downloaded.\n\n## Shipping\n\nStandard shipping takes 5 to 7 business days. Express shipping arrives the next business day."
>>> parsed = get_parser("markdown").parse(SourceFile(id="policy.md", media_type=MediaType.MARKDOWN, raw=md))
>>> chunks = get_chunker("recursive:16").chunk(parsed)
>>> len(chunks)
4
>>> [c.offsets_exact for c in chunks]
[True, True, True, True]
```

### `sentence` — `contextgrid.chunk.SentenceWindowChunker`

Chunks are a sliding window of whole sentences: cheap, boundary-respecting, and it beats more
sophisticated strategies often enough to be worth keeping as a serious arm rather than a
baseline.

| parameter | default | meaning |
|---|---|---|
| `window` | `3` | sentences per chunk |
| `stride` | `1` | sentences to advance between chunks |
| `tokenizer` | `None` | which tokenizer counts tokens on the resulting chunk |

`stride < window` produces overlapping chunks, which is usually what you want: a fact stated
across a sentence boundary survives in at least one window. Sentence detection is a regex, not a
linguistic model (`contextgrid.chunk.sentence.sentence_ranges`) — it handles common
abbreviations and decimals ("Fig. 3 shows" holds; "e.g. the party of the first part" sometimes
does not) and the failure is visible in the chunk boundaries rather than hidden, which is the
right kind of wrong. Spec: `sentence`, `sentence:3` (shorthand for `window=3`).

### `structural` — `contextgrid.chunk.StructuralChunker`

Cuts on the document's own structure: a chunk is a section, from one heading to the next. This
is the arm that usually wins on documentation and contracts, and it is the one that most sharply
depends on the parser having found the headings at all — a corpus parsed with `text` instead of
`markdown` gives `structural` nothing to work with.

| parameter | default | meaning |
|---|---|---|
| `max_size` | `512` | tokens; sections larger than this are split further |
| `min_size` | `64` | tokens; sections smaller than this are merged with the previous one |
| `keep_heading_path` | `False` | prepend the chain of headings above a chunk to its text |
| `split_tables` | `False` | allow a table to be cut across chunks |
| `tokenizer` | `None` | which tokenizer measures size |

Sections too large for `max_size` are split with `recursive` internally; a document with no
headings at all falls back to plain `recursive` chunking rather than returning nothing — a
parser that found no structure should score badly on this axis, not silently produce an empty
index. Tables are kept whole across a size-driven split unless `split_tables=True`: cutting a
table in half is one of the most damaging things a chunker can do and one of the hardest to spot
from a leaderboard alone. `keep_heading_path` reliably helps retrieval — a paragraph under
"Termination > Notice period" carries those words into its own embedding — but it also means the
chunk text is no longer a literal slice of the document, so chunks produced that way declare
`offsets_exact=False`. Spec: `structural`, `structural:512` (shorthand for `max_size=512`; note
`min_size` defaults to 64, so a `max_size` below 64 needs `min_size=` set explicitly too, or the
constructor raises).

```python
>>> chunks = get_chunker("structural:64,min_size=0,keep_heading_path=true").chunk(parsed)
>>> [c.meta["heading_prefix"] for c in chunks]
['# Refund Policy', '# Refund Policy > ## Digital goods', '# Refund Policy > ## Shipping']
>>> [c.offsets_exact for c in chunks]
[False, False, False]
```

### `semantic` — `contextgrid.chunk.SemanticChunker`

The most-hyped strategy in the field and the one with the least published measurement behind
it. Embeds each sentence and cuts where consecutive sentences stop being about the same thing,
so a boundary lands at a topic change rather than at an arbitrary token count. Not free — it
embeds the corpus once to decide where to cut and again to index the result — and on corpora
with real structure, a heading-aware chunker often finds the same boundaries for nothing.

| parameter | default | meaning |
|---|---|---|
| `embedder` | `"tfidf"` | embedder used to compare consecutive sentences |
| `percentile` | `90.0` | how aggressive the cut-off is (see below) |
| `buffer_size` | `1` | neighbouring sentences grouped before comparing, to smooth noise |
| `max_size` | `1024` | tokens; a backstop against a chunk with no natural break |
| `min_sentences` | `1` | documents at or below this become one chunk, no comparison run |
| `tokenizer` | `None` | which tokenizer measures `max_size` |

`percentile` is a percentile of **this document's own** similarity drops, not an absolute
similarity threshold — 0.7 is a big drop for one embedding model and noise for another, so a
fixed value would silently mean something different on every arm of a sweep. At 90, only the
largest tenth of this document's drops become boundaries (few, large chunks); at 50, half of
them do. `buffer_size` groups neighbouring sentences before comparing so a single one-clause
sentence between two paragraphs reads as a blip, not a topic change. `max_size` exists because
semantic similarity can run for pages without a real break, and a chunk nothing will fit in the
context window is worse than one cut slightly early. `similarity_profile` and `profile_summary`
(`contextgrid.chunk.semantic`) let you check *why* a semantic chunker cut where it did — a flat
profile (spread under 0.05) means it had nothing to work with and its score should be read that
way. Spec: `semantic`, `semantic:percentile=50`.

```python
>>> chunks = get_chunker("semantic:percentile=50").chunk(parsed)
>>> len(chunks)
3
>>> chunks[1].text
'Digital goods are not refundable once downloaded.'
```

## The library chunkers

### chonkie — `contextgrid.chunk.chonkie`

chonkie exists to do one thing and does more of it than this package ever will: nine strategies
including late chunking, neural boundary detection and AST-aware code splitting, over a Rust
core. Reimplementing that would produce a worse version of it and — worse — would mean this
package compares *its own* chunkers rather than the ones people actually deploy. Every adapted
chunk carries verified offsets: chonkie's `start_index`/`end_index` are checked against the text
on every document at runtime, never assumed, because a chunker whose offsets drift silently
moves every gold span in the corpus and the run still looks fine
(`_ChonkieChunker._checked`).

`_TokenizerBridge` (`contextgrid.chunk.chonkie._bridge`) is what makes `chonkie:recursive:512`
and this package's own `recursive:512` mean the same 512: chonkie counts characters by default,
this package hands its own tokenizer down instead. The bridge's encoding is lossless by
construction — `decode(encode(text)) == text` exactly — because chonkie's token chunker decodes
its own output to work out where a chunk starts, and an encoding that dropped whitespace would
hand back offsets quietly off by a few characters, which would be the single worst failure this
package could have.

| name | extra param | default | maps to |
|---|---|---|---|
| `chonkie:token` | `overlap` | `0` | chonkie's `TokenChunker` |
| `chonkie:recursive` | `min_characters` | `24` | chonkie's `RecursiveChunker` |
| `chonkie:sentence` | `overlap`, `min_sentences` | `0`, `1` | chonkie's `SentenceChunker` |
| `chonkie:code` | `language` | `"auto"` | chonkie's `CodeChunker` |

All four also take `size` (default `512`, the shared shorthand) and `tokenizer`. `chonkie:code`
splits on the syntax tree rather than the text — nothing hand-written in this package comes
close, since only a real parser for the language can reliably avoid cutting a function in half.
Extra: `chunk`.

```python
>>> for spec in ["chonkie:recursive:24", "chonkie:sentence:24", "chonkie:token:24"]:
...     print(spec, len(get_chunker(spec).chunk(parsed)))
chonkie:recursive:24 3
chonkie:sentence:24 2
chonkie:token:24 2
```

### LangChain — `contextgrid.chunk.langchain`

On this axis for one reason: these are what most deployed RAG systems are actually running. A
comparison that shows chonkie beating this package's own `recursive` is interesting; a
comparison that shows either of them beating *LangChain's* recursive splitter tells someone
whether it's worth changing the code they already have in production.

| name | extra param | default | maps to |
|---|---|---|---|
| `langchain:recursive` | — | — | `RecursiveCharacterTextSplitter`, the default nearly every tutorial reaches for |
| `langchain:character` | `separator` | `"\n\n"` | `CharacterTextSplitter`, one separator only — the naive baseline a lot of systems shipped with |
| `langchain:markdown` | — | — | `MarkdownTextSplitter`, recursive with Markdown's own boundaries tried first |

All three take `size` (default `512`) and `overlap` (default `0`, **not** LangChain's own
default — an explicit 0 rather than an eighth of `size` so the shorthand always works standalone:
`langchain:recursive:32` with an inherited overlap would be rejected outright, while
`chonkie:token:32` sails through. Ask for overlap explicitly:
`langchain:recursive:512,overlap=64`) and `tokenizer`.

LangChain's splitters return strings, not offsets, so the adapter asks for
`add_start_index=True` and reads `metadata["start_index"]` back — using `create_documents`
rather than the more obvious `split_text`, since without the offsets these chunks could not be
scored against character-span gold at all. `length_function` is set to this package's own
tokenizer count rather than LangChain's default `len`, for the same tokens-not-characters reason
as everywhere else on this axis.

Two behaviors worth knowing about, both LangChain's rather than this package's own:

- Its splitters **strip whitespace** by default, so chunks don't tile the document. Offsets stay
  exact — `start_index` accounts for the stripping — but a character between two chunks can
  belong to neither, and a gold span landing in that gap resolves to nothing. The run reports it
  rather than hiding it.
- **Overlap means chunks share characters.** Ordinary and intended; the scorer already handles
  overlapping chunk spans.

`start_index` cannot be trusted at face value, and the adapter doesn't
(`_LangChainChunker._locate`): LangChain rebuilds each chunk by rejoining the pieces it split,
looks the rejoined text up in the source, and reports `-1` when the rejoined text differs from
the original by so much as a newline. On this package's own fixtures that happens to roughly one
chunk in eight, always on tables. The chunk content is still a literal slice of the document in
those cases, so the offset is recoverable — the adapter searches forward from the previous
chunk's start and verifies the slice matches before trusting it. Trusting `-1` would silently
drop the chunk; trusting a wrong index would silently move every gold span landing in it.

```python
>>> for spec in ["langchain:recursive:24", "langchain:character:24", "langchain:markdown:24"]:
...     print(spec, len(get_chunker(spec).chunk(parsed)))
langchain:recursive:24 2
langchain:character:24 2
langchain:markdown:24 2
```

Extra: `chunk`, same as chonkie — `pip install "context-grid[chunk]"` covers both libraries plus
`litellm`.

## What ends up in the config

```yaml
grid:
  chunker:
    - recursive:512
    - structural:512
    - chonkie:recursive:512
    - langchain:recursive:512
```

The four most-compared arms in one sweep: this package's own default, the structural arm that
depends on the parser's headings, and the two libraries people already have in production. See
[ingestion](ingestion.md) for what a strategy does with a chunker's output before it reaches the
index, and the [axis model](README.md) for spec strings, sweep modes and how the matrix drops
redundant runs.

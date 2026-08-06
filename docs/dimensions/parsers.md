# Parsers

Set it with `grid.parser` in the config, or pass a spec string to
`contextgrid.parse.get_parser`. See the [axis model](README.md) for how spec strings and extras
work in general, and [chunkers](chunkers.md) for the axis that consumes a parser's output.

## Why the parser is an axis at all

The parser defines the text every character offset downstream refers to. For born-digital text
(plain text, Markdown) that text already exists and the parser mostly points into it. For a PDF
there is no text until the parser makes some — and that is exactly where two parsers stop
agreeing. Two parsers reading the same PDF produce different text, different structure, and
different retrievable evidence, and nothing else in the RAG field measures that end to end. That
disagreement is the whole reason the parser sits on the grid rather than in one-time setup.

**Offsets stay exact regardless.** Every block is a literal slice of the text *that parser*
produced, and the document text is assembled from the blocks
(`contextgrid.parse.builder.TextAssembler`) — so `document.text[span] == block.text` by
construction, not by careful bookkeeping. What changes between parsers is *what the text is*,
never whether the offsets into it can be trusted.

## The eight

| name | needs | extra | what it does |
|---|---|---|---|
| `text` | nothing | — | Plain text, split into paragraphs on blank lines |
| `markdown` | nothing | — | Markdown with headings, code, lists, quotes and tables |
| `pymupdf` | `pymupdf` | `parse` | Fast PDF text extraction. The speed baseline |
| `pdfplumber` | `pdfplumber` | `parse` | Table-aware PDF extraction |
| `pymupdf4llm` | `pymupdf4llm` | `parse` | Markdown from the PyMuPDF engine. Tests output format |
| `docling` | `docling` | `parse-ml` | Layout and table-structure models. PDF, DOCX, PPTX, HTML |
| `marker` | `marker-pdf` | `parse-marker` | Surya layout + OCR, 90+ languages. Most faithful, by far the slowest |
| `agno` | `agno` (+`pypdf` for PDF) | `agent` | Text through agno's readers — what most RAG stacks use without choosing to |

`text` and `markdown` are registered eagerly and need nothing installed — they are the reference
implementations of the `Parser` protocol and what the conformance suite runs against. Everything
else needs a PDF engine or a layout model and is registered **lazily**, by module path, so
`import contextgrid` stays cheap and asking for one without its extra raises
`MissingExtraError` naming the install command rather than a bare `ModuleNotFoundError`.

There is no ninth parser. `unstructured` was registered here briefly with nothing behind it —
asking for it raised `MissingExtraError` naming the `parse-ml` extra, which did not contain it
and would not have helped. It has been removed from `contextgrid.parse.PARSERS`
(`contextgrid.parse.__init__` explains why in its module docstring): advertising a plugin that
cannot run is the same failure as a config key nobody reads, worse than simply not having it.

### `text` — `contextgrid.parse.TextParser`

The simplest thing that can be a parser, and the baseline every other parse on a text corpus is
compared against. Splits on blank lines into paragraph blocks; `offsets_exact` is always `True`
because the decoded file text is kept exactly as it is. Supports `MediaType.TEXT`,
`MediaType.MARKDOWN` and `MediaType.UNKNOWN`. No parameters. Spec: `text`.

### `markdown` — `contextgrid.parse.MarkdownParser`

Markdown with headings, code fences, lists, block quotes and tables identified as their own
block kinds, walked line by line rather than with one whole-text regex — a fenced code block can
contain anything, including lines that look exactly like headings, and only a stateful pass gets
that right. Structure is the point: a chunk that knows it sits under
"Termination > Notice period" retrieves far better than a bare paragraph, and a chunker that can
see where a table starts will not cut it in half. No parameters. Spec: `markdown`.

```python
>>> from contextgrid.core.documents import SourceFile, MediaType
>>> from contextgrid.parse import get_parser
>>> md = b"# Refund Policy\n\n## Digital goods\n\nRefunds are issued within 30 days of purchase.\n"
>>> parsed = get_parser("markdown").parse(SourceFile(id="policy.md", media_type=MediaType.MARKDOWN, raw=md))
>>> [(b.kind.value, b.text[:20]) for b in parsed.blocks]
[('heading', '# Refund Policy'), ('heading', '## Digital goods'), ('paragraph', 'Refunds are issued w')]
```

### `pymupdf` — `contextgrid.parse.pymupdf.PyMuPDFParser`

The speed baseline: very fast, reliable on born-digital PDF text, and it has no idea what a
table is — a table comes out as a run of loose text in whatever order the content stream
happened to store it. That failure is the point: it is the arm every table-aware parser has to
beat, and the reason the parser belongs on the grid instead of in setup.

| parameter | default | meaning |
|---|---|---|
| `detect_headings` | `True` | infer heading levels from font size (see below) |
| `margin_ratio` | `0.0` | drop text in the top/bottom band of each page, as a fraction of page height |

A PDF has no headings, only text that happens to be set larger. `detect_headings` is what makes
structural chunking possible on a PDF at all: sizes are weighted by *characters, not lines*
(`contextgrid.parse.builder.infer_heading_levels`) — weighting by lines lets a bordered table's
dozen small two-word cells outvote the prose and drag the inferred body size down, promoting the
actual body text to a heading. `margin_ratio` exists because repeated page furniture (headers,
footers, page numbers) is one of the quietest ways to poison dense retrieval; it is off by
default because dropping real content at the edges is worse than leaving furniture in. Extra:
`parse` (`pip install "context-grid[parse]"`). Spec: `pymupdf`.

### `pdfplumber` — `contextgrid.parse.pdfplumber.PDFPlumberParser`

Slower than `pymupdf`, and it finds tables — the clearest illustration of why the parser belongs
on the grid at all. On a corpus of prose the two are nearly indistinguishable and `pdfplumber`
is simply slower; on a financial report, one of them returns the number you asked for and the
other returns a soup of digits.

| parameter | default | meaning |
|---|---|---|
| `extract_tables` | `True` | find tables and emit them as their own blocks |
| `table_format` | `"pipe"` | how cells are joined: `"pipe"`, `"tsv"` or `"plain"` |

Elements are ordered by vertical position on the page, so a table between two paragraphs comes
out between them rather than appended at the end — reading order is one of the things parsers
most often get wrong, and getting it wrong quietly destroys retrieval on any document where the
answer depends on what a sentence was next to. `table_format` is a real trade-off, not a
formatting preference: Markdown pipes give an embedder an explicit signal about where one cell
ends and the next begins, which usually helps retrieval, but a row no longer reads as
`Premium 3400 500`, so ground truth quoting a table row verbatim will not match. Use `"plain"`
when the eval set was authored against a reading of the table rather than against this parse.
Extra: `parse`. Spec: `pdfplumber`.

```python
>>> import sys; sys.path.insert(0, "tests")
>>> from pdf_fixtures import contract_pdf
>>> from support import pdf
>>> from contextgrid.parse.pymupdf import PyMuPDFParser
>>> from contextgrid.parse.pdfplumber import PDFPlumberParser
>>> src = pdf("contract", contract_pdf())  # a heading, body text, and a bordered fee table
>>> PyMuPDFParser().parse(src).text[-90:]
's in Schedule A.\n\nService\n\nMonthly fee\n\nSetup fee\n\nStandard\n\n1200\n\n500\n\nPremium\n\n3400\n\n500'
>>> PDFPlumberParser().parse(src).text[-90:]
'\n\n| Service | Monthly fee | Setup fee |\n| Standard | 1200 | 500 |\n| Premium | 3400 | 500 |'
```

Same bytes, same table. `pymupdf` hands back six loose lines with no idea they were ever a grid;
`pdfplumber` hands back a table an embedder can read as one.

### `pymupdf4llm` — `contextgrid.parse.layout.PyMuPDF4LLMParser`

Markdown output from the PyMuPDF engine already installed for `pymupdf`. The cheapest arm on
this axis and a genuinely useful control: it shares an extraction engine with `pymupdf`, so any
difference between the two is *output format alone* — headings marked as headings, tables as
pipe rows — which isolates a question the heavier parsers confound: is Markdown structure worth
anything to your retriever, separately from whether the extraction itself was any good?

| parameter | default | meaning |
|---|---|---|
| `page_chunks` | `True` | ask pymupdf4llm for per-page output, tagged with page markers |
| `table_strategy` | `"lines_strict"` | pymupdf4llm's table-detection strategy |
| `isolate` | `True` | parse each document in its own subprocess (see below) |

**Each document is parsed in its own process, and that is not optional by default.**
pymupdf4llm's output for a document depends on which documents were converted before it in the
same interpreter — state persists in MuPDF's C layer, below Python, so reloading the module and
emptying MuPDF's store both fail to clear it. Measured directly on this package's own fixtures:

```python
>>> import sys; sys.path.insert(0, "tests")
>>> from pdf_fixtures import contract_pdf, prose_pdf
>>> from support import pdf
>>> from contextgrid.parse.layout import PyMuPDF4LLMParser
>>> prose, contract = pdf("prose", prose_pdf()), pdf("contract", contract_pdf())
>>> no_isolation = PyMuPDF4LLMParser(isolate=False)
>>> alone = no_isolation.parse(prose)
>>> _ = no_isolation.parse(contract)
>>> after = no_isolation.parse(prose)
>>> len(alone.text), len(after.text)
(1182, 919)
>>> after.text[:60]
'Page 1 line 1: the notice period is thirty days.\n\nPage 1 lin'
```

The same prose PDF parses to 1182 characters alone and 919 mangled ones after a PDF with a table
has gone through the same interpreter — words missing letters, not missing pages. With
`isolate=True` (the default), the same sequence produces identical text both times:

```python
>>> isolated = PyMuPDF4LLMParser(isolate=True)
>>> a = isolated.parse(prose)
>>> _ = isolated.parse(contract)
>>> b = isolated.parse(prose)
>>> a.text == b.text
True
```

For a tool whose entire foundation is the parse, a corpus that parses differently depending on
file order is disqualifying. A process per document costs about a tenth of a second against the
several tenths the conversion already takes, and buys back determinism — `isolate=False` is an
escape hatch for anyone who has measured their own corpus and wants the speed back, and the
failure it re-enables is silent, which is why it defaults off. Extra: `parse`. Spec:
`pymupdf4llm`.

### `docling` — `contextgrid.parse.layout.DoclingParser`

IBM's docling: layout and table-structure models run over the page. The arm to reach for when
the corpus has tables, and it reads far more than PDF — DOCX, PPTX, XLSX and HTML too — which is
the only parser on this axis with anything to say about a non-PDF corpus.

| parameter | default | meaning |
|---|---|---|
| `table_structure` | `True` | run the table-structure model |
| `ocr` | `False` | run OCR |

`table_structure` and `ocr` are separate switches because they are separate costs. Turning OCR
off on a corpus with a text layer gets most of the speed back for none of the quality lost, and
that trade is worth being able to sweep rather than assume. Extra: `parse-ml`
(`pip install "context-grid[parse-ml]"`; downloads and runs vision models on first use). Spec:
`docling`.

```python
>>> import sys; sys.path.insert(0, "tests")
>>> from pdf_fixtures import contract_pdf
>>> from support import pdf
>>> from contextgrid.parse.layout import DoclingParser
>>> parsed = DoclingParser(ocr=False).parse(pdf("contract", contract_pdf()))
>>> parsed.text
'Master Services Agreement\n\n2. Termination\n\nEither party may terminate this agreement for convenience by giving thirty days written notice. Notice must be delivered to the address in Schedule A.\n\n| Service   |   Monthly fee |   Setup fee |\n|-----------|---------------|-------------|\n| Standard  |          1200 |         500 |\n| Premium   |          3400 |         500 |'
>>> [b.kind.value for b in parsed.blocks]
['heading', 'heading', 'paragraph', 'table']
```

### `marker` — `contextgrid.parse.layout.MarkerParser` — **unverified in this environment**

Surya for layout and OCR, across 90+ languages. Reported as the most faithful of the three
layout parsers to document structure, and by a wide margin the slowest — roughly a hundred times
a plain text extractor is not an unusual report. That is exactly the sort of thing this package
exists to put on one chart next to the recall it bought, rather than leave as folklore.

| parameter | default | meaning |
|---|---|---|
| `languages` | `"en"` | languages to OCR |
| `use_llm` | `False` | use an LLM pass to improve output |

Model weights download on first use — the failure message says so explicitly, because a first
run that appears to hang for several minutes is otherwise very hard to diagnose. Extra:
`parse-marker` (package `marker-pdf`) — **deliberately its own extra, not part of `parse-ml`.**
Spec: `marker`, `marker:fr` (shorthand for `languages="fr"`).

**Install `marker` alone, in its own environment, when the marker arm is the one being
measured.** It cannot be verified alongside `docling` in one environment, and that is a
statement about the packages, not about how much time was spent trying. `marker-pdf` pulls in
`surya-ocr`, which forces `transformers>=5.12.1` and `pillow<11`. `docling` — verified elsewhere
on this page — is pinned by `docling-ibm-models` to `transformers<5.9.0`, and `pdfplumber` needs
`pillow>=12.2.0`. No version of either package satisfies both sides.

The subtlety worth recording: **pip's resolver says yes.** Asking pip to install `docling` and
`marker-pdf` together does not fail — it backtracks through pillow 10.4/12.3 and transformers
4.57/5.8/5.14 and lands on one consistent set, and reports success. But the set it lands on is
the one that breaks `docling` at runtime: `docling.exceptions.ConversionError: ... KeyError:
torch.float64` on the same contract fixture used above. A clean resolve is not the same as the
combination working, and only actually running it finds the difference. I installed
`marker-pdf`, reproduced that failure, reverted the environment
(`transformers==5.8.1`, `pillow==12.3.0`, `click==8.3.3`, `pypdfium2==5.12.1`), re-verified
`docling` and `pdfplumber` both work again, and removed `marker-pdf`.

`pyproject.toml` now reflects this: `marker` lives under its own `parse-marker` extra
(`pip install "context-grid[parse-marker]"`), separate from `parse-ml`
(`docling` alone), with the same reasoning recorded next to the extra so it isn't rediscovered
the hard way twice. This section stays unverified — not "not gotten to yet," but because
verifying it here would require breaking `docling`, which this page already measured.

### `agno` — `contextgrid.parse.layout.AgnoParser`

Text extraction through agno's own readers. On the parser axis rather than the ingestion one,
because turning bytes into text is what a parser does — and what most RAG stacks reach for
without noticing they've made a parser choice at all. Putting it beside `pymupdf`, `pdfplumber`
and `docling` prices that convenience against engines built for the job.

| parameter | default | meaning |
|---|---|---|
| `reader` | `"auto"` | an agno reader name, or `"auto"` to pick by file extension |

`reader="auto"` maps `MediaType` to a file suffix and asks agno's `ReaderFactory` for a matching
reader; asking for a name explicitly that agno doesn't have raises with the list of what it
does have. Its PDF reader needs `pypdf`, and says so by name rather than failing obscurely.
Reads PDF, Markdown, HTML, DOCX and plain text. Extra: `agent` (installs `agno` and `pypdf`).
Spec: `agno`.

```python
>>> from contextgrid.core.documents import SourceFile, MediaType
>>> from contextgrid.parse import get_parser
>>> src = SourceFile(id="doc.md", media_type=MediaType.MARKDOWN, raw=b"# Title\n\nSome text here for the agno reader to pick up.")
>>> get_parser("agno").parse(src).text
'Title\n\nSome text here for the agno reader to pick up.'
```

## What ends up in the config

```yaml
grid:
  parser: [markdown, pymupdf, pdfplumber, docling]
```

A PDF corpus parsed four ways is the single clearest demonstration this package makes: the same
questions, the same chunker, the same everything else, and the only thing that changed is which
tool turned bytes into text. See [chunkers](chunkers.md) for what happens to that text next, and
the [axis model](README.md) for how spec strings and lazy extras work everywhere on the grid.

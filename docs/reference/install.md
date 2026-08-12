# Install matrix

`pip install context-grid` installs the core only: `numpy` and `pyyaml`. Nothing else comes
in until you ask for it by extra — `pip install "context-grid[chunk,index]"` and so on. This
is what each extra pulls in, roughly how heavy it is, and which plugins from
[plugins.md](plugins.md) it unlocks.

`pyproject.toml` declares eleven `[project.optional-dependencies]` keys. Ten are real,
installable extras; `dev` is the eleventh, and it's the CI/contributor extra, not something
an end user reaches for:

```bash
grep -A2 '^\[project.optional-dependencies\]' pyproject.toml   # source of truth
```

## The matrix

| Extra | Packages | Weight (installed, on disk) | Unlocks |
|---|---|---|---|
| `parse` | `pymupdf>=1.24`, `pdfplumber>=0.11`, `pymupdf4llm>=0.0.17` | Medium-heavy — `pymupdf` alone is ~107 MB installed (it bundles the MuPDF C library); `pdfplumber` and `pymupdf4llm` are small on top of it. | Parsers `pymupdf`, `pdfplumber`, `pymupdf4llm` |
| `parse-ml` | `docling>=2.0` | Heavy. `docling` is ~6.4 MB for its own code but pulls model weights and a torch-adjacent stack at runtime — expect a multi-GB environment once its model dependencies resolve. | Parser `docling` |
| `parse-marker` | `marker-pdf>=1.0` | Heavy, and deliberately its own extra rather than folded into `parse-ml` — see below. Runs Surya for layout and OCR. Not installed in this checkout, so no on-disk number here; expect the same order of magnitude as `parse-ml`. | Parser `marker` |
| `embed` | `tiktoken>=0.7` | Light — ~2.7 MB installed. | The `cl100k_base` and `o200k_base` tokenizers (exact token counts, for chunking *and* for pricing — see [caching.md](caching.md) and [cost.md](cost.md)) |
| `chunk` | `chonkie>=1.7`, `langchain-text-splitters>=0.3`, `tree-sitter-language-pack>=0.9` | Light-medium. `chonkie` ~2.1 MB, `langchain-text-splitters` ~300 KB, `tree-sitter-language-pack` ~4.5 MB (the compiled grammars `chonkie:code` splits on). | Chunkers `chonkie:recursive`, `chonkie:sentence`, `chonkie:token`, `chonkie:code`, `langchain:recursive`, `langchain:character`, `langchain:markdown` |
| `llm` | `litellm>=1.60` | Heavy for what it does — ~114 MB installed, most of it the bundled `litellm.model_cost` table (close to 3,000 models — see [cost.md](cost.md)) and its provider SDKs. One package, reused everywhere a hosted model is needed. | Embedder `litellm`, reranker `litellm-rerank`, transforms `hyde`/`multi-query`/`decompose`/`step-back`, retrieval `agentic`, ingestion `contextual`/`summary`, and the `litellm`/`openai`/`anthropic` judge/eval models |
| `index` | `faiss-cpu>=1.8`, `usearch>=2.12` | Medium. `faiss` ~18 MB installed, `usearch` ~1.9 MB. Two implementations of approximate search on purpose — see [plugins.md](plugins.md). | Indexes `faiss` (flat/hnsw/ivf/ivfpq), `usearch` (f32/f16/i8) |
| `pgvector` | `psycopg[binary]>=3.1` | Light — ~2 MB installed. The weight isn't the package, it's the Postgres server you still need running with the `pgvector` extension enabled. | Index `pgvector` |
| `agent` | `agno>=2.8`, `pypdf>=4.0` | Heavy. `agno` ~33 MB, `pypdf` ~3.3 MB. `pypdf` is pinned separately because agno's own PDF reader raises on import without it. | Parser `agno`, and it's what makes `retrieval: agentic` a reasonable choice rather than falling back to this package's own LLM protocol |
| `judge` | `deepeval>=4.0` | Heavy — ~13 MB for the package itself, plus its own sizeable dependency tree for running generation metrics. Ships usage telemetry to PostHog, which `contextgrid` [switches off](../dimensions/generation.md#deepeval-backed-generation-metrics) before importing it. | Faithfulness, answer-relevancy and the rest of the generation metrics, run through whichever model `run.model` already points at |
| `dev` | `pytest`, `pytest-cov`, `hypothesis`, `mypy`, `ruff`, `ranx`, `types-pyyaml`, plus one copy of every package above **except** `docling` and `marker-pdf` — `deepeval` and `tiktoken` are both in it. | N/A — this is the CI/contributor extra, not something an end user installs. | Everything needed to run the test suite and linters against every plugin `pyproject.toml` knows how to test, without installing each extra one at a time. `docling`/`marker-pdf` are left out on purpose (see below) — CI doesn't want a multi-GB vision-model install either. |

`dev`'s exact contents (verify against `pyproject.toml`, they drift):

```bash
sed -n '/^dev = \[/,/^\]/p' pyproject.toml
```

## Why `marker` is its own extra, not part of `parse-ml`

`parse-ml` used to hold both `docling` and `marker-pdf`. It was split because installing both
in one environment doesn't just work: `marker-pdf` pulls `surya-ocr`, which wants
`transformers>=5.12` and `pillow<11`; `docling` wants `transformers<5.9`; `pdfplumber` (from
the `parse` extra) wants `pillow>=12.2`. `pip` resolves all three onto *some* consistent set by
backtracking, and the set it lands on installs cleanly and then breaks `docling` at **runtime**
with `ConversionError: KeyError: torch.float64`. A resolver saying yes isn't the same as the
combination working — see the comment above `parse-marker` in `pyproject.toml`. Install
`marker` alone, in its own environment, when the `marker` arm is specifically what you're
measuring; don't mix it with `docling` in the same install.

Ask for `marker` without it installed and the error names the right extra and the exact
command:

```
MissingExtraError: The marker parser requires the 'parse-marker' extra (needs marker-pdf).
Install it with: pip install "context-grid[parse-marker]"
```

## Sizes, honestly

These are **installed, on-disk sizes for the package's own directory** in this checkout's
`.venv` (`du -sh` on `site-packages/<pkg>`), not download sizes and not the full dependency
tree each package pulls in transitively (httpx, pydantic, tokenizers, etc. for `litellm`;
torch-adjacent stacks for `docling`/`marker-pdf`). Treat the numbers as "which of these is
obviously heavier than the others," not as a promise about your `pip install` time.

Regenerate them:

```bash
for p in pymupdf pdfplumber pymupdf4llm docling marker_pdf chonkie \
         langchain_text_splitters litellm faiss usearch psycopg agno pypdf deepeval tiktoken; do
  dir=$(.venv/bin/python -c "
import importlib.util, os
spec = importlib.util.find_spec('$p')
print(os.path.dirname(spec.origin) if spec and spec.origin else '')
")
  [ -n "$dir" ] && du -sh "$dir" || echo "$p: not installed"
done
```

## Things that need infrastructure, not just a package

Installing the extra is necessary but not sufficient for two plugins:

- **`pgvector`** needs a running Postgres with the `pgvector` extension, reachable at
  whatever DSN the index config gives it. `psycopg` alone gets you nothing without a server.
- **`tei` (embedder) and `tei-rerank` (reranker)** need a running
  [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
  server. Neither needs an extra at all — they talk to it over plain `urllib`, so the core
  install is enough on the `context-grid` side; the weight lives entirely in running the
  server itself.

## The `embed` extra: fixed, not missing anymore

An earlier pass of this page found `cl100k_base` pointing at an `embed` extra that didn't
exist in `pyproject.toml`, and at a package (`tiktoken`) declared nowhere. Both are fixed now:
`embed = ["tiktoken>=0.7"]` is a real extra, and `src/contextgrid/tokenizers_tiktoken.py`
(newly written) implements the tokenizers it unlocks — `cl100k_base` and `o200k_base`, both
exact. See [plugins.md](plugins.md) for what "exact" buys you (real character offsets, not
just a count, verified to reconstruct the original text) and [caching.md](caching.md) for why
the tokenizer choice has to be part of a chunk's cache key.

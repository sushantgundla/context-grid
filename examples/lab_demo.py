"""Generate the precomputed demo for sushantgundla.com/lab.

The page has to reach an interesting result in ten seconds with no keys, no uploads and no
compute, which means every number on it is computed here and committed as JSON.

The corpus is written twice: once as Markdown and once as PDF carrying the same words. That
is what makes the parser axis measurable -- the same eval set, resolved against three
different readings of the same documents, one of which loses the tables.

    python examples/lab_demo.py ../personal-blog-2/app/lab/_data/demo.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lab_corpus import describe, distractor_questions, distractors

import contextgrid as cg
from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config
from contextgrid.report.manifest import build_manifest

# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------

CONTRACT = """\
# Master Services Agreement

## 1. Definitions

"Effective Date" means the date on which both parties have signed this agreement.
"Services" means the professional services described in Schedule B.
"Deliverable" means any report, software or document produced under the Services.

## 2. Term and renewal

This agreement begins on the Effective Date and continues for an initial term of twelve
months. It renews automatically for successive twelve month terms unless either party
gives notice of non-renewal at least sixty days before the end of the then-current term.

## 3. Termination

### 3.1 Termination for convenience

Either party may terminate this agreement for convenience by giving thirty days written
notice to the other party. Notice must be delivered to the address set out in Schedule A.

### 3.2 Termination for cause

A party may terminate this agreement immediately if the other party commits a material
breach and fails to remedy that breach within fifteen days of receiving written notice
of it. Insolvency of either party permits immediate termination without notice.

### 3.3 Effect of termination

On termination the Supplier shall deliver all completed Deliverables and the Customer
shall pay for Services performed up to the termination date. Clauses 6, 7 and 9 survive
termination.

## 4. Fees

| Service tier | Monthly fee | Setup fee | Support response |
|---|---|---|---|
| Standard | $1,200 | $500 | 2 business days |
| Premium | $3,400 | $500 | 4 hours |
| Enterprise | $9,750 | $2,000 | 1 hour |

Fees are payable within thirty days of the invoice date. Late payment attracts interest
at two percent per month on the outstanding balance.

## 5. Limitation of liability

Neither party's aggregate liability under this agreement shall exceed the total fees paid
in the twelve months preceding the claim. Nothing in this clause limits liability for
fraud or for death or personal injury caused by negligence.

## 6. Confidentiality

Each party shall keep the other's Confidential Information secret for a period of five
years from disclosure and shall not use it except to perform this agreement.

## 7. Intellectual property

The Customer owns all Deliverables on payment in full. The Supplier retains ownership of
any pre-existing materials and grants the Customer a perpetual licence to use them as
part of the Deliverables.
"""

API_DOCS = """\
# Widget API reference

The Widget API lets you create, read, update and delete widgets. All endpoints return
JSON and are versioned under /v2.

## Authentication

Send your key in the `X-Api-Key` header on every request. Requests without a valid key
return 401. Keys are scoped to a single workspace and can be rotated from the dashboard
without downtime; the previous key stays valid for one hour after rotation.

## Rate limits

The default limit is 600 requests per minute per key. Exceeding it returns 429 with a
`Retry-After` header giving the number of seconds to wait. Burst traffic up to 1,000
requests per minute is tolerated for a maximum of ten seconds.

## Endpoints

### POST /v2/widgets

Creates a widget. The request body must be JSON and must include a `name`. The `colour`
field is optional and defaults to grey. Returns 201 with the created widget.

### GET /v2/widgets/{id}

Returns one widget. Returns 404 when the identifier is unknown. Deleted widgets return
410 for thirty days after deletion, then 404.

### DELETE /v2/widgets/{id}

Deletes a widget. Deletion is soft for thirty days, during which the widget can be
restored with POST /v2/widgets/{id}/restore. After thirty days the record is purged and
cannot be recovered.

## Errors

Every error response carries a `code` and a human readable `message`. Codes beginning
with `WIDGET_` are safe to show to end users; codes beginning with `INTERNAL_` are not.
"""

POLICY = """\
# Information security policy

## Access control

Access to production systems requires multi-factor authentication. Access is reviewed
quarterly and revoked within one business day of an employee leaving. Shared accounts
are prohibited except for break-glass administrator accounts, which are stored in a
sealed vault and rotated after every use.

## Data retention

Customer data is retained for the life of the contract plus ninety days. Backups are
retained for twelve months and are encrypted at rest with AES-256. Logs containing
personal data are retained for thirty days.

## Incident response

Suspected incidents must be reported to the security team within one hour of discovery.
The security team acknowledges within thirty minutes and provides an initial assessment
within four hours. Customers affected by a confirmed breach are notified within
seventy-two hours.

## Vendor management

Vendors handling customer data must complete a security review before onboarding and
annually thereafter. Vendors are rated Critical, High or Standard; Critical vendors
require an on-site audit every two years.
"""

#: The three documents every question points at.
ANSWERING = {
    "contract.md": CONTRACT,
    "api-reference.md": API_DOCS,
    "security-policy.md": POLICY,
}

#: Those three plus the near-neighbour distractors. A corpus of only the answering documents
#: retrieves a quarter of its own index at k=5 and every configuration scores 1.000, which
#: the eval set's own quality score correctly called out as unable to detect anything.
DOCUMENTS = {**ANSWERING, **distractors()}

# ---------------------------------------------------------------------------
# the questions
# ---------------------------------------------------------------------------

QUESTIONS: list[tuple[str, str, str, str, str]] = [
    # id, question, source, quoted evidence, type
    (
        "q01",
        "How much notice is needed to terminate for convenience?",
        "contract.md",
        "thirty days written notice",
        "factoid",
    ),
    (
        "q02",
        "How long does a party have to remedy a material breach?",
        "contract.md",
        "fifteen days of receiving written notice",
        "factoid",
    ),
    ("q03", "What is the Premium tier monthly fee?", "contract.md", "Premium | $3,400", "tabular"),
    (
        "q04",
        "What is the support response time on Enterprise?",
        "contract.md",
        "Enterprise | $9,750 | $2,000 | 1 hour",
        "tabular",
    ),
    (
        "q05",
        "How much notice stops the agreement renewing?",
        "contract.md",
        "at least sixty days before the end",
        "numeric",
    ),
    (
        "q06",
        "What is the cap on aggregate liability?",
        "contract.md",
        "shall not exceed the total fees paid",
        "factoid",
    ),
    (
        "q07",
        "How long does confidentiality last after disclosure?",
        "contract.md",
        "five years from disclosure",
        "numeric",
    ),
    (
        "q08",
        "Who owns the deliverables?",
        "contract.md",
        "The Customer owns all Deliverables on payment in full",
        "factoid",
    ),
    (
        "q09",
        "What interest applies to late payment?",
        "contract.md",
        "two percent per month",
        "numeric",
    ),
    (
        "q10",
        "Which clauses survive termination?",
        "contract.md",
        "Clauses 6, 7 and 9 survive",
        "factoid",
    ),
    ("q11", "Which header carries the API key?", "api-reference.md", "X-Api-Key", "factoid"),
    (
        "q12",
        "What is the default rate limit?",
        "api-reference.md",
        "600 requests per minute per key",
        "numeric",
    ),
    (
        "q13",
        "How long does a rotated API key stay valid?",
        "api-reference.md",
        "previous key stays valid for one hour",
        "numeric",
    ),
    (
        "q14",
        "What does a GET return for an unknown widget id?",
        "api-reference.md",
        "Returns 404 when the identifier is unknown",
        "factoid",
    ),
    (
        "q15",
        "How long can a deleted widget be restored?",
        "api-reference.md",
        "Deletion is soft for thirty days",
        "numeric",
    ),
    (
        "q16",
        "Which error codes are safe to show end users?",
        "api-reference.md",
        "Codes beginning with `WIDGET_` are safe to show",
        "factoid",
    ),
    (
        "q17",
        "What happens when the rate limit is exceeded?",
        "api-reference.md",
        "returns 429 with a `Retry-After` header",
        "factoid",
    ),
    (
        "q18",
        "How quickly must a suspected incident be reported?",
        "security-policy.md",
        "within one hour of discovery",
        "numeric",
    ),
    (
        "q19",
        "How long are backups kept?",
        "security-policy.md",
        "Backups are retained for twelve months",
        "numeric",
    ),
    (
        "q20",
        "When are affected customers notified of a breach?",
        "security-policy.md",
        "within\nseventy-two hours",
        "numeric",
    ),
    (
        "q21",
        "How often is production access reviewed?",
        "security-policy.md",
        "reviewed\nquarterly",
        "factoid",
    ),
    (
        "q22",
        "What encryption protects backups?",
        "security-policy.md",
        "encrypted at rest with AES-256",
        "factoid",
    ),
    (
        "q23",
        "How often do Critical vendors need an on-site audit?",
        "security-policy.md",
        "on-site audit every two years",
        "numeric",
    ),
    (
        "q24",
        "How long is customer data kept after the contract ends?",
        "security-policy.md",
        "life of the contract plus ninety days",
        "numeric",
    ),
]


#: The hand-written questions plus one per distractor fact. Twenty-four questions can only
#: detect differences above 0.40 -- the eval set's own quality score says so -- and a demo
#: whose winner is indistinguishable from its runner-up demonstrates nothing.
ALL_QUESTIONS = QUESTIONS + distractor_questions()


def build_evalset() -> cg.EvalSet:
    return cg.EvalSet(
        id="lab-demo",
        items=tuple(
            cg.EvalItem(
                id=item_id,
                question=question,
                anchors=(cg.GoldAnchor(source_id=source, quote=quote),),
                qtype=qtype,
            )
            for item_id, question, source, quote, qtype in ALL_QUESTIONS
        ),
        source="hand-written + generated",
    )


# ---------------------------------------------------------------------------
# the two corpora
# ---------------------------------------------------------------------------


def markdown_corpus() -> cg.Corpus:
    return cg.Corpus.from_texts(DOCUMENTS, media_type=MediaType.MARKDOWN, name="lab-demo")


def pdf_corpus() -> cg.Corpus | None:
    """The same words, typeset as PDFs, so the parser axis has something to disagree about.

    Markdown tables are drawn as **real bordered grids** rather than as lines of text. That is
    the whole point of the parser sweep: a table-aware parser finds the grid and keeps the row
    together, while a fast text extractor emits the cells as loose fragments in whatever order
    the content stream stored them. Typesetting the table as plain lines would make both
    parsers agree and the chart would show nothing.
    """
    try:
        import pymupdf
    except ImportError:
        return None

    files: list[SourceFile] = []
    for name, text in DOCUMENTS.items():
        document = pymupdf.open()
        page = document.new_page()
        y = 60.0

        lines = text.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()

            if _is_table_row(line):
                rows = []
                while index < len(lines) and _is_table_row(lines[index].strip()):
                    cells = _table_cells(lines[index].strip())
                    if cells and not all(set(c) <= set("-: ") for c in cells):
                        rows.append(cells)
                    index += 1
                if y > 640:
                    page = document.new_page()
                    y = 60.0
                y = _draw_table(pymupdf, page, rows, top=y, left=56) + 10
                continue

            index += 1
            if not line:
                y += 6
                continue
            if y > 760:
                page = document.new_page()
                y = 60.0
            size = 16 if line.startswith("# ") else 12 if line.startswith("## ") else 10
            page.insert_text((56, y), line.lstrip("# ").strip(), fontsize=size)
            y += size + 5

        files.append(
            SourceFile(
                id=name.replace(".md", ".pdf"),
                media_type=MediaType.PDF,
                raw=document.tobytes(),
            )
        )
        document.close()

    return cg.Corpus(files=tuple(files), name="lab-demo-pdf")


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|")


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _draw_table(
    pymupdf: Any, page: Any, rows: list[list[str]], *, top: float, left: float
) -> float:
    """Draw cells inside ruled boxes, which is what a table-aware parser looks for."""
    if not rows:
        return top

    columns = max(len(row) for row in rows)
    width = min(120.0, (520.0 - left + 56) / columns)
    height = 20.0

    for row_index, row in enumerate(rows):
        for column_index in range(columns):
            cell = row[column_index] if column_index < len(row) else ""
            x = left + column_index * width
            y = top + row_index * height
            page.draw_rect(pymupdf.Rect(x, y, x + width, y + height), color=(0, 0, 0), width=0.7)
            page.insert_text((x + 4, y + 14), cell[:22], fontsize=9)

    return top + len(rows) * height


def pdf_questions(evalset: cg.EvalSet) -> cg.EvalSet:
    """The same questions, pointed at the PDF filenames."""
    from dataclasses import replace

    return evalset.with_items(
        tuple(
            replace(
                item,
                anchors=tuple(
                    replace(anchor, source_id=anchor.source_id.replace(".md", ".pdf"))
                    for anchor in item.anchors
                ),
            )
            for item in evalset
        )
    )


# ---------------------------------------------------------------------------
# the sweeps
# ---------------------------------------------------------------------------

HEADLINE = "recall@5"


def run_retrieval_sweep(corpus: cg.Corpus, evalset: cg.EvalSet) -> Any:
    """The main grid: chunker, embedder, index and reranker on the Markdown corpus."""
    grid = matrix(
        parser="markdown",
        chunker=[
            "recursive:256,overlap=32",
            "recursive:512,overlap=64",
            "sentence:3",
            "structural:400,min_size=40",
            "semantic:85",
            "fixed:256,overlap=0",
        ],
        embedder=["tfidf", "hash:512"],
        index=["dense", "bm25", "hybrid"],
        reranker=[None, "lexical"],
        k=10,
    )
    return Runner(corpus=corpus, headline=HEADLINE).run(grid, evalset, mode="factorial")


def run_parser_sweep(corpus: cg.Corpus, evalset: cg.EvalSet) -> Any:
    """The headline: the same eval set, resolved against three readings of the same words."""
    grid = matrix(
        parser=["pymupdf", "pdfplumber"],
        chunker=["recursive:512,overlap=64", "sentence:3"],
        embedder="tfidf",
        index=["dense", "bm25", "hybrid"],
        k=10,
    )
    return Runner(corpus=corpus, headline=HEADLINE).run(grid, evalset, mode="factorial")


def run_depth_sweep(corpus: cg.Corpus, evalset: cg.EvalSet) -> Any:
    """The candidate-depth curve nobody publishes."""
    grid = matrix(
        chunker="recursive:256,overlap=32",
        embedder="tfidf",
        index="hybrid",
        reranker="lexical",
        candidates=[5, 10, 20, 50, 100, 200],
        k=10,
    )
    return Runner(corpus=corpus, headline=HEADLINE).run(grid, evalset, mode="factorial")


# ---------------------------------------------------------------------------
# serialising for the page
# ---------------------------------------------------------------------------


def run_payload(run: Any) -> dict[str, Any]:
    interval = run.interval()
    return {
        "label": run.label,
        "config": run.config.as_dict(),
        "metrics": {k: round(v, 4) for k, v in run.metrics.items()},
        "chunks": run.chunk_count,
        "p95_ms": round(run.timings.percentile(0.95), 3),
        "build_ms": round(run.timings.build_ms, 1),
        "index_bytes": run.index_bytes,
        "scored": run.scored_queries,
        "unresolved": run.unresolved_gold,
        "ci": None if interval is None else [round(interval.low, 4), round(interval.high, 4)],
        "by_type": {m: {t: round(v, 4) for t, v in d.items()} for m, d in run.by_type.items()},
        "failures": None if run.failures is None else run.failures.counts(),
        "per_query": {q: round(v, 4) for q, v in run.per_query.items()},
    }


def results_payload(results: Any, metric: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric": metric,
        "mode": results.mode,
        "cache": results.cache_summary,
        "runs": [run_payload(run) for run in results],
        "axis_effects": {
            axis: {str(k): round(v, 4) for k, v in results.axis_effect(axis, metric).items()}
            for axis in ("parser", "chunker", "embedder", "index", "reranker", "candidates")
            if len(results.axis_effect(axis, metric)) > 1
        },
        "summary": results.summary(metric),
        "pareto": [r.label for r in results.pareto(metric, "p95_ms")],
    }
    verdict = results.is_the_winner_real(metric)
    if verdict is not None:
        payload["verdict"] = {**verdict.as_dict(), "text": verdict.verdict()}
    return payload


def main(destination: Path) -> int:
    evalset = build_evalset()
    markdown = markdown_corpus()

    print("running the retrieval sweep ...", file=sys.stderr)
    retrieval = run_retrieval_sweep(markdown, evalset)

    print("running the candidate-depth sweep ...", file=sys.stderr)
    depth = run_depth_sweep(markdown, evalset)

    parser_payload = None
    pdfs = pdf_corpus()
    if pdfs is not None:
        print("running the parser sweep ...", file=sys.stderr)
        parser_payload = results_payload(run_parser_sweep(pdfs, pdf_questions(evalset)), HEADLINE)

    winner = retrieval.best(HEADLINE)
    quality = cg.assess(evalset)

    payload = {
        "generated_by": f"context-grid {cg.__version__}",
        "corpus": {
            "documents": [
                {
                    "id": name,
                    "characters": len(text),
                    "words": len(text.split()),
                    "answering": name in ANSWERING,
                }
                for name, text in DOCUMENTS.items()
            ],
            "distractor_note": describe(),
            "fingerprint": _fingerprint(markdown),
        },
        "evalset": {
            "id": evalset.id,
            "questions": [
                {"id": i, "question": q, "source": s, "quote": t, "type": k}
                for i, q, s, t, k in ALL_QUESTIONS
            ],
            "quality": {
                "size": quality.size,
                "detectable_difference": round(quality.detectable_difference, 3),
                "summary": quality.summary(),
            },
        },
        "retrieval": results_payload(retrieval, HEADLINE),
        "depth": results_payload(depth, HEADLINE),
        "parser": parser_payload,
        "manifest": build_manifest(winner.config, markdown, evalset).to_dict()
        if winner is not None
        else None,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    size = destination.stat().st_size
    print(
        f"wrote {destination} ({size / 1024:.0f} KB): "
        f"{len(payload['retrieval']['runs'])} retrieval runs, "
        f"{len(payload['depth']['runs'])} depth runs, "
        f"{0 if parser_payload is None else len(parser_payload['runs'])} parser runs",
        file=sys.stderr,
    )
    return 0


def _fingerprint(corpus: cg.Corpus) -> dict[str, Any]:
    from contextgrid.pipeline import build

    parses = build(Config(parser="markdown"), corpus).parses
    profile = cg.fingerprint(corpus, parses)
    return {
        "summary": profile.summary(),
        "hints": profile.hints(),
        "table_ratio": round(profile.table_ratio, 4),
        "headings": profile.heading_count,
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo.json")
    raise SystemExit(main(target))

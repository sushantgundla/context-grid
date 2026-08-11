"""Reading and writing eval sets.

Four formats, for four different reasons.

**JSONL** is the native one. It round-trips everything, including anchors, and it is what the
review queue writes.

**CSV** is what a subject-matter expert will actually hand you, because they wrote the
questions in a spreadsheet. Accepting it without complaint removes a real barrier.

**BEIR** is the standard IR layout, and importing it lets the scorer be checked against a
published benchmark rather than only against itself.

**LegalBench-RAG** matters most. It is the only public benchmark that stores ground truth as
character spans, which is the same decision this package makes -- so it is the natural set to
validate the whole scoring chain against.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from contextgrid.core.errors import EvalSetError
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span

# ---------------------------------------------------------------------------
# JSONL -- the native format
# ---------------------------------------------------------------------------


def write_jsonl(evalset: EvalSet, path: str | Path) -> Path:
    """Write an eval set, one item per line, with a header line carrying its identity."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "_evalset": {
            "id": evalset.id,
            "version": evalset.version,
            "source": evalset.source,
            "meta": evalset.meta,
        }
    }
    with target.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for item in evalset:
            handle.write(json.dumps(item.to_dict()) + "\n")
    return target


def read_jsonl(path: str | Path) -> EvalSet:
    """Read an eval set written by `write_jsonl`, or a bare list of items."""
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise EvalSetError(f"no eval set at {source_path}")

    identity: dict[str, Any] = {}
    items: list[EvalItem] = []

    with source_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise EvalSetError(f"{source_path}:{number} is not valid JSON: {exc}") from exc

            if "_evalset" in record:
                identity = record["_evalset"]
                continue
            items.append(EvalItem.from_dict(record))

    return EvalSet(
        id=identity.get("id", source_path.stem),
        items=tuple(items),
        version=int(identity.get("version", 1)),
        source=identity.get("source", "import"),
        meta=dict(identity.get("meta", {})),
    )


# ---------------------------------------------------------------------------
# CSV -- what a domain expert actually hands you
# ---------------------------------------------------------------------------

#: Column names accepted for each field, so a spreadsheet does not have to be reformatted.
_CSV_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "question_id", "qid"),
    "question": ("question", "query", "q"),
    "source_id": ("source_id", "document", "doc", "doc_id", "file", "filename"),
    "quote": ("quote", "evidence", "answer_span", "context", "passage"),
    "answer": ("answer", "expected_answer", "gold_answer"),
    "qtype": ("qtype", "type", "question_type", "category"),
    "page": ("page", "page_hint", "page_number"),
    "grade": ("grade", "relevance", "rel"),
}


def read_csv(path: str | Path, *, evalset_id: str | None = None) -> EvalSet:
    """Read questions from a spreadsheet export.

    Column names are matched loosely -- `question`/`query`/`q` all work, as do
    `quote`/`evidence`/`passage` -- because the alternative is telling somebody their
    spreadsheet is wrong when it is perfectly clear.
    """
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise EvalSetError(f"no eval set at {source_path}")

    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EvalSetError(f"{source_path} has no header row")

        columns = _match_columns(reader.fieldnames)
        if "question" not in columns:
            raise EvalSetError(
                f"{source_path} has no question column. Expected one of: "
                f"{', '.join(_CSV_ALIASES['question'])}. Found: {', '.join(reader.fieldnames)}"
            )

        items: list[EvalItem] = []
        for number, row in enumerate(reader, start=2):
            question = (row.get(columns["question"]) or "").strip()
            if not question:
                continue

            anchors: tuple[GoldAnchor, ...] = ()
            quote = (row.get(columns.get("quote", "")) or "").strip()
            source_id = (row.get(columns.get("source_id", "")) or "").strip()
            if quote and source_id:
                anchors = (
                    GoldAnchor(
                        source_id=source_id,
                        quote=quote,
                        grade=_as_int(row.get(columns.get("grade", "")), default=2),
                        page_hint=_as_optional_int(row.get(columns.get("page", ""))),
                    ),
                )

            items.append(
                EvalItem(
                    id=(row.get(columns.get("id", "")) or f"q{number - 1}").strip(),
                    question=question,
                    anchors=anchors,
                    qtype=(row.get(columns.get("qtype", "")) or "").strip() or None,
                    answer=(row.get(columns.get("answer", "")) or "").strip() or None,
                )
            )

    return EvalSet(
        id=evalset_id or source_path.stem, items=tuple(items), source=f"csv:{source_path.name}"
    )


def write_csv(evalset: EvalSet, path: str | Path) -> Path:
    """Write an eval set back out as a spreadsheet, for hand editing."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "question", "source_id", "quote", "grade", "page", "qtype", "answer"]
        )
        for item in evalset:
            anchor = item.anchors[0] if item.anchors else None
            writer.writerow(
                [
                    item.id,
                    item.question,
                    anchor.source_id if anchor else "",
                    anchor.quote if anchor else "",
                    anchor.grade if anchor else "",
                    anchor.page_hint if anchor and anchor.page_hint is not None else "",
                    item.qtype or "",
                    item.answer or "",
                ]
            )
    return target


# ---------------------------------------------------------------------------
# picking between the two hand-written formats
# ---------------------------------------------------------------------------


def read_evalset(path: str | Path) -> EvalSet:
    """Read a hand-written eval set, whichever of the two formats it is in.

    JSONL and CSV are both documented as "an eval set you can point at", so every entry point
    that takes one from a person should accept both. The choice was made inline in the config
    loader and nowhere else, so `evalset: ./questions.csv` in a config worked while
    `contextgrid evalset questions.csv` failed with a JSON parse error -- the same file, the
    same package, two answers. This is the one decision, for everything that needs it.

    By extension, not by sniffing the contents: the extension is what the person writing the
    file chose to call it, and a CSV whose first line happens to parse as JSON is not worth
    the ambiguity.
    """
    source_path = Path(path).expanduser()
    if source_path.suffix.lower() == ".csv":
        return read_csv(source_path)
    return read_jsonl(source_path)


# ---------------------------------------------------------------------------
# BEIR -- the standard IR layout
# ---------------------------------------------------------------------------


def read_beir(
    queries_path: str | Path,
    qrels_path: str | Path,
    *,
    evalset_id: str = "beir",
) -> EvalSet:
    """Read a BEIR-format dataset: `queries.jsonl` plus a TSV of judgements.

    BEIR identifies gold by *document*, not by span, so the items this produces carry
    document-level gold. That is enough to compare retrievers and not enough to compare
    chunkers fairly -- which is exactly the limitation span-level ground truth exists to fix,
    and it is stated on the imported set rather than left for someone to discover.
    """
    queries: dict[str, str] = {}
    with Path(queries_path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            queries[str(record["_id"])] = str(record["text"])

    judgements: dict[str, list[tuple[str, int]]] = {}
    with Path(qrels_path).expanduser().open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header and header[0].lower() not in {"query-id", "query_id", "qid"}:
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, document_id, score = row[0], row[1], row[2]
            try:
                grade = int(float(score))
            except ValueError:
                continue
            if grade > 0:
                judgements.setdefault(query_id, []).append((document_id, grade))

    items = [
        EvalItem(
            id=query_id,
            question=text,
            meta={"gold_documents": judgements.get(query_id, [])},
        )
        for query_id, text in queries.items()
        if query_id in judgements
    ]

    return EvalSet(
        id=evalset_id,
        items=tuple(items),
        source="beir",
        meta={
            "granularity": "document",
            "note": (
                "BEIR gold is document-level. It compares retrievers fairly and cannot "
                "compare chunkers fairly, because every chunk of a gold document counts "
                "as relevant regardless of whether it holds the evidence."
            ),
        },
    )


# ---------------------------------------------------------------------------
# LegalBench-RAG -- the one that stores spans
# ---------------------------------------------------------------------------


def read_legalbench_rag(path: str | Path, *, evalset_id: str = "legalbench-rag") -> EvalSet:
    """Read LegalBench-RAG, whose ground truth is character spans.

    The only public benchmark that anchors evidence the way this package does, which makes it
    the natural set to validate the scoring chain against: our numbers on it should reproduce
    the published ones, and if they do not, the problem is ours.

    Expected shape: `{"tests": [{"query": ..., "snippets": [{"file_path", "span": [s, e]}]}]}`.
    """
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise EvalSetError(f"no LegalBench-RAG file at {source_path}")

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    tests = payload.get("tests", payload if isinstance(payload, list) else [])

    items: list[EvalItem] = []
    for index, test in enumerate(tests):
        question = str(test.get("query", "")).strip()
        if not question:
            continue

        gold: list[GoldSpan] = []
        for snippet in test.get("snippets", []):
            span = snippet.get("span")
            file_path = snippet.get("file_path")
            if not span or not file_path or len(span) < 2:
                continue
            gold.append(GoldSpan(span=Span(str(file_path), int(span[0]), int(span[1])), grade=2))

        items.append(
            EvalItem(
                id=str(test.get("id", f"lb{index}")),
                question=question,
                gold=tuple(gold),
                answer=test.get("answer"),
            )
        )

    return EvalSet(
        id=evalset_id,
        items=tuple(items),
        source="legalbench-rag",
        meta={"granularity": "span"},
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _match_columns(fieldnames: Sequence[str]) -> dict[str, str]:
    """Map our field names onto whatever the spreadsheet called them."""
    lowered = {name.strip().lower(): name for name in fieldnames}
    matched: dict[str, str] = {}
    for field, aliases in _CSV_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                matched[field] = lowered[alias]
                break
    return matched


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_records(items: Iterable[EvalItem]) -> list[Mapping[str, Any]]:
    """Plain dictionaries, for anything that wants to serialise its own way."""
    return [item.to_dict() for item in items]

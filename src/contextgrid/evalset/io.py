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
import io
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from contextgrid.core.errors import EvalSetError, SpanError
from contextgrid.core.evalset import CSV_ALIASES, EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span

# ---------------------------------------------------------------------------
# opening the file at all
# ---------------------------------------------------------------------------

#: What an eval set is, in one clause, for the messages that have to say it.
_WANTED = "An eval set is a JSONL or CSV file of questions"

#: Extensions worth naming when the bytes turn out not to be text. Pointing a config's
#: `evalset:` at the spreadsheet the questions were written in is an easy mistake, and
#: "invalid continuation byte" is a terrible way to be told about it.
_BINARY_FORMATS: dict[str, str] = {
    ".xlsx": "an Excel workbook -- save the sheet as CSV and point at that instead",
    ".xls": "an Excel workbook -- save the sheet as CSV and point at that instead",
    ".ods": "a spreadsheet -- save the sheet as CSV and point at that instead",
    ".numbers": "a Numbers spreadsheet -- export it as CSV and point at that instead",
    ".pdf": "a PDF. Questions come from a JSONL or CSV file, not from a document",
    ".docx": "a Word document. Questions come from a JSONL or CSV file",
    ".doc": "a Word document. Questions come from a JSONL or CSV file",
    ".parquet": "a Parquet file -- export it as CSV and point at that instead",
    ".zip": "a zip archive. Point at the file inside it",
    ".gz": "a compressed file. Uncompress it and point at what comes out",
    ".db": "a database file -- export the questions as CSV and point at that instead",
    ".sqlite": "a database file -- export the questions as CSV and point at that instead",
}

#: Extensions an eval set is actually read from. Anything else still gets tried as JSONL --
#: `read_evalset` routes by extension and `config/loader.py` does the same, so refusing an
#: unusual name here would make one of them say yes and the other no about the same file.
#: This set only decides whether a *failure* is worth explaining as a format mistake.
_TEXT_FORMATS = frozenset({"", ".jsonl", ".json", ".ndjson", ".csv", ".tsv", ".txt"})


def _read_eval_file(source_path: Path, *, encoding: str = "utf-8") -> str:
    """The text of an eval set file, or an error saying what was expected instead.

    Every way of failing to open the file used to arrive as a raw OS or Python string with an
    `error:` prefix in front of it -- `[Errno 21] Is a directory`, or `'utf-8' codec can't
    decode byte 0xca in position 0`. True, and no use: neither says what an eval set is or
    what should have been named. This is the one place both readers open their file, so both
    the config loader's route and `read_evalset`'s route give the same answer.
    """
    if not source_path.exists():
        raise EvalSetError(f"no eval set at {source_path}")
    if source_path.is_dir():
        raise EvalSetError(
            f"{source_path} is a directory, and an eval set file was expected. Name the "
            ".jsonl or .csv file of questions itself, not the folder holding it."
        )
    if not source_path.is_file():
        raise EvalSetError(f"{source_path} is not a regular file. {_WANTED}.")

    try:
        return source_path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise EvalSetError(
            f"{source_path} is not text -- {exc.reason} at byte {exc.start}. {_WANTED}"
            f".{_binary_format_hint(source_path)}"
        ) from exc
    except OSError as exc:
        raise EvalSetError(f"{source_path} could not be read: {exc.strerror or exc}") from exc


def _binary_format_hint(source_path: Path) -> str:
    """Name the format when the extension says what the file really is."""
    told = _BINARY_FORMATS.get(source_path.suffix.lower())
    return f" This looks like {told}." if told else ""


def _wrong_format_hint(source_path: Path) -> str:
    """Say what was expected when a file that is text is not either format we read.

    Only on failure. A `.yaml` pointed at `evalset:` parses as text, fails on line 1, and the
    JSON error alone leaves somebody looking for a typo in a file that was never the right
    kind of file.
    """
    suffix = source_path.suffix.lower()
    if suffix in _TEXT_FORMATS:
        return ""
    return (
        f" A `{suffix}` file is not a format an eval set is read from: expected .jsonl "
        "(one question per line) or .csv."
    )


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
    contents = _read_eval_file(source_path)

    identity: dict[str, Any] = {}
    items: list[EvalItem] = []

    for number, line in enumerate(io.StringIO(contents), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvalSetError(
                f"{source_path}:{number} is not valid JSON: {exc}.{_wrong_format_hint(source_path)}"
            ) from exc

        if not isinstance(record, Mapping):
            raise EvalSetError(
                f"{source_path}:{number} is {_json_kind(record)}, and every line of a JSONL "
                "eval set has to be an object -- one question per line."
            )
        if "_evalset" in record:
            header = record["_evalset"]
            if not isinstance(header, Mapping):
                raise EvalSetError(
                    f"{source_path}:{number}: `_evalset` is {_json_kind(header)}, expected an "
                    "object carrying the set's id, version and source."
                )
            identity = dict(header)
            continue
        try:
            items.append(EvalItem.from_dict(record))
        except EvalSetError as exc:
            # The record knows what is wrong with itself; only the reader knows where it is.
            named = f" (item {record['id']!r})" if isinstance(record.get("id"), str) else ""
            raise EvalSetError(f"{source_path}:{number}{named}: {exc}") from exc

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

#: Two columns the CSV format carries that `CSV_ALIASES` does not name, and deliberately so.
#:
#: `CSV_ALIASES` lives in `core.evalset` because the *JSON* side needs it: a reader who saw
#: `doc_id` in the CSV documentation and then hand-wrote JSONL is told that the name is real
#: in the other format rather than simply refused. That reasoning does not apply to these two.
#: `occurrence` and `meta` are spelled identically in both formats, so there is no alias to
#: disambiguate and nothing for the JSON error path to say about them.
_LOCAL_CSV_ALIASES: dict[str, tuple[str, ...]] = {
    "occurrence": ("occurrence", "occurrence_index", "nth"),
    "meta": ("meta", "metadata"),
}

#: Column names accepted for each field, so a spreadsheet does not have to be reformatted.
#: Defined in `core.evalset` because the JSON readers need to recognise these names too, in
#: order to say that an alias belongs to the other format rather than simply refusing it.
_CSV_ALIASES: dict[str, tuple[str, ...]] = {**CSV_ALIASES, **_LOCAL_CSV_ALIASES}

#: The columns `write_csv` writes, in order. `occurrence` sits with the other anchor fields;
#: `meta` goes last because it is the only one holding JSON rather than a plain value, and a
#: spreadsheet is easier to read with the awkward column at the end.
CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "question",
    "source_id",
    "quote",
    "grade",
    "page",
    "occurrence",
    "qtype",
    "answer",
    "meta",
)


def read_csv(path: str | Path, *, evalset_id: str | None = None) -> EvalSet:
    """Read questions from a spreadsheet export.

    Column names are matched loosely -- `question`/`query`/`q` all work, as do
    `quote`/`evidence`/`passage` -- because the alternative is telling somebody their
    spreadsheet is wrong when it is perfectly clear.
    """
    source_path = Path(path).expanduser()
    contents = _read_eval_file(source_path, encoding="utf-8-sig")

    with io.StringIO(contents, newline="") as handle:
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
                        occurrence=_as_int(row.get(columns.get("occurrence", "")), default=0),
                    ),
                )

            items.append(
                EvalItem(
                    id=(row.get(columns.get("id", "")) or f"q{number - 1}").strip(),
                    question=question,
                    anchors=anchors,
                    qtype=(row.get(columns.get("qtype", "")) or "").strip() or None,
                    answer=(row.get(columns.get("answer", "")) or "").strip() or None,
                    meta=_as_meta(row.get(columns.get("meta", ""))),
                )
            )

    return EvalSet(
        id=evalset_id or source_path.stem, items=tuple(items), source=f"csv:{source_path.name}"
    )


def write_csv(evalset: EvalSet, path: str | Path) -> Path:
    """Write an eval set back out as a spreadsheet, for hand editing.

    Columns: `id, question, source_id, quote, grade, page, occurrence, qtype, answer, meta`.

    `occurrence` and `meta` are here because leaving them out silently changed what the file
    meant. An anchor written with `occurrence: 2` came back as `0`, which points at a
    different passage in a document that repeats its quote; and `meta` came back empty, which
    includes `meta.reviewed` -- the flag `assess()` counts for its "% reviewed" figure. So a
    round trip through a spreadsheet reset somebody's review progress and then had the quality
    summary tell them off for not having reviewed anything.

    `meta` is written as JSON in one cell. It is the only column that is not a plain value,
    and it is not pretty in a spreadsheet, but a JSON cell round-trips and a missing column
    does not. Note that JSON has no tuples: `("a", 1)` inside `meta` comes back as `["a", 1]`.

    What still cannot survive the format is warned about rather than dropped in silence: CSV
    is one row per question, so only `item.anchors[0]` is written, and span-form `gold` has no
    column at all. Both raise a `UserWarning` naming the questions affected.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    _warn_about_what_csv_cannot_carry(evalset)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
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
                    anchor.occurrence if anchor else "",
                    item.qtype or "",
                    item.answer or "",
                    json.dumps(item.meta, sort_keys=True) if item.meta else "",
                ]
            )
    return target


def _warn_about_what_csv_cannot_carry(evalset: EvalSet) -> None:
    """Say out loud which questions lose something on the way to a spreadsheet.

    A `UserWarning` rather than a `WarningLog`: `write_csv` returns a `Path`, so there is no
    result object to hang a log on, and the caller is a person running a command rather than a
    stage of a sweep. The point is only that the loss stops being silent -- the previous
    behaviour was to write the file and say nothing, which is how somebody discovers a
    second anchor is gone by finding a question scoring zero three commands later.
    """
    extra_anchors = [item.id for item in evalset if len(item.anchors) > 1]
    with_gold = [item.id for item in evalset if item.gold]

    if extra_anchors:
        warnings.warn(
            f"CSV holds one anchor per question, so only the first was written for "
            f"{_named_items(extra_anchors)}. Use `write_jsonl` to keep all of them.",
            UserWarning,
            stacklevel=3,
        )
    if with_gold:
        warnings.warn(
            f"CSV has no column for span-form `gold`, so the resolved spans on "
            f"{_named_items(with_gold)} were not written. The anchors were; re-resolve them "
            "against a parse to get the spans back, or use `write_jsonl`.",
            UserWarning,
            stacklevel=3,
        )


def _named_items(ids: Sequence[str], limit: int = 3) -> str:
    """Up to `limit` question ids, then a count. A warning naming 400 questions is noise."""
    shown = ", ".join(repr(name) for name in ids[:limit])
    rest = len(ids) - limit
    return f"{shown} and {rest} more" if rest > 0 else shown


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
    A bare array of tests is accepted too, because that is how several published dumps of the
    benchmark are actually shaped and the documentation says so.

    Two kinds of imperfection, treated differently. A snippet with *no* `file_path` or *no*
    two-element `span` is dropped quietly -- that is documented, and it is what an
    incompletely annotated benchmark looks like. A field of the *wrong shape* -- `span` as a
    string, `tests` as an object, a snippet that is not an object -- raises, naming the file
    and what was expected, because the alternative is loading something silently wrong:
    `"span": "117,202"` used to import as characters 1 to 1 and score against nothing.

    What was dropped is counted and recorded on the returned set's `meta`, because a quiet
    drop and a silent one are not the same thing. A benchmark file whose every snippet was
    discarded imports as tests with no gold, which used to look exactly like a benchmark
    pointing at the wrong corpus. `describe_skipped(evalset)` turns the counts into a line
    somebody can act on.
    """
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise EvalSetError(f"no LegalBench-RAG file at {source_path}")

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalSetError(f"{source_path} is not valid JSON: {exc}") from exc

    tests = _legalbench_tests(payload, source_path)
    skipped: dict[str, int] = {}

    items: list[EvalItem] = []
    for index, test in enumerate(tests):
        where = f"{source_path}: test {index}"
        if not isinstance(test, Mapping):
            raise EvalSetError(
                f"{where} is {_json_kind(test)}, expected an object with a `query` and "
                "a `snippets` array."
            )

        question = _legalbench_text(test.get("query"), field="query", where=where).strip()
        if not question:
            skipped[_NO_QUERY] = skipped.get(_NO_QUERY, 0) + 1
            continue

        items.append(
            EvalItem(
                id=_legalbench_id(test.get("id"), fallback=f"lb{index}", where=where),
                question=question,
                gold=tuple(_legalbench_gold(test.get("snippets"), where=where, skipped=skipped)),
                answer=_legalbench_text(test.get("answer"), field="answer", where=where) or None,
            )
        )

    return EvalSet(
        id=evalset_id,
        items=tuple(items),
        source="legalbench-rag",
        meta={
            "granularity": "span",
            "source_file": str(source_path),
            "tests_in_file": len(tests),
            "tests_skipped": skipped.get(_NO_QUERY, 0),
            "snippets_skipped": {
                reason: count for reason, count in skipped.items() if reason != _NO_QUERY
            },
        },
    )


#: Why a test or a snippet was left out, in the words the message will use.
_NO_QUERY = "had no `query`"
_NO_FILE_PATH = "had no `file_path`"
_NO_SPAN = "had no `span`"
_SHORT_SPAN = "had a span shorter than two offsets"


def describe_skipped(evalset: EvalSet) -> str:
    """One line naming everything an import left out, or an empty string if it left nothing.

    A count of what was dropped is the difference between "this file did not give you what
    you think it did" and finding out from a percentage three commands later. It reads off
    `meta` rather than being recomputed, so it cannot drift from what was actually skipped.
    """
    parts: list[str] = []

    tests_skipped = int(evalset.meta.get("tests_skipped", 0) or 0)
    if tests_skipped:
        word = "test" if tests_skipped == 1 else "tests"
        parts.append(f"{tests_skipped} {word} skipped: no `query`")

    snippets = evalset.meta.get("snippets_skipped") or {}
    total = sum(int(count) for count in snippets.values())
    if total:
        word = "snippet" if total == 1 else "snippets"
        reasons = ", ".join(f"{count} {reason}" for reason, count in snippets.items() if int(count))
        parts.append(f"{total} {word} skipped: {reasons}")

    return ". ".join(parts) + "." if parts else ""


def _legalbench_tests(payload: Any, source_path: Path) -> Sequence[Any]:
    """The list of tests, from either accepted top-level shape."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise EvalSetError(
            f"{source_path} is {_json_kind(payload)}. Expected a JSON object with a `tests` "
            "array, or a bare array of tests."
        )
    if "tests" not in payload:
        found = ", ".join(sorted(str(key) for key in payload)) or "none"
        raise EvalSetError(
            f"{source_path} has no `tests` key. Expected a JSON object with a `tests` array, "
            f"or a bare array of tests. Keys found: {found}."
        )
    tests = payload["tests"]
    if not isinstance(tests, list):
        raise EvalSetError(
            f"{source_path}: `tests` is {_json_kind(tests)}, expected an array of tests."
        )
    return tests


def _legalbench_gold(snippets: Any, *, where: str, skipped: dict[str, int]) -> list[GoldSpan]:
    """The gold spans of one test, dropping the incomplete and rejecting the misshapen.

    `skipped` is added to, not replaced: the counts are for the whole file, and the caller
    keeps them.
    """
    if snippets is None:
        return []
    if not isinstance(snippets, list):
        raise EvalSetError(
            f"{where}: `snippets` is {_json_kind(snippets)}, expected an array of snippets, "
            'each with a `file_path` and a `span` like {"file_path": "a.md", "span": [0, 12]}.'
        )

    gold: list[GoldSpan] = []
    for position, snippet in enumerate(snippets):
        place = f"{where}, snippet {position}"
        if not isinstance(snippet, Mapping):
            raise EvalSetError(
                f"{place} is {_json_kind(snippet)}, expected an object with a `file_path` "
                "and a `span`."
            )

        file_path = snippet.get("file_path")
        span = snippet.get("span")

        # Documented drops. Counted rather than announced one by one, because a benchmark
        # with a thousand incomplete rows should report a number, not a thousand lines.
        if file_path is None:
            skipped[_NO_FILE_PATH] = skipped.get(_NO_FILE_PATH, 0) + 1
            continue
        if not isinstance(file_path, str):
            raise EvalSetError(
                f"{place}: `file_path` is {_json_kind(file_path)}, expected a string naming "
                "a document relative to the corpus directory."
            )
        if not file_path.strip():
            skipped[_NO_FILE_PATH] = skipped.get(_NO_FILE_PATH, 0) + 1
            continue
        if span is None:
            skipped[_NO_SPAN] = skipped.get(_NO_SPAN, 0) + 1
            continue
        if not isinstance(span, (list, tuple)):
            raise EvalSetError(
                f"{place}: `span` is {_json_kind(span)}, expected an array of two character "
                "offsets, like [117, 202]."
            )
        if len(span) < 2:
            skipped[_SHORT_SPAN] = skipped.get(_SHORT_SPAN, 0) + 1
            continue

        try:
            start, end = _as_offset(span[0], place=place), _as_offset(span[1], place=place)
            gold.append(GoldSpan(span=Span(file_path, start, end), grade=2))
        except SpanError as exc:
            raise EvalSetError(f"{place}: {exc}") from exc
    return gold


def _as_offset(value: Any, *, place: str) -> int:
    """One end of a span, which has to be a whole number of characters."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise EvalSetError(
            f"{place}: `span` offsets must be whole numbers counted from 0, found "
            f"{_json_kind(value)}."
        )
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise EvalSetError(
            f"{place}: `span` offsets must be whole numbers counted from 0, found {value!r}."
        ) from exc


def _legalbench_text(value: Any, *, field: str, where: str) -> str:
    """A string field, absent or otherwise."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise EvalSetError(f"{where}: `{field}` is {_json_kind(value)}, expected a string.")
    return value


def _legalbench_id(value: Any, *, fallback: str, where: str) -> str:
    """The test's id. Numbers are accepted because benchmarks number their tests."""
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise EvalSetError(f"{where}: `id` is {_json_kind(value)}, expected a string or a number.")
    return str(value)


def _json_kind(value: Any) -> str:
    """What a JSON value is, in words somebody can read back against their file."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a true/false value"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, Mapping):
        return "an object"
    return f"a {type(value).__name__}"


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


def _as_meta(value: Any) -> dict[str, Any]:
    """A `meta` cell, read as JSON, or `{}` when it is empty or is not JSON at all.

    Forgiving on purpose. The column is only there so `write_csv`'s own output survives a
    round trip, and the file in between has been open in a spreadsheet where somebody may
    well have typed a note into it. A free-text cell should not take down a file of 400
    perfectly good questions; a JSON array should not become an item's `meta` either, which
    is why the type is checked rather than trusted.
    """
    text = (str(value) if value is not None else "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def to_records(items: Iterable[EvalItem]) -> list[Mapping[str, Any]]:
    """Plain dictionaries, for anything that wants to serialise its own way."""
    return [item.to_dict() for item in items]

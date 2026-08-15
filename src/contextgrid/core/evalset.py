"""Ground truth: anchors, gold spans, questions and eval sets.

Ground truth exists in two forms here, and the distinction matters.

A **`GoldAnchor`** is portable. It says "the evidence is this quoted sentence, somewhere on
page 4 of this file". It survives re-parsing, because it does not depend on any parser's
character offsets.

A **`GoldSpan`** is resolved. It says "characters 840-1010 of this document", and it is only
meaningful against the one parse that produced that text.

Why both: chunkers all cut up the *same* text, so span-level gold compares them fairly and no
anchor is needed. Parsers produce *different* text, so span-level gold cannot survive a change
of parser -- and comparing parsers is the headline feature. Anchors are the layer that makes
the parser axis possible, and they resolve down to spans once a parse exists.

There is a pleasing side effect. When a parser mangles a table so badly that the quoted
evidence no longer appears in its output, the anchor fails to resolve -- and that failure is
itself the measurement. A parser that loses the evidence cannot retrieve it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from contextgrid.core.errors import EvalSetError
from contextgrid.core.span import Span, total_length

# `ranx` consumes qrels and runs as {query_id: {doc_id: value}}. Naming the shape here keeps
# the boundary with that library in one place.
Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]

#: Column names `read_csv` accepts for each field, so a spreadsheet does not have to be
#: reformatted before it can be used.
#:
#: It lives here rather than beside the CSV reader because the JSON side needs it too -- not to
#: accept the aliases, which it does not, but to recognise one when it sees it. A reader who
#: followed the CSV documentation and then hand-wrote JSONL will reach for `doc_id`, and being
#: told "that name is real, in the other format" is the difference between a fixed file and a
#: guess.
CSV_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "question_id", "qid"),
    "question": ("question", "query", "q"),
    "source_id": ("source_id", "document", "doc", "doc_id", "file", "filename"),
    "quote": ("quote", "evidence", "answer_span", "context", "passage"),
    "answer": ("answer", "expected_answer", "gold_answer"),
    "qtype": ("qtype", "type", "question_type", "category"),
    "page": ("page", "page_hint", "page_number"),
    "grade": ("grade", "relevance", "rel"),
}


def _required(data: Mapping[str, Any], key: str, what: str) -> Any:
    """`data[key]`, or an `EvalSetError` that says what is missing and what was there instead.

    A bare `KeyError` reaches the CLI as `error: 'source_id'` -- four characters in quotes,
    naming no file, no record and no expectation. This is the same lookup with the context a
    person needs to fix their file.
    """
    if key in data:
        return data[key]

    alias = next((name for name in CSV_ALIASES.get(key, ()) if name != key and name in data), None)
    if alias is not None:
        raise EvalSetError(
            f"every {what} needs a `{key}`, and this one has `{alias}` instead. `{alias}` is a "
            f"real name for that field -- but only as a CSV column, where `read_csv` accepts "
            f"several spellings. JSON records take `{key}` and nothing else. Rename it."
        )

    present = ", ".join(f"`{name}`" for name in sorted(data)) or "no keys at all"
    raise EvalSetError(f"every {what} needs a `{key}`. This one has {present}.")


class QuestionType:
    """The question categories the tool slices metrics by.

    Not an Enum: users label their own corpora and a closed set would fight them. These are
    the values the built-in classifier emits, and the ones the report knows how to order.
    """

    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"
    NUMERIC = "numeric"
    TABULAR = "tabular"
    SUMMARISATION = "summarisation"
    UNANSWERABLE = "unanswerable"

    ALL = (FACTOID, MULTI_HOP, COMPARATIVE, NUMERIC, TABULAR, SUMMARISATION, UNANSWERABLE)


@dataclass(frozen=True, slots=True)
class GoldAnchor:
    """Parser-independent evidence: the text that answers the question, quoted.

    `occurrence` disambiguates a quote that appears more than once in the document, counting
    from zero in reading order. `page_hint` narrows the search and, when a parser reports
    pages, catches the case where the same boilerplate sentence appears on every page.
    """

    source_id: str
    quote: str
    grade: int = 2
    page_hint: int | None = None
    occurrence: int = 0

    def __post_init__(self) -> None:
        if not self.quote.strip():
            raise EvalSetError("a gold anchor must quote some text")
        if self.grade < 0:
            raise EvalSetError(f"gold grade must be >= 0, got {self.grade}")
        if self.occurrence < 0:
            raise EvalSetError(f"occurrence must be >= 0, got {self.occurrence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "quote": self.quote,
            "grade": self.grade,
            "page_hint": self.page_hint,
            "occurrence": self.occurrence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoldAnchor:
        return cls(
            source_id=str(_required(data, "source_id", "gold anchor")),
            quote=str(_required(data, "quote", "gold anchor")),
            grade=int(data.get("grade", 2)),
            page_hint=data.get("page_hint"),
            occurrence=int(data.get("occurrence", 0)),
        )


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """A stretch of source text that answers a question, and how well it does so.

    Grades follow the usual IR convention: 2 fully answers, 1 partially relevant,
    0 irrelevant. Graded relevance is what makes nDCG mean anything; binary gold turns it
    into a noisier version of hit rate.
    """

    span: Span
    grade: int = 2

    def __post_init__(self) -> None:
        if self.span.is_empty:
            raise EvalSetError(f"gold span must cover at least one character: {self.span!r}")
        if self.grade < 0:
            raise EvalSetError(f"gold grade must be >= 0, got {self.grade}")

    @property
    def doc_id(self) -> str:
        return self.span.doc_id

    def to_dict(self) -> dict[str, Any]:
        return {**self.span.to_dict(), "grade": self.grade}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoldSpan:
        return cls(span=Span.from_dict(data), grade=int(data.get("grade", 2)))


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One question and the source text that answers it.

    Gold is a set of spans, not one, because a question can legitimately be answered by more
    than one passage -- and because a single piece of evidence can straddle a paragraph break
    and be cleaner to annotate as two spans than one.

    `anchors` is the portable form of the same evidence. An item authored against a fixed
    text can carry spans alone; an item that must survive a change of parser carries anchors,
    and `resolved_with` fills the spans in once a parse exists.
    """

    id: str
    question: str
    gold: tuple[GoldSpan, ...] = ()
    anchors: tuple[GoldAnchor, ...] = ()
    qtype: str | None = None
    answer: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise EvalSetError(f"eval item {self.id!r} has an empty question")

    @property
    def is_answerable(self) -> bool:
        """True when this question carries evidence in either form, resolved or not.

        False only for questions deliberately included with no supporting evidence at all.
        Unanswerable questions test whether a system correctly declines to answer, which
        almost no eval set measures and which is a real failure mode in production.

        Evidence counts in *either* form. The documented best practice is to write anchors
        rather than spans (`docs/guide/evalsets.md`), so a rule that looked only at `gold`
        reported every best-practice eval set as entirely unanswerable. A grade-0 anchor or
        span still counts: this asks "is there evidence here a scorer could resolve", not
        "is the evidence relevant", and `gold` has always counted grade-0 spans. Filtering
        grade 0 on the anchor side alone would make the two forms disagree again.

        Use `is_resolved` for the stricter question -- "has that evidence been located in
        *this* parse" -- which is what ranking metrics and the parse dimension need.
        """
        return bool(self.gold or self.anchors)

    #: Older name for `is_answerable`. There were once two rules for "does this question
    #: have ground truth", and they drifted: the CLI counted anchors, `is_answerable` did
    #: not, and the same file reported two different answers. One rule now, under both names.
    has_evidence = is_answerable

    @property
    def is_resolved(self) -> bool:
        """True when this item's evidence has been located as character spans in a parse.

        Stricter than `is_answerable`. A freshly authored item has anchors and no spans: it
        is answerable, and nothing has resolved it yet. Ranking metrics need spans, so they
        ask this; and `anchors and not is_resolved` is exactly "this parser lost the quoted
        evidence", which is how the parse dimension is scored.
        """
        return bool(self.gold)

    @property
    def is_portable(self) -> bool:
        """True when this item carries anchors and can be re-resolved against any parse."""
        return bool(self.anchors)

    @property
    def gold_spans(self) -> tuple[Span, ...]:
        return tuple(g.span for g in self.gold)

    @property
    def gold_length(self) -> int:
        """Total gold characters, counting overlapping gold spans once."""
        return total_length(self.gold_spans)

    def gold_documents(self) -> set[str]:
        return {g.doc_id for g in self.gold}

    def resolved_with(self, gold: tuple[GoldSpan, ...]) -> EvalItem:
        """A copy of this item with its gold spans replaced by ones resolved against a parse."""
        return replace(self, gold=gold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold": [g.to_dict() for g in self.gold],
            "anchors": [a.to_dict() for a in self.anchors],
            "qtype": self.qtype,
            "answer": self.answer,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalItem:
        return cls(
            id=str(_required(data, "id", "eval item")),
            question=str(_required(data, "question", "eval item")),
            gold=tuple(GoldSpan.from_dict(g) for g in data.get("gold", ())),
            anchors=tuple(GoldAnchor.from_dict(a) for a in data.get("anchors", ())),
            qtype=data.get("qtype"),
            answer=data.get("answer"),
            meta=dict(data.get("meta", {})),
        )


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A versioned collection of questions with span-level ground truth.

    Versioned because comparing two runs scored against different eval sets is a silent
    correctness bug, and the version is what lets a manifest catch it.
    """

    id: str
    items: tuple[EvalItem, ...]
    version: int = 1
    source: str = "manual"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise EvalSetError(f"duplicate eval item id {item.id!r} in eval set {self.id!r}")
            seen.add(item.id)

    @property
    def answerable(self) -> tuple[EvalItem, ...]:
        """Questions carrying evidence in either form, resolved or not."""
        return tuple(i for i in self.items if i.is_answerable)

    #: Older name for `answerable`, from when the two counted differently. See
    #: `EvalItem.is_answerable`.
    with_evidence = answerable

    @property
    def resolved(self) -> tuple[EvalItem, ...]:
        """Questions whose evidence has been located as spans in a particular parse."""
        return tuple(i for i in self.items if i.is_resolved)

    @property
    def is_portable(self) -> bool:
        """True when every item with evidence can be re-resolved against a different parse."""
        return all(i.is_portable for i in self.items if i.is_answerable)

    def by_type(self, qtype: str) -> tuple[EvalItem, ...]:
        return tuple(i for i in self.items if i.qtype == qtype)

    def types(self) -> set[str]:
        return {i.qtype for i in self.items if i.qtype is not None}

    def get(self, item_id: str) -> EvalItem | None:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def with_items(self, items: tuple[EvalItem, ...]) -> EvalSet:
        """A copy holding different items, keeping id, version and provenance."""
        return replace(self, items=items)

    def __iter__(self) -> Iterator[EvalItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    """A resolved judgement: for this question, this chunk is relevant at this grade."""

    item_id: str
    chunk_id: str
    grade: int

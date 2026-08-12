"""What "answerable" means, and the one rule that decides it.

There were two rules. `EvalItem.is_answerable` looked only at `gold`; `evalset.quality.assess`
looked at `gold or anchors`. On the same file they disagreed, and the disagreement pointed the
wrong way: `docs/guide/evalsets.md` tells every user to write anchors rather than spans, so the
property reported the documented best-practice eval set as entirely unanswerable while the CLI
printed it as fully answerable.

One rule now. `is_answerable` means "there is evidence here a scorer could resolve, in either
form". The stricter question -- "has this parse located that evidence as character spans" --
is `is_resolved`, and it is a real and separate thing: it is how the parser axis is scored.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span
from contextgrid.evalset.io import read_csv, read_jsonl
from contextgrid.evalset.quality import assess


def anchor(quote: str = "within 30 days of purchase", grade: int = 2) -> GoldAnchor:
    return GoldAnchor(source_id="refunds.md", quote=quote, grade=grade)


def span(grade: int = 2) -> GoldSpan:
    return GoldSpan(span=Span("refunds.md", 0, 26), grade=grade)


# -- the four combinations ---------------------------------------------------


def test_gold_only_is_answerable() -> None:
    item = EvalItem(id="q1", question="How long do refunds take?", gold=(span(),))
    assert item.is_answerable
    assert item.is_resolved
    assert not item.is_portable


def test_anchor_only_is_answerable() -> None:
    """The documented best practice, and the shape the bug got wrong."""
    item = EvalItem(id="q1", question="How long do refunds take?", anchors=(anchor(),))
    assert item.is_answerable
    assert not item.is_resolved  # nothing has located it in a parse yet
    assert item.is_portable


def test_both_forms_is_answerable() -> None:
    item = EvalItem(
        id="q1", question="How long do refunds take?", gold=(span(),), anchors=(anchor(),)
    )
    assert item.is_answerable
    assert item.is_resolved


def test_neither_form_is_not_answerable() -> None:
    """Deliberately unanswerable questions are the point of this being a real property."""
    item = EvalItem(id="q1", question="What is the CEO's phone number?")
    assert not item.is_answerable
    assert not item.is_resolved


# -- the grade-0 decision ----------------------------------------------------


def test_a_grade_0_anchor_still_counts_as_answerable() -> None:
    """Decided: grade 0 counts.

    `is_answerable` asks whether there is evidence a scorer could resolve, not whether that
    evidence is relevant. `gold` has always counted grade-0 spans, so filtering grade 0 on
    the anchor side alone would put the two forms back into disagreement -- which is the
    exact failure being fixed. An item whose only evidence is graded 0 will score zero on
    ranking metrics, and that is the metric's job to say, not this property's.
    """
    item = EvalItem(id="q1", question="How long do refunds take?", anchors=(anchor(grade=0),))
    assert item.is_answerable


def test_a_grade_0_span_counts_too_so_the_two_forms_agree() -> None:
    item = EvalItem(id="q1", question="How long do refunds take?", gold=(span(grade=0),))
    assert item.is_answerable


# -- one rule, not two -------------------------------------------------------


def test_assess_counts_answerable_with_the_same_rule_as_the_property() -> None:
    """`assess` used to keep its own copy of this rule, and the copies disagreed."""
    items = (
        EvalItem(id="gold", question="a", gold=(span(),)),
        EvalItem(id="anchor", question="b", anchors=(anchor(),)),
        EvalItem(id="both", question="c", gold=(span(),), anchors=(anchor(),)),
        EvalItem(id="neither", question="d"),
    )
    evalset = EvalSet(id="mixed", items=items)
    quality = assess(evalset)

    assert quality.answerable == sum(1 for i in items if i.is_answerable) == 3
    assert quality.unanswerable == 1
    assert len(evalset.answerable) == 3


def test_has_evidence_and_with_evidence_are_the_same_rule_under_the_old_names() -> None:
    item = EvalItem(id="q1", question="a", anchors=(anchor(),))
    assert item.has_evidence is item.is_answerable is True

    evalset = EvalSet(id="e", items=(item,))
    assert evalset.with_evidence == evalset.answerable


def test_resolved_is_the_separate_stricter_question() -> None:
    """The parse dimension needs this: anchors present, spans not found."""
    evalset = EvalSet(
        id="e",
        items=(
            EvalItem(id="found", question="a", gold=(span(),), anchors=(anchor(),)),
            EvalItem(id="lost", question="b", anchors=(anchor("a quote this parse mangled"),)),
        ),
    )
    assert len(evalset.answerable) == 2
    assert len(evalset.resolved) == 1
    assert [i.id for i in evalset.resolved] == ["found"]


# -- through the readers -----------------------------------------------------


def test_the_documented_jsonl_example_is_answerable(tmp_path: Path) -> None:
    """`docs/guide/getting-started.md`'s own anchor-only example."""
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"_evalset": {"id": "policy-questions", "version": 1, "source": "manual"}})
        + "\n"
        + json.dumps(
            {
                "id": "q1",
                "question": "How long do refunds take?",
                "gold": [],
                "anchors": [
                    {
                        "source_id": "refunds.md",
                        "quote": "within 30 days of purchase",
                        "grade": 2,
                        "page_hint": None,
                        "occurrence": 0,
                    }
                ],
            }
        )
        + "\n"
    )

    (item,) = read_jsonl(path)
    assert item.anchors
    assert not item.gold
    assert item.is_answerable


def test_csv_rows_with_a_quote_and_a_document_are_answerable(tmp_path: Path) -> None:
    """`docs/guide/evalsets.md`'s own CSV example."""
    path = tmp_path / "questions.csv"
    path.write_text(
        "question,document,evidence\n"
        "How long do refunds take?,refunds.md,within 30 days of purchase\n"
        "How fast is express shipping?,shipping.md,arrives the next business day\n"
    )

    evalset = read_csv(path, evalset_id="from-csv")
    assert len(evalset) == 2
    assert all(i.is_answerable for i in evalset)
    assert assess(evalset).answerable == 2


def test_a_csv_row_with_no_quote_stays_unanswerable(tmp_path: Path) -> None:
    """Documented at `docs/guide/evalsets.md`: only `question` is required, and a row
    without a quote/document is a question with no evidence yet."""
    path = tmp_path / "questions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "document", "evidence"])
        writer.writerow(["How long do refunds take?", "refunds.md", "within 30 days"])
        writer.writerow(["What is the CEO's phone number?", "", ""])

    evalset = read_csv(path, evalset_id="from-csv")
    answerable = {i.question: i.is_answerable for i in evalset}
    assert answerable == {
        "How long do refunds take?": True,
        "What is the CEO's phone number?": False,
    }
    assert assess(evalset).answerable == 1

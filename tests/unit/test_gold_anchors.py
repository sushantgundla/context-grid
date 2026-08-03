"""Unit tests for portable ground truth.

Ground truth exists in two forms, and the distinction is what makes the parser axis possible.

A gold *span* says "characters 840-1010", and is only meaningful against the one parse that
produced that text. Chunkers all cut up the same text, so spans compare them perfectly well.
Parsers produce *different* text, so spans cannot survive a change of parser.

A gold *anchor* says "the evidence is this quoted sentence". It survives re-parsing, and it
resolves down to a span once a parse exists.
"""

from __future__ import annotations

import pytest

from contextgrid.core.errors import EvalSetError
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan, QuestionType
from contextgrid.core.span import Span


def anchor(quote: str = "thirty days", **kwargs: object) -> GoldAnchor:
    return GoldAnchor(source_id="contract", quote=quote, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GoldAnchor
# ---------------------------------------------------------------------------


def test_an_anchor_must_quote_something() -> None:
    with pytest.raises(EvalSetError, match="must quote some text"):
        GoldAnchor(source_id="c", quote="   ")


def test_anchor_grade_cannot_be_negative() -> None:
    with pytest.raises(EvalSetError, match="grade must be >= 0"):
        GoldAnchor(source_id="c", quote="x", grade=-1)


def test_anchor_occurrence_cannot_be_negative() -> None:
    with pytest.raises(EvalSetError, match="occurrence must be >= 0"):
        GoldAnchor(source_id="c", quote="x", occurrence=-1)


def test_anchor_defaults_to_the_first_occurrence_fully_answering() -> None:
    a = anchor()
    assert a.grade == 2
    assert a.occurrence == 0
    assert a.page_hint is None


def test_anchor_round_trips_through_dict() -> None:
    a = anchor(grade=1, page_hint=4, occurrence=2)
    assert GoldAnchor.from_dict(a.to_dict()) == a


# ---------------------------------------------------------------------------
# portability
# ---------------------------------------------------------------------------


def test_an_item_with_anchors_is_portable() -> None:
    item = EvalItem(id="q1", question="How long is notice?", anchors=(anchor(),))
    assert item.is_portable


def test_an_item_with_only_spans_is_not_portable() -> None:
    """It can still compare chunkers perfectly well. It cannot survive a change of parser."""
    item = EvalItem(id="q1", question="q", gold=(GoldSpan(Span("contract", 100, 200)),))
    assert not item.is_portable
    assert item.is_answerable


def test_resolving_fills_in_the_spans_and_keeps_everything_else() -> None:
    item = EvalItem(
        id="q1",
        question="How long is notice?",
        anchors=(anchor(),),
        qtype=QuestionType.FACTOID,
        meta={"reviewed": True},
    )
    resolved = item.resolved_with((GoldSpan(Span("contract", 840, 1010)),))

    assert resolved.gold_spans == (Span("contract", 840, 1010),)
    assert resolved.anchors == item.anchors  # still re-resolvable against another parse
    assert resolved.qtype == QuestionType.FACTOID
    assert resolved.meta == {"reviewed": True}
    assert item.gold == ()  # the original is untouched


def test_the_same_anchors_resolve_to_different_spans_under_different_parses() -> None:
    """The whole point. Two parsers produce different text, so the same evidence lives at
    different offsets -- and the eval set does not need rewriting."""
    item = EvalItem(id="q1", question="q", anchors=(anchor(),))
    under_parser_a = item.resolved_with((GoldSpan(Span("contract", 840, 1010)),))
    under_parser_b = item.resolved_with((GoldSpan(Span("contract", 902, 1072)),))

    assert under_parser_a.gold_spans != under_parser_b.gold_spans
    assert under_parser_a.anchors == under_parser_b.anchors


def test_an_eval_set_is_portable_when_every_answerable_item_is() -> None:
    portable = EvalSet(
        id="es",
        items=(
            EvalItem(id="q1", question="a", anchors=(anchor(),)),
            EvalItem(id="q2", question="b"),  # unanswerable, so it does not count against it
        ),
    )
    assert portable.is_portable


def test_an_eval_set_is_not_portable_if_any_answerable_item_is_span_only() -> None:
    mixed = EvalSet(
        id="es",
        items=(
            EvalItem(id="q1", question="a", anchors=(anchor(),)),
            EvalItem(id="q2", question="b", gold=(GoldSpan(Span("contract", 0, 10)),)),
        ),
    )
    assert not mixed.is_portable


def test_with_items_keeps_the_identity_of_the_set() -> None:
    """A resolved eval set must stay comparable to the one it came from, so id, version and
    provenance survive."""
    original = EvalSet(id="es", items=(EvalItem(id="q1", question="a"),), version=3, source="auto")
    replaced = original.with_items((EvalItem(id="q1", question="a", qtype="factoid"),))
    assert replaced.id == "es"
    assert replaced.version == 3
    assert replaced.source == "auto"


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_an_item_round_trips_with_both_forms_of_gold() -> None:
    item = EvalItem(
        id="q1",
        question="How long is the notice period?",
        gold=(GoldSpan(Span("contract", 840, 1010), grade=2),),
        anchors=(anchor(page_hint=2),),
        qtype=QuestionType.FACTOID,
        answer="Thirty days.",
    )
    assert EvalItem.from_dict(item.to_dict()) == item


def test_question_types_are_open_but_the_common_ones_are_named() -> None:
    """Users label their own corpora, so this is not a closed enum -- but the built-in
    classifier and the report both need a shared vocabulary."""
    assert QuestionType.TABULAR in QuestionType.ALL
    assert EvalItem(id="q", question="q", qtype="my-own-label").qtype == "my-own-label"

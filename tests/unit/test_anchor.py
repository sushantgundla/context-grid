"""Unit tests for locating portable ground truth inside a parse.

The scenarios that matter are the ones a real PDF extractor creates: the same sentence
reflowed across a line break, the same boilerplate repeated on every page, and evidence that
one parser simply loses.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import Block, BlockKind, Document, ParsedDocument
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.span import Span
from contextgrid.core.warnings import WarningCode
from contextgrid.score.anchor import (
    AnchorResolver,
    MatchStrategy,
    collapse_whitespace,
)

CLEAN = (
    "Either party may terminate this agreement for convenience by giving thirty days "
    "written notice."
)

# What a PDF extractor typically produces from the same paragraph: a line break mid-sentence
# and doubled spaces after the column gutter.
REFLOWED = (
    "Either party may terminate this agreement\nfor convenience by giving  thirty days\n"
    "written notice."
)


def parse(text: str, parser: str = "text", doc_id: str = "contract") -> ParsedDocument:
    return ParsedDocument(document=Document(id=doc_id, text=text), parser=parser)


def anchor(quote: str, **kwargs: object) -> GoldAnchor:
    return GoldAnchor(source_id="contract", quote=quote, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# collapse_whitespace
# ---------------------------------------------------------------------------


def test_collapse_maps_every_character_back_to_the_original() -> None:
    text = "a  b\n\nc"
    collapsed, positions = collapse_whitespace(text)
    assert collapsed == "a b c"
    assert len(collapsed) == len(positions)
    for index, char in enumerate(collapsed):
        original = text[positions[index]]
        assert char == original or (char == " " and original.isspace())


def test_collapse_leaves_text_without_runs_alone() -> None:
    collapsed, positions = collapse_whitespace("a b c")
    assert collapsed == "a b c"
    assert positions == [0, 1, 2, 3, 4]


def test_collapse_of_empty_text() -> None:
    assert collapse_whitespace("") == ("", [])


def test_collapse_handles_leading_and_trailing_runs() -> None:
    collapsed, positions = collapse_whitespace("  hi  ")
    assert collapsed == " hi "
    assert positions[0] == 0


# ---------------------------------------------------------------------------
# exact matching
# ---------------------------------------------------------------------------


def test_a_verbatim_quote_is_found_exactly() -> None:
    parsed = parse(f"Preamble. {CLEAN} More text.")
    match = AnchorResolver().locate(anchor("thirty days written notice"), parsed)

    assert match.strategy is MatchStrategy.EXACT
    assert match.span is not None
    assert parsed.document.slice(match.span) == "thirty days written notice"


def test_the_resolved_span_points_at_real_characters() -> None:
    """The whole contract: a relaxed comparison must still yield exact offsets."""
    parsed = parse(f"Preamble. {CLEAN}")
    match = AnchorResolver().locate(anchor("thirty days"), parsed)
    assert match.span is not None
    assert parsed.text[match.span.start : match.span.end] == "thirty days"


def test_surrounding_whitespace_in_the_quote_is_ignored() -> None:
    parsed = parse(CLEAN)
    match = AnchorResolver().locate(anchor("  thirty days  "), parsed)
    assert match.found
    assert parsed.document.slice(match.span) == "thirty days"  # type: ignore[arg-type]


def test_an_empty_parse_finds_nothing() -> None:
    assert not AnchorResolver().locate(anchor("thirty days"), parse("")).found


# ---------------------------------------------------------------------------
# normalised matching -- the PDF case
# ---------------------------------------------------------------------------


def test_a_reflowed_quote_is_found_after_collapsing_whitespace() -> None:
    """The single most common real-world case. A PDF extractor breaks the sentence across a
    line and doubles a space; the evidence is still there and must still resolve."""
    parsed = parse(REFLOWED, parser="pymupdf")
    match = AnchorResolver().locate(anchor("for convenience by giving thirty days"), parsed)

    assert match.strategy is MatchStrategy.NORMALISED
    assert match.span is not None
    # The span covers the real characters, doubled space and all -- not a tidied copy.
    assert parsed.document.slice(match.span) == "for convenience by giving  thirty days"


def test_a_quote_spanning_a_line_break_resolves_to_the_real_characters() -> None:
    parsed = parse(REFLOWED, parser="pymupdf")
    match = AnchorResolver().locate(anchor("this agreement for convenience"), parsed)

    assert match.strategy is MatchStrategy.NORMALISED
    assert match.span is not None
    found = parsed.document.slice(match.span)
    assert "\n" in found  # the newline is inside the span, where it belongs
    assert " ".join(found.split()) == "this agreement for convenience"


def test_normalisation_can_be_turned_off() -> None:
    parsed = parse(REFLOWED)
    strict = AnchorResolver(allow_normalised=False)
    assert not strict.locate(anchor("for convenience by giving thirty days"), parsed).found


def test_case_sensitivity_is_the_default_and_can_be_relaxed() -> None:
    parsed = parse(CLEAN)
    assert not AnchorResolver().locate(anchor("THIRTY DAYS"), parsed).found
    assert AnchorResolver(case_sensitive=False).locate(anchor("THIRTY DAYS"), parsed).found


# ---------------------------------------------------------------------------
# bounded matching -- damage in the middle
# ---------------------------------------------------------------------------


def test_bounded_matching_survives_corruption_in_the_middle() -> None:
    """What a bad table extraction does: the ends of the passage survive, something in the
    middle is mangled. Off by default, because a wrong span is worse than a missing one."""
    damaged = parse("Either party may terminate ??? ?? giving thirty days written notice.")
    quote = "Either party may terminate this agreement for convenience by giving thirty days"

    assert not AnchorResolver().locate(anchor(quote), damaged).found

    lenient = AnchorResolver(allow_bounded=True)
    match = lenient.locate(anchor(quote), damaged)
    assert match.strategy is MatchStrategy.BOUNDED
    assert match.span is not None


def test_bounded_matching_refuses_a_span_far_longer_than_the_quote() -> None:
    """Two unrelated occurrences of the opening and closing words are not one damaged
    passage, and stitching them together would score entirely the wrong text as correct."""
    text = "Either party may terminate. " + ("filler " * 200) + "giving thirty days notice."
    lenient = AnchorResolver(allow_bounded=True)
    quote = "Either party may terminate this agreement by giving thirty days notice"
    assert not lenient.locate(anchor(quote), parse(text)).found


def test_bounded_matching_needs_enough_words_to_be_meaningful() -> None:
    lenient = AnchorResolver(allow_bounded=True)
    assert not lenient.locate(anchor("thirty days"), parse("nothing similar here")).found


# ---------------------------------------------------------------------------
# choosing between repeated occurrences
# ---------------------------------------------------------------------------


REPEATED = "Confidential. Page one.\nConfidential. Page two.\nConfidential. Page three."


def test_a_repeated_quote_defaults_to_the_first_occurrence() -> None:
    match = AnchorResolver().locate(anchor("Confidential."), parse(REPEATED))
    assert match.candidates == 3
    assert match.span == Span("contract", 0, 13)


def test_occurrence_selects_a_later_one() -> None:
    match = AnchorResolver().locate(anchor("Confidential.", occurrence=2), parse(REPEATED))
    assert match.span is not None
    assert match.span.start == REPEATED.rindex("Confidential.")


def test_an_occurrence_beyond_the_last_one_is_not_found() -> None:
    assert not AnchorResolver().locate(anchor("Confidential.", occurrence=9), parse(REPEATED)).found


def test_a_page_hint_narrows_repeated_boilerplate() -> None:
    """Header text repeated on every page is the classic ambiguous anchor, and the page is
    the only thing that distinguishes the occurrences."""
    text = "Confidential\nBody one.\nConfidential\nBody two."
    parsed = ParsedDocument(
        document=Document(id="contract", text=text),
        blocks=(
            Block(span=Span("contract", 0, 22), text=text[0:22], kind=BlockKind.PARAGRAPH, page=1),
            Block(
                span=Span("contract", 23, len(text)),
                text=text[23:],
                kind=BlockKind.PARAGRAPH,
                page=2,
            ),
        ),
    )
    match = AnchorResolver().locate(anchor("Confidential", page_hint=2), parsed)
    assert match.span is not None
    assert match.span.start == text.rindex("Confidential")


def test_a_page_hint_that_matches_nothing_falls_back_rather_than_failing() -> None:
    match = AnchorResolver().locate(anchor("Confidential.", page_hint=7), parse(REPEATED))
    assert match.found


# ---------------------------------------------------------------------------
# resolving items and eval sets
# ---------------------------------------------------------------------------


def item(*anchors: GoldAnchor, iid: str = "q1") -> EvalItem:
    return EvalItem(id=iid, question="How long is the notice period?", anchors=anchors)


def test_resolving_an_item_fills_in_its_gold_spans() -> None:
    parsed = parse(f"Preamble. {CLEAN}")
    resolved, log = AnchorResolver().resolve_item(item(anchor("thirty days")), {"contract": parsed})

    assert len(resolved.gold) == 1
    assert parsed.document.slice(resolved.gold[0].span) == "thirty days"
    assert not log.of_code(WarningCode.ANCHOR_NOT_FOUND)


def test_the_grade_carries_from_the_anchor_to_the_span() -> None:
    parsed = parse(CLEAN)
    resolved, _ = AnchorResolver().resolve_item(
        item(anchor("thirty days", grade=1)), {"contract": parsed}
    )
    assert resolved.gold[0].grade == 1


def test_an_item_without_anchors_is_returned_untouched() -> None:
    """Authored against a fixed text, its spans are already right for it."""
    from contextgrid.core.evalset import GoldSpan

    original = EvalItem(id="q1", question="q", gold=(GoldSpan(Span("contract", 0, 10)),))
    resolved, log = AnchorResolver().resolve_item(original, {})
    assert resolved is original
    assert not log


def test_a_missing_parse_is_reported_rather_than_ignored() -> None:
    resolved, log = AnchorResolver().resolve_item(item(anchor("thirty days")), {})
    assert resolved.gold == ()
    assert log.of_code(WarningCode.NO_PARSE_FOR_SOURCE)


def test_evidence_a_parser_lost_is_reported_as_a_fact_about_the_parser() -> None:
    """Not a bug in the eval set. A parser that loses the evidence cannot retrieve it, and
    that is exactly what the parser axis exists to measure."""
    mangled = parse("Either party may term|nate th|s agreement", parser="bad-ocr")
    resolved, log = AnchorResolver().resolve_item(
        item(anchor("terminate this agreement")), {"contract": mangled}
    )

    assert resolved.gold == ()
    assert not resolved.is_answerable
    lost = log.of_code(WarningCode.ANCHOR_NOT_FOUND)
    assert lost
    assert "bad-ocr" in lost[0].message


def test_a_reflowed_match_is_noted_but_not_treated_as_a_problem() -> None:
    resolved, log = AnchorResolver().resolve_item(
        item(anchor("for convenience by giving thirty days")),
        {"contract": parse(REFLOWED, parser="pymupdf")},
    )
    assert resolved.is_answerable
    assert log.of_code(WarningCode.ANCHOR_NORMALISED)
    assert log.is_sound


def test_an_ambiguous_anchor_says_how_to_disambiguate_it() -> None:
    _, log = AnchorResolver().resolve_item(
        item(anchor("Confidential.")), {"contract": parse(REPEATED)}
    )
    warnings = log.of_code(WarningCode.ANCHOR_AMBIGUOUS)
    assert warnings
    assert "occurrence" in warnings[0].message


def test_an_explicit_occurrence_silences_the_ambiguity_warning() -> None:
    _, log = AnchorResolver().resolve_item(
        item(anchor("Confidential.", occurrence=1)), {"contract": parse(REPEATED)}
    )
    assert not log.of_code(WarningCode.ANCHOR_AMBIGUOUS)


# ---------------------------------------------------------------------------
# the point of the whole layer
# ---------------------------------------------------------------------------


def test_one_eval_set_resolves_against_two_different_parsers() -> None:
    """This is what makes the parser axis a comparison rather than a re-annotation exercise.

    The same authored eval set, resolved twice, gives correct offsets into two texts that
    are not the same text.
    """
    evalset = EvalSet(
        id="es",
        items=(
            item(anchor("thirty days written notice"), iid="q1"),
            item(anchor("terminate this agreement"), iid="q2"),
        ),
    )

    clean = parse(f"Preamble. {CLEAN}", parser="markdown")
    reflowed = parse(REFLOWED, parser="pymupdf")

    for parsed in (clean, reflowed):
        resolved, _ = AnchorResolver().resolve(evalset, {"contract": parsed})
        for resolved_item in resolved:
            for gold in resolved_item.gold:
                # Whatever the text looks like, the span points at the real evidence in it.
                found = " ".join(parsed.document.slice(gold.span).split())
                expected = " ".join(resolved_item.anchors[0].quote.split())
                assert found == expected


def test_a_parser_that_loses_evidence_is_summarised_at_the_set_level() -> None:
    evalset = EvalSet(
        id="es",
        items=(
            item(anchor("thirty days written notice"), iid="q1"),
            item(anchor("this text is simply absent"), iid="q2"),
        ),
    )
    _, log = AnchorResolver().resolve(evalset, {"contract": parse(CLEAN, parser="lossy")})

    summary = log.of_code(WarningCode.ANCHOR_NOT_FOUND)[-1]
    assert "lost 1 of 2" in summary.message
    assert summary.detail == {"lost": 1, "total": 2}


def test_resolving_a_set_keeps_its_identity() -> None:
    evalset = EvalSet(id="es", items=(item(anchor("thirty days")),), version=4, source="auto")
    resolved, _ = AnchorResolver().resolve(evalset, {"contract": parse(CLEAN)})
    assert resolved.id == "es"
    assert resolved.version == 4
    assert resolved.source == "auto"


def test_an_empty_set_resolves_to_an_empty_set() -> None:
    resolved, log = AnchorResolver().resolve(EvalSet(id="es", items=()), {})
    assert len(resolved) == 0
    assert not log


@pytest.mark.parametrize("quote", ["", "   "])
def test_a_blank_quote_cannot_be_constructed(quote: str) -> None:
    from contextgrid.core.errors import EvalSetError

    with pytest.raises(EvalSetError):
        anchor(quote)

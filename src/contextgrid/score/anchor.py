"""Locating portable ground truth inside a particular parse.

This is the layer that makes the parser axis possible.

Character-span ground truth compares *chunkers* perfectly well, because every chunker cuts up
the same text. It cannot survive a change of *parser*, because two parsers produce different
text -- and comparing parsers is the headline feature of this tool. So ground truth is
authored as a `GoldAnchor`: a quoted passage, optionally with a page hint. Resolving it
against a parse turns it into the `GoldSpan` everything downstream already understands.

The failure case is the interesting one. When a parser mangles a table so badly that the
quoted evidence no longer appears in its output, the anchor does not resolve -- and that is
not a bug in the eval set, it is the measurement. A parser that loses the evidence cannot
retrieve it, and the run should say so out loud rather than quietly scoring zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from contextgrid.core.documents import ParsedDocument
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span
from contextgrid.core.warnings import Severity, WarningCode, WarningLog


class MatchStrategy(str, Enum):
    """How an anchor was located, in decreasing order of confidence.

    Recorded on every match and surfaced in the results, because a run where half the gold
    was located by a heuristic deserves to be read differently from one where it was all
    found verbatim.
    """

    EXACT = "exact"
    NORMALISED = "normalised"
    BOUNDED = "bounded"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class AnchorMatch:
    """Where one anchor ended up in one parse, and how sure we are about it.

    `candidates` is how many times the quote was located, and it is filled in whether or not
    a span came back. That is the whole point: a failure with `candidates == 0` is evidence
    the parse does not contain, and a failure with `candidates > 0` is evidence it does
    contain, under an `occurrence` that names a repetition which is not there. The two used
    to arrive downstream as the same `None` and were reported with the same sentence.
    """

    anchor: GoldAnchor
    span: Span | None
    strategy: MatchStrategy
    candidates: int = 0

    @property
    def found(self) -> bool:
        return self.span is not None

    @property
    def occurrence_out_of_range(self) -> bool:
        """The quote is in the text; `anchor.occurrence` counts past the last copy of it."""
        return self.span is None and self.candidates > 0

    @property
    def is_verbatim(self) -> bool:
        return self.strategy is MatchStrategy.EXACT

    def to_gold(self) -> GoldSpan | None:
        return None if self.span is None else GoldSpan(span=self.span, grade=self.anchor.grade)


@dataclass(frozen=True, slots=True)
class AnchorResolver:
    """Finds quoted evidence in a parse.

    Three strategies are tried in order, and the first that succeeds is recorded:

    **exact** -- the quote appears verbatim. The common case for born-digital text.

    **normalised** -- the quote appears once runs of whitespace are collapsed on both sides.
    Extremely common with PDFs, where extraction inserts line breaks mid-sentence and turns
    a single space into three. The span returned still points at real characters in the
    original text; only the comparison was relaxed.

    **bounded** -- the quote's opening and closing words are both present, in order, close
    enough together to plausibly be the same passage. This catches the case where a parser
    corrupted something in the middle of the evidence -- a table cell, a ligature, a footnote
    marker -- while keeping the ends intact. It is a heuristic and it is off by default,
    because a wrong span is worse than a missing one: a missing anchor is visible, a wrong
    one silently scores the wrong text as correct.
    """

    allow_normalised: bool = True
    allow_bounded: bool = False
    case_sensitive: bool = True
    #: How much longer than the quote a bounded match may be before it is rejected.
    bounded_slack: float = 1.5
    #: Words taken from each end of the quote when matching by boundary.
    boundary_words: int = 3

    # -- one anchor ----------------------------------------------------------

    def locate(self, anchor: GoldAnchor, parsed: ParsedDocument) -> AnchorMatch:
        """Find one anchor in one parse.

        Every strategy reports how many copies of the quote it saw, even when none of them
        was the one asked for. A failure then carries the count, so the caller can tell
        "this parse does not contain the evidence" from "it contains two copies and the
        anchor asked for the eighth".
        """
        quote = anchor.quote.strip()
        text = parsed.text
        if not quote or not text:
            return AnchorMatch(anchor, None, MatchStrategy.NOT_FOUND)

        #: The most copies any strategy managed to see. Taken across strategies rather than
        #: from the last one tried, because a quote found twice by exact match and not at all
        #: by the normalised pass was still found twice.
        seen = 0

        exact, exact_seen = self._find_exact(quote, text, anchor, parsed)
        seen = max(seen, exact_seen)
        if exact is not None:
            return exact

        if self.allow_normalised:
            normalised, normalised_seen = self._find_normalised(quote, text, anchor, parsed)
            seen = max(seen, normalised_seen)
            if normalised is not None:
                return normalised

        if self.allow_bounded:
            bounded = self._find_bounded(quote, text, anchor)
            if bounded is not None:
                return bounded

        return AnchorMatch(anchor, None, MatchStrategy.NOT_FOUND, candidates=seen)

    def _find_exact(
        self, quote: str, text: str, anchor: GoldAnchor, parsed: ParsedDocument
    ) -> tuple[AnchorMatch | None, int]:
        """The match, and how many copies of the quote were there to choose from."""
        haystack, needle = self._case(text), self._case(quote)
        spans = [
            Span(parsed.id, start, start + len(quote))
            for start in _all_occurrences(haystack, needle)
        ]
        chosen = self._choose(spans, anchor, parsed)
        if chosen is None:
            return None, len(spans)
        return AnchorMatch(anchor, chosen, MatchStrategy.EXACT, candidates=len(spans)), len(spans)

    def _find_normalised(
        self, quote: str, text: str, anchor: GoldAnchor, parsed: ParsedDocument
    ) -> tuple[AnchorMatch | None, int]:
        flat_text, positions = collapse_whitespace(text)
        flat_quote, _ = collapse_whitespace(quote)
        flat_quote = flat_quote.strip()
        if not flat_quote:
            return None, 0

        haystack, needle = self._case(flat_text), self._case(flat_quote)
        spans: list[Span] = []
        for start in _all_occurrences(haystack, needle):
            end = start + len(flat_quote)
            spans.append(Span(parsed.id, positions[start], positions[end - 1] + 1))

        chosen = self._choose(spans, anchor, parsed)
        if chosen is None:
            return None, len(spans)
        return (
            AnchorMatch(anchor, chosen, MatchStrategy.NORMALISED, candidates=len(spans)),
            len(spans),
        )

    def _find_bounded(self, quote: str, text: str, anchor: GoldAnchor) -> AnchorMatch | None:
        """Match on the opening and closing words, tolerating damage in between."""
        words = quote.split()
        if len(words) < self.boundary_words * 2:
            return None

        flat_text, positions = collapse_whitespace(text)
        head = self._case(" ".join(words[: self.boundary_words]))
        tail = self._case(" ".join(words[-self.boundary_words :]))
        haystack = self._case(flat_text)

        start = haystack.find(head)
        if start == -1:
            return None
        tail_start = haystack.find(tail, start + len(head))
        if tail_start == -1:
            return None

        end = tail_start + len(tail)
        if end - start > len(quote) * self.bounded_slack:
            # Almost certainly two unrelated occurrences rather than one damaged passage.
            return None

        span = Span(anchor.source_id, positions[start], positions[end - 1] + 1)
        return AnchorMatch(anchor, span, MatchStrategy.BOUNDED, candidates=1)

    # -- picking between candidates ------------------------------------------

    def _choose(
        self, spans: Sequence[Span], anchor: GoldAnchor, parsed: ParsedDocument
    ) -> Span | None:
        if not spans:
            return None

        if anchor.page_hint is not None:
            on_page = [s for s in spans if parsed.page_at(s.start) == anchor.page_hint]
            if on_page:
                spans = on_page

        if anchor.occurrence < len(spans):
            return spans[anchor.occurrence]
        return None

    def _case(self, value: str) -> str:
        return value if self.case_sensitive else value.casefold()

    # -- whole items and eval sets -------------------------------------------

    def resolve_item(
        self, item: EvalItem, parses: Mapping[str, ParsedDocument]
    ) -> tuple[EvalItem, WarningLog]:
        """Resolve one question's anchors against a set of parses, one per source file.

        An item with no anchors is returned unchanged -- it was authored against a fixed
        text and its spans are already correct for it.
        """
        log = WarningLog()
        if not item.anchors:
            return item, log

        gold: list[GoldSpan] = []
        for anchor in item.anchors:
            parsed = parses.get(anchor.source_id)
            if parsed is None:
                log.add(
                    WarningCode.NO_PARSE_FOR_SOURCE,
                    f"no parse for source {anchor.source_id!r}, so the evidence for "
                    f"{item.id!r} cannot be located",
                    severity=Severity.CAUTION,
                    stage="anchor",
                    subject=item.id,
                    source_id=anchor.source_id,
                )
                continue

            match = self.locate(anchor, parsed)
            self._record(match, item, parsed, log)
            resolved = match.to_gold()
            if resolved is not None:
                gold.append(resolved)

        return item.resolved_with(tuple(gold)), log

    def resolve(
        self, evalset: EvalSet, parses: Mapping[str, ParsedDocument]
    ) -> tuple[EvalSet, WarningLog]:
        """Resolve a whole eval set against one parser's reading of the corpus.

        Run this once per parser. The same eval set, re-resolved, is what makes the parser
        axis a fair comparison instead of a re-annotation exercise.
        """
        log = WarningLog()
        items: list[EvalItem] = []
        for item in evalset:
            resolved, item_log = self.resolve_item(item, parses)
            items.append(resolved)
            log.extend(item_log)

        failures = log.of_code(WarningCode.ANCHOR_NOT_FOUND)
        if failures:
            parser = next(iter(parses.values())).parser if parses else "unknown"
            absent = sum(1 for w in failures if w.detail.get("reason") == _NOT_PRESENT)
            misindexed = sum(1 for w in failures if w.detail.get("reason") == _OUT_OF_RANGE)
            log.add(
                WarningCode.ANCHOR_NOT_FOUND,
                _aggregate_message(
                    parser, absent=absent, misindexed=misindexed, total=_anchor_count(evalset)
                ),
                severity=Severity.CAUTION,
                stage="anchor",
                subject=parser,
                lost=absent,
                out_of_range=misindexed,
                total=_anchor_count(evalset),
            )

        return evalset.with_items(tuple(items)), log

    def _record(
        self,
        match: AnchorMatch,
        item: EvalItem,
        parsed: ParsedDocument,
        log: WarningLog,
    ) -> None:
        quote = _abbreviate(match.anchor.quote)

        if match.occurrence_out_of_range:
            # "does not appear" was wrong, and wrong in the direction that costs the most
            # time: it sent somebody to check whether the parser had mangled a passage that
            # the parser had in fact read perfectly. The quote is there; the index is not.
            # `/evalsets/overview` says only that `occurrence` "picks which repetition of
            # `quote` is meant", so the count and the highest usable index both belong here.
            copies = match.candidates
            how_often = "once" if copies == 1 else f"{copies} times"
            log.add(
                WarningCode.ANCHOR_NOT_FOUND,
                f"the evidence for {item.id!r} appears {how_often} in {parsed.parser!r}'s "
                f"reading of {match.anchor.source_id!r}, but the anchor asks for occurrence "
                f"{match.anchor.occurrence}, which is out of range: they are numbered from 0, "
                f"so the last one is {copies - 1}. Nothing was scored for it: {quote}",
                severity=Severity.CAUTION,
                stage="anchor",
                subject=item.id,
                parser=parsed.parser,
                reason=_OUT_OF_RANGE,
                occurrence=match.anchor.occurrence,
                found=copies,
            )
            return

        if not match.found:
            log.add(
                WarningCode.ANCHOR_NOT_FOUND,
                f"the evidence for {item.id!r} does not appear in {parsed.parser!r}'s "
                f"reading of {match.anchor.source_id!r}: {quote}",
                severity=Severity.CAUTION,
                stage="anchor",
                subject=item.id,
                parser=parsed.parser,
                reason=_NOT_PRESENT,
            )
            return

        if match.strategy is MatchStrategy.NORMALISED:
            log.add(
                WarningCode.ANCHOR_NORMALISED,
                f"the evidence for {item.id!r} was found only after collapsing whitespace, "
                f"so {parsed.parser!r} reflowed it: {quote}",
                # CAUTION, not INFO, and the reason is where INFO ends up rather than what it
                # means. The CLI drops INFO warnings whenever a run produced results -- which is
                # always, for this one -- so at INFO a command-line user was never told. The
                # hard anchor failures printed loudly right beside it, which made the silence
                # read as "your evidence matched literally" when it had in fact been reflowed
                # to fit. Markdown hard-wraps, so this is the common case, not the exotic one.
                severity=Severity.CAUTION,
                stage="anchor",
                subject=item.id,
                parser=parsed.parser,
            )
        elif match.strategy is MatchStrategy.BOUNDED:
            log.add(
                WarningCode.ANCHOR_BOUNDED,
                f"the evidence for {item.id!r} was matched on its opening and closing words "
                f"only, so {parsed.parser!r} corrupted something in the middle: {quote}",
                severity=Severity.CAUTION,
                stage="anchor",
                subject=item.id,
                parser=parsed.parser,
            )

        if match.candidates > 1 and match.anchor.occurrence == 0:
            log.add(
                WarningCode.ANCHOR_AMBIGUOUS,
                f"the evidence for {item.id!r} appears {match.candidates} times in "
                f"{match.anchor.source_id!r}; using the first. Set `occurrence` or a "
                f"`page_hint` to say which one is meant",
                severity=Severity.CAUTION,
                stage="anchor",
                subject=item.id,
                candidates=match.candidates,
            )


# ---------------------------------------------------------------------------
# whitespace normalisation with an offset map
# ---------------------------------------------------------------------------


def collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse runs of whitespace to one space, and map each result character back.

    `positions[i]` is the index in the original text of the character that became
    `collapsed[i]`. That map is what lets a match found in the tidied text be reported as a
    span in the real one -- the comparison is relaxed, the offsets never are.
    """
    collapsed: list[str] = []
    positions: list[int] = []
    previous_was_space = False

    for index, char in enumerate(text):
        if char.isspace():
            if previous_was_space:
                continue
            collapsed.append(" ")
            positions.append(index)
            previous_was_space = True
        else:
            collapsed.append(char)
            positions.append(index)
            previous_was_space = False

    return "".join(collapsed), positions


def _all_occurrences(haystack: str, needle: str) -> list[int]:
    """Every start index at which `needle` appears, including overlapping ones."""
    if not needle:
        return []
    found: list[int] = []
    cursor = haystack.find(needle)
    while cursor != -1:
        found.append(cursor)
        cursor = haystack.find(needle, cursor + 1)
    return found


#: Why one anchor produced no gold, recorded on the warning's `detail` so the summary can
#: count the two apart instead of adding them together.
_NOT_PRESENT = "not_present"
_OUT_OF_RANGE = "occurrence_out_of_range"


def _aggregate_message(parser: str, *, absent: int, misindexed: int, total: int) -> str:
    """One line for a whole eval set, claiming only what the counts can support.

    The old wording ended "which is a fact about the parser, not the eval set". It is not.
    The same warning fires for a quote nobody wrote into the document, for a quote naming the
    wrong `source_id`, and for an `occurrence` past the last copy -- and the resolver cannot
    tell an invented quote from a mangled table, because in both cases the text simply is not
    there. So the sentence that can be defended is the one about the measurement: these
    questions cannot be scored. Who to blame is left open where it is open, and named where
    the code genuinely knows it -- an out-of-range index is proof the text was found.
    """
    sentences: list[str] = []

    if absent:
        sentences.append(
            f"parser {parser!r} could not locate {absent} of {total} {_pieces(total)} of "
            "evidence. Those questions cannot be answered under this parse, whatever the "
            "retriever does"
        )
        sentences.append(
            "Why is not something this can tell: either the parser lost the text, or the eval "
            "set quotes something that is not in the document it names. Check one of the "
            "quotes against the file before blaming either"
        )

    # "a further N" once the sentence above has already said how many there were in total,
    # "N of total" when this is the only sentence in the message.
    if misindexed:
        count = (
            f"a further {misindexed} {_pieces(misindexed)}"
            if absent
            else f"{misindexed} of {total} {_pieces(total)}"
        )
        sentences.append(
            f"{parser!r} did find {count} of evidence and skipped "
            f"{'them' if misindexed > 1 else 'it'} anyway: the `occurrence` on "
            f"{'those anchors counts' if misindexed > 1 else 'that anchor counts'} past the "
            "last copy of the quote in the document. That part is an eval set fix, not a "
            "parser problem, and those questions score nothing until it is made"
        )

    return ". ".join(sentences)


def _pieces(count: int) -> str:
    return "piece" if count == 1 else "pieces"


def _anchor_count(evalset: EvalSet) -> int:
    return sum(len(item.anchors) for item in evalset)


def _abbreviate(quote: str, limit: int = 60) -> str:
    flat = " ".join(quote.split())
    return repr(flat if len(flat) <= limit else flat[: limit - 1] + "…")

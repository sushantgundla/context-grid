"""The review queue.

Ragas generates eval sets. AutoRAG generates eval sets. RAGBuilder generates eval sets. None
of them let you fix the bad ones, and everyone in the field knows auto-generated ground truth
is the weak link. This is the missing piece.

The design constraint is the only one that matters: **under five seconds per question**. A
reviewer who has to think about the interface will do thirty and stop, and thirty reviewed
questions is not an eval set. So the state lives here, in a plain object with one method per
keystroke, and the terminal or the browser is a thin thing on top that can be swapped without
touching any of this.

Keeping the state machine separate also means it can be tested without a terminal, which is
the difference between this being covered and being hoped about.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from contextgrid.core.errors import EvalSetError
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor


class Verdict(str, Enum):
    """What a reviewer decided about one question."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Decision:
    """One reviewer action, kept so it can be undone and so the set records who agreed."""

    item_id: str
    verdict: Verdict
    note: str = ""
    replacement: EvalItem | None = None


@dataclass(slots=True)
class ReviewQueue:
    """Questions awaiting judgement, and the decisions made so far.

    Every method is one keystroke's worth of work. Nothing here does I/O.
    """

    items: list[EvalItem]
    position: int = 0
    decisions: dict[str, Decision] = field(default_factory=dict)
    history: list[Decision] = field(default_factory=list)

    @classmethod
    def from_evalset(cls, evalset: EvalSet, *, skip_reviewed: bool = True) -> ReviewQueue:
        """Build a queue, optionally leaving out questions already looked at."""
        items = [item for item in evalset if not (skip_reviewed and item.meta.get("reviewed"))]
        return cls(items=items)

    # -- where we are --------------------------------------------------------

    @property
    def current(self) -> EvalItem | None:
        """The question in front of the reviewer, or None when the queue is done."""
        if 0 <= self.position < len(self.items):
            return self.items[self.position]
        return None

    @property
    def remaining(self) -> int:
        return max(0, len(self.items) - self.position)

    @property
    def reviewed(self) -> int:
        return len(self.decisions)

    @property
    def is_done(self) -> bool:
        return self.position >= len(self.items)

    def progress(self) -> str:
        return f"{self.position} of {len(self.items)} · {self.remaining} left"

    def counts(self) -> dict[str, int]:
        counts = {verdict.value: 0 for verdict in Verdict if verdict is not Verdict.PENDING}
        for decision in self.decisions.values():
            counts[decision.verdict.value] += 1
        return counts

    # -- one keystroke each --------------------------------------------------

    def accept(self, note: str = "") -> EvalItem:
        """Keep this question as it stands."""
        return self._decide(Verdict.ACCEPTED, note)

    def reject(self, note: str = "") -> EvalItem:
        """Drop this question. The note is why, and it is worth writing."""
        return self._decide(Verdict.REJECTED, note)

    def skip(self, note: str = "") -> EvalItem:
        """Leave it undecided and move on. Skipped questions stay out of the final set."""
        return self._decide(Verdict.SKIPPED, note)

    def edit(
        self,
        *,
        question: str | None = None,
        quote: str | None = None,
        qtype: str | None = None,
        grade: int | None = None,
        note: str = "",
    ) -> EvalItem:
        """Fix the question, the evidence, the type or the grade, and keep it.

        Editing the quote is the common case: the generator picked a passage that only
        half answers the question, and moving it is a two-second fix that turns a rejected
        question into a good one.
        """
        item = self._require_current()
        updated = item

        if question is not None:
            updated = replace(updated, question=question)
        if qtype is not None:
            updated = replace(updated, qtype=qtype)
        if quote is not None or grade is not None:
            updated = replace(updated, anchors=_revise_anchors(updated, quote, grade))

        return self._decide(Verdict.EDITED, note, replacement=updated)

    def mark(self, qtype: str, note: str = "") -> EvalItem:
        """Accept the question but correct its type. The most common single-key correction."""
        return self.edit(qtype=qtype, note=note)

    def undo(self) -> EvalItem | None:
        """Take back the last decision and go back to that question.

        Non-negotiable in a queue running at five seconds an item: the reviewer will hit the
        wrong key, and without an undo they will slow down to avoid it.
        """
        if not self.history:
            return None
        last = self.history.pop()
        self.decisions.pop(last.item_id, None)
        self.position = max(0, self.position - 1)
        return self.current

    def _decide(self, verdict: Verdict, note: str, replacement: EvalItem | None = None) -> EvalItem:
        item = self._require_current()
        decision = Decision(item_id=item.id, verdict=verdict, note=note, replacement=replacement)
        self.decisions[item.id] = decision
        self.history.append(decision)
        self.position += 1
        return replacement or item

    def _require_current(self) -> EvalItem:
        item = self.current
        if item is None:
            raise EvalSetError("the review queue is finished; there is nothing to decide on")
        return item

    # -- the result ----------------------------------------------------------

    def result(self, original: EvalSet) -> EvalSet:
        """The reviewed eval set: accepted and edited questions, marked as reviewed.

        Rejected and skipped questions are left out. Undecided ones are kept as they were,
        so stopping halfway through a queue loses nothing.
        """
        kept: list[EvalItem] = []
        for item in original:
            decision = self.decisions.get(item.id)
            if decision is None:
                kept.append(item)
                continue
            if decision.verdict in {Verdict.REJECTED, Verdict.SKIPPED}:
                continue
            final = decision.replacement or item
            kept.append(
                replace(
                    final,
                    meta={**final.meta, "reviewed": True, "verdict": decision.verdict.value},
                )
            )

        return replace(original, items=tuple(kept), version=original.version + 1)

    def rejections(self) -> list[Decision]:
        """Everything dropped, with the reasons. Worth reading before regenerating."""
        return [d for d in self.decisions.values() if d.verdict is Verdict.REJECTED]

    def __iter__(self) -> Iterator[EvalItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _revise_anchors(item: EvalItem, quote: str | None, grade: int | None) -> tuple[GoldAnchor, ...]:
    """Apply an edit to the first anchor, which is the one on screen."""
    if not item.anchors:
        raise EvalSetError(
            f"item {item.id!r} has no evidence to edit. Reject it, or add an anchor first."
        )
    first = item.anchors[0]
    changes: dict[str, Any] = {}
    if quote is not None:
        changes["quote"] = quote
    if grade is not None:
        changes["grade"] = grade
    return (replace(first, **changes), *item.anchors[1:])


def review_summary(queue: ReviewQueue, elapsed_seconds: float | None = None) -> str:
    """What the session achieved, for the end of a review run."""
    counts = queue.counts()
    parts = [
        f"{queue.reviewed} reviewed",
        f"{counts['accepted']} kept",
        f"{counts['edited']} fixed",
        f"{counts['rejected']} dropped",
    ]
    if counts["skipped"]:
        parts.append(f"{counts['skipped']} skipped")
    if elapsed_seconds and queue.reviewed:
        parts.append(f"{elapsed_seconds / queue.reviewed:.1f}s per question")
    return ", ".join(parts)


def pending(evalset: EvalSet) -> Sequence[EvalItem]:
    """Questions nobody has looked at yet."""
    return [item for item in evalset if not item.meta.get("reviewed")]

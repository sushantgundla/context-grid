"""Scoring the eval set itself.

Every conclusion this tool produces rests on the ground truth underneath it, and nothing in
the field tells a user that theirs is too weak to support what they are about to claim. A
sweep over 12 auto-generated questions will happily report that one configuration beats
another by 0.04, and that number means nothing whatsoever.

So the eval set gets scored too: how big it is, how much of it a human has actually looked at,
whether the question types are balanced, how much of it separates configurations at all --
and, most usefully, the smallest difference it could detect if there were one.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from contextgrid.core.evalset import EvalSet, QuestionType
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.evalset.classify import type_distribution

#: Below this, a difference has to be very large before it means anything.
SMALL_SET = 30
#: Around here, differences of about 0.1 become detectable.
COMFORTABLE_SET = 100
#: Decimal places every message about the noise floor prints it to.
FLOOR_PLACES = 2


def round_up(value: float, places: int = FLOOR_PLACES) -> float:
    """`value` rounded *away from zero* to `places` decimals.

    Up rather than to-nearest, because the number this rounds is a floor. Printing 0.40 for a
    real floor of 0.404145 states something the set cannot do; printing 0.41 states something
    it can, and errs by less than a hundredth.

    The `round` inside the `ceil` is not decoration: `0.28` is held as 0.28000000000000003, so
    a bare `ceil(value * 100)` would report a 24-question set's floor as 0.29 on the strength
    of a rounding error in the seventeenth decimal place.
    """
    # Annotated because `10**places` is `Any` to a type checker -- a negative exponent would
    # make it a float -- and an `Any` here would silently widen the return type.
    factor: int = 10**places
    return math.ceil(round(value * factor, 9)) / factor


def minimum_detectable_difference(n: int, *, power: float = 0.8, alpha: float = 0.05) -> float:
    """The smallest difference in a proportion this many questions could detect.

    Standard two-sided test at the worst-case variance (p = 0.5), which is the honest
    assumption when you do not yet know what the scores will be. Approximate, and its job is
    to make the size of an eval set concrete: "you can detect 0.28" lands harder than
    "n is small".

    Returns 1.0 for a set too small to detect anything short of a total reversal.
    """
    if n < 2:
        return 1.0
    z_alpha = 1.96 if alpha <= 0.05 else 1.645
    z_power = 0.84 if power <= 0.8 else 1.28
    difference = (z_alpha + z_power) * math.sqrt(2 * 0.25 / n)
    return min(1.0, difference)


@dataclass(frozen=True, slots=True)
class EvalSetQuality:
    """What this eval set can and cannot support."""

    size: int
    answerable: int
    reviewed: int
    portable: int
    types: dict[str, int] = field(default_factory=dict)
    mean_discriminating_power: float | None = None
    non_discriminating: int = 0

    @property
    def reviewed_fraction(self) -> float:
        return self.reviewed / self.size if self.size else 0.0

    @property
    def unanswerable(self) -> int:
        """Questions with no evidence. Some are deliberate; too many is a generator problem."""
        return self.size - self.answerable

    @property
    def detectable_difference(self) -> float:
        return minimum_detectable_difference(self.answerable)

    @property
    def reported_detectable_difference(self) -> float:
        """The noise floor as every message here prints it: rounded up to two places.

        `summary()` used to print `{detectable_difference:.2f}`, which rounds to *nearest*.
        On 24 questions the real floor is 0.404145, so the sentence read "detects differences
        of 0.40 and above" -- and `can_support(0.40)` then returned False, because it tests
        the unrounded value. Somebody read a number out of the tool's own summary, handed it
        straight back, and was told no.

        Rounding up is the half of that pair worth fixing. Rounding the *predicate* down to
        match the print would make `can_support` agree by promising a resolution the eval set
        has not got, which is the failure this whole module exists to prevent.
        """
        return round_up(self.detectable_difference)

    @property
    def is_portable(self) -> bool:
        """True when every answerable question can be re-resolved against another parser."""
        return self.portable == self.answerable

    def can_support(self, difference: float) -> bool:
        """Whether a claimed difference of this size is bigger than the noise floor.

        Tested against the exact floor, not the printed one. The number in `summary()` is
        rounded up precisely so that handing it back here answers True.
        """
        return difference >= self.detectable_difference

    def summary(self) -> str:
        # "with evidence", not "answerable". Nothing here has seen a corpus: this counts the
        # questions that carry an anchor, and says nothing about whether that anchor's quote can
        # actually be found in any document. A stranger read "14 answerable" and reasonably took
        # it to mean 14 questions could be scored, while one of them quoted a sentence that
        # appears nowhere at all. The word the tool uses for *scorable* had been lent to a
        # weaker claim. `contextgrid run` is where evidence meets documents, and it says so
        # there with `anchor_not_found`.
        parts = [
            f"{self.size} questions ({self.answerable} with evidence, unchecked against a corpus)",
            f"{self.reviewed_fraction:.0%} reviewed",
            f"detects differences of {self.reported_detectable_difference:.2f} and above",
        ]
        return ", ".join(parts)

    def warnings(self) -> WarningLog:
        """Everything about this set that should change how its results are read.

        Nothing here says "answerable" either, and that is the same fix `summary()` got in
        0.9.1 rather than a second opinion about wording. That release retired the word from
        the line above these on the grounds that the codebase spends it on *can be scored*,
        while this number counts *carries an anchor* -- and then left it in the caution
        printed directly beneath, where it went on making the identical claim about the
        identical count. The field keeps its name: `answerable` is what the attribute is
        called, `EvalSetQuality`'s repr is documented with it, and `detail["answerable"]` is
        read by things that are not prose. Only the sentences overstated the case.
        """
        log = WarningLog()

        if self.answerable < SMALL_SET:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{self.answerable} questions carry evidence, unchecked against a corpus, so "
                f"this set can only detect differences of about "
                f"{self.reported_detectable_difference:.2f} or larger. Anything smaller than "
                "that on a leaderboard built from this set is noise",
                severity=Severity.CAUTION,
                stage="evalset",
                answerable=self.answerable,
            )
        elif self.answerable < COMFORTABLE_SET:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{self.answerable} questions carry evidence, unchecked against a corpus, so "
                f"this set detects differences of about "
                f"{self.reported_detectable_difference:.2f}. Enough to choose between clearly "
                "different configurations, not enough to rank close ones",
                severity=Severity.INFO,
                stage="evalset",
            )

        if self.reviewed_fraction < 0.2 and self.size:
            # Says how to clear it, because it was previously unclearable. `reviewed` reads
            # `item.meta["reviewed"]`, which only the review queue ever wrote -- so a set
            # somebody had written by hand reported 0% forever and was told off on every run
            # for work it had already done. The flag was readable from the file all along;
            # nothing said so.
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"only {self.reviewed_fraction:.0%} of this set is marked as checked by a "
                "human. Ground truth nobody has read is the weakest link in any retrieval "
                "comparison. If you wrote these questions yourself, say so with "
                '`"meta": {"reviewed": true}` on each one; otherwise the review queue is the '
                "cheapest place to fix it",
                severity=Severity.CAUTION,
                stage="evalset",
                reviewed=self.reviewed,
            )

        if self.non_discriminating:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{self.non_discriminating} questions are answered perfectly by the baseline, "
                "so they separate nothing and raise every configuration's score equally",
                severity=Severity.CAUTION,
                stage="evalset",
            )

        if self.answerable and not self.is_portable:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{self.answerable - self.portable} questions carry character spans rather "
                "than quoted evidence. They compare chunkers correctly and cannot be "
                "re-resolved against a different parser, so the parser axis is not available",
                severity=Severity.CAUTION,
                stage="evalset",
            )

        dominant = _dominant_type(self.types)
        if dominant is not None:
            label, share = dominant
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{share:.0%} of this set is {label!r} questions. Results will describe how "
                "configurations handle that one kind of question and little else",
                severity=Severity.INFO,
                stage="evalset",
            )

        return log


def assess(
    evalset: EvalSet,
    *,
    baseline_scores: dict[str, float] | None = None,
    reviewed_key: str = "reviewed",
) -> EvalSetQuality:
    """Score an eval set, using a baseline run to judge discriminating power where available."""
    items = list(evalset)
    # `EvalItem.is_answerable` is the one rule for this, and this line used to be a second
    # copy of it that counted anchors when the property did not. Same file, two answers.
    answerable = [item for item in items if item.is_answerable]

    discriminating: list[float] = []
    non_discriminating = 0
    if baseline_scores:
        for item in answerable:
            score = baseline_scores.get(item.id)
            if score is None:
                continue
            # A question the baseline aces separates nothing; one it fails entirely may be
            # broken. Power is highest in the middle.
            discriminating.append(1.0 - abs(score - 0.5) * 2)
            if score >= 1.0:
                non_discriminating += 1

    return EvalSetQuality(
        size=len(items),
        answerable=len(answerable),
        reviewed=sum(1 for item in items if item.meta.get(reviewed_key)),
        portable=sum(1 for item in answerable if item.is_portable),
        types=type_distribution(evalset),
        mean_discriminating_power=(statistics.fmean(discriminating) if discriminating else None),
        non_discriminating=non_discriminating,
    )


def _dominant_type(types: dict[str, int]) -> tuple[str, float] | None:
    """The type covering more than 70% of the set, if there is one."""
    total = sum(types.values())
    if total < 10:
        return None
    for label, count in types.items():
        if label == QuestionType.UNANSWERABLE:
            continue
        share = count / total
        if share > 0.7:
            return label, share
    return None

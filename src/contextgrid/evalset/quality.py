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
    def is_portable(self) -> bool:
        """True when every answerable question can be re-resolved against another parser."""
        return self.portable == self.answerable

    def can_support(self, difference: float) -> bool:
        """Whether a claimed difference of this size is bigger than the noise floor."""
        return difference >= self.detectable_difference

    def summary(self) -> str:
        parts = [
            f"{self.size} questions ({self.answerable} answerable)",
            f"{self.reviewed_fraction:.0%} reviewed",
            f"detects differences of {self.detectable_difference:.2f} and above",
        ]
        return ", ".join(parts)

    def warnings(self) -> WarningLog:
        """Everything about this set that should change how its results are read."""
        log = WarningLog()

        if self.answerable < SMALL_SET:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{self.answerable} answerable questions can only detect differences of about "
                f"{self.detectable_difference:.2f} or larger. Anything smaller than that on a "
                "leaderboard built from this set is noise",
                severity=Severity.CAUTION,
                stage="evalset",
                answerable=self.answerable,
            )
        elif self.answerable < COMFORTABLE_SET:
            log.add(
                WarningCode.SMALL_EVAL_SET,
                f"{self.answerable} answerable questions detect differences of about "
                f"{self.detectable_difference:.2f}. Enough to choose between clearly different "
                "configurations, not enough to rank close ones",
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
                f"{self.answerable - self.portable} answerable questions carry character spans "
                "rather than quoted evidence. They compare chunkers correctly and cannot be "
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
    # Evidence in either form counts: a freshly generated set has anchors and no
    # spans yet, and reporting it as zero answerable questions would be nonsense.
    answerable = [item for item in items if item.has_evidence]

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

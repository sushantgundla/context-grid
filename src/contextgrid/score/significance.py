"""Confidence intervals and paired significance testing.

A leaderboard reporting 0.71 against 0.68 invites exactly one conclusion, and on 40 questions
that conclusion is usually wrong. The difference is well inside what you would get by asking
40 different questions, and nothing on the screen says so.

Two configurations here are always run on **identical questions**, which is a gift: a paired
test uses each question as its own control and removes the enormous variance that comes from
some questions simply being harder than others. It is far more sensitive than comparing two
independent means, and it is the right test for this data.

The headline output is deliberately blunt. "These two configurations are not distinguishable
on this eval set (n=87)" is a more useful sentence than a p-value, and publishing it is worth
more than any leaderboard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from contextgrid.core.errors import ContextGridError

#: Enough for a stable interval without making a sweep feel slow.
DEFAULT_RESAMPLES = 2000


class SignificanceError(ContextGridError, ValueError):
    """A comparison was asked for that cannot be computed."""


@dataclass(frozen=True, slots=True)
class Interval:
    """An estimate and the range it could plausibly sit in."""

    estimate: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval is on one side of zero."""
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass(frozen=True, slots=True)
class Comparison:
    """Whether two configurations actually differ, and how sure we can be.

    `distinguishable` is the field to read. It is deliberately conservative: it requires both
    a small p-value and a confidence interval that stays on one side of zero, because two
    weak signals agreeing is a better basis for a decision than either alone.
    """

    left: str
    right: str
    metric: str
    n: int
    left_mean: float
    right_mean: float
    difference: Interval
    p_value: float
    alpha: float = 0.05
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def distinguishable(self) -> bool:
        return self.p_value < self.alpha and self.difference.excludes_zero

    @property
    def winner(self) -> str | None:
        if not self.distinguishable:
            return None
        return self.left if self.difference.estimate > 0 else self.right

    def verdict(self) -> str:
        """The result as a sentence somebody can act on.

        The negative case gets the longer explanation on purpose. A reader who sees "not
        distinguishable" needs to know what would change that, or they will simply ignore it
        and read the leaderboard order instead.
        """
        if not self.distinguishable:
            return (
                f"{self.left} and {self.right} are not distinguishable on this eval set "
                f"(n={self.n}). The gap of {self.difference.estimate:+.3f} on {self.metric} "
                f"sits inside the confidence interval {self.difference.low:+.3f} to "
                f"{self.difference.high:+.3f}, so it is consistent with no difference at all. "
                + _sample_size_note(self.difference.estimate, self.n, self.ties)
            )

        direction = "beats" if self.difference.estimate > 0 else "loses to"
        return (
            f"{self.left} {direction} {self.right} by {abs(self.difference.estimate):.3f} on "
            f"{self.metric} (95% CI {self.difference.low:+.3f} to {self.difference.high:+.3f}, "
            f"p={self.p_value:.3f}, n={self.n}). It wins on {self.wins} questions, loses on "
            f"{self.losses} and ties on {self.ties}."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "metric": self.metric,
            "n": self.n,
            "left_mean": self.left_mean,
            "right_mean": self.right_mean,
            "difference": self.difference.estimate,
            "ci_low": self.difference.low,
            "ci_high": self.difference.high,
            "p_value": self.p_value,
            "distinguishable": self.distinguishable,
            "winner": self.winner,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
        }


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------


def bootstrap_interval(
    scores: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """A confidence interval for a mean score, by resampling the questions.

    Bootstrapping rather than a normal approximation because per-question retrieval scores
    are nothing like normal -- recall@5 on a single question is one of a handful of discrete
    values, often just 0 or 1. Resampling makes no assumption about the shape.
    """
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        raise SignificanceError("cannot build an interval from no scores")
    if values.size == 1:
        single = float(values[0])
        return Interval(single, single, single, confidence)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)

    tail = (1 - confidence) / 2
    return Interval(
        estimate=float(values.mean()),
        low=float(np.quantile(means, tail)),
        high=float(np.quantile(means, 1 - tail)),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# two configurations
# ---------------------------------------------------------------------------


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """A confidence interval for the *difference*, resampling questions in pairs.

    Pairing is what makes this sensitive. Both configurations answered the same questions, so
    resampling them together holds question difficulty constant and leaves only the part of
    the variance that is actually about the configurations.
    """
    a, b = _aligned(left, right)
    differences = a - b

    if differences.size == 1:
        single = float(differences[0])
        return Interval(single, single, single, confidence)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[draws].mean(axis=1)

    tail = (1 - confidence) / 2
    return Interval(
        estimate=float(differences.mean()),
        low=float(np.quantile(means, tail)),
        high=float(np.quantile(means, 1 - tail)),
        confidence=confidence,
    )


def randomisation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    permutations: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> float:
    """A two-sided p-value from a paired randomisation test.

    The null hypothesis is that the two configurations are interchangeable, so on any given
    question their two scores could equally well have been swapped. Swapping them at random
    many times builds the distribution of differences you would see if that were true, and
    the p-value is how often chance alone beats what was actually observed.

    No distributional assumption, exact in the limit, and the standard test in information
    retrieval for exactly this situation.
    """
    a, b = _aligned(left, right)
    differences = a - b
    observed = abs(float(differences.mean()))

    if differences.size == 0:
        raise SignificanceError("cannot test two empty score sets")
    if np.allclose(differences, 0):
        return 1.0  # identical everywhere; there is nothing to detect

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, differences.size))
    means = np.abs((signs * differences).mean(axis=1))

    # The +1 on both sides is the standard correction: the observed arrangement is itself one
    # of the possible ones, and leaving it out can report p=0, which is never true.
    return float((np.count_nonzero(means >= observed) + 1) / (permutations + 1))


def compare(
    left_scores: Mapping[str, float],
    right_scores: Mapping[str, float],
    *,
    left: str = "left",
    right: str = "right",
    metric: str = "recall@5",
    alpha: float = 0.05,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Comparison:
    """Compare two configurations question by question.

    Only questions both configurations answered are used. Comparing on different question
    sets would confound the difference between the configurations with the difference between
    the questions, which is the thing pairing exists to remove.
    """
    shared = sorted(set(left_scores) & set(right_scores))
    if not shared:
        raise SignificanceError(
            f"{left!r} and {right!r} have no questions in common, so there is nothing to "
            "compare them on"
        )

    a = [left_scores[key] for key in shared]
    b = [right_scores[key] for key in shared]

    difference = paired_bootstrap(a, b, confidence=confidence, resamples=resamples, seed=seed)
    p_value = randomisation_test(a, b, permutations=resamples, seed=seed)

    wins = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    losses = sum(1 for x, y in zip(a, b, strict=True) if x < y)

    return Comparison(
        left=left,
        right=right,
        metric=metric,
        n=len(shared),
        left_mean=float(np.mean(a)),
        right_mean=float(np.mean(b)),
        difference=difference,
        p_value=p_value,
        alpha=alpha,
        wins=wins,
        losses=losses,
        ties=len(shared) - wins - losses,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _aligned(
    left: Sequence[float], right: Sequence[float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if a.size != b.size:
        raise SignificanceError(
            f"paired tests need one score per question on both sides, got {a.size} and {b.size}"
        )
    if a.size == 0:
        raise SignificanceError("cannot compare two empty score sets")
    return a, b


def _sample_size_note(difference: float, n: int, ties: int) -> str:
    """What it would take to settle this, as a finished sentence.

    Three genuinely different situations, and stitching one phrase into a template produced
    sentences like "About many more than questions would be needed" -- which is the kind of
    thing that makes a reader stop trusting everything around it.
    """
    gap = abs(difference)

    if gap < 1e-6:
        if ties == n:
            return (
                "They scored identically on every single question, so this is not a close "
                "call between two different configurations -- they are behaving the same way."
            )
        return (
            "They average exactly the same score while disagreeing on individual questions, "
            "so no number of questions like these would separate them."
        )

    # The same worst-case power calculation the eval-set quality score uses, so the two
    # numbers agree with each other.
    needed = int(((1.96 + 0.84) ** 2 * 2 * 0.25) / (gap**2)) + 1
    if needed <= n:
        return (
            f"A gap this size would usually be detectable with about {needed} questions, so "
            f"the {n} here are being defeated by how much the scores vary between them."
        )
    return f"About {needed} questions would be needed to settle a gap this size."

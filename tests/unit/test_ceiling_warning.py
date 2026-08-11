"""Saying so when a sweep ranked nothing because everything was already perfect.

The most expensive way to learn nothing: the baseline answers every question, no arm has any
headroom, the whole grid ties at 1.000, and the leaderboard reads like a clean result. A blind
evaluator hit exactly this -- three of six arms at 1.000 -- and called it "the one warning this
tool most needs and does not have".
"""

from __future__ import annotations

from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus
from contextgrid.grid import Runner, matrix

CORPUS = Corpus.from_texts(
    {
        f"doc{index}.md": (
            f"# Topic {index}\n\nThe distinctive answer for topic {index} is marker{index}.\n"
            + " ".join(f"padding{index}word{n}" for n in range(60))
            + "\n"
        )
        for index in range(8)
    },
    media_type=MediaType.MARKDOWN,
    name="ceiling",
)

EVALSET = EvalSet(
    id="ceiling",
    items=tuple(
        EvalItem(
            id=f"q{index}",
            question=f"what is the distinctive answer for topic {index}?",
            anchors=(
                GoldAnchor(
                    quote=f"The distinctive answer for topic {index} is marker{index}.",
                    source_id=f"doc{index}.md",
                ),
            ),
        )
        for index in range(8)
    ),
)


def ceiling_warnings(results: object) -> list[str]:
    return [
        warning.message
        for warning in results.warnings  # type: ignore[attr-defined]
        if warning.code is WarningCode.EVALSET_AT_CEILING
    ]


def test_a_sweep_where_everything_ties_at_the_top_says_so() -> None:
    """Questions this easy are answered by every arm, so the grid measured no difference
    rather than finding none -- and those are not the same claim."""
    results = Runner(corpus=CORPUS, headline="recall@5").run(
        matrix(
            chunker="recursive:128",
            embedder="tfidf",
            index="dense",
            ingestion=[None, "parent-document:2"],
        ),
        EVALSET,
        mode="factorial",
    )

    assert all(run.metric("recall@5") == 1.0 for run in results.runs)
    warnings = ceiling_warnings(results)
    assert warnings, "a sweep that ranked nothing said nothing about it"
    # It has to say what to do next, or it is just an observation.
    assert "harder questions" in warnings[0]
    assert "recall@1" in warnings[0]


def test_a_sweep_that_actually_separates_is_not_warned_about() -> None:
    """The guard against a warning nobody reads: it must be silent whenever the sweep did its
    job. `recall@1` on the same corpus and arms genuinely discriminates."""
    results = Runner(corpus=CORPUS, headline="recall@1").run(
        matrix(chunker=["recursive:128", "fixed:32"], embedder="tfidf", index="dense"),
        EVALSET,
        mode="factorial",
    )

    scores = {run.metric("recall@1") for run in results.runs}
    if len(scores) == 1 and scores.pop() >= 0.999:  # pragma: no cover - corpus-dependent
        return  # This corpus happened to tie here too; nothing to assert.
    assert not ceiling_warnings(results)


def test_one_configuration_alone_is_never_called_a_ceiling() -> None:
    """A single arm at 1.000 is a result, not a ceiling -- nothing was being compared, so
    there was no ranking to fail to produce."""
    results = Runner(corpus=CORPUS, headline="recall@5").run(
        matrix(chunker="recursive:128", embedder="tfidf", index="dense"),
        EVALSET,
        mode="factorial",
    )

    assert len(results.runs) == 1
    assert results.runs[0].metric("recall@5") == 1.0
    assert not ceiling_warnings(results)

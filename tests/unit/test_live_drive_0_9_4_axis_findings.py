"""Three axis faults a stranger driving the published 0.9.4 wheel found.

All three share a shape: the user named something on an axis, the tool accepted the name, and
then answered about something else without saying so.

1. `Lab.run(evalset, headline="recall@1")` reached `RunResult.headline` and stopped there.
   `Results` had no such field, so `summary()` and `best()` -- and every other view that takes
   a metric -- fell back to their own `recall@5` default. The paragraph then *named* the metric
   it had used, so a sentence reading "scored best on recall@5 at 0.615" answered a question
   nobody had asked, in the authoritative voice of one that had been.

2. The `expand` transform could not be configured from a spec string by any route.
   `expand` alone built `ExpandAcronyms(expansions={})`, which is the identity, so the arm
   scored a tie with plain search and the leaderboard still labelled it `+expand`.
   `expand:RPO=recovery point objective` parsed correctly and then died on a bare
   `TypeError: ExpandAcronyms.__init__() got an unexpected keyword argument 'RPO'`, and
   `expand:expansions=RPO` built happily with a str where a dict belongs and died later on a
   bare `AttributeError`. Every one of those is a builtins exception from a package that
   promises a `ContextGridError` family.

3. `grid(candidates=[5, 20, 50])` with no reranker ran one configuration and said nothing.
   The depth axis means nothing without something to rerank, so folding it is right -- but two
   arms the user asked for vanished between `estimate()`'s `shape ... = 3` and a one-row
   leaderboard, with `results.warnings` empty. `ARM_NOT_MEASURED` already exists and already
   fires for exactly this on the `widened` arm.
"""

from __future__ import annotations

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.core.errors import ContextGridError
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus
from contextgrid.grid.matrix import matrix as build_matrix
from contextgrid.grid.runner import Runner
from contextgrid.lab import Lab
from contextgrid.pipeline import Config
from contextgrid.report.results import Results, RunResult
from contextgrid.transform import ExpandAcronyms, get_transform

# ---------------------------------------------------------------------------
# a corpus small enough to run in a unit test and long enough to separate two arms
# ---------------------------------------------------------------------------

DOCS = {
    "refunds.md": (
        "# Refunds\n\n"
        "Refunds are issued to the original payment method within five business days.\n\n"
        "## Exceptions\n\n"
        "Sale items marked final sale cannot be returned at all.\n"
    ),
    "shipping.md": (
        "# Shipping\n\n"
        "Standard shipping takes three to seven business days and costs five dollars.\n\n"
        "## Express\n\n"
        "Express shipping arrives the next working day and costs twenty dollars.\n"
    ),
}


def _corpus() -> Corpus:
    return Corpus.from_texts(DOCS, media_type=MediaType.MARKDOWN)


def _evalset() -> EvalSet:
    return EvalSet(
        id="axis-findings",
        items=(
            EvalItem(
                id="q1",
                question="How long do refunds take?",
                anchors=(GoldAnchor(source_id="refunds.md", quote="within five business days"),),
            ),
            EvalItem(
                id="q2",
                question="What does standard shipping cost?",
                anchors=(GoldAnchor(source_id="shipping.md", quote="costs five dollars"),),
            ),
        ),
    )


def _two_runs() -> Results:
    """Two hand-built runs that disagree about which is better at k=1 and at k=5."""
    left = RunResult(
        config=Config(chunker="recursive:128"),
        metrics={"recall@1": 0.9, "recall@5": 0.4},
        per_query_by_metric={
            "recall@1": {"q1": 1.0, "q2": 0.8},
            "recall@5": {"q1": 0.4, "q2": 0.4},
        },
        scored_queries=2,
        headline="recall@1",
    )
    right = RunResult(
        config=Config(chunker="sentence"),
        metrics={"recall@1": 0.1, "recall@5": 0.9},
        per_query_by_metric={
            "recall@1": {"q1": 0.1, "q2": 0.1},
            "recall@5": {"q1": 0.9, "q2": 0.9},
        },
        scored_queries=2,
        headline="recall@1",
    )
    return Results(runs=[left, right], headline="recall@1", planned=2)


# ---------------------------------------------------------------------------
# 1. the headline the sweep was asked for is the one the sweep answers about
# ---------------------------------------------------------------------------


def test_results_carries_the_headline_it_was_run_with() -> None:
    """`Results` remembers the metric, the way `RunResult` already did."""
    results = _two_runs()
    assert results.headline == "recall@1"


def test_best_defaults_to_the_headline_not_recall_at_5() -> None:
    """`best()` picked the recall@5 winner however the sweep had been ranked."""
    results = _two_runs()
    winner = results.best()
    assert winner is not None
    # recursive:128 wins at recall@1 and loses at recall@5. The headline is recall@1.
    assert winner.config.chunker == "recursive:128"


def test_summary_names_and_uses_the_headline_it_was_run_with() -> None:
    """The sentence a reader quotes said `recall@5` whatever they asked to be judged on."""
    text = _two_runs().summary()
    assert "recall@1" in text
    assert "recall@5" not in text
    assert "0.900" in text


def test_leaderboard_ranks_on_the_headline_by_default() -> None:
    rows = _two_runs().leaderboard()
    assert [row["config"].split(" · ")[1] for row in rows] == ["recursive:128", "sentence"]
    assert "recall@1" in rows[0]


def test_an_explicit_metric_still_overrides_the_headline() -> None:
    """The override is the whole point of the argument, so it has to keep working."""
    results = _two_runs()
    winner = results.best("recall@5")
    assert winner is not None
    assert winner.config.chunker == "sentence"
    assert "recall@5" in results.summary("recall@5")
    assert results.leaderboard("recall@5")[0]["config"].split(" · ")[1] == "sentence"


def test_a_sweep_with_no_headline_given_still_reads_as_recall_at_5() -> None:
    """The default has to survive, or every existing caller changes meaning."""
    assert Results().headline == "recall@5"


def test_lab_run_headline_reaches_the_summary() -> None:
    """End to end, through the API the quickstart teaches."""
    lab = Lab(_corpus())
    lab.grid(chunker=["recursive:128", "sentence"])
    results = lab.run(_evalset(), headline="recall@1", mode="factorial")

    assert results.headline == "recall@1"
    assert "recall@1" in results.summary()
    assert "recall@5" not in results.summary()


def test_runner_headline_reaches_the_results_it_returns() -> None:
    """The `Runner` is the other public door into the same object."""
    runner = Runner(corpus=_corpus(), headline="mrr@3")
    results = runner.run(build_matrix(chunker=["recursive:128", "sentence"]), _evalset())
    assert results.headline == "mrr@3"
    assert "mrr@3" in results.summary()


# ---------------------------------------------------------------------------
# 2. the expand transform can be configured, and fails like a contextgrid error
# ---------------------------------------------------------------------------


def test_expand_takes_its_acronyms_from_the_spec_string() -> None:
    """The documented grammar -- `name:key=value` -- now reaches `expansions`."""
    transform = get_transform("expand:RPO=recovery point objective")
    assert isinstance(transform, ExpandAcronyms)
    assert transform.expansions == {"RPO": "recovery point objective"}

    result = transform.transform("What is our RPO?")
    assert result.queries == ("What is our RPO recovery point objective?",)


def test_expand_takes_several_acronyms_at_once() -> None:
    transform = get_transform("expand:RPO=recovery point objective,RTO=recovery time objective")
    assert isinstance(transform, ExpandAcronyms)
    assert transform.expansions == {
        "RPO": "recovery point objective",
        "RTO": "recovery time objective",
    }


def test_expand_with_no_acronyms_is_still_the_identity() -> None:
    """Unchanged behaviour, stated so the fix cannot quietly alter the bare spec."""
    transform = get_transform("expand")
    assert isinstance(transform, ExpandAcronyms)
    assert transform.expansions == {}
    assert transform.transform("What is our RPO?").queries == ("What is our RPO?",)


def test_a_numeric_looking_expansion_stays_text() -> None:
    """`_coerce` turns `3` into an int, and an acronym table holds strings."""
    transform = get_transform("expand:SLA=3 business days")
    assert isinstance(transform, ExpandAcronyms)
    assert transform.expansions == {"SLA": "3 business days"}


def test_expansions_given_as_a_bare_string_is_a_contextgrid_error() -> None:
    """It used to build, then die on `'str' object has no attribute 'items'` much later."""
    with pytest.raises(ContextGridError) as caught:
        get_transform("expand:expansions=RPO")
    message = str(caught.value)
    assert "expansions" in message
    # The message has to carry the form that does work, or it only says "no".
    assert "expand:RPO=recovery point objective" in message


def test_an_unknown_parameter_is_a_contextgrid_error_not_a_bare_typeerror() -> None:
    """Every family shared this: an unknown key reached `__init__` and raised builtins."""
    with pytest.raises(ContextGridError) as caught:
        get_transform("none:nonsense=1")
    message = str(caught.value)
    assert message == (
        "transform 'none:nonsense=1': this transform takes no parameters, but 'nonsense' was given."
    )


def test_the_unknown_parameter_error_names_what_the_plugin_does_take() -> None:
    from contextgrid.chunk import CHUNKERS

    with pytest.raises(ContextGridError) as caught:
        CHUNKERS.create("recursive:512,overlop=64")
    assert str(caught.value) == (
        "chunker 'recursive:512,overlop=64': unknown parameter 'overlop'. "
        "Did you mean 'overlap'? This chunker takes size, overlap, separators, tokenizer."
    )


def test_two_unknown_parameters_are_listed_without_a_guess() -> None:
    """One typo gets a suggestion; two get the accepted list, which is the useful answer."""
    from contextgrid.chunk import CHUNKERS

    with pytest.raises(ContextGridError) as caught:
        CHUNKERS.create("recursive:512,overlop=64,seperators=x")
    assert str(caught.value) == (
        "chunker 'recursive:512,overlop=64,seperators=x': unknown parameters 'overlop' and "
        "'seperators'. This chunker takes size, overlap, separators, tokenizer."
    )


@pytest.mark.parametrize(
    ("family", "spec"),
    [
        ("transform", "none:nonsense=1"),
        ("chunker", "recursive:512,overlop=64"),
        ("chunker", "recursive:512,overlop=64,seperators=x"),
        ("transform", "expand:expansions=RPO"),
        ("index", "bm25:k=2"),
    ],
)
def test_a_spec_error_never_shows_python_internals_or_a_double_stop(family: str, spec: str) -> None:
    """The two things wrong with the first version of this message.

    `__init__` and the class name told the reader only that this package is made of classes.
    And Python's own "Did you mean 'overlap'?" already ends in a question mark, so appending a
    full stop to it produced `?.`.
    """
    from contextgrid.chunk import CHUNKERS
    from contextgrid.index import INDEXES
    from contextgrid.transform import TRANSFORMS

    registry = {"transform": TRANSFORMS, "chunker": CHUNKERS, "index": INDEXES}[family]

    with pytest.raises(ContextGridError) as caught:
        registry.create(spec)
    message = str(caught.value)
    assert "__init__" not in message
    assert "?." not in message
    assert message.startswith(f"{family} '")
    assert message.endswith(".")


# ---------------------------------------------------------------------------
# 3. a depth axis that cannot do anything says so
# ---------------------------------------------------------------------------


def test_sweeping_candidates_with_no_reranker_warns_that_nothing_measured_it() -> None:
    lab = Lab(_corpus())
    lab.grid(chunker="recursive:128", candidates=[5, 20, 50])
    results = lab.run(_evalset(), mode="factorial")

    # The fold itself is right: one row, because the three depths are one configuration.
    assert len(results.leaderboard()) == 1

    notes = [w for w in results.warnings if w.code is WarningCode.ARM_NOT_MEASURED]
    assert notes, "three depths collapsed to one row with nothing said"
    message = str(notes[0])
    assert "candidates" in message
    assert "reranker" in message


def test_candidates_with_a_reranker_is_a_real_axis_and_is_not_warned_about() -> None:
    lab = Lab(_corpus())
    lab.grid(chunker="recursive:128", reranker="mmr", candidates=[5, 50])
    results = lab.run(_evalset(), mode="factorial")

    assert len(results.leaderboard()) == 2
    depth_notes = [
        w
        for w in results.warnings
        if w.code is WarningCode.ARM_NOT_MEASURED and "candidates" in str(w)
    ]
    assert not depth_notes


def test_a_single_candidates_value_with_no_reranker_is_not_warned_about() -> None:
    """Nothing was swept, so nothing went missing. A warning here would be noise."""
    lab = Lab(_corpus())
    lab.grid(chunker="recursive:128", candidates=20)
    results = lab.run(_evalset(), mode="factorial")

    depth_notes = [
        w
        for w in results.warnings
        if w.code is WarningCode.ARM_NOT_MEASURED and "candidates" in str(w)
    ]
    assert not depth_notes

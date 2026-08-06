"""Metrics as a plugin family: the `Metric` protocol, `METRICS`, and a custom metric swept
for real -- the same shape `tests/unit/test_retrieve_agentic.py` uses to register a scripted
strategy into the real `RETRIEVERS` for a test.

The six built-ins keep their own dedicated cross-check against `ranx`
(`test_metrics_vs_ranx.py`, untouched by this file). What's new here is that a *custom*
metric, registered by someone who never touches `contextgrid/score/metrics.py`, is computed
by `evaluate()`/`per_query()` exactly like a built-in one, can be named as `run.headline`, and
survives a metric that raises without taking the whole run down with it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from contextgrid.config.loader import loads
from contextgrid.config.schema import ConfigError
from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config
from contextgrid.score import METRICS, Metric
from contextgrid.score.metrics import (
    BUILTIN_METRIC_NAMES,
    available_metrics,
    evaluate,
    per_query,
)
from tests.support import API_DOCS, CONTRACT

QUESTIONS = [
    ("q1", "How much notice is needed to terminate for convenience?", "contract.md", "thirty days"),
    ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
    ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )


@pytest.fixture
def evalset() -> EvalSet:
    return EvalSet(
        id="es",
        items=tuple(
            EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
            for i, q, s, t in QUESTIONS
        ),
    )


# ---------------------------------------------------------------------------
# a custom metric: grade-weighted recall
# ---------------------------------------------------------------------------
#
# Plain recall_at_k counts a "fully answers" chunk and a "partially relevant" one the same,
# as long as both come back -- it only counts chunks, not how much evidence they carry. This
# weighs by grade instead: what fraction of the total relevance *mass* landed in the top k.


@dataclass(frozen=True, slots=True)
class WeightedRecall:
    """Recall weighted by grade rather than by chunk count."""

    name: ClassVar[str] = "test_weighted_recall"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        total = sum(grade for grade in judgements.values() if grade > 0)
        if total == 0:
            return 0.0
        top = set(ranked[:k])
        found = sum(grade for cid, grade in judgements.items() if grade > 0 and cid in top)
        return found / total


@dataclass(frozen=True, slots=True)
class AlwaysFails:
    """A metric with a bug, for testing that one does not take the whole run down."""

    name: ClassVar[str] = "test_always_fails"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        raise RuntimeError("this metric is broken on purpose")


def _register(name: str, cls: type[Metric], doc: str = "") -> None:
    """Idempotent registration into the real `METRICS`, the same guard
    `test_retrieve_agentic.py` uses for `RETRIEVERS` -- so re-running this module (or pytest
    re-collecting it) doesn't hit `Registry`'s "already registered" error.

    `WeightedRecall` is registered here, permanently, because it's a metric that actually
    works -- leaving it in `METRICS` for the rest of the session is harmless, the same as any
    other registered plugin. `AlwaysFails` is *not* registered here: see the `failing_metric`
    fixture below for why a metric registered only to prove it fails needs to come back out.
    """
    if name not in METRICS:
        METRICS.register(name, doc=doc)(cls)


_register("test_weighted_recall", WeightedRecall, doc="Recall weighted by grade, for tests.")


@pytest.fixture
def failing_metric() -> Iterator[str]:
    """Registers `AlwaysFails` into the real `METRICS` for one test, then removes it.

    A metric registered only to prove it fails must not leak into `available_metrics()` for
    the rest of the session: `test_metrics_vs_ranx.py`'s
    `test_available_metrics_matches_what_evaluate_accepts` computes every registered metric
    and expects each one to actually produce a score, which `AlwaysFails` never will. Using
    `Registry.unregister` here (added alongside `Metric` for exactly this) keeps that test
    honest without this file having to know it exists.
    """
    METRICS.register("test_always_fails", doc="Always raises, for tests.")(AlwaysFails)
    try:
        yield "test_always_fails"
    finally:
        METRICS.unregister("test_always_fails")


# ---------------------------------------------------------------------------
# the protocol and the registry
# ---------------------------------------------------------------------------


def test_the_six_builtins_satisfy_the_metric_protocol() -> None:
    for name in BUILTIN_METRIC_NAMES:
        instance = METRICS.create(name)
        assert isinstance(instance, Metric)
        assert instance.name == name
        assert instance.version


def test_a_custom_metric_satisfies_the_metric_protocol_too() -> None:
    instance = METRICS.create("test_weighted_recall")
    assert isinstance(instance, Metric)
    assert isinstance(instance, WeightedRecall)


def test_available_metrics_reflects_registrations_made_after_import() -> None:
    """`available_metrics()` reads the registry at call time, not at import time -- a metric
    registered by user code after `import contextgrid` still shows up."""
    assert "test_weighted_recall" in available_metrics()


def test_unregistering_a_metric_removes_it_from_available_metrics(
    failing_metric: str,
) -> None:
    assert failing_metric in available_metrics()
    METRICS.unregister(failing_metric)
    assert failing_metric not in available_metrics()
    # put it back so the fixture's own teardown (a second `unregister`) stays a no-op
    METRICS.register(failing_metric, doc="Always raises, for tests.")(AlwaysFails)


# ---------------------------------------------------------------------------
# evaluate() and per_query() resolve through the registry
# ---------------------------------------------------------------------------


def test_evaluate_computes_a_custom_metric_alongside_the_builtins() -> None:
    qrels = {"q1": {"a": 2, "b": 1, "z": 0}}
    run = {"q1": ["x", "a", "y", "b", "w"]}

    scores = evaluate(qrels, run, ks=(5,), metrics=("recall", "test_weighted_recall"))

    assert scores["recall@5"] == 1.0  # both relevant chunks are in the top 5
    # grade 2 (a) + grade 1 (b) found, out of grade 2 + grade 1 total -- everything found
    assert scores["test_weighted_recall@5"] == 1.0


def test_evaluate_gives_a_different_number_from_plain_recall_when_grades_differ() -> None:
    """The point of writing a new metric instead of reusing recall: this one is sensitive to
    *which* relevant chunk was found, not just whether one was."""
    qrels = {"q1": {"a": 2, "b": 1}}  # a matters twice as much as b
    run_finds_the_big_one = {"q1": ["a"]}
    run_finds_the_small_one = {"q1": ["b"]}

    wanted = ("recall", "test_weighted_recall")
    big = evaluate(qrels, run_finds_the_big_one, ks=(1,), metrics=wanted)
    small = evaluate(qrels, run_finds_the_small_one, ks=(1,), metrics=wanted)

    # plain recall can't tell these apart: one relevant chunk out of two, either way
    assert big["recall@1"] == small["recall@1"] == 0.5
    # weighted recall can: finding the grade-2 chunk captures twice the evidence
    assert big["test_weighted_recall@1"] == pytest.approx(2 / 3)
    assert small["test_weighted_recall@1"] == pytest.approx(1 / 3)


def test_per_query_resolves_a_custom_metric() -> None:
    qrels = {"q1": {"a": 2, "b": 1}}
    run = {"q1": ["b"]}
    assert per_query(qrels, run, "test_weighted_recall", 1) == {"q1": pytest.approx(1 / 3)}


def test_an_unregistered_metric_name_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate({"q": {"a": 1}}, {"q": ["a"]}, metrics=["not_a_real_metric"])
    with pytest.raises(ValueError, match="unknown metric"):
        per_query({"q": {"a": 1}}, {"q": ["a"]}, "not_a_real_metric", 5)


# ---------------------------------------------------------------------------
# a metric that fails does not take the run down, and does not silently score zero
# ---------------------------------------------------------------------------


def test_a_failing_metric_is_left_out_not_zeroed(failing_metric: str) -> None:
    from contextgrid.core.warnings import WarningLog

    qrels = {"q1": {"a": 1}}
    run = {"q1": ["a"]}
    log = WarningLog()

    scores = evaluate(qrels, run, ks=(5,), metrics=("recall", failing_metric), warnings=log)

    assert scores == {"recall@5": 1.0}  # the good metric is unaffected
    assert f"{failing_metric}@5" not in scores  # not present -- and definitely not 0.0
    failed = log.of_code(WarningCode.METRIC_FAILED)
    assert len(failed) == 1
    assert failing_metric in failed[0].message


def test_a_failing_metric_without_a_warning_log_still_does_not_crash(failing_metric: str) -> None:
    scores = evaluate({"q1": {"a": 1}}, {"q1": ["a"]}, ks=(5,), metrics=("recall", failing_metric))
    assert scores == {"recall@5": 1.0}


# ---------------------------------------------------------------------------
# run.headline and run.metrics in a config
# ---------------------------------------------------------------------------


def test_headline_accepts_a_registered_custom_metric() -> None:
    config = loads("corpus: ./docs\nrun:\n  headline: test_weighted_recall@5\n")
    assert config.run.headline == "test_weighted_recall@5"


def test_metrics_key_accepts_one_value_or_a_list() -> None:
    one = loads("corpus: ./docs\nrun:\n  metrics: test_weighted_recall\n")
    assert one.run.metrics == ("test_weighted_recall",)

    many = loads("corpus: ./docs\nrun:\n  metrics: [test_weighted_recall, ndcg]\n")
    assert many.run.metrics == ("test_weighted_recall", "ndcg")


def test_an_unknown_name_in_metrics_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"unknown metric.*run\.metrics"):
        loads("corpus: ./docs\nrun:\n  metrics: not_a_real_metric\n")


# ---------------------------------------------------------------------------
# a real sweep: run.headline, extra_metrics, and the leaderboard
# ---------------------------------------------------------------------------


def test_runner_metric_names_always_includes_the_headline() -> None:
    """The same guarantee `ks` already makes for cut-offs (see the comment in
    `Runner.__post_init__`), now made for the metric name too."""
    runner = Runner(corpus=Corpus.from_texts({}), headline="test_weighted_recall@5")
    assert "test_weighted_recall" in runner.metric_names
    assert set(BUILTIN_METRIC_NAMES) <= set(runner.metric_names)


def test_a_custom_metric_can_be_the_headline_of_a_real_sweep(
    corpus: Corpus, evalset: EvalSet
) -> None:
    runner = Runner(corpus=corpus, headline="test_weighted_recall@5")
    results = runner.run(matrix(chunker="sentence:1", embedder="tfidf"), evalset, mode="factorial")

    winner = results.best("test_weighted_recall@5")
    assert winner is not None
    assert winner.has("test_weighted_recall@5")
    assert 0.0 <= winner.metric("test_weighted_recall@5") <= 1.0

    row = results.leaderboard("test_weighted_recall@5", extra=["recall@5"])[0]
    assert "test_weighted_recall@5" in row
    assert "recall@5" in row


def test_extra_metrics_computes_alongside_the_headline(corpus: Corpus, evalset: EvalSet) -> None:
    """`run.metrics` (the config key) becomes `Runner.extra_metrics` -- a metric that isn't
    the headline still gets computed and shows up on `RunResult.metrics`."""
    runner = Runner(corpus=corpus, extra_metrics=("test_weighted_recall",))
    result = runner.run_one(Config(chunker="sentence:1", embedder="tfidf"), evalset)
    assert result.has("test_weighted_recall@5")
    assert result.has("recall@5")  # the built-ins are still there too


def test_the_config_metrics_key_reaches_the_runner_end_to_end(
    tmp_path: Path, evalset: EvalSet
) -> None:
    """`run.metrics` in a config file reaches `Runner.extra_metrics` through `build_runner`,
    without `test_weighted_recall` needing to be the headline at all."""
    from contextgrid.config.loader import build_runner

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "contract.md").write_text(CONTRACT, encoding="utf-8")
    (corpus_dir / "api.md").write_text(API_DOCS, encoding="utf-8")

    config = loads(f"corpus: {corpus_dir}\nrun:\n  metrics: test_weighted_recall\n")
    runner = build_runner(config, Corpus.from_dir(corpus_dir))
    assert "test_weighted_recall" in runner.metric_names
    assert runner.extra_metrics == ("test_weighted_recall",)

    result = runner.run_one(Config(chunker="sentence:1", embedder="tfidf"), evalset)
    assert result.has("test_weighted_recall@5")

"""What a cost-and-budget drive of 0.9.2 found, and two gaps left over from the CLI drive.

The theme is money that is either invisible or unrepeatable.

* A sweep stopped by its budget printed a leaderboard that looked finished. The only signal
  went to stderr, and the documentation tells people to redirect stdout and keep that.
* `budget_usd` charged wall-clock machine time, which falls as a cache warms, so the same
  budget bought a different number of configurations on every run.
* `machine_usd_per_hour` changed that hidden arithmetic and appeared in nothing anybody reads.
* `BUDGET_REACHED` meant three unrelated things, one of which the docs tell users to filter on.
* A folded `widened` arm was loud in the warnings and invisible in the row.
* And `cg.Lab` still scored `contextual` ingestion as if a model had run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import contextgrid as cg
from contextgrid.core.warnings import Severity, WarningCode
from contextgrid.cost.model import CostBreakdown, CostModel

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# a corpus and an eval set small enough to sweep several times
# ---------------------------------------------------------------------------

DOCS = {
    "return-policy.md": (
        "# Return Policy\n\nItems may be returned within 30 days of delivery for a full "
        "refund.\n\nThe item must be unused and in its original packaging.\n"
    ),
    "shipping.md": (
        "# Shipping\n\nStandard shipping takes 3 to 7 business days and costs $5.99.\n\n"
        "Orders over $50 ship free within the mainland.\n"
    ),
}


def corpus() -> Any:
    return cg.Corpus.from_texts(DOCS, media_type=cg.MediaType.MARKDOWN)


def evalset() -> Any:
    return cg.EvalSet(
        id="tiny",
        items=(
            cg.EvalItem(
                id="q1",
                question="How many days do I have to return an item?",
                anchors=(
                    cg.GoldAnchor(
                        source_id="return-policy.md", quote="returned within 30 days of delivery"
                    ),
                ),
            ),
            cg.EvalItem(
                id="q2",
                question="How much does standard shipping cost?",
                anchors=(cg.GoldAnchor(source_id="shipping.md", quote="costs $5.99"),),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1. a truncated leaderboard that looked finished
# ---------------------------------------------------------------------------


def _stopped_early(after: int = 2) -> Any:
    """A sweep the budget cuts short after `after` configurations.

    The stop is forced rather than timed. A real `budget_seconds` small enough to bite is a
    race -- it stops after two configurations on a warm machine and none on a cold one -- and
    what is being pinned here is what the results object then *says*, not the clock.
    """
    from unittest.mock import patch

    from contextgrid.grid.runner import Budget

    calls = {"n": 0}

    def exceeded(self: Budget) -> str | None:
        calls["n"] += 1
        return "the 0.35s budget ran out" if calls["n"] > after else None

    lab = cg.Lab(corpus())
    lab.grid(chunker=["recursive:512", "recursive:256", "sentence:2", "sentence:3"])
    with patch.object(Budget, "exceeded", exceeded):
        return lab.run(evalset(), budget_seconds=0.35)


def test_results_know_they_are_partial() -> None:
    """Nothing on the object said so, so nothing printed from it could either."""
    results = _stopped_early()

    assert results.is_partial, "a sweep the budget cut short is not a complete sweep"
    assert results.planned > len(results.runs)
    assert results.stopped is not None
    assert "budget" in results.stopped


def test_a_complete_sweep_is_not_marked_partial() -> None:
    lab = cg.Lab(corpus())
    lab.grid(chunker=["recursive:512", "recursive:256"])
    results = lab.run(evalset())

    assert not results.is_partial
    assert results.stopped is None
    assert results.planned == len(results.runs)


def test_the_summary_says_so_and_the_summary_is_stdout() -> None:
    """`/reference/cli` recommends `contextgrid run config.yaml > leaderboard.txt` to keep
    "just the results". Follow that and the only note that the table is a fraction of the
    matrix went to the terminal and was thrown away."""
    results = _stopped_early()
    summary = results.summary("recall@5")

    assert "partial" in summary.lower(), summary
    assert str(results.planned) in summary, summary


def test_the_partial_warning_does_not_point_below_itself() -> None:
    """In `report.md` the warnings block sits under the leaderboard, so "the leaderboard below
    is partial" points at nothing. The fact does not depend on where it is printed."""
    results = _stopped_early()
    stopped = results.warnings.of_code(WarningCode.BUDGET_REACHED)

    assert stopped, "the budget stop is still recorded as a warning"
    assert "below" not in stopped[0].message, stopped[0].message


def test_the_cli_says_it_above_the_table_it_is_about(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On stdout, before the leaderboard, where "the rows below" is true."""
    from unittest.mock import patch

    from contextgrid.cli import main
    from contextgrid.evalset import write_jsonl
    from contextgrid.grid.runner import Budget

    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    write_jsonl(evalset(), tmp_path / "evalset.jsonl")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "corpus: ./docs\nevalset: ./evalset.jsonl\n"
        "grid:\n  chunker: [recursive:512, recursive:256, sentence:2]\n"
        "run:\n  budget_seconds: 30\n",
        encoding="utf-8",
    )

    calls = {"n": 0}

    def exceeded(self: Budget) -> str | None:
        calls["n"] += 1
        return "the 30s budget ran out" if calls["n"] > 1 else None

    with patch.object(Budget, "exceeded", exceeded):
        assert main(["run", str(config)]) == 0
    printed = capsys.readouterr().out

    assert "partial" in printed.lower(), printed
    # The rule under the leaderboard's own header, rather than the metric name -- which also
    # appears in the plan line the command prints before anything runs.
    assert printed.lower().index("partial") < printed.index("-----"), (
        "it has to be above the table it is about, not under it"
    )


def test_the_manifest_records_it(tmp_path: Path) -> None:
    """A bundle is what somebody keeps. Read six months later, an 11-row leaderboard with a
    complete-looking manifest is indistinguishable from a finished experiment."""
    from contextgrid.report import build_manifest

    results = _stopped_early()
    manifest = build_manifest(
        results.runs[0].config, corpus(), evalset(), notes=results.manifest_note()
    )

    assert "partial" in manifest.notes.lower(), manifest.notes
    assert str(results.planned) in manifest.notes


def test_a_complete_run_leaves_the_manifest_note_empty() -> None:
    lab = cg.Lab(corpus())
    lab.grid(chunker=["recursive:512"])

    assert lab.run(evalset()).manifest_note() == ""


# ---------------------------------------------------------------------------
# 2. a budget that bought more configurations on a warm cache
# ---------------------------------------------------------------------------


def _breakdown(compute_seconds: float, machine_usd: float) -> CostBreakdown:
    """One configuration's cost, with machine time already folded into `index_usd` the way
    `CostModel.estimate` folds it."""
    return CostBreakdown(
        index_usd=0.01 + machine_usd,
        query_usd_per_1k=0.0,
        compute_seconds=compute_seconds,
        machine_usd=machine_usd,
    )


def test_a_budget_charges_the_same_whether_the_cache_was_warm() -> None:
    """`/reference/caching` says never to compare `compute_seconds` across warm-cache runs.
    `Budget.charge` compared them against a spending limit, so the same money bought two,
    four, then two configurations on one set of attempts and two, two, three on the next."""
    from contextgrid.grid.runner import Budget
    from contextgrid.report.results import RunResult

    cold = Budget(usd=1.0)
    warm = Budget(usd=1.0)
    cold.charge(RunResult(config=cg.Config(), cost=_breakdown(40.0, 1.11)), queries=2)
    warm.charge(RunResult(config=cg.Config(), cost=_breakdown(0.4, 0.011)), queries=2)

    assert cold.spent_usd == pytest.approx(warm.spent_usd), (
        "the two runs did identical work and paid identical token prices"
    )


def test_the_machine_time_is_still_in_the_bill() -> None:
    """Not deleted -- `spent_now` is what a configuration cost, machine time included. It is
    the *budget* that cannot compare it between runs."""
    breakdown = _breakdown(40.0, 1.11)

    assert breakdown.spent_now(2) > breakdown.metered_now(2)
    assert breakdown.spent_now(2) - breakdown.metered_now(2) == pytest.approx(1.11)


def test_a_budget_with_a_machine_rate_says_what_it_is_not_counting() -> None:
    """Silently ignoring a number the user set is how the old behaviour would come back."""
    lab = cg.Lab(corpus(), machine_usd_per_hour=100.0)
    lab.grid(chunker=["recursive:512"])
    results = lab.run(evalset(), budget_usd=10.0)

    said = [w for w in results.warnings if "machine" in w.message and "budget" in w.message]
    assert said, [w.message for w in results.warnings]


def test_no_such_warning_without_a_machine_rate() -> None:
    lab = cg.Lab(corpus())
    lab.grid(chunker=["recursive:512"])
    results = lab.run(evalset(), budget_usd=10.0)

    assert not [w for w in results.warnings if "machine" in w.message and "budget" in w.message]


# ---------------------------------------------------------------------------
# 3. a machine rate that showed up nowhere
# ---------------------------------------------------------------------------


def test_the_breakdown_names_the_machine_time_it_charged() -> None:
    """`index_usd` was "token cost plus machine time" with no way to see the split, so a
    local embedder's entire index cost was machine time nobody could identify."""
    breakdown = CostModel(machine_usd_per_hour=36.0).estimate(
        embedder="tfidf", index_tokens=1000, query_tokens_per_query=12, compute_seconds=100.0
    )

    assert breakdown.machine_usd == pytest.approx(1.0)
    assert breakdown.index_usd == pytest.approx(1.0)
    assert breakdown.as_dict()["machine_usd"] == pytest.approx(1.0)


def test_the_summary_paragraph_stops_calling_a_priced_machine_free() -> None:
    """ "It runs locally at no cost per query" is true about tokens and false about the $100
    an hour the user just told the tool their machine costs."""
    lab = cg.Lab(corpus(), machine_usd_per_hour=100.0)
    lab.grid(chunker=["recursive:512"])
    summary = lab.run(evalset()).summary("recall@5")

    assert "machine" in summary, summary
    assert "at no cost" not in summary, summary


def test_a_free_machine_reads_exactly_as_before() -> None:
    lab = cg.Lab(corpus())
    lab.grid(chunker=["recursive:512"])
    summary = lab.run(evalset()).summary("recall@5")

    assert "at no cost per query" in summary, summary
    assert "machine" not in summary, summary


def test_estimate_reports_the_rate_it_was_given() -> None:
    """`/lab/grid` says to pass `machine_usd_per_hour` "if your compute isn't actually free",
    next to an `estimated_usd` that never moves when you do -- there is no way to predict
    seconds before running. Echo the setting rather than implying it was applied."""
    lab = cg.Lab(corpus(), machine_usd_per_hour=100.0)
    lab.grid(chunker=["recursive:512", "recursive:256"])
    estimate = lab.estimate()

    assert estimate["machine_usd_per_hour"] == 100.0
    assert "estimated_usd" in estimate


def test_the_grid_page_no_longer_promises_estimate_prices_machine_time() -> None:
    page = (REPO_ROOT / "docs-site" / "lab" / "grid.mdx").read_text(encoding="utf-8")
    machine = page[page.index("estimated_usd") :]

    assert "measured per run" in machine or "cannot predict" in machine, (
        "the page told people to set a rate to make this number real; it never does"
    )


# ---------------------------------------------------------------------------
# 4. one code for three unrelated facts
# ---------------------------------------------------------------------------


def test_an_unpriced_model_has_its_own_code() -> None:
    """`/lab/running` tells people to detect a budget stop with
    `w.code.name == "BUDGET_REACHED"`. That filter caught this too."""
    model = CostModel()
    model.pricing_for("some-unknown-hosted-model")

    assert model.warnings.of_code(WarningCode.MODEL_NOT_PRICED)
    assert not model.warnings.of_code(WarningCode.BUDGET_REACHED)
    assert model.warnings.entries[0].severity is Severity.CAUTION


class ScriptedLLM:
    """Enough of an `LLM` for a strategy that plans with one. No key, no network."""

    name = "scripted"

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        return '["how long do returns take?"]'


def test_a_sweep_with_no_ceiling_raises_its_own_code() -> None:
    """Nothing was reached: the sweep has no ceiling at all. `runner.py` already carried a
    comment in `_why_nothing_ran` working around the collision this caused."""
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    runner = Runner(corpus=corpus(), llm=ScriptedLLM())
    results = runner.run(matrix(retrieval=["agentic"]), evalset())

    assert results.warnings.of_code(WarningCode.NO_COST_CEILING)
    assert not results.warnings.of_code(WarningCode.BUDGET_REACHED)


def test_a_sweep_with_a_ceiling_says_nothing() -> None:
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    runner = Runner(corpus=corpus(), llm=ScriptedLLM())
    results = runner.run(matrix(retrieval=["agentic"]), evalset(), budget_usd=1.0)

    assert not results.warnings.of_code(WarningCode.NO_COST_CEILING)


def test_budget_reached_now_means_one_thing() -> None:
    """A stopped sweep, and nothing else."""
    stopped = _stopped_early().warnings.of_code(WarningCode.BUDGET_REACHED)

    assert stopped
    assert all("budget" in w.message for w in stopped)


def test_the_running_page_filters_on_a_code_that_means_one_thing() -> None:
    page = (REPO_ROOT / "docs-site" / "lab" / "running.mdx").read_text(encoding="utf-8")

    assert 'w.code.name == "BUDGET_REACHED"' in page, "the recipe is still recommended"
    assert "MODEL_NOT_PRICED" in page or "NO_COST_CEILING" in page, (
        "the page has to say what the filter no longer catches"
    )


# ---------------------------------------------------------------------------
# 5. the folded arm, invisible in the row
# ---------------------------------------------------------------------------


def test_a_folded_arm_is_visible_in_the_row() -> None:
    """`simple`, `decomposed` and `relevance-feedback` all label with the strategy. A folded
    `widened` labelled as plain search reads as a row that measured plain search on purpose."""
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    results = Runner(corpus=corpus()).run(matrix(retrieval=["widened:factor=8"]), evalset())

    assert len(results.runs) == 1
    row = results.runs[0]
    assert row.config.retrieval is None, "it really does run as plain search"
    assert row.folded_from == "widened:factor=8"
    assert "widened:factor=8" in row.label, row.label
    assert "widened:factor=8" in results.leaderboard("recall@5")[0]["config"]
    assert "widened" in results.manifest_note(), results.manifest_note()


def test_the_fold_marker_is_not_a_config_field() -> None:
    """`Config` is what `winning-config.yaml` round-trips, and every field on it has to be an
    axis somebody can write and re-run. Provenance about one row is not that, and three
    existing tests in `test_export_roundtrip` exist to keep it out."""
    from dataclasses import fields

    from contextgrid.grid.matrix import AXIS_ORDER
    from contextgrid.pipeline import Config

    homeless = [f.name for f in fields(Config) if f.name not in AXIS_ORDER and f.name != "k"]
    assert homeless == []


def test_the_fold_marker_does_not_split_a_row_in_two() -> None:
    """The whole point of folding is one run, so the marker must not make two configurations
    that are the same run look different to the dedupe."""
    from contextgrid.grid.matrix import matrix

    configs, report = matrix(retrieval=[None, "widened:factor=8"]).expand_with_report("factorial")

    assert len(configs) == 1
    assert report.collapsed == 1


def test_a_row_nobody_folded_carries_no_marker() -> None:
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    results = Runner(corpus=corpus()).run(matrix(retrieval=["decomposed"]), evalset())
    row = results.runs[0]

    assert "~decomposed" in row.label
    assert row.folded_from is None
    assert "plain search" not in row.label
    assert results.manifest_note() == ""


# ---------------------------------------------------------------------------
# 6. `cg.Lab` still scored `contextual` as if a model had run
# ---------------------------------------------------------------------------


def test_the_lab_scores_no_row_for_a_paid_strategy_with_no_model() -> None:
    """`contextgrid check` and `contextgrid run` refuse this before anything starts. The
    Python path fell back to plain chunks on every chunk and scored a row labelled
    `contextual` anyway.

    It now fails that row rather than the sweep, which is the rule item 7 sets for every
    configuration that cannot run: in a grid where only one arm lacks a model, the other arms
    are still worth measuring. Nothing is scored, and the warning says why.
    """
    lab = cg.Lab(corpus())
    lab.grid(ingestion=["contextual"])
    results = lab.run(evalset())

    assert results.runs == [], "no row may be scored for a strategy that never called a model"
    failed = results.warnings.of_code(WarningCode.CONFIGURATION_FAILED)
    assert failed, [w.code.value for w in results.warnings]
    assert "needs a model" in failed[0].message, failed[0].message
    assert "run.model" in failed[0].message, failed[0].message


def test_a_paid_arm_with_no_model_does_not_take_the_free_arms_with_it() -> None:
    """The reason it is a failed row and not a failed sweep."""
    lab = cg.Lab(corpus())
    lab.grid(ingestion=["parent-document", "contextual"])
    results = lab.run(evalset())

    assert [run.config.ingestion for run in results.runs] == ["parent-document"]
    assert results.warnings.of_code(WarningCode.CONFIGURATION_FAILED)


def test_the_free_strategies_still_run_on_the_lab_path() -> None:
    lab = cg.Lab(corpus())
    lab.grid(ingestion=["parent-document"])

    assert lab.run(evalset()).runs


def test_a_model_makes_it_run() -> None:
    """The refusal is about having no model, not about `contextual` itself."""
    from contextgrid.pipeline import Config, build

    class ScriptedLLM:
        name = "scripted"

        def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
            return "This chunk is from the shipping page, about delivery times."

    built = build(
        Config(ingestion="contextual", chunker="recursive:128"), corpus(), llm=ScriptedLLM()
    )

    assert built.ingested is not None
    assert built.ingested.model_calls > 0


def test_sweep_can_bound_money_and_not_only_time(tmp_path: Path) -> None:
    """`sweep` had `--budget-seconds` and no `--budget-usd`, while `lab.run` and a config file
    both take either. The one command that spends money from a single line could only be
    bounded by the clock."""
    from contextgrid.cli.__main__ import _build_parser

    args = _build_parser().parse_args(
        ["sweep", str(tmp_path), str(tmp_path / "q.jsonl"), "--budget-usd", "0.5"]
    )

    assert args.budget_usd == 0.5


def test_the_message_is_the_same_one_check_prints() -> None:
    """Two wordings for one refusal is how somebody fails to recognise the second as the
    same problem."""
    from contextgrid.config.plugins import model_missing_for
    from contextgrid.ingest.base import needs_model_error

    assert str(model_missing_for("contextual", None)) == str(needs_model_error("contextual"))


# ---------------------------------------------------------------------------
# 7. one unscorable configuration must not throw the whole sweep away
# ---------------------------------------------------------------------------


def _explodes(*_args: Any, **_kwargs: Any) -> dict[str, float]:
    """What `evaluate()` now does to a run whose ranking repeats a chunk id."""
    raise ValueError("the run for query 'q1' returns the same chunk id more than once: 'c1'")


def test_one_unscorable_configuration_does_not_kill_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`evaluate()` rejecting a repeated chunk id is right -- it used to produce `recall@3 =
    1.5`. But `run_one` was called with no `try`, so that rejection propagated out of `_flat`
    and discarded every configuration already measured. One silently wrong number traded for
    eighteen configurations of work thrown away is not a trade.
    """
    from contextgrid.grid import runner as runner_module
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    real = runner_module.evaluate
    calls = {"n": 0}

    def sometimes(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return _explodes() if calls["n"] == 2 else real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate", sometimes)
    results = Runner(corpus=corpus()).run(
        matrix(chunker=["recursive:512", "recursive:256", "sentence:2"]), evalset()
    )

    assert len(results.runs) == 2, "the two configurations that could be scored are still here"
    failed = results.warnings.of_code(WarningCode.CONFIGURATION_FAILED)
    assert len(failed) == 1, [w.code.value for w in results.warnings]
    assert "recursive:256" in failed[0].message, failed[0].message
    assert "same chunk id more than once" in failed[0].message, failed[0].message
    assert failed[0].severity is Severity.CAUTION


def test_a_sweep_that_lost_a_row_is_reported_as_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two of three rows is not the leaderboard the header promised, whatever cut it short."""
    from contextgrid.grid import runner as runner_module
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    real = runner_module.evaluate
    calls = {"n": 0}

    def sometimes(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return _explodes() if calls["n"] == 1 else real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate", sometimes)
    results = Runner(corpus=corpus()).run(
        matrix(chunker=["recursive:512", "recursive:256"]), evalset()
    )

    assert results.is_partial
    assert "partial" in results.summary("recall@5").lower()


def test_a_staged_sweep_survives_it_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The staged path calls `run_one` from its own loop, and had the same bare call."""
    from contextgrid.grid import runner as runner_module
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    real = runner_module.evaluate
    calls = {"n": 0}

    def sometimes(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return _explodes() if calls["n"] == 1 else real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate", sometimes)
    results = Runner(corpus=corpus()).run(
        matrix(chunker=["recursive:512", "recursive:256"]), evalset(), mode="staged"
    )

    assert results.runs, "the stage carried on to the value that could be scored"
    assert results.warnings.of_code(WarningCode.CONFIGURATION_FAILED)


def test_a_sweep_where_everything_fails_is_still_an_empty_leaderboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a crash, and not a green build either -- `contextgrid run` already treats "nothing
    measured" as exit 1."""
    from contextgrid.grid import runner as runner_module
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    monkeypatch.setattr(runner_module, "evaluate", _explodes)
    results = Runner(corpus=corpus()).run(matrix(chunker=["recursive:512"]), evalset())

    assert results.runs == []
    assert results.warnings.of_code(WarningCode.CONFIGURATION_FAILED)


# ---------------------------------------------------------------------------
# 8. the run-level warning that asserted a cause it could not know
# ---------------------------------------------------------------------------


def test_the_unreachable_evidence_warning_does_not_pick_a_culprit() -> None:
    """It fires just as readily when the eval set quotes something the document does not
    contain. Saying "nothing to do with retrieval" is true and leaves the reader nowhere to
    look; naming the parser would be a guess."""
    lab = cg.Lab(corpus())
    lab.grid(chunker=["recursive:512"])
    results = lab.run(
        cg.EvalSet(
            id="wrong",
            items=(
                cg.EvalItem(
                    id="q1",
                    question="What is the refund window?",
                    anchors=(cg.GoldAnchor(source_id="return-policy.md", quote="not in here"),),
                ),
            ),
        )
    )

    unreachable = [
        w
        for w in results.warnings.of_code(WarningCode.GOLD_SPAN_UNREACHABLE)
        if w.subject and "·" in w.subject
    ]
    assert unreachable, [w.subject for w in results.warnings]
    message = unreachable[0].message
    assert "cannot tell" in message, message
    assert "anchor_not_found" in message, message
    assert "eval set" in message, message


# ---------------------------------------------------------------------------
# 9. Ctrl-C, and the promise the traceback broke
# ---------------------------------------------------------------------------


def _interrupt(*_args: Any, **_kwargs: Any) -> None:
    raise KeyboardInterrupt


def _workspace(tmp_path: Path, body: str = "") -> Path:
    from contextgrid.evalset import write_jsonl

    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    write_jsonl(evalset(), tmp_path / "evalset.jsonl")
    config = tmp_path / "experiment.yaml"
    config.write_text(f"corpus: ./docs\nevalset: ./evalset.jsonl\n{body}", encoding="utf-8")
    return config


def test_ctrl_c_prints_one_line_and_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/reference/cli`: "Every command catches its own exceptions and prints `error: <message>`
    to stderr rather than a Python traceback". Ctrl-C printed 51 lines of traceback ending in
    `tokens.py`, which is a promise broken in the one moment a user is already unhappy."""
    from contextgrid.cli import main

    monkeypatch.setattr("contextgrid.config.run", _interrupt)
    code = main(["run", str(_workspace(tmp_path))])

    errors = capsys.readouterr().err
    assert code == 130, "128 + SIGINT, the convention every shell already knows"
    assert "Traceback" not in errors, errors
    assert "Ctrl-C" in errors, errors


def test_ctrl_c_says_what_happened_to_the_work_just_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A disk cache survives an interrupt, so a re-run resumes rather than starting again.
    That is worth one clause and is the difference between "I lost ten minutes" and "I lost
    nothing"."""
    from contextgrid.cli import main

    monkeypatch.setattr("contextgrid.config.run", _interrupt)
    main(["run", str(_workspace(tmp_path, "run:\n  cache: disk\n"))])

    errors = capsys.readouterr().err
    assert "re-run" in errors or "rerun" in errors, errors
    assert "cache" in errors, errors


def test_a_memory_cache_does_not_claim_the_work_was_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run.cache` defaults to `memory`, which dies with the process. Telling that user their
    work is kept would be the more comfortable sentence and a false one."""
    from contextgrid.cli import main

    monkeypatch.setattr("contextgrid.config.run", _interrupt)
    main(["run", str(_workspace(tmp_path))])

    errors = capsys.readouterr().err
    assert "run.cache: disk" in errors, errors


def test_the_interrupt_still_stops_the_sweep_rather_than_one_row() -> None:
    """`_run_one_or_report` catches `Exception`. `KeyboardInterrupt` is not one, by language
    definition, and must keep travelling."""
    from contextgrid.grid.matrix import matrix
    from contextgrid.grid.runner import Runner

    with pytest.raises(KeyboardInterrupt):
        Runner(corpus=corpus()).run(
            matrix(chunker=["recursive:512", "recursive:256"]),
            evalset(),
            on_progress=_interrupt,
        )


# ---------------------------------------------------------------------------
# 10. `check` passing an output directory it cannot write
# ---------------------------------------------------------------------------

not_root = pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 0)() == 0,
    reason="root can write anywhere, so a read-only directory proves nothing",
)


@not_root
def test_check_refuses_an_output_directory_it_cannot_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The sweep ran to the end and died on `manifest.json`, with everything discarded. On a
    hosted embedder that is money spent and thrown away, and `check` exists to catch it."""
    from contextgrid.cli import main

    out = tmp_path / "ro_out"
    out.mkdir()
    out.chmod(0o555)
    try:
        assert main(["check", str(_workspace(tmp_path, "report:\n  out: ./ro_out\n"))]) == 1
        errors = capsys.readouterr().err
        assert "ro_out" in errors, errors
        assert "report.out" in errors, errors
    finally:
        out.chmod(0o755)


@not_root
def test_check_refuses_an_output_path_whose_parent_cannot_be_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`write_report` calls `mkdir(parents=True)`, so a directory that does not exist yet is
    fine -- unless nothing above it can be created either."""
    from contextgrid.cli import main

    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        config = _workspace(tmp_path, "report:\n  out: ./locked/results\n")
        assert main(["check", str(config)]) == 1
        assert "locked" in capsys.readouterr().err
    finally:
        parent.chmod(0o755)


def test_check_accepts_an_output_directory_that_does_not_exist_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ordinary case: `report.out: ./results` on a first run. `write_report` will create it."""
    from contextgrid.cli import main

    assert main(["check", str(_workspace(tmp_path, "report:\n  out: ./results\n"))]) == 0
    assert "config is valid." in capsys.readouterr().out


def test_check_does_not_create_the_output_directory_as_a_side_effect(tmp_path: Path) -> None:
    """Checking a config must not leave anything behind -- a `results/` a sweep never ran is a
    directory somebody has to wonder about."""
    from contextgrid.cli import main

    main(["check", str(_workspace(tmp_path, "report:\n  out: ./results\n"))])

    assert not (tmp_path / "results").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "docs",
        "evalset.jsonl",
        "experiment.yaml",
    ]


def test_ctrl_c_in_a_command_with_no_config_is_still_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run` tailors the sentence because it knows the cache setting. Every other command --
    `profile`, `sweep`, `validate` -- has no config to read, and still owes the user a line
    rather than a traceback."""
    from contextgrid.cli import main
    from contextgrid.lab import Lab

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(DOCS["return-policy.md"])
    monkeypatch.setattr(Lab, "fingerprint", _interrupt)

    code = main(["profile", str(docs)])
    errors = capsys.readouterr().err

    assert code == 130
    assert errors.strip() == "error: stopped by Ctrl-C."
    assert "Traceback" not in errors

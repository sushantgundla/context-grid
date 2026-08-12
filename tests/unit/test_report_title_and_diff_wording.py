"""The report's title, and what a manifest diff claims when nothing changed.

Two things the documentation promised and the tool did not do.

`docs/reference/reports.md:182` says the report's H1 carries the experiment's name. The
machinery was there -- `results_to_markdown(..., name=...)` -- but `write_report` never passed
it, so a directory of sweeps was a directory of files all titled
`# Retrieval configuration comparison`, with nothing above the fold to say which sweep produced
which.

`docs/guide/cli.md:408` documents that `contextgrid diff` compares the two *winning*
configurations' manifests, not the config files. The caveat was in the docs; the message
overclaimed anyway. It said "these two runs should have produced identical numbers", which is
more than a manifest can support: a 7-configuration `ofat` run and a 27-configuration
`factorial` run over the same grid write byte-identical manifests whenever they happen to pick
the same winner. A reader told those two runs were the same goes hunting for nondeterminism
that is not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.core.documents import MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.corpus import Corpus
from contextgrid.evalset import write_jsonl
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config
from contextgrid.report import build_manifest, explain_diff, results_to_markdown
from contextgrid.report.results import Results, RunResult
from tests.support import API_DOCS, CONTRACT


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )


@pytest.fixture
def evalset() -> EvalSet:
    rows = [
        ("q1", "How much notice to terminate for convenience?", "contract.md", "thirty days"),
        ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
        ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
        ("q4", "What happens on a material breach?", "contract.md", "fifteen days"),
    ]
    return EvalSet(
        id="demo",
        items=tuple(
            EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
            for i, q, s, t in rows
        ),
    )


@pytest.fixture
def one_result() -> Results:
    return Results(runs=[RunResult(config=Config(), metrics={"recall@5": 0.5}, scored_queries=10)])


# ---------------------------------------------------------------------------
# the title, straight through `results_to_markdown`
#
# This is the half `report/export.py` owns, and it is green on its own -- it does not wait on
# whatever the caller decides to pass.
# ---------------------------------------------------------------------------


def test_a_name_passed_in_becomes_the_title(one_result: Results) -> None:
    report = results_to_markdown(one_result, name="northwind-sweep")
    assert report.startswith("# northwind-sweep — retrieval configuration comparison")


def test_no_name_keeps_the_documented_fallback(one_result: Results) -> None:
    assert results_to_markdown(one_result).startswith("# Retrieval configuration comparison")


def test_the_bare_word_experiment_is_not_a_name(one_result: Results) -> None:
    """`ExperimentConfig` fills in `experiment` when no config file was read. It names no sweep
    in particular, so `# experiment — ...` would say less than the generic title does."""
    assert results_to_markdown(one_result, name="experiment").startswith(
        "# Retrieval configuration comparison"
    )


def test_a_blank_name_is_not_a_name(one_result: Results) -> None:
    assert results_to_markdown(one_result, name="   ").startswith(
        "# Retrieval configuration comparison"
    )


def test_a_short_name_still_gets_the_descriptive_half(one_result: Results) -> None:
    """A one-letter name is never a bare `# n` -- the heading always appends the rest, so even
    the shortest name produces a title that says what the document is."""
    assert results_to_markdown(one_result, name="n").startswith(
        "# n — retrieval configuration comparison"
    )


# ---------------------------------------------------------------------------
# the title, for the name a real config file actually yields
#
# `ExperimentConfig.name` defaults to the config's filename stem, so what a user gets depends on
# what their file is called. These pin that down without waiting on the caller.
# ---------------------------------------------------------------------------


def _config_named(directory: Path, *, filename: str, name: str | None) -> str:
    """`ExperimentConfig.name` for a real config file. `name=None` omits the `name:` key."""
    from contextgrid.config.loader import load

    docs = directory / "documents"
    docs.mkdir(exist_ok=True)
    (docs / "contract.md").write_text(CONTRACT)
    write_jsonl(
        EvalSet(
            id="demo",
            items=(
                EvalItem(
                    id="q1",
                    question="How much notice to terminate for convenience?",
                    anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
                ),
            ),
        ),
        directory / "questions.jsonl",
    )

    named = "" if name is None else f"name: {name}\n"
    path = directory / filename
    path.write_text(f"{named}corpus: ./documents\nevalset: ./questions.jsonl\n")
    return load(path).name


def test_a_named_config_titles_the_report_after_the_name(
    tmp_path: Path, one_result: Results
) -> None:
    name = _config_named(tmp_path, filename="sweep.yaml", name="northwind-sweep")
    assert name == "northwind-sweep"
    assert results_to_markdown(one_result, name=name).startswith(
        "# northwind-sweep — retrieval configuration comparison"
    )


def test_a_config_named_by_its_filename_titles_the_report_after_the_file(
    tmp_path: Path, one_result: Results
) -> None:
    """`name` defaults to the config's filename (`configuration.md:56`). A stem is still the
    user's own word for the sweep, and it is already what the console banner, `experiment.yaml`
    and `winning-config.yaml` print -- so the report agrees with them rather than falling
    back."""
    name = _config_named(tmp_path, filename="northwind.yaml", name=None)
    assert name == "northwind"
    assert results_to_markdown(one_result, name=name).startswith(
        "# northwind — retrieval configuration comparison"
    )


def test_a_config_with_no_name_key_at_all_falls_back(tmp_path: Path, one_result: Results) -> None:
    """First way to the sentinel: no `name:` in a file called `experiment.yaml`, so the stem
    default lands on `experiment`."""
    name = _config_named(tmp_path, filename="experiment.yaml", name=None)
    assert name == "experiment"
    assert results_to_markdown(one_result, name=name).startswith(
        "# Retrieval configuration comparison"
    )


def test_a_config_that_writes_name_experiment_by_hand_falls_back(
    tmp_path: Path, one_result: Results
) -> None:
    """Second way to the same sentinel: `name: experiment` typed out in a file called something
    else. The user gets the title they would have had anyway."""
    name = _config_named(tmp_path, filename="northwind.yaml", name="experiment")
    assert name == "experiment"
    assert results_to_markdown(one_result, name=name).startswith(
        "# Retrieval configuration comparison"
    )


# ---------------------------------------------------------------------------
# the caller
# ---------------------------------------------------------------------------


def test_the_written_report_is_titled_after_the_experiment(tmp_path: Path) -> None:
    """The name reached the console banner, `experiment.yaml` and `winning-config.yaml`, and was
    dropped on the one file a human reads first."""
    from contextgrid.config.loader import load, run, write_report

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    (docs / "api.md").write_text(API_DOCS)
    write_jsonl(
        EvalSet(
            id="demo",
            items=(
                EvalItem(
                    id="q1",
                    question="How much notice to terminate for convenience?",
                    anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
                ),
                EvalItem(
                    id="q2",
                    question="Which header carries the API key?",
                    anchors=(GoldAnchor(source_id="api.md", quote="X-Api-Key"),),
                ),
            ),
        ),
        tmp_path / "questions.jsonl",
    )
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "name: northwind-sweep\ncorpus: ./documents\nevalset: ./questions.jsonl\n"
        "run:\n  headline: recall@3\nreport:\n  out: ./out\n"
    )

    config = load(config_path)
    write_report(config, run(config))
    title = (tmp_path / "out" / "report.md").read_text().splitlines()[0]

    assert title == "# northwind-sweep — retrieval configuration comparison"


# ---------------------------------------------------------------------------
# what the diff claims when nothing changed
# ---------------------------------------------------------------------------


def test_identical_manifests_do_not_claim_the_two_runs_were_the_same(
    corpus: Corpus, evalset: EvalSet
) -> None:
    manifest = build_manifest(Config(), corpus, evalset)
    explanation = explain_diff(manifest, manifest)

    # What was compared, said out loud.
    assert "winning configuration" in explanation
    assert "not the sweep that found it" in explanation
    # The claim scoped to the one configuration, rather than to the runs.
    assert "identical numbers for this one configuration" in explanation
    # The warning that makes the manifest worth keeping is still there.
    assert "something outside the manifest is affecting results" in explanation


def test_an_ofat_and_a_factorial_run_over_one_grid_are_not_called_identical(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """The repro. Same grid, two modes, very different amounts of work -- and if they agree on a
    winner, the manifests are byte-identical, because a manifest records the winner and not the
    sweep."""
    grid = matrix(chunker=["sentence:1", "fixed:12,overlap=0"], index=["dense", "bm25"])
    runner = Runner(corpus=corpus, headline="recall@3")

    ofat = runner.run(grid, evalset, mode="ofat")
    factorial = runner.run(grid, evalset, mode="factorial")
    assert len(ofat) < len(factorial), "the two modes must do different amounts of work"

    ofat_winner = ofat.best("recall@3")
    factorial_winner = factorial.best("recall@3")
    assert ofat_winner is not None
    assert factorial_winner is not None
    if ofat_winner.config.as_dict() != factorial_winner.config.as_dict():
        pytest.skip("the two modes picked different winners, so there is a diff to show")

    explanation = explain_diff(
        build_manifest(ofat_winner.config, corpus, evalset),
        build_manifest(factorial_winner.config, corpus, evalset),
    )

    assert "Nothing in these two manifests is different" in explanation
    # The sentence that sent a reader hunting for nondeterminism that was not there.
    assert "these two runs should have produced identical numbers." not in explanation
    assert "That is not the same as the two runs being identical." in explanation

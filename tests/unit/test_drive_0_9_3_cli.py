"""What a documentation-driven drive of 0.9.2 found in the CLI, the matrix and the config.

Seven findings, and the theme is the same in six of them: the tool knew something and either
said nothing, or said it about the wrong thing.

* A warning blamed the chunker for a collapse an *ingestion* strategy had caused, and offered
  advice ("sweep smaller sizes") that could not have helped. A false alarm from the warning
  whose whole job is to be believed.
* `widened` was folded onto plain search -- correctly -- and nothing anywhere said so, so a
  sweep naming it read as a measured tie against a row that never ran.
* `check` accepted `candidates: -3` while rejecting `run.k: -1` and `recursive:-5` in the same
  breath, and the nonsense reached a leaderboard label.
* `check` accepted a model-backed *ingestion* strategy with no `run.model`, where the same
  omission on `transform`, `retrieval` and `generator` is a clean refusal.
* `plugins` promised its listing reflected what is installed here and never asked.
* A malformed YAML file was reported against `<unicode string>` rather than its own name.
* And a Markdown table in the CLI reference had a paragraph inserted into the middle of it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import contextgrid as cg
from contextgrid.cli import main
from contextgrid.core.warnings import Severity, WarningCode
from contextgrid.evalset import write_jsonl
from tests.support import API_DOCS, CONTRACT

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# a workspace `check` and `run` can be pointed at
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    (docs / "api.md").write_text(API_DOCS)
    write_jsonl(
        cg.EvalSet(
            id="es",
            items=(
                cg.EvalItem(
                    id="q1",
                    question="How much notice to terminate for convenience?",
                    anchors=(cg.GoldAnchor(source_id="contract.md", quote="thirty days"),),
                ),
                cg.EvalItem(
                    id="q2",
                    question="Which header carries the API key?",
                    anchors=(cg.GoldAnchor(source_id="api.md", quote="X-Api-Key"),),
                ),
            ),
        ),
        tmp_path / "evalset.jsonl",
    )
    return tmp_path


def write_config(workspace: Path, body: str) -> Path:
    path = workspace / "experiment.yaml"
    path.write_text(f"corpus: ./docs\nevalset: ./evalset.jsonl\n{body}", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. the table the 0.9.2 documentation broke in half
# ---------------------------------------------------------------------------


def _table_rows(lines: list[str], start: int) -> list[str]:
    """The run of table lines beginning at `start`. A table ends at the first line that is not
    one, which is exactly the rule the renderer follows and the reason this broke."""
    rows: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_the_six_commands_are_all_in_one_table() -> None:
    """A paragraph and a fenced block were inserted between two rows, so the four rows below
    them fell out of the table and render as literal pipe-text."""
    page = (REPO_ROOT / "docs-site" / "reference" / "cli.mdx").read_text(encoding="utf-8")
    lines = page.splitlines()

    heading = next(i for i, line in enumerate(lines) if line.startswith("## The other six"))
    start = next(i for i in range(heading, len(lines)) if lines[i].startswith("| Command"))
    rows = "\n".join(_table_rows(lines, start))

    for command in ("profile", "sweep", "plugins", "evalset", "diff", "validate"):
        assert f"`contextgrid {command}" in rows, (
            f"the {command} row is outside the table, so it renders as literal pipe-text"
        )


def test_every_table_in_the_cli_reference_is_unbroken() -> None:
    """The general form of the same mistake: no table may hold a non-table line."""
    page = (REPO_ROOT / "docs-site" / "reference" / "cli.mdx").read_text(encoding="utf-8")

    stray = [
        line
        for previous, line in zip(page.splitlines(), page.splitlines()[1:], strict=False)
        if line.startswith("|") and not previous.startswith("|") and "---" not in line
    ]
    # Every table starts with exactly one such line: its header row.
    headers = [line for line in page.splitlines() if re.match(r"^\|\s*(Command|Exit)\b", line)]
    assert stray == headers, f"a table row is separated from its table: {stray}"


# ---------------------------------------------------------------------------
# 2. the warning that blamed the chunker for an ingestion strategy's doing
# ---------------------------------------------------------------------------

LONG_DOCS = {
    "return-policy.md": (
        "# Return Policy\n\n"
        "Items may be returned within 30 days of delivery for a full refund.\n\n"
        "The item must be unused and in its original packaging.\n\n"
        "Refunds are issued to the original payment method within ten working days.\n\n"
        "Items marked final sale cannot be returned under any circumstances.\n"
    ),
    "shipping.md": (
        "# Shipping\n\n"
        "Standard shipping takes 3 to 7 business days and costs $5.99.\n\n"
        "Orders over $50 ship free within the mainland.\n\n"
        "Express shipping arrives the next working day and costs $14.99.\n\n"
        "We do not ship to post office boxes or freight forwarders.\n"
    ),
    "warranty.md": (
        "# Warranty\n\n"
        "All electronics carry a 1 year manufacturer warranty.\n\n"
        "The warranty covers defects in materials and workmanship.\n\n"
        "Accidental damage and normal wear are not covered by the warranty.\n\n"
        "Claims are made through the support portal with the order number.\n"
    ),
}


def _run(chunker: str, ingestion: str | None = None) -> Any:
    corpus = cg.Corpus.from_texts(LONG_DOCS, media_type=cg.MediaType.MARKDOWN)
    evalset = cg.EvalSet(
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
    lab = cg.Lab(corpus)
    lab.grid(chunker=[chunker], ingestion=[ingestion])
    return lab.run(evalset)


def _one_chunk_warnings(results: Any) -> list[Any]:
    return [w for w in results.runs[0].warnings if w.code is WarningCode.ONE_CHUNK_PER_DOCUMENT]


def test_an_ingestion_strategy_that_collapses_units_is_not_blamed_on_the_chunker() -> None:
    """`parent-document` returns the passage its small chunks came from -- by design, and
    documented. With four chunks a document and a group of four, one passage comes back per
    document, and the chunker cut perfectly well on the way there.

    The warning named the chunker and said "sweep smaller sizes", which is advice that cannot
    work: smaller chunks make more of them, and the strategy groups them back to the same
    passage. A warning that exists to be believed cannot point at the wrong axis.
    """
    plain = _run("fixed:20")
    assert plain.runs[0].chunk_count > 3, "fixture assumes this chunker splits; it stopped"

    grouped = _run("fixed:20", "parent-document:4")
    assert grouped.runs[0].chunk_count <= 3, "fixture assumes the group collapses to one a doc"

    raised = _one_chunk_warnings(grouped)
    assert raised, "one passage per document is still worth saying"
    message = raised[0].message
    assert "parent-document" in message, f"name the axis that collapsed them: {message}"
    assert "sweep smaller sizes" not in message, (
        f"smaller chunks cannot help; the strategy groups them back: {message}"
    )
    assert raised[0].subject is not None
    assert "fixed:20" not in raised[0].subject, (
        "the chunker is not the subject of a fact about the ingestion axis"
    )


def test_the_chunker_is_still_blamed_when_the_chunker_is_the_cause() -> None:
    """The original finding, unchanged: a chunk size above the documents leaves them whole."""
    raised = _one_chunk_warnings(_run("recursive:512"))

    assert raised, "a leaderboard that cannot see the chunker axis has to say so"
    assert "recursive:512" in raised[0].message
    assert "sweep smaller sizes" in raised[0].message
    assert raised[0].severity is not Severity.INFO


def test_a_strategy_that_returns_the_chunker_s_own_units_still_blames_the_chunker() -> None:
    """`sentence-window` is scored on the chunks themselves, so nothing was merged away and
    the chunker really is the one axis that could have made a difference."""
    raised = _one_chunk_warnings(_run("recursive:512", "sentence-window:1"))

    assert raised
    assert "recursive:512" in raised[0].message


# ---------------------------------------------------------------------------
# 3. `widened` folded onto plain search, and nothing said so
# ---------------------------------------------------------------------------


def test_a_widened_arm_that_becomes_plain_search_is_reported() -> None:
    """Folding it is right -- with no reranker, no transform, no ingestion and an exact index
    the surplus is provably thrown away. Doing it in silence is not: the row is labelled as
    plain search, the manifest records `retrieval: null`, and in a sweep naming both arms the
    result reads as a measured tie against a configuration that never ran."""
    from contextgrid.grid.matrix import matrix

    configs, report = matrix(retrieval=["widened:factor=8"]).expand_with_report("ofat")

    assert len(configs) == 1
    assert configs[0].retrieval is None, "the fold itself is correct and stays"
    assert report.rewrites, "a row that quietly became a different configuration has to be said"
    note = " ".join(report.rewrites)
    assert "widened:factor=8" in note, note
    assert "plain search" in note, note


def test_a_widened_arm_that_survives_is_not_reported() -> None:
    """With a reranker downstream the wider net is exactly what gets reordered, so nothing was
    rewritten and there is nothing to say. A note here would be noise."""
    from contextgrid.grid.matrix import matrix

    _, report = matrix(retrieval=["widened:factor=8"], reranker=["lexical"]).expand_with_report(
        "ofat"
    )

    assert not report.rewrites


def test_a_staged_sweep_reports_the_fold_for_the_stage_it_happened_in() -> None:
    """Staged freezes a different configuration in front of each stage, so the same arm is
    plain search before a reranker is chosen and a real arm after it. The report is per stage
    for that reason, rather than once for a matrix that never has one shape."""
    from contextgrid.grid.matrix import matrix
    from contextgrid.pipeline import Config

    grid = matrix(retrieval=["simple", "widened:factor=8"])

    _, bare = grid.stage_configs_with_report("retrieval", Config())
    _, reranked = grid.stage_configs_with_report("retrieval", Config(reranker="lexical"))

    assert bare.rewrites, "with nothing downstream the arm is plain search"
    assert not reranked.rewrites, "with a reranker frozen in front of it, it is a real arm"


def test_a_run_says_out_loud_that_widened_ran_as_plain_search(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console is where the leaderboard is read, so it is where this belongs."""
    config = write_config(workspace, "grid:\n  retrieval: [widened:factor=8]\n")
    assert main(["run", str(config)]) == 0

    errors = capsys.readouterr().err
    assert "widened:factor=8" in errors, errors
    assert "plain search" in errors, errors


def test_the_fold_has_a_code_of_its_own(workspace: Path) -> None:
    """It was first raised as `NON_DETERMINISTIC_STAGE`, which is the wrong fact about it:
    nothing here is non-deterministic, the arm folded onto plain search deliberately and would
    do it again on every run.

    A code borrowed because it is roughly the right shape is how `BUDGET_REACHED` came to mean
    both "the sweep stopped" and "this model has no published price", which broke the filter
    `/lab/running` tells people to write. One code, one fact.
    """
    from contextgrid.config import load, run

    results = run(load(write_config(workspace, "grid:\n  retrieval: [widened:factor=8]\n")))

    folded = results.warnings.of_code(WarningCode.ARM_NOT_MEASURED)
    assert len(folded) == 1, [w.code.value for w in results.warnings]
    assert "widened:factor=8" in folded[0].message
    assert folded[0].severity is Severity.CAUTION, (
        "the CLI drops INFO whenever a run produced results, and this one always has"
    )
    assert not [
        w
        for w in results.warnings
        if w.code is WarningCode.NON_DETERMINISTIC_STAGE and "plain search" in w.message
    ]


# ---------------------------------------------------------------------------
# 4. a negative candidate depth that reached a leaderboard label
# ---------------------------------------------------------------------------


def test_check_rejects_a_negative_candidate_depth(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run.k: -1` and `chunker: recursive:-5` are both refused by the same command in the same
    words. `candidates: -3` validated, then reached the leaderboard as `... · lexical@-3`."""
    config = write_config(workspace, "grid:\n  candidates: [-3]\n  reranker: [lexical]\n")
    assert main(["check", str(config)]) == 1

    errors = capsys.readouterr().err
    assert "candidates must be at least 1, got -3" in errors, errors


def test_zero_candidates_is_refused_too(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "grid:\n  candidates: 0\n  reranker: [lexical]\n")
    assert main(["check", str(config)]) == 1
    assert "at least 1, got 0" in capsys.readouterr().err


def test_the_python_api_refuses_it_in_the_same_words() -> None:
    """`Matrix` has always checked `k` here. Depth is the same kind of number."""
    from contextgrid.grid.matrix import Matrix, MatrixError, matrix

    with pytest.raises(MatrixError, match="candidates must be at least 1, got -3"):
        Matrix(candidates=(-3,))
    with pytest.raises(MatrixError, match="candidates must be at least 1"):
        matrix(candidates=[50, -1])


def test_a_real_candidate_depth_is_untouched() -> None:
    from contextgrid.grid.matrix import matrix

    assert matrix(candidates=[10, 50]).candidates == (10, 50)


# ---------------------------------------------------------------------------
# 5. an ingestion strategy that needs a model and never says so
# ---------------------------------------------------------------------------


def test_check_refuses_a_model_backed_ingestion_strategy_with_no_model(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`transform: hyde`, `retrieval: agentic` and `generator: llm` all refuse cleanly here.
    `ingestion: contextual` validated, ran, fell back to plain chunking on every chunk, and
    scored a row labelled `contextual` that a user would act on."""
    config = write_config(workspace, "grid:\n  ingestion: [contextual]\n")
    assert main(["check", str(config)]) == 1

    errors = capsys.readouterr().err
    assert "contextual" in errors
    assert "needs a model" in errors, errors
    assert "run.model" in errors, errors
    # The way out, named -- the same courtesy the transform axis's error extends.
    assert "plain" in errors, errors


def test_the_free_ingestion_strategies_are_unaffected(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "grid:\n  ingestion: [parent-document, sentence-window]\n")
    assert main(["check", str(config)]) == 0
    assert "config is valid." in capsys.readouterr().out


def test_a_spec_that_names_nothing_is_left_for_the_builder_to_report() -> None:
    """Two errors about one typo, in two wordings, help nobody -- and the other one lists the
    names that do exist."""
    from contextgrid.config.plugins import model_missing_for

    assert model_missing_for("not-a-real-strategy", None) is None


def test_a_model_makes_the_paid_strategy_checkable(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `run.model` set there is nothing to refuse -- exactly as for the transform axis."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    config = write_config(
        workspace, "grid:\n  ingestion: [contextual]\nrun:\n  model: openai:gpt-4o-mini\n"
    )
    assert main(["check", str(config)]) == 0
    assert "config is valid." in capsys.readouterr().out


def test_run_refuses_it_too_rather_than_scoring_a_row_that_never_ran(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`check` catching it is only half the promise: `run` is what spends the time, and a
    leaderboard row labelled `contextual` that made no model call is the actual damage."""
    config = write_config(workspace, "grid:\n  ingestion: [contextual]\n")
    assert main(["run", str(config)]) == 1

    captured = capsys.readouterr()
    assert "needs a model" in captured.err, captured.err
    assert "1.000" not in captured.out, "no row may be scored for a strategy that never ran"


# ---------------------------------------------------------------------------
# 6. `plugins` claims to reflect this installation and never asked
# ---------------------------------------------------------------------------


def _line_for(printed: str, name: str) -> str:
    return next(line for line in printed.splitlines() if line.strip().startswith(f"{name} "))


def test_plugins_marks_a_name_whose_extra_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/reference/cli`: "for this installation specifically -- which extras are installed
    changes what shows up." Nothing in the command ever asked, so `marker` was listed
    unqualified on a machine with no `marker` module at all."""
    from contextgrid.config import plugins as plugin_support

    real = plugin_support._dependency_present
    monkeypatch.setattr(
        plugin_support,
        "_dependency_present",
        lambda package: False if package == "marker-pdf" else real(package),
    )

    assert main(["plugins", "--family", "parser"]) == 0
    printed = capsys.readouterr().out

    assert "-" in _line_for(printed, "marker").replace("marker", "", 1)
    assert 'pip install "context-grid[parse-marker]"' in printed, printed
    # The ones that are installed stay clean: a marker on every line marks nothing.
    assert "-" not in _line_for(printed, "markdown").replace("markdown", "", 1)


def test_plugins_asks_the_same_question_check_and_init_ask(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One fact read in three places, rather than a third opinion that can drift from the
    other two -- `extra_missing_for` is what `check` and the starter template both use."""
    from contextgrid.config import plugins as plugin_support

    asked: list[str] = []
    real = plugin_support._dependency_present

    def watched(package: str) -> bool:
        asked.append(package)
        return real(package)

    monkeypatch.setattr(plugin_support, "_dependency_present", watched)
    assert main(["plugins", "--family", "index"]) == 0

    assert "faiss-cpu" in asked, asked


def test_plugins_still_stars_the_ones_that_need_a_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The existing column keeps its meaning: `*` is about a model, not about an install."""
    assert main(["plugins", "--family", "transform"]) == 0
    printed = capsys.readouterr().out

    assert "*" in _line_for(printed, "hyde")
    assert "* needs a model. Set `run.model` in your config to use it." in printed


# ---------------------------------------------------------------------------
# 7. the placeholder PyYAML puts where the filename belongs
# ---------------------------------------------------------------------------


def test_a_malformed_config_is_reported_against_its_own_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Everything else in the message was usable -- the line, the column, the caret. Only
    `in "<unicode string>"` said nothing to anybody who did not write PyYAML."""
    bad = tmp_path / "bad_yaml.yaml"
    bad.write_text("corpus: ./docs\ngrid:\n  chunker: [recursive:128\n", encoding="utf-8")

    assert main(["check", str(bad)]) == 1

    errors = capsys.readouterr().err
    assert "<unicode string>" not in errors, errors
    assert "bad_yaml.yaml" in errors
    # The useful half is untouched.
    assert "line 3" in errors, errors
    assert "column" in errors, errors


def test_a_config_parsed_from_a_string_says_so(tmp_path: Path) -> None:
    """`loads()` has no file to name, and `<unicode string>` is no better an answer there."""
    from contextgrid.config import loads
    from contextgrid.config.schema import ConfigError

    with pytest.raises(ConfigError) as caught:
        loads("corpus: ./docs\ngrid:\n  chunker: [recursive:128\n")

    assert "<unicode string>" not in str(caught.value)

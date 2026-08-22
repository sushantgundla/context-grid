"""What `check` catches before a sweep starts, and what `run` calls a failure.

Both commands were quietly optimistic. `check` validated the shape of a config and never
built anything the config named, so a typo in a spec string -- the place typos actually
happen -- reached the user as a failure minutes into an expensive `run`. And a `run` that
measured nothing at all still exited 0, which in CI is a green build for an experiment that
produced no numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.cli import main
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.evalset import write_jsonl
from tests.support import API_DOCS, CONTRACT

QUESTIONS = [
    ("q1", "How much notice to terminate for convenience?", "contract.md", "thirty days"),
    ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A directory holding a real corpus and a real eval set, and nothing else."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    (docs / "api.md").write_text(API_DOCS)
    write_jsonl(
        EvalSet(
            id="es",
            items=tuple(
                EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
                for i, q, s, t in QUESTIONS
            ),
        ),
        tmp_path / "evalset.jsonl",
    )
    return tmp_path


def write_config(workspace: Path, body: str, *, corpus: str = "./docs") -> Path:
    path = workspace / "experiment.yaml"
    path.write_text(f"corpus: {corpus}\nevalset: ./evalset.jsonl\n{body}", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_passes_a_config_that_can_actually_run(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "grid:\n  chunker: [recursive:512, sentence:2]\n")
    assert main(["check", str(config)]) == 0
    assert "config is valid." in capsys.readouterr().out


def test_check_rejects_a_plugin_name_nothing_is_registered_under(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this command exists for: `recursve:512` for `recursive:512`."""
    config = write_config(workspace, "grid:\n  chunker: banana:999\n")
    assert main(["check", str(config)]) == 1

    errors = capsys.readouterr().err
    assert "no chunker named 'banana'" in errors
    # The same text `run` produces, so nobody has to learn two vocabularies for one mistake.
    assert "Available:" in errors


def test_check_rejects_parameters_the_plugin_will_not_accept(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "grid:\n  chunker: recursive:-5\n")
    assert main(["check", str(config)]) == 1

    errors = capsys.readouterr().err
    assert "chunk size must be positive, got -5" in errors
    # Which axis value said it. Six chunkers in a config means six candidates for the blame.
    assert "recursive:-5" in errors


def test_check_rejects_a_corpus_directory_with_nothing_readable_in_it(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty `./documents` is a clone without its data, or a path off by one directory."""
    empty = workspace / "empty"
    empty.mkdir()
    (empty / "notes.bin").write_bytes(b"\x00\x01")

    config = write_config(workspace, "", corpus="./empty")
    assert main(["check", str(config)]) == 1
    assert "no files under" in capsys.readouterr().err


def test_check_reports_every_bad_axis_at_once(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fixing one typo per run of `check` is the slow way to fix four of them."""
    config = write_config(
        workspace, "grid:\n  chunker: banana:999\n  index: pineapple\n  parser: kiwi\n"
    )
    assert main(["check", str(config)]) == 1

    errors = capsys.readouterr().err
    assert "no chunker named 'banana'" in errors
    assert "no index named 'pineapple'" in errors
    assert "no parser named 'kiwi'" in errors


def test_check_does_not_complain_about_a_combination_the_sweep_will_never_run(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`embedder: null` is legitimate beside `bm25`, and the matrix drops it beside `dense`."""
    config = write_config(workspace, "grid:\n  embedder: [tfidf, null]\n  index: [dense, bm25]\n")
    assert main(["check", str(config)]) == 0
    assert "config is valid." in capsys.readouterr().out


def test_check_still_reports_a_missing_path(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "", corpus="./absent")
    assert main(["check", str(config)]) == 1
    assert "corpus not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_succeeds_when_it_measured_something(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "run:\n  headline: recall@2\n")
    assert main(["run", str(config), "--quiet"]) == 0
    assert "scored best" in capsys.readouterr().out


def test_run_fails_when_the_budget_stopped_it_before_anything_ran(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A green CI build for an experiment that measured nothing is the failure here."""
    config = write_config(workspace, "run:\n  budget_usd: 0.0\n")
    assert main(["run", str(config), "--quiet"]) == 1

    errors = capsys.readouterr().err
    assert "no configurations were run" in errors
    # And why. The reason was recorded as a warning, which only ever reached the report file.
    assert "budget ran out" in errors


def test_run_fails_when_the_matrix_had_no_runnable_combination(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace, "grid:\n  embedder: null\n  index: dense\n")
    assert main(["run", str(config), "--quiet"]) == 1

    errors = capsys.readouterr().err
    assert "no configurations were run" in errors
    assert "cannot be built and were skipped" in errors


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
#
# `sweep` is the same experiment as `run` with the axes on the command line instead of in a
# file, and it printed the same empty leaderboard for the same reasons -- and exited 0 while
# `run` exited 1. `_measured_something` was written for this rule and wired into one of the
# two commands. `sweep` is the flags-only one-shot, which is the one most likely to be sitting
# in a CI job, so it is the worse of the two places to leave a green build for a sweep that
# measured nothing.


def test_sweep_succeeds_when_it_measured_something(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["sweep", str(workspace / "docs"), str(workspace / "evalset.jsonl"), "--metric", "recall@2"]
    )
    assert exit_code == 0
    assert "scored best" in capsys.readouterr().out


def test_sweep_fails_when_no_configuration_scored_a_question(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An eval set with no questions in it swept the whole matrix and exited 0."""
    empty = workspace / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert main(["sweep", str(workspace / "docs"), str(empty)]) == 1

    captured = capsys.readouterr()
    assert "no configuration scored a single question" in captured.err
    # And the reason, on stdout, under the leaderboard it explains -- `sweep` writes no report
    # unless `--bundle` is passed, so stderr and the console are all there is.
    assert "has no questions in it" in captured.out


def test_sweep_still_succeeds_when_the_budget_stopped_it_partway(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial leaderboard is a real result, and `--budget-seconds` has to stay usable in CI.

    The stop is forced rather than timed, for the reason `test_drive_cost_findings` gives: a
    budget small enough to bite is a race between a warm machine and a cold one, and what is
    being pinned here is the exit code, not the clock.
    """
    from unittest.mock import patch

    from contextgrid.grid.runner import Budget

    calls = {"n": 0}

    def exceeded(self: Budget) -> str | None:
        calls["n"] += 1
        return "the 30s budget ran out" if calls["n"] > 1 else None

    with patch.object(Budget, "exceeded", exceeded):
        exit_code = main(
            [
                "sweep",
                str(workspace / "docs"),
                str(workspace / "evalset.jsonl"),
                "--metric",
                "recall@2",
                "--chunker",
                "recursive:512",
                "--chunker",
                "sentence:2",
                "--chunker",
                "sentence:3",
                "--budget-seconds",
                "30",
            ]
        )

    assert exit_code == 0, "the leaderboard it printed is real, just partial"
    assert "budget ran out" in capsys.readouterr().err

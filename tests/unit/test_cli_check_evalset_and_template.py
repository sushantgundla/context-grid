"""What `check` knows about the questions file, and what `init` writes about the plugins.

Both commands stopped one step short of the promise made for them.

`check` resolved the eval set's *path* and never opened the file, so a JSONL of something
that is not JSON, or a spreadsheet whose question column is called `foo`, got "config is
valid." and then stopped `run` a second later -- with an error `check` had been standing
right next to. `cli.md` says every message `check` prints is "the one `run` would have
printed" and only arrives earlier, which was true of every message except the ones about the
file the whole sweep is scored against.

`init` wrote a template that contradicted the three things the docs say it writes. Plugins
this installation runs perfectly well were demoted to a comment; `marker`, the one parser it
genuinely cannot run, was left out of the file altogether; and no comment named an extra or
an install command, which is the entire point of writing the comment.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from contextgrid.cli import main
from contextgrid.config import render
from contextgrid.config.loader import loads
from tests.support import API_DOCS, CONTRACT

HEADER = '{"_evalset": {"id": "es", "version": 1, "source": "test"}}'
ITEM = '{"id": "q1", "question": "How much notice to terminate?"}'


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A real corpus, and no eval set -- each test writes the one it is about."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    (docs / "api.md").write_text(API_DOCS)
    return tmp_path


def config_for(workspace: Path, evalset: str) -> Path:
    path = workspace / "experiment.yaml"
    path.write_text(f"corpus: ./docs\nevalset: {evalset}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# check opens the eval set
# ---------------------------------------------------------------------------


def test_check_rejects_a_jsonl_that_is_not_json(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The repro: `check` said valid, and `run` a second later said line 1 is not JSON."""
    (workspace / "bad.jsonl").write_text("garbage not jsonl\n", encoding="utf-8")

    assert main(["check", str(config_for(workspace, "./bad.jsonl"))]) == 1
    assert "is not valid JSON" in capsys.readouterr().err


def test_check_rejects_a_csv_with_no_question_column(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """And says which columns it would have accepted, exactly as `run` does."""
    (workspace / "nocol.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")

    assert main(["check", str(config_for(workspace, "./nocol.csv"))]) == 1
    error = capsys.readouterr().err
    assert "has no question column" in error
    assert "Expected one of: question, query, q" in error


def test_check_says_the_same_words_run_says(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cli.md`: "Each message is the one `run` would have printed". Not a paraphrase of it.

    A second reader here would drift from the one `run` uses, and the two messages would then
    have to be recognised as the same problem rather than read as the same sentence.
    """
    (workspace / "bad.jsonl").write_text("garbage not jsonl\n", encoding="utf-8")
    config = config_for(workspace, "./bad.jsonl")

    assert main(["check", str(config)]) == 1
    checked = capsys.readouterr().err
    assert main(["run", str(config)]) == 1
    assert capsys.readouterr().err.strip() == checked.strip()


def test_check_rejects_an_eval_set_with_no_questions_in_it(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty file parses fine, and `run` then sweeps the whole matrix to score nothing.

    This is the one problem here that `run` does not raise. It parses, indexes every document,
    reports every metric as zero and warns about unresolvable evidence -- the most expensive
    way this tool has of saying the questions file is empty.
    """
    (workspace / "empty.jsonl").write_text("", encoding="utf-8")

    assert main(["check", str(config_for(workspace, "./empty.jsonl"))]) == 1
    assert "has no questions in it" in capsys.readouterr().err


def test_check_rejects_a_jsonl_that_is_only_its_header_line(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `_evalset` header carries the identity and no questions. A file of just that is empty."""
    (workspace / "header.jsonl").write_text(HEADER + "\n", encoding="utf-8")

    assert main(["check", str(config_for(workspace, "./header.jsonl"))]) == 1
    assert "has no questions in it" in capsys.readouterr().err


def test_check_rejects_a_csv_with_a_header_and_no_rows(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "header.csv").write_text("id,question\n", encoding="utf-8")

    assert main(["check", str(config_for(workspace, "./header.csv"))]) == 1
    assert "has no questions in it" in capsys.readouterr().err


def test_check_still_passes_a_real_jsonl_eval_set(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the bug: reading the file must not make `check` stricter by accident."""
    (workspace / "questions.jsonl").write_text(f"{HEADER}\n{ITEM}\n", encoding="utf-8")

    assert main(["check", str(config_for(workspace, "./questions.jsonl"))]) == 0
    assert "config is valid." in capsys.readouterr().out


def test_check_still_passes_a_real_csv_eval_set(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A spreadsheet is a documented eval set, and `check` accepting it is not a new leniency."""
    (workspace / "questions.csv").write_text(
        "question,document,quote\nHow much notice?,contract.md,thirty days\n", encoding="utf-8"
    )

    assert main(["check", str(config_for(workspace, "./questions.csv"))]) == 0
    assert "config is valid." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# and still reads nothing expensive
# ---------------------------------------------------------------------------


def test_check_opens_the_eval_set_and_none_of_the_documents(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cli.md:105`: check "reads no documents, embeds nothing, indexes nothing".

    Reading the eval set is the new work, and this is what stops it becoming reading the
    corpus. Every `open` is recorded, so the assertion is about what was opened rather than
    about what the code looks like: the eval set yes, `contract.md` and `api.md` no.
    """
    import builtins

    opened: list[str] = []
    real_builtin_open = builtins.open
    real_path_open = Path.open

    def recording_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_builtin_open(file, *args, **kwargs)

    def recording_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    (workspace / "questions.jsonl").write_text(f"{HEADER}\n{ITEM}\n", encoding="utf-8")
    config = config_for(workspace, "./questions.jsonl")
    # Patch `Path.open` rather than `io.open`, because how `pathlib` reaches the underlying
    # opener is a private detail that changed between the versions this package supports. On
    # 3.11+ `Path.read_text` calls `io.open` at call time, so patching `io.open` catches it; on
    # 3.10 `pathlib` binds its opener once at import, so a later patch of `io.open` never fires
    # and this test recorded nothing at all -- it passed on three interpreters and failed on the
    # oldest. `Path.open` is the public door every one of those versions goes through.
    monkeypatch.setattr(Path, "open", recording_path_open)
    monkeypatch.setattr(builtins, "open", recording_open)

    assert main(["check", str(config)]) == 0

    assert any(path.endswith("questions.jsonl") for path in opened), opened
    assert not [path for path in opened if path.endswith(("contract.md", "api.md"))]


def test_check_still_catches_an_empty_corpus(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading nothing must not cost the check that the corpus has something in it.

    The glob is what answers "is anything here", and it runs before the file limit does -- so
    a limit of zero asks the same question and reads none of the answer.
    """
    (workspace / "empty-corpus").mkdir()
    (workspace / "questions.jsonl").write_text(f"{HEADER}\n{ITEM}\n", encoding="utf-8")
    config = workspace / "experiment.yaml"
    config.write_text("corpus: ./empty-corpus\nevalset: ./questions.jsonl\n", encoding="utf-8")

    assert main(["check", str(config)]) == 1
    assert "no files under" in capsys.readouterr().err


def test_check_makes_no_network_call(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cli.md:105`: check "calls no model". Proved by taking the socket away.

    Not by reading the code: the point of the promise is that `check` is safe to run on a
    laptop on a train, and no amount of reading proves that about nine plugin constructors.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("check opened a socket")

    (workspace / "questions.jsonl").write_text(f"{HEADER}\n{ITEM}\n", encoding="utf-8")
    config = config_for(workspace, "./questions.jsonl")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert main(["check", str(config)]) == 0
    assert "config is valid." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the starter config init writes
# ---------------------------------------------------------------------------


def axis_block(text: str, axis: str) -> list[str]:
    """The axis line and the comment lines under it, up to the next blank line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"  {axis}:"))
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.strip().startswith("#"):
            break
        block.append(line)
    return block


def test_the_starter_config_names_the_extra_a_missing_plugin_needs() -> None:
    """The promise: everything else "as a comment showing how to unlock it (which extra to
    install)". No comment named an extra or an install command at all, which is the point."""
    from contextgrid.config.plugins import extra_missing_for
    from contextgrid.parse import PARSERS

    blocked = [entry for entry in PARSERS if extra_missing_for(entry) is not None]
    if not blocked:
        pytest.skip("every parser extra is installed here, so there is nothing to unlock")

    block = "\n".join(axis_block(render(), "parser"))
    for entry in blocked:
        assert entry.name in block, f"{entry.name} cannot run here and is not mentioned"
        assert f'pip install "context-grid[{entry.extra}]"' in block


def test_a_plugin_that_cannot_run_here_is_never_a_chosen_value() -> None:
    """`init` writing a config that `check` then rejects is the worst of both commands."""
    from contextgrid.config.plugins import missing_extra
    from contextgrid.parse import PARSERS

    line = axis_block(render(), "parser")[0]
    for value in line.split("[", 1)[1].rstrip("]").split(","):
        assert missing_extra(PARSERS, value.strip()) is None, value


def test_the_starter_config_tells_installed_apart_from_needs_installing() -> None:
    """Two different states needing two different things from the reader: type this name, or
    run this command. One list covering both tells them neither."""
    block = axis_block(render(), "parser")
    available = next(line for line in block if "also available:" in line)
    assert "marker" not in available, "marker needs an extra and is not simply available"
    assert "text" in available, "text needs no extra at all"


def test_the_starter_config_lists_a_plugin_whose_distribution_is_named_differently() -> None:
    """`faiss` installs as `faiss-cpu` and imports as `faiss`. The template looked for a module
    called `faiss_cpu`, found none, and left a working index out of the file."""
    from contextgrid.config.plugins import extra_missing_for
    from contextgrid.index import INDEXES

    for entry in INDEXES:
        if extra_missing_for(entry) is None:
            assert entry.name in "\n".join(axis_block(render(), "index")), entry.name


def test_the_starter_config_says_which_plugins_want_a_model() -> None:
    """A third state, and the one move this comment invites: `hyde` is installed, is blocked by
    no extra, and still will not build without `run.model`. Moving it up to the axis line on
    the strength of "also available" is a config `check` refuses."""
    from contextgrid.transform import MODEL_BACKED

    block = "\n".join(axis_block(render(), "transform"))
    assert "`run.model` set" in block
    for name in MODEL_BACKED:
        assert name in block


def test_the_generated_starter_config_still_parses(tmp_path: Path) -> None:
    """Whatever else changes about the comments, the file has to load."""
    assert loads(render(), base=tmp_path).corpus.name == "documents"


def test_every_line_of_the_starter_config_stays_readable() -> None:
    for line in render().splitlines():
        assert len(line) <= 100, line


def test_init_writes_a_config_that_check_accepts(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end, and the reason all of the above matters: the two commands agree."""
    (workspace / "questions.jsonl").write_text(f"{HEADER}\n{ITEM}\n", encoding="utf-8")
    path = workspace / "starter.yaml"

    assert main(["init", str(path), "--corpus", "./docs", "--evalset", "./questions.jsonl"]) == 0
    capsys.readouterr()
    assert main(["check", str(path)]) == 0
    assert "config is valid." in capsys.readouterr().out

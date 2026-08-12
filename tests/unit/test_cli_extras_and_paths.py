"""Three ways the CLI said yes to something it could not do, or no to something it could.

**A plugin that cannot run passed `check`.** Building a plugin is how `check` proves the
config will work, and for most plugins it is proof: the plugin's module imports the package it
needs, so a missing package fails the import and `check` reports it. `marker` breaks that. It
lives in `contextgrid.parse.layout`, which is in this tree and imports fine; `import marker`
happens inside `parse()`, where the Surya models are actually wanted. So `check` built it, said
"config is valid.", and the sweep it approved died on the first document with `The marker
parser requires the 'parse-marker' extra`. The expensive command found what the cheap one is
for.

**`check` on a directory printed the operating system's sentence.** `contextgrid check
./documents` -- pointing it at the corpus rather than the config, which is easy when every
other subcommand takes a corpus -- gave `error: [Errno 21] Is a directory: 'documents'`.

**`profile` refused a single file.** `configuration.md` defines a corpus as "a directory of
documents, or a single file", and `check` and `run` both honour it. The one command whose
whole job is looking at a corpus did not.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from contextgrid.cli import main
from contextgrid.config.plugins import _dependency_present, missing_extra
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.evalset import write_jsonl
from contextgrid.parse import PARSERS
from tests.support import API_DOCS, CONTRACT

#: A distribution name no index will ever carry, for asserting the absent case without
#: depending on which extras happen to be installed where the suite runs.
NOT_INSTALLED = "context-grid-no-such-package-9f3a"


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
            items=(
                EvalItem(
                    id="q1",
                    question="How much notice to terminate for convenience?",
                    anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
                ),
            ),
        ),
        tmp_path / "evalset.jsonl",
    )
    return tmp_path


def write_config(workspace: Path, body: str = "", *, corpus: str = "./docs") -> Path:
    path = workspace / "experiment.yaml"
    path.write_text(f"corpus: {corpus}\nevalset: ./evalset.jsonl\n{body}", encoding="utf-8")
    return path


@pytest.fixture
def lazy_parser() -> object:
    """A parser shaped exactly like `marker`: in-tree module, third-party package absent.

    Registered into the real `PARSERS` rather than a local registry, because what is being
    tested is the whole command -- `main(["check", ...])` looks plugins up by name in the
    registries the CLI itself imports. `unregister` in the teardown, so a plugin registered
    only to prove it fails does not leak into every later test that lists parsers.
    """
    PARSERS.register_lazy(
        "pretend-heavy",
        module="contextgrid.parse.text",  # imports fine, like contextgrid.parse.layout
        attr="MarkdownParser",  # and builds fine, like MarkerParser
        extra="pretend",
        package=NOT_INSTALLED,
        doc="A parser whose package is not installed.",
    )
    yield
    PARSERS.unregister("pretend-heavy")


# ---------------------------------------------------------------------------
# a plugin whose extra is not installed
# ---------------------------------------------------------------------------


def test_check_rejects_a_plugin_whose_package_is_not_installed(
    workspace: Path, lazy_parser: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bug: this plugin builds without complaint, so building cannot be the only test."""
    from contextgrid.parse import get_parser

    assert get_parser("pretend-heavy") is not None, "the premise: it builds, and still cannot run"

    config = write_config(workspace, "grid:\n  parser: pretend-heavy\n")
    assert main(["check", str(config)]) == 1

    errors = capsys.readouterr().err
    assert "requires the 'pretend' extra" in errors
    assert 'pip install "context-grid[pretend]"' in errors
    assert f"needs {NOT_INSTALLED}" in errors


def test_the_message_check_prints_is_the_one_run_prints(lazy_parser: object) -> None:
    """Word for word, so a user who has seen one recognises the other rather than filing two.

    `check` cannot reuse `run`'s error by catching it -- the whole problem is that `run` raises
    it hours later, from inside `parse()`. So it is built from the registration, and the thing
    worth pinning is that both spellings agree, including `The marker parser` unquoted.
    """
    from contextgrid.core.errors import MissingExtraError

    from_check = str(missing_extra(PARSERS, "pretend-heavy"))
    from_run = str(MissingExtraError("The pretend-heavy parser", "pretend", package=NOT_INSTALLED))
    assert from_check == from_run


def test_a_plugin_whose_extra_is_installed_still_passes(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The regression that matters more than the fix: refusing a sweep that would have worked."""
    config = write_config(workspace, "grid:\n  parser: [markdown, text]\n  index: [dense, bm25]\n")
    assert main(["check", str(config)]) == 0
    assert "config is valid." in capsys.readouterr().out


@pytest.mark.parametrize("spec", ["pretend-heavy", "pretend-heavy:tables=false"])
def test_parameters_on_the_spec_do_not_hide_the_missing_package(
    spec: str, lazy_parser: object
) -> None:
    """`marker:languages=en` is the same plugin as `marker`, and just as unrunnable."""
    assert missing_extra(PARSERS, spec) is not None


def test_a_name_nothing_is_registered_under_is_left_to_the_builder() -> None:
    """`create` already reports unknown names, and lists the ones that exist. One report."""
    assert missing_extra(PARSERS, "banana") is None


def test_a_package_installed_under_another_module_name_counts_as_present() -> None:
    """`faiss-cpu` imports as `faiss`, `marker-pdf` as `marker`.

    Checking only whether a module of the distribution's name is importable would call `faiss`
    missing on a machine that has it -- refusing a sweep that runs. So the distribution is
    asked for first, by the name in `pip install`, which is the name the registration records.
    """
    assert _dependency_present("pytest") is True
    assert _dependency_present(NOT_INSTALLED) is False


@pytest.mark.skipif(
    _dependency_present("marker-pdf"), reason="marker-pdf is installed, so it is not missing"
)
def test_marker_itself_is_reported(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The originally reported case, on the real registration."""
    config = write_config(workspace, "grid:\n  parser: marker\n")
    assert main(["check", str(config)]) == 1
    assert 'pip install "context-grid[parse-marker]"' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# config paths that are not config files
# ---------------------------------------------------------------------------


def test_check_on_a_directory_says_a_config_file_was_expected(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", str(workspace / "docs")]) == 1

    errors = capsys.readouterr().err
    assert "is a directory" in errors
    assert "config file was expected" in errors
    assert "Errno" not in errors, "the operating system's wording, not ours"


def test_run_on_a_directory_says_the_same_thing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fixed in `load`, not in `check`, so every command taking a config path benefits."""
    assert main(["run", str(workspace / "docs")]) == 1
    assert "config file was expected" in capsys.readouterr().err


def test_check_on_a_path_that_does_not_exist(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", str(workspace / "nope.yaml")]) == 1
    assert "no config file at" in capsys.readouterr().err


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file with mode 000")
def test_check_on_a_config_it_cannot_read(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(workspace)
    config.chmod(0o000)
    try:
        assert main(["check", str(config)]) == 1
        errors = capsys.readouterr().err
        assert "no permission to read" in errors
        assert "Errno" not in errors
    finally:
        config.chmod(0o644)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def test_profile_accepts_a_single_file(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """ "Corpus" has to mean the same thing in `profile corpus` as in `corpus:`."""
    assert main(["profile", str(workspace / "docs" / "contract.md")]) == 0
    assert "1 files" in capsys.readouterr().out


def test_profile_still_accepts_a_directory(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(workspace / "docs")]) == 0
    assert "2 files" in capsys.readouterr().out


def test_profile_on_a_path_that_does_not_exist(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", str(workspace / "nope")]) == 1
    assert "no corpus at" in capsys.readouterr().err


def test_the_profile_help_string_is_the_one_the_docs_print(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`docs/guide/cli.md` prints a usage block, and a usage block that is wrong is worse than
    none. It said "Measure a corpus and flag settings its shape rules out."; argparse said
    "Profile a corpus and say which axes will matter.", which also contradicts the same page
    saying `profile` "does not rank the axes for you".

    `COLUMNS` because argparse wraps to the terminal, and under pytest that is narrow enough to
    break the line in the middle of the sentence. 90 is the width the block in the docs was
    rendered at.
    """
    from contextgrid.cli.__main__ import _build_parser

    monkeypatch.setenv("COLUMNS", "90")
    help_text = _build_parser().format_help()

    assert "    profile             Measure a corpus and flag settings its shape rules out.\n" in (
        help_text
    )
    assert "which axes will matter" not in help_text

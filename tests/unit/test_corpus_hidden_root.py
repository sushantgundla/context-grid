"""The corpus root is whatever the user named -- dots and all.

Two defects lived here. A corpus under a dot-prefixed path (`.data/`, `.claude/`,
`~/.local/share/...`) loaded zero files, because the hidden-entry filter looked at the whole
absolute path instead of the part below the root. The error message that followed then
claimed "The directory holds no files at all" about a directory holding eight `.md` files.

Hidden filtering *below* the root is correct and must survive every change here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.corpus import Corpus, CorpusError


def _error(path: Path, **kwargs: object) -> str:
    with pytest.raises(CorpusError) as caught:
        Corpus.from_dir(path, **kwargs)  # type: ignore[arg-type]
    return str(caught.value)


# -- defect 1: the root the user named ---------------------------------------


def test_dot_prefixed_root_loads(tmp_path: Path) -> None:
    """`.claude/skills/.../documents` is a corpus, not something to filter away."""
    root = tmp_path / ".claude" / "data" / "documents"
    root.mkdir(parents=True)
    (root / "billing.md").write_text("# Billing\n")
    (root / "pricing.md").write_text("# Pricing\n")

    corpus = Corpus.from_dir(root)

    assert corpus.ids == ("billing.md", "pricing.md")


def test_root_named_like_a_build_directory_loads(tmp_path: Path) -> None:
    """A checkout that happens to sit under `node_modules` is still what the user pointed at."""
    root = tmp_path / "node_modules" / "docs"
    root.mkdir(parents=True)
    (root / "readme.md").write_text("# Readme\n")

    assert Corpus.from_dir(root).ids == ("readme.md",)


def test_dot_prefixed_root_with_custom_patterns(tmp_path: Path) -> None:
    """`Corpus.from_dir(path, patterns=[...])` had the same root confusion."""
    root = tmp_path / ".data"
    root.mkdir()
    (root / "notes.rst").write_text("notes")

    assert Corpus.from_dir(root, patterns=["*.rst"]).ids == ("notes.rst",)


def test_dot_prefixed_root_non_recursive(tmp_path: Path) -> None:
    root = tmp_path / ".data"
    root.mkdir()
    (root / "a.md").write_text("a")

    assert Corpus.from_dir(root, recursive=False).ids == ("a.md",)


def test_single_file_under_a_hidden_path(tmp_path: Path) -> None:
    """`corpus: /some/.hidden/file.md` -- the config single-file path."""
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    target = hidden / "file.md"
    target.write_text("# One file\n")

    assert Corpus.from_files([target]).ids == ("file.md",)


# -- hidden filtering below the root still works -----------------------------


def test_hidden_directory_inside_the_corpus_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("real")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "x.md").write_text("not corpus content")

    assert Corpus.from_dir(tmp_path).ids == ("real.md",)


def test_hidden_file_inside_the_corpus_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("real")
    (tmp_path / ".hidden.md").write_text("not corpus content")

    assert Corpus.from_dir(tmp_path).ids == ("real.md",)


def test_build_directory_inside_the_corpus_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("real")
    for name in (".venv", "node_modules", "__pycache__"):
        sub = tmp_path / name
        sub.mkdir()
        (sub / "x.md").write_text("not corpus content")

    assert Corpus.from_dir(tmp_path).ids == ("real.md",)


def test_hidden_filtering_still_applies_under_a_dot_prefixed_root(tmp_path: Path) -> None:
    """The root being hidden must not switch the below-root filter off."""
    root = tmp_path / ".claude" / "documents"
    root.mkdir(parents=True)
    (root / "real.md").write_text("real")
    (root / ".hidden.md").write_text("no")
    cache = root / ".contextgrid-cache"
    cache.mkdir()
    (cache / "stale.md").write_text("no")

    assert Corpus.from_dir(root).ids == ("real.md",)


# -- defect 2: the message must not assert what it did not check -------------


def test_truly_empty_directory_says_so(tmp_path: Path) -> None:
    message = _error(tmp_path)

    assert "holds no files at all" in message
    assert "Python-API only" in message


def test_directory_of_rst_files_is_not_called_empty(tmp_path: Path) -> None:
    (tmp_path / "a.rst").write_text("a")
    (tmp_path / "b.rst").write_text("b")

    message = _error(tmp_path)

    assert "holds no files at all" not in message
    assert "2 files" in message
    assert ".rst" in message
    assert "Python-API only" in message


def test_matching_files_filtered_as_hidden_say_so(tmp_path: Path) -> None:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "x.md").write_text("x")
    (git / "y.md").write_text("y")

    message = _error(tmp_path)

    assert "holds no files at all" not in message
    assert "2 files matched the patterns but were skipped as hidden" in message
    assert "never skipped" in message


def test_one_hidden_match_reads_as_singular(tmp_path: Path) -> None:
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "x.md").write_text("x")

    assert "1 file matched the patterns but was skipped as hidden" in _error(tmp_path)


def test_non_matching_hidden_files_are_not_called_empty(tmp_path: Path) -> None:
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "a.rst").write_text("a")

    message = _error(tmp_path)

    assert "holds no files at all" not in message
    assert "every one hidden" in message


def test_files_without_extensions_are_not_called_empty(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text("license")

    message = _error(tmp_path)

    assert "holds no files at all" not in message
    assert "none with a file extension" in message


def test_extension_list_is_capped(tmp_path: Path) -> None:
    for index in range(7):
        (tmp_path / f"f{index}.x{index}").write_text("x")

    message = _error(tmp_path)

    assert "and 2 more" in message

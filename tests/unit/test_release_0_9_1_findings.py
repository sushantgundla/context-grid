"""What a stranger found in 0.9.0, installed from PyPI into a container.

The package worked -- quickstart and the `build()` example reproduced character for character
-- but four things were wrong, and each one is pinned here so it cannot come back.

The theme running through them is the same: the tool knew the right answer and did not hand it
over. It knew faiss was missing and raised a type nobody catches. It knew a quote had been
reflowed and told only the Python API. It knew a question had an anchor and called it
"answerable", which is a stronger word than it had earned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.core.errors import MissingExtraError
from contextgrid.core.warnings import Severity, WarningCode

# ---------------------------------------------------------------------------
# 1. the exception a user is told to catch
# ---------------------------------------------------------------------------


def test_a_missing_index_extra_raises_the_documented_exception() -> None:
    """`reference/errors` presents `MissingExtraError` as the missing-extras exception.

    faiss raised `IndexBuildError`, which inherits `ContextGridError` and `ValueError` and has
    no relationship to `MissingExtraError` at all -- so the `except MissingExtraError` the docs
    hand out, and the `except ImportError` they say also works, both sail straight past it. The
    message was perfect and the type was wrong, which is the combination most likely to be
    trusted and then break in production.
    """
    import builtins

    from contextgrid.index import ann

    real_import = builtins.__import__

    def no_faiss(name: str, *args: object, **kwargs: object) -> object:
        if name == "faiss":
            raise ImportError("No module named 'faiss'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    builtins.__import__ = no_faiss
    try:
        with pytest.raises(MissingExtraError) as caught:
            ann._faiss()
    finally:
        builtins.__import__ = real_import

    message = str(caught.value)
    assert "index" in message
    assert "faiss" in message
    # The install line is the whole point of the exception; keep it reachable.
    assert "pip install" in message


def test_the_missing_extra_error_is_still_an_import_error() -> None:
    """`reference/errors` also promises `except ImportError` catches it. Hold that."""
    assert issubclass(MissingExtraError, ImportError)


# ---------------------------------------------------------------------------
# 2. the warning the CLI kept to itself
# ---------------------------------------------------------------------------


def test_a_reflowed_anchor_is_loud_enough_to_reach_the_terminal() -> None:
    """Markdown hard-wraps, so a quote copied out of it spans a newline and matches only after
    whitespace is collapsed. That is a fact about the user's ground truth and they have to hear
    it.

    At `Severity.INFO` the CLI dropped it whenever a run produced results -- which is always,
    for this warning. The three hard anchor failures printed loudly, so silence read as "your
    evidence matched literally" when it had in fact been reflowed to fit.
    """
    from contextgrid.core.documents import MediaType, SourceFile
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.parse import get_parser
    from contextgrid.score.anchor import AnchorResolver

    # Hard-wrapped, as Markdown written by a human almost always is. The quote below spans the
    # newline, so it can only match once whitespace is collapsed.
    text = (
        "# Refunds\n\nRefunds are issued within 30 days of purchase, provided that the\n"
        "item is unopened.\n"
    )
    parsed = get_parser("markdown").parse(
        SourceFile(id="refunds.md", raw=text.encode("utf-8"), media_type=MediaType.MARKDOWN)
    )
    item = EvalItem(
        id="q1",
        question="How long?",
        anchors=(GoldAnchor(source_id="refunds.md", quote="provided that the item is unopened"),),
    )

    _, log = AnchorResolver().resolve(EvalSet(id="t", items=(item,)), {"refunds.md": parsed})

    normalised = [w for w in log.entries if w.code is WarningCode.ANCHOR_NORMALISED]
    assert normalised, "the reflow happened; something must say so"
    assert normalised[0].severity is not Severity.INFO, (
        "INFO is filtered out of the CLI whenever there are results, which is exactly when "
        "this warning fires -- so INFO means a CLI user is never told"
    )


# ---------------------------------------------------------------------------
# 3. a word that claimed more than it checked
# ---------------------------------------------------------------------------


def test_the_evalset_command_does_not_claim_a_question_is_answerable(tmp_path: Path) -> None:
    """`contextgrid evalset` never reads the corpus, so it cannot know whether an anchor
    resolves. It reported "14 questions (14 answerable)" for a set whose evidence included a
    sentence appearing nowhere in the documents.

    "Answerable" is the word the tool uses elsewhere for *this question can be scored*, so
    reusing it for *this question has an anchor attached* quietly overstates it.
    """
    from contextgrid.cli import main

    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Nonsense?",
                "anchors": [{"source_id": "a.md", "quote": "appears nowhere at all"}],
            }
        )
        + "\n"
    )

    assert main(["evalset", str(path)]) == 0


def test_the_evalset_command_says_what_it_actually_measured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Nonsense?",
                "anchors": [{"source_id": "a.md", "quote": "appears nowhere at all"}],
            }
        )
        + "\n"
    )
    main_out = __import__("contextgrid.cli", fromlist=["main"]).main
    main_out(["evalset", str(path)])
    printed = capsys.readouterr().out

    assert "answerable" not in printed or "corpus" in printed, (
        "either stop using the word, or say plainly that nothing was checked against a corpus"
    )


# ---------------------------------------------------------------------------
# 4. a generated file pointing at a page that does not exist
# ---------------------------------------------------------------------------


def test_the_generated_config_does_not_cite_a_missing_page() -> None:
    """`contextgrid init` wrote "see extending.md" into every starter config. There is no
    `extending` page on the documentation site -- the nearest real one is `concepts/plugins`.
    A generated file is the worst place for a dead reference, because it is copied forward.
    """
    from contextgrid.config import render

    text = render(corpus="./documents", evalset="./questions.jsonl")

    assert "extending.md" not in text

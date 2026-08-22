"""What a stranger found in 0.9.4 about the sentences this package writes when something is missing.

Installed from PyPI into a container and driven from the documentation site alone. Every number
on `/scoring/significance`, `/evalsets/*` and `/reference/errors` reproduced exactly, and every
missing-extra path raised the type the docs promise. What did not hold was the *prose*.

Two findings, and they are the same finding twice: the message is assembled in more than one
place, and the copies drifted.

* Three plugins say "Install with:" where the other twelve say "Install it with:". A user who
  greps their logs for the sentence the documentation shows them misses three of the fifteen.
* `MissingExtraError` renders `f"{feature} requires ..."`, so a `feature` that is a bare plural
  reads "faiss indexes requires the 'index' extra". Six call sites did that.

Neither could fail a test before this file existed, and the reason is worth keeping: every
missing-extra message is only reachable when the extra is *absent*, and the test environment
installs `[dev]`. So the suites that cover these code paths never render their strings. The
tests below block the import instead, which reproduces a bare install inside a full one.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

import contextgrid as cg
from contextgrid.core.errors import MissingExtraError

SRC = Path(__file__).resolve().parents[2] / "src" / "contextgrid"

# The sentence `/installation` and `/reference/errors` both show. Nothing may say it differently.
INSTALL_HINT = "Install it with:"


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


@pytest.fixture
def block_import(monkeypatch: pytest.MonkeyPatch):
    """Make `import <name>` raise ImportError, the way a bare install would.

    `sys.modules[name] = None` is the documented way to do this: the import system treats a
    `None` entry as "this module is known to be absent" and raises rather than searching.
    """

    def blocker(*names: str) -> None:
        for name in names:
            for loaded in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
                monkeypatch.delitem(sys.modules, loaded, raising=False)
            monkeypatch.setitem(sys.modules, name, None)  # type: ignore[typeddict-item]

    return blocker


# --------------------------------------------------------------------------------------------
# Finding 3a: three plugins spell the install hint differently from every other plugin.
# --------------------------------------------------------------------------------------------


def test_no_source_file_spells_the_install_hint_the_short_way() -> None:
    """A source scan, because the string is what drifted and the string is what users grep.

    A behavioural test can only reach the plugins whose extras this environment happens to be
    missing. This one reaches all of them, including any added tomorrow.
    """
    offenders = [
        f"{path.relative_to(SRC)}:{number}"
        for path in _sources()
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"Install with:", line)
    ]
    assert offenders == [], (
        f"these lines say 'Install with:' where the rest of the package and the documentation "
        f"say {INSTALL_HINT!r}: {offenders}"
    )


def test_every_install_hint_in_the_package_reads_the_same(
    block_import,
) -> None:
    """The three that drifted, rendered for real rather than read off the page."""
    from contextgrid.core.documents import MediaType, SourceFile
    from contextgrid.parse import TextParser

    parsed = TextParser().parse(
        SourceFile(id="a.txt", media_type=MediaType.TEXT, raw=b"One sentence. And a second one.")
    )

    # Each case names the call that reaches the import, because every one of these plugins
    # constructs happily and only looks for its package when asked to do the work.
    cases = [
        (
            "chonkie:token chunker",
            ("chonkie",),
            lambda: cg.get_chunker("chonkie:token").chunk(parsed),
        ),
        (
            "langchain:recursive chunker",
            ("langchain_text_splitters",),
            lambda: cg.get_chunker("langchain:recursive").chunk(parsed),
        ),
        (
            "litellm embedder",
            ("litellm",),
            lambda: cg.get_embedder("litellm:bge-base-en-v1.5").embed_queries(["a query"]),
        ),
        (
            "litellm reranker",
            ("litellm",),
            # A real candidate, because `rerank` returns `[]` for an empty list without ever
            # reaching the import.
            lambda: cg.get_reranker("litellm-rerank:rerank-v3").rerank(
                "a query", [cg.Chunk("c0", cg.Span("d0", 0, 1), "a passage")], 5
            ),
        ),
    ]

    for label, modules, call in cases:
        block_import(*modules)
        with pytest.raises(cg.ContextGridError) as caught:
            call()
        message = str(caught.value)
        assert INSTALL_HINT in message, (
            f"{label} said {message!r}, which does not contain {INSTALL_HINT!r}"
        )


# --------------------------------------------------------------------------------------------
# Finding 3b: `f"{feature} requires ..."` with a plural `feature` reads "indexes requires".
# --------------------------------------------------------------------------------------------


def _as_text(node: ast.expr) -> str | None:
    """The `feature` argument as a readable string, f-strings included.

    An earlier version of this walk only understood `ast.Constant`, so the one call site that
    built its feature with an f-string was invisible to every rule below -- which is exactly the
    site that turned out to hold a whole sentence. Interpolated values stand in as `X`; none of
    the rules care what goes in the hole, only about the shape of the sentence around it.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue):
                parts.append("X")
        return "".join(parts)
    return None


def _missing_extra_features() -> list[tuple[str, str]]:
    """Every `feature` argument handed to `MissingExtraError` in the package.

    Read off the syntax tree rather than by running the code, for the same reason as above: most
    of these lines are unreachable while `[dev]` is installed.
    """
    found: list[tuple[str, str]] = []
    for path in _sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name != "MissingExtraError" or not node.args:
                continue
            text = _as_text(node.args[0])
            if text is not None:
                found.append((f"{path.relative_to(SRC)}:{node.args[0].lineno}", text))
    return found


def test_the_call_sites_are_discoverable() -> None:
    """Guards the two tests below: if the walk finds nothing, they pass by being empty."""
    features = _missing_extra_features()
    assert len(features) >= 10, (
        f"expected the package's MissingExtraError call sites, got {features}"
    )


def test_no_missing_extra_message_says_a_plural_requires() -> None:
    """`MissingExtraError` puts `feature` straight in front of a singular verb.

    So `feature` has to be something that *requires*, not something that *require*. The check is
    the crude one -- the last word before ' requires' must not end in 's' -- which is exactly
    what went wrong ('indexes', 'metrics', 'tokenizers', 'calls') and is worth catching cheaply.
    A singular that genuinely ends in 's' would need this rule relaxed; none does today.
    """
    offenders = []
    for where, feature in _missing_extra_features():
        subject = feature.rstrip().split()[-1] if feature.strip() else ""
        if subject.endswith("s") and not subject.endswith("ss"):
            rendered = str(MissingExtraError(feature, "index", package="x"))
            offenders.append(f"{where}: {rendered.split(' extra')[0]!r}")
    assert offenders == [], (
        "these read as a plural subject in front of the singular 'requires'; give the feature a "
        f"singular or mass-noun form: {offenders}"
    )


# The longest legitimate feature today is `f"The {registration.name} {registration.family}"`,
# which renders as about 25 characters. 60 leaves room for a longer plugin name without leaving
# room for a sentence.
LONGEST_NOUN_PHRASE = 60


def test_no_missing_extra_message_jams_a_sentence_into_the_feature_slot() -> None:
    """`feature` is the subject of a sentence the template finishes, so it must be a noun phrase.

    `MissingExtraError` renders `f"{feature} requires the '{extra}' extra..."`. Hand it a whole
    sentence and the join reads "...so this needs network once requires the 'embed' extra",
    which is not English. Two cheap rules catch it: no sentence-ending punctuation inside the
    feature, and no feature long enough to be prose. Anything explanatory goes in `detail`,
    which is appended after the install hint where a full sentence belongs.
    """
    offenders = []
    for where, feature in _missing_extra_features():
        reason = None
        if re.search(r"[.!?](\s|$)", feature):
            reason = "contains sentence-ending punctuation"
        elif len(feature) > LONGEST_NOUN_PHRASE:
            reason = f"is {len(feature)} characters, past the {LONGEST_NOUN_PHRASE} allowed"
        if reason:
            offenders.append(f"{where}: {feature!r} {reason}")
    assert offenders == [], (
        "a MissingExtraError feature is the subject of 'X requires the ... extra', so it has to "
        f"be a noun phrase; move the explanation to the `detail` argument: {offenders}"
    )


def test_the_detail_argument_carries_the_explanation_after_the_install_hint() -> None:
    """`detail` exists so an explanation has somewhere to go that is not the subject slot."""
    plain = MissingExtraError("The 'x' encoding", "embed", package="tiktoken")
    assert str(plain).endswith('Install it with: pip install "context-grid[embed]"')

    with_detail = MissingExtraError(
        "The 'x' encoding", "embed", package="tiktoken", detail="It downloads on first use."
    )
    # The hint ends on a quote, so the detail has to arrive after a full stop, not a bare space.
    assert str(with_detail) == f"{plain}. It downloads on first use."


# --------------------------------------------------------------------------------------------
# The exit-code table on `/reference/cli` listed 0, 1 and 130, and the CLI also exits 2.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["run", "--wat"], id="unknown-flag"),
        pytest.param(["frobnicate"], id="unknown-subcommand"),
        pytest.param(["run"], id="missing-positional"),
    ],
)
def test_a_command_line_that_does_not_parse_exits_2(argv: list[str]) -> None:
    """`argparse` exits 2 by convention, and the documented table has to say so.

    Nothing here is a code change -- the CLI already does this. The row was missing from
    `/reference/cli`, and a table of exit codes that omits one is worse for CI than no table.
    """
    from contextgrid.cli import main

    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("spec", "module", "expected"),
    [
        ("faiss", "faiss", "The faiss index requires the 'index' extra (needs faiss-cpu)."),
        ("usearch", "usearch", "The usearch index requires the 'index' extra (needs usearch)."),
    ],
)
def test_the_ann_indexes_name_themselves_in_the_singular(
    spec: str, module: str, expected: str, block_import
) -> None:
    """The two the drive actually read on a terminal, rendered end to end."""
    import numpy as np

    from contextgrid import Chunk, Span, get_index

    index = get_index(spec)
    vectors = np.eye(4, dtype="float32")
    chunks = [Chunk(f"c{i}", Span(f"d{i}", 0, 1), f"chunk {i}") for i in range(4)]

    block_import(module)
    with pytest.raises(MissingExtraError) as caught:
        index.build(chunks, vectors)
    assert str(caught.value).startswith(expected), str(caught.value)

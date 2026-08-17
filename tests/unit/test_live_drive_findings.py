"""What a stranger found in 0.9.1, installed from PyPI into a container.

Four things, and three of them share a shape: the tool held the right answer and printed
something else. It knew the confidence interval stayed clear of zero and said the gap was
"consistent with no difference at all". It knew `size` had to be a number and handed a string
to the arithmetic. It knew -- in `profile`, and only in `profile` -- that a chunk size above
the median document length cannot differentiate, then ran the sweep without repeating it.

The fourth is the one that costs the most: a leaderboard where every document became a single
chunk still prints a number, sorts it, and calls the top row a winner.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from contextgrid.core.errors import ContextGridError
from contextgrid.core.warnings import Severity, WarningCode
from contextgrid.score.significance import compare

# ---------------------------------------------------------------------------
# 1. a verdict that argued against its own numbers
# ---------------------------------------------------------------------------


def _p_value_only_failure() -> object:
    """Two configurations whose interval clears zero while the p-value does not clear alpha.

    These are the exact scores `scoring/significance` uses for its worked example, so the case
    the page prints and the case pinned here are the same case.
    """
    left = {"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.5, "q6": 1.0, "q7": 0.0, "q8": 1.0}
    right = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.0, "q6": 1.0, "q7": 0.0, "q8": 0.5}
    return compare(left, right, left="bm25", right="dense", metric="recall@5")


def test_the_example_really_is_the_p_value_only_case() -> None:
    """Guard the fixture itself. If resampling ever moves these numbers the test below stops
    testing what it claims to, and would keep passing while doing it."""
    result = _p_value_only_failure()
    assert result.difference.excludes_zero is True
    assert result.p_value >= result.alpha
    assert result.distinguishable is False


def test_a_verdict_does_not_call_an_interval_zero_when_it_excludes_zero() -> None:
    """`distinguishable` fails for two different reasons and there was one sentence for both.

    With a confidence interval of +0.062 to +0.500 the tool said the gap "sits inside the
    confidence interval ... so it is consistent with no difference at all". Zero is not in that
    interval. The reader is being told the opposite of what the two printed bounds say, by the
    same sentence that prints them, which is worse than saying nothing.
    """
    verdict = _p_value_only_failure().verdict()

    assert "consistent with no difference at all" not in verdict, (
        "the interval excludes zero -- no difference at all is exactly what these bounds rule out"
    )


def test_a_verdict_names_the_p_value_when_that_is_what_failed() -> None:
    """Saying less would be honest but useless. The reader needs the reason to act on it."""
    result = _p_value_only_failure()
    verdict = result.verdict()

    assert f"{result.p_value:.3f}" in verdict, "the p-value is the reason; print it"
    assert f"{result.difference.low:+.3f}" in verdict
    assert f"{result.difference.high:+.3f}" in verdict


def test_a_verdict_still_says_consistent_with_zero_when_the_interval_includes_zero() -> None:
    """The other branch is correct and is quoted verbatim on six documentation pages.

    Changing its wording would invalidate all of them for no gain, so this pins it in place.
    """
    identical = {f"q{i}": 1.0 for i in range(10)}
    result = compare(identical, dict(identical), left="a", right="b")

    assert result.difference.excludes_zero is False
    assert "consistent with no difference at all" in result.verdict()


# ---------------------------------------------------------------------------
# 2. a spec parameter that was never checked for being a number
# ---------------------------------------------------------------------------

#: Every parameterised axis in the tree, one bad value each. Written out rather than generated
#: so that a plugin gaining a numeric parameter does not silently join the list untested.
BAD_SPECS: list[tuple[str, str]] = [
    ("chunker", "recursive:banana"),
    ("chunker", "recursive:1.5"),
    ("chunker", "recursive:256,overlap=banana"),
    ("chunker", "fixed:banana"),
    ("chunker", "sentence:banana"),
    ("chunker", "structural:banana"),
    ("embedder", "hash:banana"),
]

#: The raw interpreter text each of these used to leak, and must never leak again.
INTERNALS = (
    "unsupported operand type",
    "not supported between instances of",
    "object cannot be interpreted as an integer",
    "Traceback",
)


def _build(axis: str, spec: str) -> object:
    from contextgrid.chunk import get_chunker
    from contextgrid.embed import get_embedder

    return {"chunker": get_chunker, "embedder": get_embedder}[axis](spec)


@pytest.mark.parametrize(("axis", "spec"), BAD_SPECS, ids=[spec for _, spec in BAD_SPECS])
def test_a_non_numeric_spec_parameter_is_a_sentence_not_a_traceback(axis: str, spec: str) -> None:
    """A bad plugin *name* has always produced an excellent error, and so has a bad *sign*:
    `no chunker named 'recursiv'. Available: ...` and `chunk size must be positive, got -5`.

    Only a bad *type* fell through to the arithmetic, so the user got
    `unsupported operand type(s) for //: 'str' and 'int'` -- a message about the inside of a
    chunker, from a tool whose entire premise is that you never have to read its source.
    """
    with pytest.raises(ContextGridError) as caught:
        _build(axis, spec)

    message = str(caught.value)
    for internal in INTERNALS:
        assert internal not in message, f"{spec!r} still leaks interpreter text: {message}"
    assert "banana" in message or "1.5" in message, (
        f"{spec!r} must quote the value that is wrong, got: {message}"
    )
    assert spec in message, f"{spec!r} must name the spec it came from, got: {message}"


def test_a_good_spec_parameter_still_builds() -> None:
    """The check must not become a fourth way for a valid config to fail."""
    from contextgrid.chunk import get_chunker
    from contextgrid.embed import get_embedder

    assert get_chunker("recursive:256").size == 256
    assert get_chunker("recursive:256,overlap=32").overlap == 32
    assert get_chunker("recursive").size == 512
    # `overlap` is `int | None`, so an explicit none is a real value and not a type error.
    assert get_chunker("recursive:256,overlap=none").overlap == 32
    assert get_embedder("hash:64").dimensions == 64
    # A float parameter takes a whole number without complaint; the tower goes one way only.
    assert get_chunker("semantic:90").percentile == 90


def test_check_reports_a_bad_spec_parameter_without_the_interpreter_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`reference/cli` promises `check` reports a typo'd or out-of-range spec "now, in the same
    pass". It did report it, wearing a Python internal: `chunker 'recursive:banana':
    unsupported operand type(s) for //: 'str' and 'int'`."""
    from contextgrid.cli import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nSome text about refunds and returns.\n")
    evalset = tmp_path / "q.jsonl"
    evalset.write_text(json.dumps({"id": "q1", "question": "Refunds?"}) + "\n")

    config = tmp_path / "cg.yaml"
    config.write_text(
        f"corpus: {corpus}\nevalset: {evalset}\ngrid:\n  chunker: [recursive:banana]\n"
    )

    assert main(["check", str(config)]) == 1

    printed = capsys.readouterr().err
    for internal in INTERNALS:
        assert internal not in printed, f"check still leaks interpreter text: {printed}"
    assert "size must be a whole number" in printed
    # The message already opens with `chunker 'recursive:banana':`, and `check` prefixes every
    # other build failure with exactly that pair. Prefixing this one too printed it twice.
    assert printed.count("recursive:banana") == 1, f"axis and spec named twice: {printed}"


def test_sweep_refuses_a_bad_spec_before_it_starts_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It printed the plan and the `[1/1]` progress line, then died. Everything it had already
    said was about a configuration that could never be built."""
    from contextgrid.cli import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nSome text about refunds and returns.\n")
    evalset = tmp_path / "q.jsonl"
    evalset.write_text(json.dumps({"id": "q1", "question": "Refunds?"}) + "\n")

    assert main(["sweep", str(corpus), str(evalset), "--chunker", "recursive:banana"]) == 1

    captured = capsys.readouterr()
    assert "on paper" not in captured.out, "do not print a plan for a sweep that cannot run"
    assert "[1/1]" not in captured.err, "do not start a configuration that cannot be built"


def test_a_list_written_with_commas_is_told_how_to_write_a_list() -> None:
    """`--chunker recursive:128,recursive:256` is the obvious guess and it is wrong. The error
    was about `key=value`, which is true and answers a question nobody asked."""
    from contextgrid.chunk import CHUNKERS

    with pytest.raises(ContextGridError) as caught:
        CHUNKERS.parse_spec("recursive:128,recursive:256")

    message = str(caught.value)
    assert "one value" in message or "repeat" in message, (
        f"say how to sweep several values, got: {message}"
    )


# ---------------------------------------------------------------------------
# 3. a leaderboard measuring documents while labelled as measuring chunks
# ---------------------------------------------------------------------------


def _tiny_run(chunker: str) -> object:
    """Three short documents and three questions -- the quickstart's own shape."""
    import contextgrid as cg

    docs = {
        "return-policy.md": (
            "# Return Policy\n\nItems may be returned within 30 days of delivery for a full "
            "refund. The item must be unused and in its original packaging.\n"
        ),
        "shipping.md": (
            "# Shipping\n\nStandard shipping takes 3 to 7 business days and costs $5.99. "
            "Orders over $50 ship free.\n"
        ),
        "warranty.md": (
            "# Warranty\n\nAll electronics carry a 1 year manufacturer warranty covering "
            "defects in materials and workmanship.\n"
        ),
    }
    corpus = cg.Corpus.from_texts(docs, media_type=cg.MediaType.MARKDOWN)
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
    lab.grid(chunker=[chunker])
    return lab.run(evalset)


def test_one_chunk_per_document_is_warned_about() -> None:
    """Every document became one chunk, so recall@5 out of three chunks is a measurement of
    document ranking. It read 1.000, exited 0 and said nothing.

    The machinery was already there -- the same run happily raised `anchor_ambiguous` -- and
    `contextgrid profile` already knew the rule ("Chunk sizes above that cannot differentiate").
    Only the run path was silent, which is the one place the number gets printed.
    """
    results = _tiny_run("recursive:512")
    run = results.runs[0]

    assert run.chunk_count <= 3, "fixture assumes one chunk per document; it stopped being that"

    raised = [w for w in run.warnings if w.code is WarningCode.ONE_CHUNK_PER_DOCUMENT]
    assert raised, "a leaderboard that cannot see the chunker axis has to say so"
    assert raised[0].severity is not Severity.INFO, (
        "INFO is filtered out of the CLI whenever there are results, and there are always "
        "results here -- so INFO means a CLI user is never told"
    )
    assert "3" in raised[0].message


def test_a_chunker_that_actually_splits_is_not_warned_about() -> None:
    """The warning has to stay rare enough to mean something."""
    results = _tiny_run("recursive:16")
    run = results.runs[0]

    assert run.chunk_count > 3, "fixture assumes this size splits; it stopped doing that"
    assert not [w for w in run.warnings if w.code is WarningCode.ONE_CHUNK_PER_DOCUMENT]


# ---------------------------------------------------------------------------
# 4. a worked example whose "Real output" had stopped being real
# ---------------------------------------------------------------------------


def test_the_extending_page_prints_what_the_example_actually_produces() -> None:
    """`docs/internals/extending.md` labels its last block "Real output" and it had drifted.

    It claimed the winner scored `1.000  1.000` and a gap of `+0.222` inside `+0.000 to
    +0.667`, and closed with "About 80 questions would be needed" -- a sentence
    `_sample_size_note` has not produced in a long time. The run really gives `0.889  0.833`
    and `+0.111` inside `+0.000 to +0.333`.

    A page that promises real output and prints invented output is worse than one that prints
    nothing, because it is the page a plugin author checks their own numbers against.
    """
    from contextgrid.core.documents import MediaType
    from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
    from contextgrid.corpus import Corpus
    from contextgrid.grid import Runner, matrix
    from contextgrid.score import METRICS

    page = Path(__file__).resolve().parents[2] / "docs" / "internals" / "extending.md"
    if not page.is_file():  # pragma: no cover - the installed package ships no docs tree
        pytest.skip("documentation tree is not present")

    @dataclass(frozen=True, slots=True)
    class WeightedRecall:
        name: ClassVar[str] = "weighted_recall"
        version: ClassVar[str] = "1"

        def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
            total = sum(grade for grade in judgements.values() if grade > 0)
            if total == 0:
                return 0.0
            top = set(ranked[:k])
            return sum(g for cid, g in judgements.items() if g > 0 and cid in top) / total

    contract = (
        "# Master Services Agreement\n\n## 2. Termination\n\n### 2.1 Notice period\n\n"
        "Either party may terminate this agreement for convenience by giving thirty days\n"
        "written notice. Notice must be delivered to the address in Schedule A.\n\n"
        "### 2.2 Termination for cause\n\n"
        "A party may terminate immediately if the other party commits a material breach\n"
        "and fails to remedy it within fifteen days of written notice.\n"
    )
    api_docs = (
        "# Widget API\n\n## Authentication\n\n"
        "Every request needs an `X-Api-Key` header. Requests without one return 401.\n"
    )
    corpus = Corpus.from_texts(
        {"contract.md": contract, "api.md": api_docs}, media_type=MediaType.MARKDOWN
    )
    evalset = EvalSet(
        id="es",
        items=(
            EvalItem(
                id="q1",
                question="How much notice is needed to terminate for convenience, and where "
                "must it be sent?",
                anchors=(
                    GoldAnchor(
                        source_id="contract.md", quote="thirty days\nwritten notice", grade=2
                    ),
                    GoldAnchor(source_id="contract.md", quote="Schedule A", grade=1),
                ),
            ),
            EvalItem(
                id="q2",
                question="What happens on a material breach?",
                anchors=(
                    GoldAnchor(
                        source_id="contract.md", quote="fifteen days of written notice", grade=2
                    ),
                ),
            ),
            EvalItem(
                id="q3",
                question="Which header carries the API key?",
                anchors=(GoldAnchor(source_id="api.md", quote="X-Api-Key", grade=2),),
            ),
        ),
    )

    # `METRICS` is the one shared registry a custom metric has to reach, so this leaves a name
    # behind unless it is taken out again -- and a stray `weighted_recall` would follow every
    # later test in the process that lists or validates against the registry.
    registered_here = "weighted_recall" not in METRICS
    if registered_here:
        METRICS.register("weighted_recall", doc="Recall weighted by grade, not by chunk count.")(
            WeightedRecall
        )
    try:
        results = Runner(corpus=corpus, headline="weighted_recall@5").run(
            matrix(chunker=["sentence:1", "fixed:20,overlap=0"], embedder="tfidf", k=1),
            evalset,
            mode="factorial",
        )
        rows = [
            f"{row['config']:52} {row['weighted_recall@5']:6.3f} {row['recall@5']:6.3f}"
            for row in results.leaderboard("weighted_recall@5", extra=["recall@5"])
        ]
        summary = results.summary("weighted_recall@5")
        comparison = results.is_the_winner_real("weighted_recall@5")
    finally:
        if registered_here:
            METRICS.unregister("weighted_recall")

    text = page.read_text()
    for row in rows:
        assert row in text, f"leaderboard line is not what the page prints:\n{row}"
    assert summary in text, f"summary is not what the page prints:\n{summary}"

    # The interval includes zero here, so this block goes through the branch of `verdict()`
    # whose wording did not change. If that ever flips, the page needs the other sentence.
    assert comparison.difference.excludes_zero is False


def test_the_documented_version_is_the_version_that_ships() -> None:
    """Five places across four pages printed `0.9.0` after 0.9.1 shipped. Each one is a
    transcript of a real command, which is exactly why they are worth keeping right."""
    import contextgrid

    root = Path(__file__).resolve().parents[2]
    docs = root / "docs-site"
    if not docs.is_dir():  # pragma: no cover - the installed package ships no docs tree
        pytest.skip("documentation tree is not present")

    current = contextgrid.__version__
    # Any release-shaped number on a page, however it is quoted: `context-grid 0.9.0`,
    # `'contextgrid': '0.9.0'`, or a bare `0.9.0` on its own line under a `print()`.
    looks_like_a_version = re.compile(r"\b\d+\.\d+\.\d+\b")

    stale: list[str] = []
    for page in sorted([*docs.rglob("*.mdx"), *(root / "docs").rglob("*.md")]):
        # `docs/drives` are transcripts of drives against releases that have already shipped.
        # The version in them is part of the record, not a claim about what is current.
        if "drives" in page.parts:
            continue
        for number, line in enumerate(page.read_text().splitlines(), start=1):
            found = looks_like_a_version.findall(line)
            # Only lines that are talking about *this* package. A pinned `numpy 2.4.6` or a
            # `pymupdf>=1.24` is somebody else's version and is not ours to keep current.
            if not found or not _is_about_contextgrid(line, page.name):
                continue
            if current not in found:
                stale.append(f"{page.relative_to(root)}:{number}: {line.strip()}")

    assert not stale, "these lines print a version this package is not:\n" + "\n".join(stale)


def _is_about_contextgrid(line: str, page: str) -> bool:
    """Whether a version-shaped number on this line belongs to this package."""
    lowered = line.lower()
    if "context-grid" in lowered or "contextgrid" in lowered:
        return True
    # `reference/api` prints `cg.__version__` and the output line is the bare number, with the
    # only context being the `print` two lines above it.
    return page == "api.mdx" and line.strip().count(".") == 2 and line.strip()[0].isdigit()

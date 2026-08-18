"""What a stranger found in 0.9.3 about the things this package says it does not know.

Six findings, installed from PyPI into a container and driven from the documentation site
alone. Five of them are one failure wearing different clothes: somewhere between the number
and the reader, "we did not measure this" turns into a confident zero, an empty directory, or
silence.

* Every renderer but one filled an unmeasured metric with `0.000`. `RunResult.row` leaves it
  out on purpose -- `results.json` for a sweep that scored nothing carries no `recall@5` key
  at all -- and then the terminal table, the Markdown report and the summary sentence each
  reached for it with a default of zero and printed five configurations tied on nought.
* A sweep over an eval set with no questions in it exited `0`. Every configuration built,
  indexed and scored nothing, and CI went green.
* The one thing that explained an empty leaderboard -- the missing extra and its install
  command -- went only to stderr, which is the stream `/reference/cli` tells people to drop.
* `contextgrid diff` read two manifests, one of them stamped `PARTIAL RUN`, and reported
  nothing different between them.
* An unreadable directory was reported as an empty one, with advice that could not work.
* And the eval set's own noise floor was printed as a promise: "detects differences of 0.45
  and above", for a gap of 0.45 that the paired test it feeds calls indistinguishable.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import pytest

import contextgrid as cg
from contextgrid.cli import main
from contextgrid.core.warnings import WarningCode
from contextgrid.corpus import Corpus, CorpusError
from contextgrid.evalset import assess, minimum_detectable_difference, write_jsonl
from contextgrid.pipeline import Config
from contextgrid.report.export import format_leaderboard, results_to_markdown
from contextgrid.report.manifest import Manifest, explain_diff
from contextgrid.report.results import Results, RunResult
from contextgrid.score.significance import compare

#: Two documents of this file's own, rather than `tests.support`'s.
#:
#: `tests.support` imports `tests.pdf_fixtures`, which calls `pytest.importorskip("pymupdf")`
#: at module level -- so importing it for two strings made *this whole file* skip on any
#: machine without the `parse` extra, the guards below included. A container with the base
#: install collected zero tests here and reported success. That is the same "reported success
#: while measuring nothing" this file exists to catch, so the dependency is gone.
CONTRACT = """\
# Master Services Agreement

## Termination

Either party may terminate this agreement for convenience by giving thirty days
written notice. Notice must be delivered to the address in Schedule A.

## Fees

Fees are payable within 30 days of invoice.
"""

API_DOCS = """\
# Widget API

## Authentication

Every request carries its credential in the X-Api-Key header. A request without
one is rejected before it reaches any handler.

## Endpoints

GET /widgets returns every widget the caller can see.
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs-site"


# ---------------------------------------------------------------------------
# a workspace the CLI can be pointed at
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


def write_config(workspace: Path, body: str, *, evalset: str = "./evalset.jsonl") -> Path:
    path = workspace / "experiment.yaml"
    path.write_text(f"corpus: ./docs\nevalset: {evalset}\n{body}", encoding="utf-8")
    return path


def unmeasured_run(label: str, *, p95: float = 0.1) -> RunResult:
    """A configuration that ran and scored nothing -- what a ghost eval set produces.

    `metrics` carries what the run really did measure and no `recall@5`, which is the exact
    shape `results.json` holds after a sweep whose every anchor pointed at a document that is
    not in the corpus.
    """
    del p95
    return RunResult(
        config=Config(parser="markdown", chunker=label, embedder="tfidf", index="dense"),
        metrics={"evidence_resolvable": 0.0, "embedding_quality": 0.65},
        scored_queries=0,
    )


# ---------------------------------------------------------------------------
# 1. a metric nobody computed, printed as 0.000
#
# `RunResult.row` has left an unmeasured metric out of the row since 0.9.1, and its docstring
# says why: "A number nobody measured is the most dangerous thing this package can print."
# Every renderer downstream then put it back with `row.get(metric, 0)`.
# ---------------------------------------------------------------------------


@pytest.fixture
def nothing_measured() -> Results:
    return Results(
        runs=[unmeasured_run("recursive:512"), unmeasured_run("sentence:3")],
        planned=2,
    )


def test_the_row_itself_has_no_recall_key(nothing_measured: Results) -> None:
    """The premise. Everything below is about renderers disagreeing with this."""
    for row in nothing_measured.leaderboard("recall@5"):
        assert "recall@5" not in row


def metric_cells(table: str, metric: str) -> list[str]:
    """The metric column of a fixed-width leaderboard, one entry per configuration row.

    Split from the right, because the configuration label is the one cell that holds spaces
    and the three columns after it never do. A substring search over the whole table cannot
    answer this: the `$/1k` column legitimately prints `0.0000`, which contains `0.000`.
    """
    del metric
    _header, _rule, *rows = table.splitlines()
    return [row.rsplit(maxsplit=3)[1] for row in rows]


def test_the_terminal_table_does_not_print_a_score_nobody_measured(
    nothing_measured: Results,
) -> None:
    table = format_leaderboard(nothing_measured, "recall@5")

    assert metric_cells(table, "recall@5") == ["NOT_MEASURED", "NOT_MEASURED"], (
        "the leaderboard printed a score for a metric no run computed:\n" + table
    )


def test_the_markdown_report_does_not_print_a_score_nobody_measured(
    nothing_measured: Results,
) -> None:
    report = results_to_markdown(nothing_measured, metric="recall@5")
    leaderboard = report.split("## Leaderboard", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in leaderboard.splitlines() if line.startswith("| `")]
    scores = [row.split("|")[2].strip() for row in rows]

    assert scores == ["NOT_MEASURED", "NOT_MEASURED"], (
        "report.md printed a score for a metric no run computed:\n" + leaderboard
    )


def test_the_summary_does_not_name_a_winner_at_0_000(nothing_measured: Results) -> None:
    """ "scored best on recall@5 at 0.000" is a ranking claim about an unranked field."""
    summary = nothing_measured.summary("recall@5")

    assert "at 0.000" not in summary, summary
    assert "recall@5" in summary, "the reader still has to be told which metric is missing"


def test_a_measured_zero_is_still_printed() -> None:
    """The guard is about absence, not about the digit. A real zero is a real result."""
    scored = RunResult(
        config=Config(parser="markdown", chunker="recursive:512", embedder="tfidf", index="dense"),
        metrics={"recall@5": 0.0},
        scored_queries=4,
        per_query={"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
    )
    results = Results(runs=[scored], planned=1)

    assert metric_cells(format_leaderboard(results, "recall@5"), "recall@5") == ["0.000"]
    assert "at 0.000" in results.summary("recall@5")


# ---------------------------------------------------------------------------
# 2. an eval set with no questions in it, and a green build
#
# `check` catches this and exits 1. `run` swept the whole matrix, scored nothing, printed a
# leaderboard and exited 0 -- which is the "green build for a sweep that measured nothing"
# that `/reference/cli` says `run` exists to refuse.
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_evalset(workspace: Path) -> Path:
    """A file that parses perfectly and holds nothing to ask."""
    path = workspace / "empty.jsonl"
    path.write_text(
        json.dumps({"_evalset": {"id": "empty", "version": 1, "source": "manual", "meta": {}}})
        + "\n",
        encoding="utf-8",
    )
    return path


def test_a_sweep_that_scored_nothing_fails(
    workspace: Path, empty_evalset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        workspace,
        "grid:\n  chunker: [recursive:512]\n  index: [bm25]\n",
        evalset=f"./{empty_evalset.name}",
    )

    code = main(["run", str(config)])

    assert code == 1, "a sweep that scored no questions exited 0, so CI went green"


def test_the_reason_a_sweep_scored_nothing_reaches_stdout(
    workspace: Path, empty_evalset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`> leaderboard.txt` has to carry it. Stderr is the stream the docs say to drop."""
    config = write_config(
        workspace,
        "grid:\n  chunker: [recursive:512]\n  index: [bm25]\n",
        evalset=f"./{empty_evalset.name}",
    )

    main(["run", str(config)])
    out = capsys.readouterr().out

    assert "no questions" in out, (
        "stdout never says the eval set was empty, so a captured leaderboard cannot:\n" + out
    )


def test_an_empty_eval_set_is_not_blamed_on_the_parser(
    workspace: Path, empty_evalset: Path
) -> None:
    """The warning named two causes and neither was the real one.

    It blamed the parse or the eval set's quotes. There are no quotes: there are no questions.
    """
    corpus = Corpus.from_dir(workspace / "docs")
    lab = cg.Lab(corpus)
    lab.grid(chunker="recursive:512", index="bm25")

    results = lab.run(cg.EvalSet(id="empty", items=()))

    unreachable = list(results.warnings.of_code(WarningCode.GOLD_SPAN_UNREACHABLE))
    assert unreachable, "the sweep has to say something about scoring nothing"
    said = " ".join(w.message for w in unreachable)
    assert "no questions" in said, said
    assert "parse lost the text" not in said, (
        "an empty eval set was blamed on the parser and on quotes it does not have: " + said
    )


# ---------------------------------------------------------------------------
# 3. the only reason an empty leaderboard had, on the wrong stream
#
# `contextgrid run fx.yaml > fx.out` with an index whose extra is not installed captured
# exactly `No configurations were run.` The `pip install "context-grid[index]"` that fixes it
# was on stderr alone. The budget case already echoes its reason on stdout.
# ---------------------------------------------------------------------------


def test_a_configuration_that_could_not_be_built_says_why_on_stdout(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextgrid.core.errors import MissingExtraError
    from contextgrid.grid import runner as runner_module

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise MissingExtraError("The 'faiss' index", "index", package="faiss-cpu")

    # The pipeline refuses to build, which is what a missing extra does. Faked rather than
    # named, so this says the same thing on a machine where every extra happens to be
    # installed -- and `faiss` is exactly the one the drive hit.
    monkeypatch.setattr(runner_module, "build", refuse)

    config = write_config(workspace, "grid:\n  chunker: [recursive:512]\n  index: [bm25]\n")
    code = main(["run", str(config)])
    captured = capsys.readouterr()

    assert code == 1
    assert 'pip install "context-grid[index]"' in captured.out, (
        "the install command that fixes an empty leaderboard was on stderr only, so "
        "`> leaderboard.txt` captured a failure with no reason:\n" + captured.out
    )


# ---------------------------------------------------------------------------
# 4. `diff` on a manifest stamped PARTIAL RUN
#
# `notes` is the field that says a bundle does not describe the whole matrix, and `diff`
# compared everything except it -- then told the reader the two runs "should have produced
# identical numbers".
# ---------------------------------------------------------------------------


def manifest(**overrides: object) -> Manifest:
    fields: dict[str, object] = {
        "config": {"parser": "markdown", "chunker": "recursive:512"},
        "corpus_hash": "c" * 64,
        "corpus_files": 2,
        "evalset_id": "es",
        "evalset_version": 1,
        "evalset_hash": "e" * 64,
        "resolution": {"policy": "coverage", "threshold": 0.5},
        "versions": {"contextgrid": "0.9.3"},
        "seeds": {"run": 0},
        "created_at": "2026-08-18T00:00:00+00:00",
        "notes": "",
    }
    fields.update(overrides)
    return Manifest(**fields)  # type: ignore[arg-type]


PARTIAL = (
    "PARTIAL RUN: 1 of 3 configurations ran -- the 0.0001s budget ran out. "
    "This bundle does not describe the whole matrix."
)


def test_diff_says_when_one_side_is_a_partial_run() -> None:
    text = explain_diff(manifest(), manifest(notes=PARTIAL))

    assert "PARTIAL RUN" in text or "partial" in text.lower(), (
        "diff read a manifest stamped PARTIAL RUN and never mentioned it:\n" + text
    )


def test_diff_does_not_call_a_partial_run_identical() -> None:
    """The worst version: identical configs, one truncated sweep, and a claim of sameness."""
    text = explain_diff(manifest(), manifest(notes=PARTIAL))

    assert "Nothing in these two manifests is different" not in text, text


def test_diff_on_two_whole_runs_still_reads_as_before() -> None:
    """The note is absent from both, so nothing new should appear."""
    text = explain_diff(manifest(), manifest())

    assert "Nothing in these two manifests is different" in text
    assert "partial" not in text.lower()


# ---------------------------------------------------------------------------
# 5. an unreadable directory, reported as an empty one
#
# `Corpus.from_dir` on a directory the user cannot list said "The directory holds no files at
# all" and told them to rename their files. The per-file message one function away is exactly
# right, and this is the same package saying it.
# ---------------------------------------------------------------------------


def refuse_to_walk(*blocked: Path) -> Any:
    """A stand-in for `os.walk` that reports `blocked` as unlistable and yields nothing.

    `_unlistable` learns about permissions from `os.walk`'s `onerror` callback and ignores
    what the walk yields, so this is the whole surface it depends on.

    Faked rather than chmodded because these guards previously carried
    `skipif(os.geteuid() == 0)`, and `root` can list a mode-000 directory: in a root container
    -- which is where this package gets installed and driven -- they reported success while
    running nothing at all. That is the failure this release exists to fix, so it should not
    ship inside it. `test_a_real_unreadable_directory_says_the_same_thing` below keeps one
    honest filesystem check that this fake matches reality.
    """

    def walk(top: Any, onerror: Any = None, **_kwargs: Any) -> Any:
        if onerror is not None:
            for where in blocked:
                onerror(PermissionError(13, "Permission denied", str(where)))
        return iter(())

    return walk


@pytest.fixture
def unlistable_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """What `Corpus.from_dir` says about a directory this user cannot open.

    The directory is genuinely empty, so the glob really does come back with nothing and the
    error path is reached for real; only the reason it came back empty is faked.
    """
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    monkeypatch.setattr(os, "walk", refuse_to_walk(blocked))

    with pytest.raises(CorpusError) as caught:
        Corpus.from_dir(blocked)
    return str(caught.value)


def test_an_unreadable_directory_is_not_called_empty(unlistable_message: str) -> None:
    assert "holds no files at all" not in unlistable_message, (
        "a directory nobody could open was reported as empty: " + unlistable_message
    )
    assert "ermission" in unlistable_message, (
        "the message never mentions permissions: " + unlistable_message
    )


def test_the_unreadable_directory_message_does_not_contradict_itself(
    unlistable_message: str,
) -> None:
    """It opened "no files ... matched [patterns]" and then said the patterns were never
    matched against anything. Both cannot be true, and the first one is the false one: the
    glob returned nothing because the directory could not be opened, not because its contents
    were read and rejected.

    So the whole first clause has to go, not just get a correction bolted on after it. Same
    for the trailing note about widening the pattern list -- advice for a case this is not.
    """
    assert "matched" not in unlistable_message, (
        "the message still claims something about matching patterns it never ran: "
        + unlistable_message
    )
    for pattern in ("*.txt", "*.md", "*.pdf"):
        assert pattern not in unlistable_message, (
            f"the pattern list is irrelevant when nothing was matched: {unlistable_message}"
        )
    assert "patterns=[...]" not in unlistable_message, (
        "widening the patterns cannot fix a permission bit: " + unlistable_message
    )


def test_an_unreadable_subdirectory_is_admitted_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root lists fine; what the corpus was looking for sits under a directory it cannot
    open. Saying "It holds 1 file: .log" is true and hides the part that matters."""
    root = tmp_path / "corpus"
    (root / "hidden-away").mkdir(parents=True)
    (root / "build.log").write_text("nothing to index")
    monkeypatch.setattr(os, "walk", refuse_to_walk(root / "hidden-away"))

    with pytest.raises(CorpusError) as caught:
        Corpus.from_dir(root)

    message = str(caught.value)
    assert "hidden-away" in message, (
        "a directory that could not be listed went unmentioned, so the file count reads as "
        "the whole story: " + message
    )
    # Still the ordinary no-match message, because the root itself was readable and its one
    # file really was checked against the patterns. Only the unlistable *root* drops these.
    assert "matched" in message
    assert "*.txt" in message


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root can list a mode-000 directory, so there is nothing to refuse"
)
def test_a_real_unreadable_directory_says_the_same_thing(tmp_path: Path) -> None:
    """One honest filesystem check, so the fake above cannot drift away from reality.

    This one has to skip under root -- the kernel simply does not refuse root -- which is
    exactly why it is not the test carrying the contradiction guard.
    """
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "notes.md").write_text("# Notes\n\nSomething worth indexing.\n")
    blocked.chmod(0o000)

    try:
        with pytest.raises(CorpusError) as caught:
            Corpus.from_dir(blocked)
    finally:
        blocked.chmod(stat.S_IRWXU)

    message = str(caught.value)
    assert "could not be listed" in message
    assert "ermission" in message
    assert "matched" not in message
    assert "holds no files at all" not in message


def test_an_actually_empty_directory_still_says_so(tmp_path: Path) -> None:
    """The old sentence was wrong about one case, not about every case."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(CorpusError, match="holds no files at all"):
        Corpus.from_dir(empty)


# ---------------------------------------------------------------------------
# 6. the noise floor, printed as a promise
#
# "detects differences of 0.45 and above" on 20 questions. A gap of exactly 0.45, with the
# most variance a 0-1 score allows, comes back `distinguishable=False` from the paired test
# this number is supposed to describe. `minimum_detectable_difference` is a two-independent-
# sample formula at p=0.5; `compare()` runs a paired sign-flip test on the same questions.
# ---------------------------------------------------------------------------


def worst_case_pair(n: int, gap: float) -> tuple[dict[str, float], dict[str, float]]:
    """`n` paired scores whose mean difference is about `gap`, with every difference at ±1.

    The widest a per-question difference can be, which is what "worst-case variance" has to
    mean for a paired test.
    """
    wins = round((gap * n + n) / 2)
    left = {f"q{i}": (1.0 if i < wins else 0.0) for i in range(n)}
    right = {f"q{i}": (0.0 if i < wins else 1.0) for i in range(n)}
    return left, right


@pytest.mark.parametrize("n", [20, 30, 50, 100])
def test_the_printed_floor_is_not_a_gap_the_paired_test_can_call(n: int) -> None:
    """The finding itself, kept as a standing fact rather than as a bug to fix.

    The arithmetic is unchanged -- it is a useful magnitude and every warning built on it
    still reads true. What changed is the sentence around it, and this test is here so that
    nobody restores a promise the number cannot keep.
    """
    left, right = worst_case_pair(n, minimum_detectable_difference(n))

    assert not compare(left, right).distinguishable


@pytest.mark.parametrize("n", [20, 30, 50, 100])
def test_the_floor_charges_the_difference_an_unpaired_standard_deviation(n: int) -> None:
    """Why the floor understates, pinned as arithmetic rather than as prose.

    `(z_alpha + z_power) * sqrt(2 * 0.25 / n)` is the standard error for two *independent*
    proportions, so the standard deviation it charges the difference is `sqrt(0.5)` = 0.707.
    A paired difference scores -1, 0 or +1 per question and can reach 1.0 -- which is the
    whole reason a gap at the printed floor can still fail the test above.

    Here so that changing the constants forces somebody to revisit the sentences built on
    them, in `quality.mdx` and in the docstring.
    """
    z = 1.96 + 0.84
    implied_sd = minimum_detectable_difference(n) * math.sqrt(n) / z

    assert implied_sd == pytest.approx(math.sqrt(0.5), abs=1e-9)
    assert minimum_detectable_difference(n) < z * 1.0 / math.sqrt(n), (
        "the printed floor is meant to sit below the paired worst case, not above it"
    )


def test_the_summary_does_not_promise_the_floor_is_detectable() -> None:
    evalset = cg.EvalSet(
        id="twenty",
        items=tuple(
            cg.EvalItem(
                id=f"q{i}",
                question=f"Question number {i} about the contract?",
                anchors=(cg.GoldAnchor(source_id="contract.md", quote="thirty days"),),
            )
            for i in range(20)
        ),
    )

    summary = assess(evalset).summary()

    assert "and above" not in summary, (
        "the summary promises every gap at or above the floor is detectable, and a gap of "
        "exactly the floor is not: " + summary
    )
    assert "0.45" in summary, "the number itself is still worth printing: " + summary


def test_the_formula_does_not_claim_to_be_the_paired_worst_case() -> None:
    """The docstring said "the worst-case variance (p = 0.5)". It is not the worst case: it
    charges the difference `sqrt(0.5)` where a paired difference can reach 1.0."""
    doc = (minimum_detectable_difference.__doc__ or "").lower()

    assert "worst-case variance" not in doc
    assert "unpaired" in doc, "the docstring has to name the test it assumes: " + doc
    assert "paired" in doc, "and the test that actually runs: " + doc


def test_the_verdict_does_not_claim_a_worst_case_it_is_not() -> None:
    """The third home of the same arithmetic, and the one a reader pastes into a PR.

    `_sample_size_note` inverts the identical unpaired formula to say "roughly 63 questions",
    and claimed it assumed "per-question scores vary as much as a 0-1 score possibly can".
    They can vary more than that: 0.25 is the largest variance of one 0-1 *score*, while the
    paired *difference* the test resamples ranges over -1 to +1. So the count is a lower
    bound, and saying it is worst case points the reader the wrong way.
    """
    left = {"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.5, "q6": 1.0, "q7": 0.0, "q8": 1.0}
    right = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.0, "q6": 1.0, "q7": 0.0, "q8": 0.5}
    verdict = compare(left, right).verdict()

    assert "vary as much as a 0-1 score possibly can" not in verdict, verdict
    assert "at least" in verdict, (
        "the question count is a lower bound and the sentence has to say so: " + verdict
    )


def test_the_significance_page_matches_that_verdict() -> None:
    """`/scoring/significance` prints this sentence, so it moves with it.

    The page shows it inside a `#`-prefixed Python comment block, so the prefixes come off
    before the comparison -- otherwise no rendered output in a comment could ever be checked
    against the string that produced it.
    """
    left = {"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.5, "q6": 1.0, "q7": 0.0, "q8": 1.0}
    right = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.0, "q6": 1.0, "q7": 0.0, "q8": 0.5}
    tail = compare(left, right).verdict().split("Settling a gap this size", 1)[1]
    page = (DOCS / "scoring" / "significance.mdx").read_text(encoding="utf-8")
    uncommented = re.sub(r"\s+", " ", re.sub(r"^# ?", "", page, flags=re.MULTILINE))

    assert tail in uncommented


def test_every_page_about_the_floor_says_which_test_it_assumes() -> None:
    """`quality.mdx` called it "worst-case variance"; `evalsets.md` explains the same number."""
    for page in (DOCS / "evalsets" / "quality.mdx", REPO_ROOT / "docs" / "guide" / "evalsets.md"):
        text = re.sub(r"\s+", " ", page.read_text(encoding="utf-8")).lower()
        assert "worst-case variance" not in text, page.name
        assert "unpaired" in text, f"{page.name} never says the floor assumes an unpaired test"


# ---------------------------------------------------------------------------
# 7. documentation describing a 0.9.3 that does not exist
#
# The drive read the published site, which still serves 0.9.0 and had four of these. Three
# were already corrected in this tree and only the deploy is behind; the fourth is real, and
# the rest of these tests are the regression guards that keep all four fixed.
#
# Every one reads the page with its whitespace collapsed. These sentences wrap, and a
# substring check against the raw file passes on a phrase that is present but hyphenated
# across two lines -- which is exactly how three of these looked fixed when they were not.
# ---------------------------------------------------------------------------


def page_text(*parts: str) -> str:
    """One documentation page as a single line, so a wrapped sentence still matches."""
    return re.sub(r"\s+", " ", DOCS.joinpath(*parts).read_text(encoding="utf-8"))


def test_the_disk_cache_warning_is_gone_from_running() -> None:
    """`Lab(corpus, cache=DiskCache(...))` writes to disk and reuses across processes."""
    page = page_text("lab", "running.mdx")

    assert "is silently swapped out for an in-memory one" not in page
    assert "Nothing is ever written to disk through this path" not in page


def test_a_lab_with_an_empty_disk_cache_really_does_write_to_it(tmp_path: Path) -> None:
    """The behaviour the page described, checked rather than assumed."""
    corpus = Corpus.from_texts({"a.md": CONTRACT}, media_type=cg.MediaType.MARKDOWN)
    root = tmp_path / "cache"
    cache = cg.DiskCache(root=root)
    assert not cache, "the premise: an empty DiskCache is falsy"

    lab = cg.Lab(corpus, cache=cache)
    lab.grid(chunker="recursive:512", index="bm25")
    lab.run(
        cg.EvalSet(
            id="es",
            items=(
                cg.EvalItem(
                    id="q1",
                    question="How much notice to terminate for convenience?",
                    anchors=(cg.GoldAnchor(source_id="a.md", quote="thirty days"),),
                ),
            ),
        )
    )

    assert len(cache), "nothing was written to the DiskCache the Lab was given"


def test_running_no_longer_says_a_bad_spec_stops_the_sweep() -> None:
    """It is caught per configuration and the sweep carries on -- with a warning saying the
    row is absent rather than scored zero."""
    page = page_text("lab", "running.mdx")

    assert "takes the whole sweep down with it" not in page
    assert "lab.run()` raises" not in page
    assert "configuration_failed" in page, (
        "the page has to name the code the sweep actually logs, so it can be grepped for"
    )


def test_a_bad_spec_does_not_stop_the_sweep() -> None:
    corpus = Corpus.from_texts({"a.md": CONTRACT}, media_type=cg.MediaType.MARKDOWN)
    lab = cg.Lab(corpus)
    lab.grid(chunker=["recursive:512", "not-a-real-chunker:512"], index="bm25")

    results = lab.run(
        cg.EvalSet(
            id="es",
            items=(
                cg.EvalItem(
                    id="q1",
                    question="How much notice to terminate for convenience?",
                    anchors=(cg.GoldAnchor(source_id="a.md", quote="thirty days"),),
                ),
            ),
        )
    )

    assert len(results.runs) == 1
    assert list(results.warnings.of_code(WarningCode.CONFIGURATION_FAILED))


def test_the_unpriced_model_warning_is_documented_under_its_real_code() -> None:
    """`/reference/cost` printed the warning with `'code': 'budget_reached'`, which is the
    code `/lab/running` tells people to filter on to detect a budget stop."""
    block = page_text("reference", "cost.mdx").split("Where zero means", 1)[1]

    assert "'code': 'budget_reached'" not in block
    assert "model_not_priced" in block


def test_the_unpriced_model_warning_really_carries_that_code() -> None:
    from contextgrid.cost import CostModel

    model = CostModel()
    model.estimate(
        embedder="a-model-nobody-published-a-price-for",
        index_tokens=1000,
        query_tokens_per_query=12,
    )

    codes = [entry["code"] for entry in model.warnings.to_list()]
    assert codes == [WarningCode.MODEL_NOT_PRICED.value]


def test_the_significance_verdict_example_does_not_misread_its_own_interval() -> None:
    """The page's `.verdict()` output said a gap "sits inside the confidence interval +0.062
    to +0.500, so it is consistent with no difference at all". That interval excludes zero;
    the shipped sentence says so and says the p-value is what failed."""
    page = page_text("scoring", "significance.mdx")

    assert "+0.062 to +0.500, so it is consistent with no difference at all" not in page


def test_the_significance_verdict_example_matches_what_ships() -> None:
    left = {"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.5, "q6": 1.0, "q7": 0.0, "q8": 1.0}
    right = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0, "q5": 0.0, "q6": 1.0, "q7": 0.0, "q8": 0.5}
    verdict = compare(left, right, left="bm25", right="dense", metric="recall@5").verdict()

    # The distinctive clause, rather than the whole paragraph, because the page wraps it.
    assert "the p-value of 0.245 is not below alpha 0.05" in verdict
    assert "p-value of 0.245 is not below alpha 0.05" in page_text("scoring", "significance.mdx")

"""What a stranger driving 0.9.2 found in the scoring, eval-set and report paths.

Six of these are the same shape: a number or a sentence that is true of one output and not
of the one beside it. A bundle whose docstring promises a manifest and writes four files. A
warning that reaches `results.json` and not `report.md`. A message that names the parser as
the culprit when the eval set was wrong. A summary line whose own number is rejected when
you hand it back. A CSV writer that drops two fields and says nothing.

The seventh is the documentation printing output the tool stopped producing two releases
ago, which is the same bug with a slower fuse.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from contextgrid.core.documents import Document, ParsedDocument
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor, GoldSpan
from contextgrid.core.span import Span
from contextgrid.core.types import Chunk
from contextgrid.core.warnings import Severity, WarningCode, WarningLog
from contextgrid.evalset.io import read_csv, write_csv
from contextgrid.evalset.quality import EvalSetQuality, assess
from contextgrid.pipeline import Config
from contextgrid.report.export import results_to_markdown, write_bundle
from contextgrid.report.results import Results, RunResult
from contextgrid.score.anchor import AnchorResolver
from contextgrid.score.resolve import SpanResolver

# ---------------------------------------------------------------------------
# shared scaffolding
# ---------------------------------------------------------------------------

CONTRACT = """\
# Master Services Agreement

Either party may terminate this agreement for convenience by giving thirty days
written notice.
"""

#: A sentence that appears twice, so `occurrence` has something to point past.
REPEATED = "The fee is payable within 30 days."
TWICE = f"{REPEATED} Interest accrues after that. {REPEATED}"


def parse(text: str, parser: str = "markdown", doc_id: str = "a.md") -> ParsedDocument:
    return ParsedDocument(document=Document(id=doc_id, text=text), parser=parser)


def one_run() -> Results:
    return Results(runs=[RunResult(config=Config(index="bm25"), metrics={"recall@3": 0.5})])


# ---------------------------------------------------------------------------
# 1. a bundle that cannot be re-derived from itself
# ---------------------------------------------------------------------------


def test_a_bundle_told_where_the_inputs_are_writes_a_manifest(tmp_path: Path) -> None:
    """`write_bundle`'s docstring promises the manifest, and a default call wrote four files.

    `contextgrid diff` is documented as reading two `manifest.json` files "from earlier
    bundled runs", so a bundle without one is a documented workflow with no input. Everything
    a manifest needs is already in the arguments: the winning `Config`, the corpus directory
    and the eval set file.
    """
    from contextgrid.evalset import write_jsonl

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    questions = write_jsonl(
        EvalSet(id="quiz", items=(EvalItem(id="q1", question="What is the notice period?"),)),
        tmp_path / "quiz.jsonl",
    )

    written = write_bundle(
        one_run(), tmp_path / "bundle", metric="recall@3", corpus=docs, evalset=questions
    )

    assert (tmp_path / "bundle" / "manifest.json").is_file()
    assert "manifest.json" in {path.name for path in written}

    payload = json.loads((tmp_path / "bundle" / "results.json").read_text())
    assert "manifest" in payload
    assert payload["manifest"]["corpus_files"] == 1
    assert payload["manifest"]["evalset_id"] == "quiz"
    assert "## Reproducing this" in (tmp_path / "bundle" / "report.md").read_text()


def test_a_bundle_built_by_hand_matches_the_one_the_cli_builds(tmp_path: Path) -> None:
    """The hash has to be the CLI's hash, or the two routes describe the same run differently.

    `contextgrid sweep --bundle` builds the manifest itself and passes it in. If the fallback
    inside `write_bundle` reached a different hash for the same inputs, `contextgrid diff`
    would report a change between two runs that were identical.
    """
    from contextgrid.corpus import Corpus
    from contextgrid.evalset import write_jsonl
    from contextgrid.report.manifest import Manifest, build_manifest

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    evalset = EvalSet(id="quiz", items=(EvalItem(id="q1", question="What is the notice?"),))
    questions = write_jsonl(evalset, tmp_path / "quiz.jsonl")

    write_bundle(one_run(), tmp_path / "bundle", metric="recall@3", corpus=docs, evalset=questions)

    by_hand = build_manifest(Config(index="bm25"), Corpus.from_dir(docs), evalset)
    in_bundle = Manifest.load(tmp_path / "bundle" / "manifest.json")
    assert in_bundle.hash() == by_hand.hash()


def test_a_bundle_with_no_inputs_named_still_writes_the_rest(tmp_path: Path) -> None:
    """Nothing to build a manifest from is not a reason to lose the report."""
    written = write_bundle(one_run(), tmp_path / "bundle", metric="recall@3")
    assert {path.name for path in written} == {
        "report.md",
        "results.json",
        "winning-config.yaml",
        "use_winning_config.py",
    }


def test_a_bundle_whose_corpus_has_moved_still_writes_the_rest(tmp_path: Path) -> None:
    """A corpus path that no longer resolves loses the manifest, not the whole bundle.

    Both paths are named here on purpose: with only one of them the manifest is skipped
    before anything is read, and the case worth covering is the read that fails.
    """
    from contextgrid.evalset import write_jsonl

    questions = write_jsonl(
        EvalSet(id="quiz", items=(EvalItem(id="q1", question="Gone?"),)), tmp_path / "q.jsonl"
    )
    written = write_bundle(
        one_run(),
        tmp_path / "bundle",
        metric="recall@3",
        corpus=tmp_path / "gone",
        evalset=questions,
    )
    assert "manifest.json" not in {path.name for path in written}
    assert (tmp_path / "bundle" / "report.md").is_file()


def test_a_bundle_whose_evalset_has_moved_still_writes_the_rest(tmp_path: Path) -> None:
    """The other half of the same read: the documents are there, the questions are not."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    written = write_bundle(
        one_run(),
        tmp_path / "bundle",
        metric="recall@3",
        corpus=docs,
        evalset=tmp_path / "gone.jsonl",
    )
    assert "manifest.json" not in {path.name for path in written}
    assert (tmp_path / "bundle" / "report.md").is_file()


# ---------------------------------------------------------------------------
# 1b. one directory describing two different runs
# ---------------------------------------------------------------------------


def _bundle_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    return docs


def _second_run() -> Results:
    """A different experiment: different chunker, different index, no embedder."""
    return Results(
        runs=[
            RunResult(
                config=Config(chunker="fixed:128", index="bm25", embedder=None),
                metrics={"recall@3": 0.25},
            )
        ]
    )


def test_a_second_run_does_not_leave_the_first_runs_winning_config(tmp_path: Path) -> None:
    """Run 2 into run 1's directory left `use_winning_config.py` describing run 1.

    Nothing was removed and nothing warned, so `winning-config.yaml` described a sweep that
    `manifest.json` beside it said had not happened. Somebody re-running that config
    reproduces the wrong experiment with no way to notice.
    """
    docs = _bundle_docs(tmp_path)
    bundle = tmp_path / "bundle"

    first = Results(
        runs=[RunResult(config=Config(chunker="recursive:512"), metrics={"recall@3": 0.9})]
    )
    write_bundle(first, bundle, metric="recall@3", corpus=docs)
    assert "recursive:512" in (bundle / "winning-config.yaml").read_text()

    write_bundle(_second_run(), bundle, metric="recall@3", corpus=docs)

    written = (bundle / "winning-config.yaml").read_text()
    assert "recursive:512" not in written
    assert "fixed:128" in written
    assert "recursive:512" not in (bundle / "use_winning_config.py").read_text()


def test_a_run_with_no_winner_does_not_inherit_the_last_ones_files(tmp_path: Path) -> None:
    """The case `formats` cannot fix: with no winner those two files are never written.

    Writing every format every time would still leave run 1's copies here, because the
    writer skips them for want of a winner rather than for want of a format.
    """
    docs = _bundle_docs(tmp_path)
    bundle = tmp_path / "bundle"

    write_bundle(one_run(), bundle, metric="recall@3", corpus=docs)
    assert (bundle / "winning-config.yaml").is_file()

    with pytest.warns(UserWarning, match="winning-config.yaml"):
        write_bundle(Results(runs=[]), bundle, metric="recall@3", corpus=docs)

    assert not (bundle / "winning-config.yaml").exists()
    assert not (bundle / "use_winning_config.py").exists()
    assert (bundle / "report.md").is_file()


def test_a_run_that_cannot_build_a_manifest_does_not_keep_the_last_ones(tmp_path: Path) -> None:
    """`manifest.json` is the file `contextgrid diff` reads. A stale one is worse than none."""
    docs = _bundle_docs(tmp_path)
    bundle = tmp_path / "bundle"
    from contextgrid.evalset import write_jsonl

    questions = write_jsonl(
        EvalSet(id="quiz", items=(EvalItem(id="q1", question="Q?"),)), tmp_path / "q.jsonl"
    )
    write_bundle(one_run(), bundle, metric="recall@3", corpus=docs, evalset=questions)
    assert (bundle / "manifest.json").is_file()

    with pytest.warns(UserWarning, match="manifest.json"):
        write_bundle(_second_run(), bundle, metric="recall@3", corpus=docs)

    assert not (bundle / "manifest.json").exists()


def test_clearing_a_bundle_leaves_everything_that_is_not_a_bundle_file(tmp_path: Path) -> None:
    """Only the fixed set of names a bundle can consist of. Nothing else is touched."""
    docs = _bundle_docs(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "notes.md").write_text("my own notes")
    (bundle / "report.md").write_text("stale")
    charts = bundle / "charts"
    charts.mkdir()
    (charts / "recall.png").write_bytes(b"png")

    write_bundle(one_run(), bundle, metric="recall@3", corpus=docs)

    assert (bundle / "notes.md").read_text() == "my own notes"
    assert (charts / "recall.png").is_file()
    assert (bundle / "report.md").read_text() != "stale"


def test_a_re_run_that_replaces_every_file_says_nothing(tmp_path: Path) -> None:
    """The ordinary iterate loop must not grow a warning. Only genuine leftovers do."""
    import warnings as _warnings

    docs = _bundle_docs(tmp_path)
    bundle = tmp_path / "bundle"
    write_bundle(one_run(), bundle, metric="recall@3", corpus=docs)

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        write_bundle(_second_run(), bundle, metric="recall@3", corpus=docs)


def test_clear_bundle_says_what_it_removed(tmp_path: Path) -> None:
    from contextgrid.report.export import BUNDLE_FILES, clear_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in BUNDLE_FILES:
        (bundle / name).write_text("old")
    (bundle / "keep.txt").write_text("mine")

    removed = clear_bundle(bundle)

    assert {path.name for path in removed} == set(BUNDLE_FILES)
    assert not any((bundle / name).exists() for name in BUNDLE_FILES)
    assert (bundle / "keep.txt").is_file()


def test_clearing_a_directory_that_is_not_there_is_not_an_error(tmp_path: Path) -> None:
    from contextgrid.report.export import clear_bundle

    assert clear_bundle(tmp_path / "nothing-here") == []


def test_clearing_does_not_follow_a_symlink_out_of_the_bundle(tmp_path: Path) -> None:
    """The docstring promises nothing outside the directory is touched. Worth proving.

    `report.md` symlinked at somebody's real notes would otherwise have deleted the link and,
    on a careless implementation, the file behind it.
    """
    from contextgrid.report.export import clear_bundle

    outside = tmp_path / "somewhere-else.md"
    outside.write_text("not the bundle's to delete")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "report.md").symlink_to(outside)

    assert clear_bundle(bundle) == []
    assert outside.read_text() == "not the bundle's to delete"
    assert (bundle / "report.md").is_symlink()


def test_clearing_leaves_a_directory_that_shares_a_bundle_name(tmp_path: Path) -> None:
    """`unlink` on a directory raises. A folder called `results.json` is not ours to remove."""
    from contextgrid.report.export import clear_bundle

    bundle = tmp_path / "bundle"
    (bundle / "results.json").mkdir(parents=True)

    assert clear_bundle(bundle) == []
    assert (bundle / "results.json").is_dir()


def test_experiment_yaml_is_one_of_the_files_a_bundle_owns() -> None:
    """`write_report` copies it, so a stale one describes the wrong experiment too."""
    from contextgrid.report.export import BUNDLE_FILES

    assert set(BUNDLE_FILES) == {
        "report.md",
        "results.json",
        "winning-config.yaml",
        "use_winning_config.py",
        "manifest.json",
        "experiment.yaml",
    }


#: The corpus and questions both `write_report` runs below score against.
_RUN_CORPUS = "Refunds are issued within 30 days of purchase. Express shipping is next day.\n"


def _experiment(workspace: Path, *, grid: str, formats: str) -> Path:
    """An `experiment.yaml` writing into the one `./results` directory."""
    docs = workspace / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "a.md").write_text(_RUN_CORPUS)
    (workspace / "questions.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "How long do refunds take?",
                "anchors": [{"source_id": "a.md", "quote": "within 30 days"}],
            }
        )
        + "\n"
    )
    path = workspace / "experiment.yaml"
    path.write_text(
        "corpus: ./docs\nevalset: ./questions.jsonl\n"
        f"grid:\n{grid}"
        "run:\n  k: 3\n  headline: recall@3\n"
        f"report:\n  out: ./results\n  formats: [{formats}]\n",
        encoding="utf-8",
    )
    return path


def test_two_runs_into_one_directory_cannot_describe_two_experiments(tmp_path: Path) -> None:
    """The reproduction: run 1 all four formats, run 2 `formats: [json]` into the same place.

    Run 2 exited 0 and said "wrote 3 files", and the directory afterwards held run 2's
    `manifest.json` beside run 1's `winning-config.yaml`, `report.md` and
    `use_winning_config.py`. `contextgrid run` goes through `write_report`, not
    `write_bundle`, so both writers need the same guard.
    """
    from contextgrid.config.loader import load, run, write_report

    first = _experiment(
        tmp_path,
        grid="  chunker: [recursive:512]\n  index: [dense]\n  embedder: [tfidf]\n",
        formats="markdown, json, yaml, python",
    )
    config = load(first)
    write_report(config, run(config))
    results_dir = tmp_path / "results"
    assert "recursive:512" in (results_dir / "winning-config.yaml").read_text()

    second = _experiment(
        tmp_path,
        grid="  chunker: [fixed:128]\n  index: [bm25]\n  embedder: [null]\n",
        formats="json",
    )
    config = load(second)
    with pytest.warns(UserWarning, match="winning-config.yaml"):
        write_report(config, run(config))

    # Run 2 asked for json only, so these are run 1's files or they are nothing.
    assert not (results_dir / "winning-config.yaml").exists()
    assert not (results_dir / "report.md").exists()
    assert not (results_dir / "use_winning_config.py").exists()

    manifest = json.loads((results_dir / "manifest.json").read_text())
    assert manifest["config"]["chunker"] == "fixed:128"
    assert manifest["config"]["index"] == "bm25"


# ---------------------------------------------------------------------------
# 2. a warning that reaches results.json and not report.md
# ---------------------------------------------------------------------------


def test_the_report_says_which_combinations_were_skipped() -> None:
    """`impossible_combination` is INFO, and the report drops INFO whenever runs exist.

    The CLI pulls this code by name regardless of severity -- `_why_nothing_ran` -- so the
    terminal said combinations had been skipped and the written report did not. Anybody
    reading `report.md` on its own never learned the grid was smaller than they asked for.
    """
    results = one_run()
    results.warnings.add(
        WarningCode.IMPOSSIBLE_COMBINATION,
        "`bm25` takes no embedder, so 3 combinations were skipped",
        severity=Severity.INFO,
        stage="grid",
    )

    report = results_to_markdown(results, metric="recall@3")
    assert "impossible_combination" in report
    assert "3 combinations were skipped" in report


def test_the_report_still_hides_ordinary_info_once_something_ran() -> None:
    """The INFO filter is not being removed, only pierced for the one code that needs it."""
    results = one_run()
    results.warnings.add(
        WarningCode.SMALL_EVAL_SET,
        "80% of this set is 'factoid' questions",
        severity=Severity.INFO,
        stage="evalset",
    )
    assert "small_eval_set" not in results_to_markdown(results, metric="recall@3")


# ---------------------------------------------------------------------------
# 3. blaming the parser for an eval set's mistake
# ---------------------------------------------------------------------------


def _aggregate(log: WarningLog) -> str:
    """The one summary line `resolve` adds after every item, not the per-item ones."""
    entries = log.of_code(WarningCode.ANCHOR_NOT_FOUND)
    return next(entry.message for entry in entries if "total" in entry.detail)


def _resolve(anchor: GoldAnchor, *, text: str = CONTRACT) -> WarningLog:
    evalset = EvalSet(id="e", items=(EvalItem(id="q1", question="Q?", anchors=(anchor,)),))
    _, log = AnchorResolver().resolve(evalset, {"a.md": parse(text)})
    return log


def test_an_invented_quote_is_not_reported_as_a_fact_about_the_parser() -> None:
    """The aggregate asserted "a fact about the parser, not the eval set".

    It fires for an invented quote, a quote naming the wrong `source_id` and an out-of-range
    `occurrence` just as readily as for a parser that mangled a table. Nothing in the code
    can tell those apart, so nothing in the message may claim to.
    """
    message = _aggregate(_resolve(GoldAnchor(source_id="a.md", quote="No such sentence.")))
    assert "a fact about the parser, not the eval set" not in message
    assert "eval set" in message


def test_the_aggregate_still_says_the_questions_cannot_be_answered() -> None:
    """The measurement survives the hedge: unresolved evidence is unanswerable either way."""
    message = _aggregate(_resolve(GoldAnchor(source_id="a.md", quote="No such sentence.")))
    assert "cannot be answered" in message
    assert "1 of 1" in message


def test_an_out_of_range_occurrence_is_counted_apart_from_missing_evidence() -> None:
    """Two different mistakes, and the aggregate no longer adds them together."""
    message = _aggregate(
        _resolve(
            GoldAnchor(source_id="a.md", quote="The fee is payable within 30 days.", occurrence=7),
            text=TWICE,
        )
    )
    assert "occurrence" in message
    assert "lost" not in message


# ---------------------------------------------------------------------------
# 3b. the same misattribution, one warning over
# ---------------------------------------------------------------------------


def _unreachable(item: EvalItem, chunks: list[Chunk]) -> str:
    """The `gold_span_unreachable` message `SpanResolver` produces for one item."""
    log = SpanResolver().resolve_item(item, chunks).warnings
    return next(entry.message for entry in log.of_code(WarningCode.GOLD_SPAN_UNREACHABLE))


def _chunk(cid: str, start: int, end: int, doc: str = "a.md") -> Chunk:
    return Chunk(id=cid, span=Span(doc, start, end), text="x" * (end - start))


def test_evidence_that_did_not_resolve_is_not_called_a_parser_measurement() -> None:
    """`gold_span_unreachable` still said "a measurement of the parser, not of the retriever".

    It fires for exactly the cases `ANCHOR_NOT_FOUND` fires for -- an invented quote, the
    wrong `source_id`, an out-of-range `occurrence` -- so it cannot name the parser either.
    The half that *is* supportable is the negative one: whatever went wrong, it was not the
    retriever.
    """
    item = EvalItem(
        id="dup7",
        question="Q?",
        anchors=(GoldAnchor(source_id="a.md", quote="The fee is payable."),),
    )
    message = _unreachable(item, [_chunk("c1", 0, 20)])

    assert "measurement of the parser" not in message
    assert "retriever" in message
    assert "eval set" in message
    # Points at the warning that does distinguish the causes, rather than guessing here.
    assert "anchor_not_found" in message


def test_evidence_that_did_not_resolve_still_says_it_was_excluded() -> None:
    """The measurement survives the hedge."""
    item = EvalItem(
        id="dup7", question="Q?", anchors=(GoldAnchor(source_id="a.md", quote="The fee."),)
    )
    message = _unreachable(item, [_chunk("c1", 0, 20)])
    assert "none of it was located in this parse" in message
    assert "excluded from ranking metrics" in message


def test_a_gold_span_past_the_end_of_the_text_is_not_blamed_on_the_chunker() -> None:
    """ "cannot be answered under this chunking" fires for a span that is not in the text.

    A gold span at 900-950 of a 100-character document matches no chunk, exactly as a real
    chunking gap does. Blaming the chunker sends somebody to sweep a chunk size that was
    never the problem.
    """
    item = EvalItem(id="q1", question="Q?", gold=(GoldSpan(span=Span("a.md", 900, 950)),))
    message = _unreachable(item, [_chunk("c1", 0, 50), _chunk("c2", 50, 100)])

    assert "under this chunking" not in message
    assert "0-100" in message
    assert "eval set" in message


def test_a_gold_span_in_a_document_that_produced_no_chunks_says_so() -> None:
    item = EvalItem(id="q1", question="Q?", gold=(GoldSpan(span=Span("missing.md", 0, 10)),))
    message = _unreachable(item, [_chunk("c1", 0, 50)])
    assert "no chunk" in message
    assert "missing.md" in message


def test_a_real_chunking_gap_still_names_the_chunking() -> None:
    """The honest case is left alone: the span is inside the chunked text and still missed."""
    item = EvalItem(id="q1", question="Q?", gold=(GoldSpan(span=Span("a.md", 60, 70)),))
    message = _unreachable(item, [_chunk("c1", 0, 50), _chunk("c2", 100, 150)])

    assert "under this chunking" in message
    assert "60-70" in message


# ---------------------------------------------------------------------------
# 4. an out-of-range occurrence reported as absent evidence
# ---------------------------------------------------------------------------


def test_an_out_of_range_occurrence_says_how_many_were_found() -> None:
    """The message said the evidence "does not appear". It appears twice; the index is wrong.

    `_choose` returns `None` both when there were no spans and when `occurrence` indexes past
    the ones there were, and downstream could not tell which had happened. `/evalsets/overview`
    only says `occurrence` "picks which repetition of `quote` is meant", so nothing tells a
    user the number can be too big.
    """
    log = _resolve(
        GoldAnchor(source_id="a.md", quote="The fee is payable within 30 days.", occurrence=7),
        text=TWICE,
    )
    message = next(
        entry.message
        for entry in log.of_code(WarningCode.ANCHOR_NOT_FOUND)
        if entry.subject == "q1"
    )
    assert "does not appear" not in message
    assert "2 times" in message
    assert "occurrence 7" in message
    # Numbered from zero, so the highest usable index is worth saying outright.
    assert "1" in message


def test_evidence_genuinely_absent_still_says_it_does_not_appear() -> None:
    log = _resolve(GoldAnchor(source_id="a.md", quote="No such sentence."))
    message = next(
        entry.message
        for entry in log.of_code(WarningCode.ANCHOR_NOT_FOUND)
        if entry.subject == "q1"
    )
    assert "does not appear" in message


def test_a_match_that_failed_on_its_index_knows_the_quote_was_there() -> None:
    """The distinction is on `AnchorMatch`, so a caller can act on it without parsing prose."""
    anchor = GoldAnchor(source_id="a.md", quote="The fee is payable within 30 days.", occurrence=7)
    match = AnchorResolver().locate(anchor, parse(TWICE))
    assert not match.found
    assert match.occurrence_out_of_range
    assert match.candidates == 2

    missing = AnchorResolver().locate(GoldAnchor(source_id="a.md", quote="Nope."), parse(TWICE))
    assert not missing.found
    assert not missing.occurrence_out_of_range
    assert missing.candidates == 0


# ---------------------------------------------------------------------------
# 5. the printed floor and the predicate disagreeing
# ---------------------------------------------------------------------------


def _printed_floor(quality: EvalSetQuality) -> float:
    match = re.search(r"detects differences of ([0-9.]+) and above", quality.summary())
    assert match is not None, quality.summary()
    return float(match.group(1))


def test_the_number_the_summary_prints_is_a_number_can_support_accepts() -> None:
    """On 24 questions the summary said 0.40 and `can_support(0.40)` returned False.

    The real floor is 0.404145: the sentence rounded it down, the predicate tested the
    unrounded value. Rounding the printed number *up* is the fix that keeps the sentence
    true -- rounding the predicate down would have it promise a resolution the set has not
    got.
    """
    for n in range(2, 400):
        quality = EvalSetQuality(size=n, answerable=n, reviewed=0, portable=n)
        printed = _printed_floor(quality)
        assert quality.can_support(printed), (
            f"n={n}: printed {printed}, floor {quality.detectable_difference}"
        )


def test_the_printed_floor_is_never_below_the_real_one() -> None:
    for n in range(2, 400):
        quality = EvalSetQuality(size=n, answerable=n, reviewed=0, portable=n)
        assert _printed_floor(quality) >= quality.detectable_difference - 1e-12


def test_the_printed_floor_is_the_real_one_to_two_places() -> None:
    """Rounded up, not inflated: never more than a hundredth above the truth."""
    for n in range(2, 400):
        quality = EvalSetQuality(size=n, answerable=n, reviewed=0, portable=n)
        assert _printed_floor(quality) - quality.detectable_difference < 0.01


def test_the_small_set_warning_prints_the_same_floor_as_the_summary() -> None:
    """A user reading 0.40 in the warning and 0.41 in the summary is no better off."""
    quality = EvalSetQuality(size=24, answerable=24, reviewed=24, portable=24)
    warned = next(
        entry.message
        for entry in quality.warnings().of_code(WarningCode.SMALL_EVAL_SET)
        if "detect" in entry.message
    )
    assert f"{_printed_floor(quality):.2f}" in warned


def test_a_capped_floor_still_reads_as_one() -> None:
    """Rounding up must not push the capped 1.0 to 1.01."""
    quality = EvalSetQuality(size=3, answerable=3, reviewed=0, portable=3)
    assert _printed_floor(quality) == 1.0
    assert quality.can_support(1.0)


# ---------------------------------------------------------------------------
# 6. a CSV round trip that loses two fields
# ---------------------------------------------------------------------------


def _round_trip(item: EvalItem, tmp_path: Path) -> EvalItem:
    write_csv(EvalSet(id="e", items=(item,)), tmp_path / "q.csv")
    return next(iter(read_csv(tmp_path / "q.csv")))


def test_a_csv_round_trip_keeps_the_occurrence(tmp_path: Path) -> None:
    """`occurrence=2` came back as `0`, which points at a different passage."""
    item = EvalItem(
        id="q1",
        question="Q?",
        anchors=(GoldAnchor(source_id="a.md", quote="The fee is payable.", occurrence=2),),
    )
    assert _round_trip(item, tmp_path).anchors[0].occurrence == 2


def test_a_csv_round_trip_keeps_the_review_flag(tmp_path: Path) -> None:
    """`meta.reviewed` is what `assess()` counts, so losing it resets a user's progress.

    The quality summary then reports a "% reviewed" that is wrong, and tells them off for
    work they have already done.
    """
    item = EvalItem(
        id="q1",
        question="Q?",
        anchors=(GoldAnchor(source_id="a.md", quote="The fee is payable."),),
        meta={"reviewed": True, "verdict": "accepted"},
    )
    back = _round_trip(item, tmp_path)
    assert back.meta == {"reviewed": True, "verdict": "accepted"}
    assert assess(EvalSet(id="e", items=(back,))).reviewed == 1


def test_a_csv_written_and_read_back_is_the_same_set(tmp_path: Path) -> None:
    item = EvalItem(
        id="q1",
        question="How much leave accrues per month?",
        anchors=(
            GoldAnchor(source_id="a.md", quote="1.5 days", grade=1, page_hint=4, occurrence=2),
        ),
        qtype="factoid",
        answer="1.5 days",
        meta={"reviewed": True},
    )
    assert _round_trip(item, tmp_path) == item


def test_a_spreadsheet_without_the_new_columns_still_reads(tmp_path: Path) -> None:
    """The columns are additions, not requirements. A hand-made sheet has neither."""
    (tmp_path / "hand.csv").write_text("question,doc,quote\nQ?,a.md,The fee is payable.\n")
    item = next(iter(read_csv(tmp_path / "hand.csv")))
    assert item.anchors[0].occurrence == 0
    assert item.meta == {}


def test_a_meta_cell_that_is_not_json_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    """Somebody will type a note in that column. It must not take the whole file down."""
    (tmp_path / "hand.csv").write_text("question,meta\nQ?,ask Priya about this one\n")
    assert next(iter(read_csv(tmp_path / "hand.csv"))).meta == {}


def test_writing_a_csv_says_out_loud_what_it_cannot_carry(tmp_path: Path) -> None:
    """One anchor per row and no span-form gold. Documented, and previously silent."""
    item = EvalItem(
        id="q1",
        question="Q?",
        anchors=(
            GoldAnchor(source_id="a.md", quote="first"),
            GoldAnchor(source_id="a.md", quote="second"),
        ),
    )
    with pytest.warns(UserWarning, match="anchor"):
        write_csv(EvalSet(id="e", items=(item,)), tmp_path / "q.csv")


def test_a_csv_that_loses_nothing_warns_about_nothing(tmp_path: Path) -> None:
    import warnings as _warnings

    item = EvalItem(id="q1", question="Q?", anchors=(GoldAnchor(source_id="a.md", quote="only"),))
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        write_csv(EvalSet(id="e", items=(item,)), tmp_path / "q.csv")


# ---------------------------------------------------------------------------
# 7. the documentation printing output the tool stopped producing
# ---------------------------------------------------------------------------

QUALITY_PAGE = Path(__file__).resolve().parents[2] / "docs-site" / "evalsets" / "quality.mdx"


def test_the_quality_page_prints_what_the_tool_prints() -> None:
    """Line 60 still showed the 0.9.0 wording, two releases after `summary()` changed."""
    page = QUALITY_PAGE.read_text()

    items = tuple(
        EvalItem(
            id=name,
            question="How much leave do employees accrue per month?",
            anchors=(
                GoldAnchor(
                    source_id="handbook.pdf",
                    quote="Employees accrue 1.5 days of leave per month.",
                ),
            ),
            qtype="factoid",
        )
        for name in ("q1", "q2", "q3")
    )
    quality = assess(EvalSet(id="handbook-quiz", items=items))

    assert quality.summary() in page
    assert repr(quality) in page
    for entry in quality.warnings().entries:
        assert str(entry) in page


def test_the_quality_page_no_longer_claims_three_answerable_questions() -> None:
    page = QUALITY_PAGE.read_text()
    assert "3 questions (3 answerable)" not in page


# ---------------------------------------------------------------------------
# 7b. "answerable" retired from the summary and left in the line beneath it
# ---------------------------------------------------------------------------

#: Every page that prints `EvalSetQuality.warnings()` output. `docs/drives/*` is deliberately
#: absent: those are transcripts of past runs and the old wording is the record.
QUALITY_PAGES = [
    QUALITY_PAGE,
    Path(__file__).resolve().parents[2] / "docs" / "guide" / "cli.md",
    Path(__file__).resolve().parents[2] / "docs" / "guide" / "evalsets.md",
]


def _quality(size: int, *, portable: int | None = None) -> EvalSetQuality:
    return EvalSetQuality(
        size=size,
        answerable=size,
        reviewed=size,
        portable=size if portable is None else portable,
    )


@pytest.mark.parametrize("size", [3, 24, 50, 99])
def test_no_warning_calls_a_question_answerable(size: int) -> None:
    """0.9.1 retired "answerable" from `summary()` and left it in `warnings()` beneath.

    The reason it was retired holds here word for word: `answerable` counts questions that
    carry an anchor, and the codebase uses that word for *scorable*. Same output, same
    count, same overstatement, one line lower.
    """
    for entry in _quality(size).warnings().entries:
        assert "answerable" not in entry.message, entry.message


def test_the_portability_warning_does_not_say_answerable_either() -> None:
    """The third one. It only fires when some evidence is span-form rather than quoted."""
    messages = [e.message for e in _quality(40, portable=10).warnings().entries]
    carrier = next(m for m in messages if "character spans" in m)
    assert "answerable" not in carrier
    assert "30" in carrier


def test_the_small_set_warning_matches_the_words_the_summary_uses() -> None:
    """Not merely different from "answerable" -- the same phrase as the line above it."""
    quality = _quality(3)
    warned = next(
        entry.message
        for entry in quality.warnings().of_code(WarningCode.SMALL_EVAL_SET)
        if "detect" in entry.message
    )
    assert "carry evidence" in warned
    assert "unchecked against a corpus" in warned
    assert "unchecked against a corpus" in quality.summary()


def test_the_detail_key_still_reports_the_field_by_its_real_name() -> None:
    """`answerable` is the field's name and the documented `EvalSetQuality` repr keeps it.

    Only the prose overstated things. Renaming the field would break the repr on
    `/evalsets/quality` and every reader of `warning.detail`.
    """
    entry = next(iter(_quality(3).warnings().of_code(WarningCode.SMALL_EVAL_SET)))
    assert entry.detail["answerable"] == 3
    assert "answerable=3" in repr(_quality(3))


@pytest.mark.parametrize("page", QUALITY_PAGES, ids=lambda p: p.name)
def test_every_page_printing_this_output_was_regenerated(page: Path) -> None:
    """One `assess()` call, three pages, and all of them must show what it really says."""
    text = page.read_text()
    # `reviewed=0` because that is the set all three pages were generated from.
    quality = EvalSetQuality(size=3, answerable=3, reviewed=0, portable=3)

    for entry in quality.warnings().entries:
        assert entry.message in text, f"{page.name} does not print: {entry.message}"
    assert "answerable questions can only detect" not in text


# ---------------------------------------------------------------------------
# the rounding helper itself
# ---------------------------------------------------------------------------


def test_rounding_up_is_not_fooled_by_binary_representation() -> None:
    from contextgrid.evalset.quality import round_up

    assert round_up(0.28000000000000003) == 0.28
    assert round_up(0.404145) == 0.41
    assert round_up(1.0) == 1.0
    assert round_up(0.4) == 0.4
    assert math.isclose(round_up(0.001), 0.01)

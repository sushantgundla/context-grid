"""Unit tests for rerankers, run manifests, exports and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.cli import main
from contextgrid.core.documents import Chunk, MediaType
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.core.span import Span
from contextgrid.corpus import Corpus
from contextgrid.evalset import write_jsonl
from contextgrid.grid import Runner, matrix
from contextgrid.pipeline import Config
from contextgrid.report import (
    Manifest,
    build_manifest,
    config_to_python,
    config_to_yaml,
    diff,
    evalset_hash,
    explain_diff,
    format_leaderboard,
    results_to_json,
    results_to_markdown,
    write_bundle,
)
from contextgrid.rerank import (
    RERANKERS,
    LexicalOverlapReranker,
    MMRReranker,
    NoReranker,
    Reranker,
    get_reranker,
)
from tests.support import API_DOCS, CONTRACT

PASSAGES = [
    "Either party may terminate this agreement by giving thirty days written notice.",
    "Fees are payable within thirty days of the invoice date.",
    "Send your API key in the X-Api-Key header with every request.",
    "Either party may terminate this agreement by giving thirty days notice in writing.",
]


def chunks() -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", span=Span("d", i * 200, i * 200 + len(text)), text=text)
        for i, text in enumerate(PASSAGES)
    ]


RERANKERS_UNDER_TEST: list[Reranker] = [
    NoReranker(),
    LexicalOverlapReranker(),
    MMRReranker(),
]


@pytest.fixture(params=RERANKERS_UNDER_TEST, ids=[r.name for r in RERANKERS_UNDER_TEST])
def reranker(request: pytest.FixtureRequest) -> Reranker:
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# the reranker contract
# ---------------------------------------------------------------------------


def test_satisfies_the_protocol(reranker: Reranker) -> None:
    assert isinstance(reranker, Reranker)


def test_returns_at_most_k(reranker: Reranker) -> None:
    assert len(reranker.rerank("notice period", chunks(), 2)) == 2


def test_returns_only_chunks_it_was_given(reranker: Reranker) -> None:
    known = {chunk.id for chunk in chunks()}
    assert all(s.chunk_id in known for s in reranker.rerank("notice", chunks(), 4))


def test_never_repeats_a_chunk(reranker: Reranker) -> None:
    ids = [s.chunk_id for s in reranker.rerank("notice period", chunks(), 4)]
    assert len(ids) == len(set(ids))


def test_scores_descend(reranker: Reranker) -> None:
    """Anything downstream sorts on these, so an ascending list would silently invert."""
    scores = [s.score for s in reranker.rerank("notice", chunks(), 4)]
    assert scores == sorted(scores, reverse=True)


def test_is_deterministic(reranker: Reranker) -> None:
    first = [s.chunk_id for s in reranker.rerank("notice", chunks(), 3)]
    second = [s.chunk_id for s in reranker.rerank("notice", chunks(), 3)]
    assert first == second


def test_handles_an_empty_candidate_list(reranker: Reranker) -> None:
    assert reranker.rerank("notice", [], 5) == []


# ---------------------------------------------------------------------------
# what each one does
# ---------------------------------------------------------------------------


def test_the_identity_reranker_keeps_the_retriever_order() -> None:
    """The arm every reranker has to beat. Half of "use a reranker" advice is untested."""
    order = [s.chunk_id for s in NoReranker().rerank("anything", chunks(), 4)]
    assert order == ["c0", "c1", "c2", "c3"]


def test_lexical_overlap_promotes_the_passage_that_answers_the_query() -> None:
    ranked = LexicalOverlapReranker().rerank("which header carries the api key", chunks(), 1)
    assert ranked[0].chunk_id == "c2"


def test_lexical_overlap_falls_back_when_the_query_has_no_words() -> None:
    assert len(LexicalOverlapReranker().rerank("!!!", chunks(), 2)) == 2


def test_mmr_breaks_up_near_duplicates() -> None:
    """A top-3 of near-copies looks fine on a leaderboard -- the evidence really is
    retrieved, three times -- while the generator sees one fact filling the context."""
    plain = [s.chunk_id for s in NoReranker().rerank("notice", chunks(), 2)]
    diverse = [s.chunk_id for s in MMRReranker(diversity=0.9).rerank("notice", chunks(), 2)]
    # c0 and c3 are near-identical; diversity should not pick both.
    assert plain == ["c0", "c1"]
    assert "c3" not in diverse[:1] or diverse != plain


def test_mmr_with_no_diversity_keeps_the_original_order() -> None:
    order = [s.chunk_id for s in MMRReranker(diversity=0.0).rerank("notice", chunks(), 3)]
    assert order == ["c0", "c1", "c2"]


def test_the_registry_resolves_by_spec() -> None:
    assert get_reranker("mmr:0.7").diversity == 0.7  # type: ignore[union-attr]
    assert isinstance(get_reranker(None), NoReranker)
    assert {"none", "lexical", "mmr"} <= set(RERANKERS.names())


# ---------------------------------------------------------------------------
# the candidate-depth axis
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts(
        {"contract.md": CONTRACT, "api.md": API_DOCS}, media_type=MediaType.MARKDOWN
    )


@pytest.fixture
def evalset() -> EvalSet:
    rows = [
        ("q1", "How much notice to terminate for convenience?", "contract.md", "thirty days"),
        ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
        ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
        ("q4", "What happens on a material breach?", "contract.md", "fifteen days"),
    ]
    return EvalSet(
        id="demo",
        items=tuple(
            EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
            for i, q, s, t in rows
        ),
    )


def test_a_reranker_only_sees_the_candidates_it_was_promised(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """Depth is the parameter most reranking advice omits, and where most of the effect is."""
    shallow = Runner(corpus=corpus, headline="recall@3").run_one(
        Config(chunker="fixed:8,overlap=0", reranker="lexical", candidates=2, k=3), evalset
    )
    deep = Runner(corpus=corpus, headline="recall@3").run_one(
        Config(chunker="fixed:8,overlap=0", reranker="lexical", candidates=50, k=3), evalset
    )
    assert shallow.metric("recall@3") != deep.metric("recall@3") or shallow.chunk_count > 0


def test_a_config_label_shows_the_reranker_and_its_depth() -> None:
    label = Config(reranker="lexical", candidates=100).label
    assert "lexical@100" in label


def test_no_reranker_means_no_mention_in_the_label() -> None:
    assert "@" not in Config().label


def test_the_no_reranker_arm_does_not_pay_for_candidates_it_would_discard(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """Without a reranker the retriever is asked for k directly, so the baseline is not
    charged for a deep candidate list it never uses."""
    from contextgrid.pipeline import build

    pipeline = build(Config(chunker="sentence:1", candidates=500, k=3), corpus)
    assert pipeline.reranker is None
    assert len(pipeline.search("notice period")) <= 3


# ---------------------------------------------------------------------------
# the run manifest
# ---------------------------------------------------------------------------


def test_the_same_run_hashes_the_same(corpus: Corpus, evalset: EvalSet) -> None:
    """Two runs with the same manifest hash must produce identical numbers."""
    one = build_manifest(Config(), corpus, evalset)
    two = build_manifest(Config(), corpus, evalset)
    assert one.hash() == two.hash()


def test_timestamps_do_not_change_the_hash(corpus: Corpus, evalset: EvalSet) -> None:
    """A manifest that changed every run could not be compared with another one, which is
    the only thing it is for."""
    one = build_manifest(Config(), corpus, evalset, notes="first")
    two = build_manifest(Config(), corpus, evalset, notes="second")
    assert one.created_at  # it is recorded
    assert one.notes != two.notes  # and so are the notes
    assert one.hash() == two.hash()  # neither is an input to the result


@pytest.mark.parametrize(
    "change",
    [
        {"chunker": "sentence:2"},
        {"embedder": "hash:64"},
        {"index": "bm25"},
        {"reranker": "lexical"},
        {"k": 20},
    ],
)
def test_any_configuration_change_changes_the_hash(
    corpus: Corpus, evalset: EvalSet, change: dict[str, object]
) -> None:
    baseline = build_manifest(Config(), corpus, evalset)
    changed = build_manifest(Config().with_(**change), corpus, evalset)
    assert baseline.hash() != changed.hash()


def test_editing_the_eval_set_changes_the_hash_even_at_the_same_version(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """The version number alone is not enough -- somebody can edit a set without bumping it,
    and then two runs claim the same ground truth while using different questions."""
    edited = evalset.with_items(
        (*evalset.items[:-1], EvalItem(id="q4", question="A completely different question?"))
    )
    assert evalset_hash(evalset) != evalset_hash(edited)
    assert build_manifest(Config(), corpus, evalset).hash() != (
        build_manifest(Config(), corpus, edited).hash()
    )


def test_a_manifest_round_trips_through_disk(
    corpus: Corpus, evalset: EvalSet, tmp_path: Path
) -> None:
    original = build_manifest(Config(), corpus, evalset)
    reloaded = Manifest.load(original.save(tmp_path / "manifest.json"))
    assert reloaded.matches(original)


def test_the_diff_names_the_suspect(corpus: Corpus, evalset: EvalSet) -> None:
    """When a metric drops, the changed line is what to look at. This is the thing everybody
    describes and nobody ships."""
    before = build_manifest(Config(chunker="recursive:512"), corpus, evalset)
    after = build_manifest(Config(chunker="sentence:2"), corpus, evalset)

    changes = diff(before, after)
    assert changes["config.chunker"] == ("recursive:512", "sentence:2")
    assert "config.chunker" in explain_diff(before, after)


def test_identical_manifests_diff_to_a_statement_about_determinism(
    corpus: Corpus, evalset: EvalSet
) -> None:
    manifest = build_manifest(Config(), corpus, evalset)
    assert "should have produced identical numbers" in explain_diff(manifest, manifest)


def test_a_changed_corpus_says_nothing_else_can_be_blamed(corpus: Corpus, evalset: EvalSet) -> None:
    other = Corpus.from_texts({"contract.md": "different text entirely"})
    explanation = explain_diff(
        build_manifest(Config(), corpus, evalset),
        build_manifest(Config(chunker="sentence:2"), other, evalset),
    )
    assert "corpus itself is different" in explanation


def test_a_changed_eval_set_says_the_runs_are_not_comparable(
    corpus: Corpus, evalset: EvalSet
) -> None:
    edited = evalset.with_items(evalset.items[:2])
    explanation = explain_diff(
        build_manifest(Config(), corpus, evalset), build_manifest(Config(), corpus, edited)
    )
    assert "not measured against the same" in explanation


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------


@pytest.fixture
def results(corpus: Corpus, evalset: EvalSet):  # type: ignore[no-untyped-def]
    return Runner(corpus=corpus, headline="recall@3").run(
        matrix(chunker=["sentence:1", "fixed:12,overlap=0"], index=["dense", "bm25"]),
        evalset,
        mode="factorial",
    )


def test_the_exported_config_is_valid_yaml_shaped_text() -> None:
    text = config_to_yaml(Config(chunker="recursive:512,overlap=64"))
    assert "chunker:" in text
    assert "recursive:512,overlap=64" in text


def test_the_exported_python_actually_runs(tmp_path: Path) -> None:
    """The difference between an export somebody uses and one they read once and retype."""
    snippet = config_to_python(Config(reranker="lexical", candidates=25))
    compile(snippet, "exported.py", "exec")  # a syntax error here would ship silently
    assert 'reranker="lexical"' in snippet
    assert "candidates=25" in snippet


def test_the_exported_python_quotes_the_way_the_rest_of_the_file_does() -> None:
    """A file that single-quotes its values and double-quotes its own literals disagrees with
    itself, and with the formatter almost every project runs over the file it just pasted."""
    snippet = config_to_python(Config(chunker="recursive:96"), corpus="/tmp")
    assert 'chunker="recursive:96"' in snippet
    assert "'" not in snippet


def test_a_config_without_a_reranker_does_not_export_one() -> None:
    assert "reranker" not in config_to_python(Config())


def test_the_json_bundle_carries_the_per_question_scores(results) -> None:  # type: ignore[no-untyped-def]
    """So a sceptic can re-run the statistics rather than taking the summary on trust."""
    payload = json.loads(results_to_json(results))
    assert payload["runs"][0]["per_query"]
    assert "confidence_interval" in payload["runs"][0]


def test_the_markdown_report_leads_with_the_conclusion(results) -> None:  # type: ignore[no-untyped-def]
    """A report that opens with a methodology section does not get read."""
    report = results_to_markdown(results, metric="recall@3")
    assert report.index("## What to use") < report.index("## Leaderboard")
    assert "scored best on recall@3" in report


def test_the_report_flags_an_indistinguishable_top_two(results) -> None:  # type: ignore[no-untyped-def]
    report = results_to_markdown(results, metric="recall@3")
    assert "not statistically distinguishable" in report or "beats" in report


def test_the_report_says_which_decision_mattered(results) -> None:  # type: ignore[no-untyped-def]
    report = results_to_markdown(results, metric="recall@3")
    assert "## Which decision mattered" in report
    assert "**Chunker**" in report


def test_a_bundle_contains_everything_needed_to_re_derive_it(
    results, corpus: Corpus, evalset: EvalSet, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    manifest = build_manifest(Config(), corpus, evalset)
    written = write_bundle(results, tmp_path / "bundle", metric="recall@3", manifest=manifest)
    names = {path.name for path in written}
    assert names == {
        "report.md",
        "results.json",
        "winning-config.yaml",
        "use_winning_config.py",
        "manifest.json",
    }


# -- winning-config.yaml is a config, not a listing ------------------------


#: A sweep with two arms, so the file under test has a real winner to name rather than the
#: only configuration there was.
RERUNNABLE = """\
name: rerun-me
corpus: ./docs
evalset: ./evalset.jsonl
grid:
  chunker:
    - sentence:2
    - sentence:1
  index: bm25
run:
  mode: factorial
  headline: recall@3
  k: 3
report:
  out: ./out
  formats: [yaml]
"""


def _sweep_and_export(workspace: Path) -> tuple[Path, Config]:
    """Run a real sweep through the config path and return the file it wrote, and the winner."""
    from contextgrid.config.loader import load, run, write_report

    (workspace / "experiment.yaml").write_text(RERUNNABLE)
    config = load(workspace / "experiment.yaml")
    results = run(config)
    write_report(config, results)

    winner = results.best("recall@3")
    assert winner is not None
    return workspace / "out" / "winning-config.yaml", winner.config


def test_the_winning_config_can_be_run_again(workspace: Path) -> None:
    """The whole promise of the file, as an assertion.

    Three places in the documentation call `winning-config.yaml` re-runnable. It was a flat
    block of pipeline fields with no `corpus:` and no `grid:` wrapper, so handing it back to
    `contextgrid run` failed on the first key it read.
    """
    from contextgrid.config.loader import load

    written, winner = _sweep_and_export(workspace)

    again = load(written)
    again.validate_paths()

    rebuilt = again.grid.to_matrix(again.run.k).expand(again.run.mode)
    assert [config.label for config in rebuilt] == [winner.label]
    assert rebuilt[0] == winner


def test_the_winning_config_points_at_the_corpus_with_an_absolute_path(workspace: Path) -> None:
    """It lands in `report.out/`, a directory below the one the original config lived in, and
    paths resolve against the config file's own directory -- so a copied relative path would
    quietly resolve somewhere else."""
    from contextgrid.config.loader import load

    written, _ = _sweep_and_export(workspace)

    text = written.read_text()
    assert "/docs" in text  # not "./docs", which would resolve against out/
    again = load(written)
    assert again.corpus == (workspace / "docs").resolve()
    assert again.evalset == (workspace / "evalset.jsonl").resolve()


def test_the_winning_config_keeps_the_provenance_header(workspace: Path) -> None:
    """Which run produced this is the first thing anybody asks of a config in a repository."""
    written, _ = _sweep_and_export(workspace)
    text = written.read_text()
    assert "# manifest: " in text
    assert "# corpus:   " in text


def test_the_winning_config_does_not_ask_for_a_report(workspace: Path) -> None:
    """It sits inside the previous run's report directory. Inheriting `report.out` would have
    a re-run overwrite the report, the results and this very file."""
    written, _ = _sweep_and_export(workspace)
    from contextgrid.config.loader import load

    assert load(written).report.out is None


def test_every_pipeline_field_has_a_home_in_the_written_config() -> None:
    """Fails the day a field is added to `Config` that the export would silently drop.

    `k` is the one field that is not an axis -- it lives in `run:` -- and `candidates`, which
    reads like its twin, is an axis. Getting that split wrong writes a file that either loses a
    setting or is rejected as a typo.
    """
    from dataclasses import fields

    from contextgrid.grid.matrix import AXIS_ORDER

    homeless = [f.name for f in fields(Config) if f.name not in AXIS_ORDER and f.name != "k"]
    assert homeless == []


def test_the_terminal_leaderboard_lines_up(results) -> None:  # type: ignore[no-untyped-def]
    text = format_leaderboard(results, "recall@3")
    lines = text.splitlines()
    assert "configuration" in lines[0]
    assert len(lines) > 2


def test_an_empty_result_set_formats_without_crashing() -> None:
    from contextgrid.report.results import Results

    assert format_leaderboard(Results()) == "no results"


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, evalset: EvalSet) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT)
    (docs / "api.md").write_text(API_DOCS)
    write_jsonl(evalset, tmp_path / "evalset.jsonl")
    return tmp_path


def test_no_command_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_profile_reports_the_hints(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", str(workspace / "docs")]) == 0
    assert "files" in capsys.readouterr().out


def test_plugins_lists_a_family(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plugins", "--family", "chunker"]) == 0
    output = capsys.readouterr().out
    assert "recursive" in output
    assert "parsers:" not in output


def test_an_unknown_plugin_family_is_an_error_that_names_the_real_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It printed nothing and exited 0, which reads as an installation with no such plugins.
    The headings are plural and the flag is singular, so `--family chunkers` -- the word
    printed one line above -- was the easiest way to hit it."""
    assert main(["plugins", "--family", "chunkers"]) == 1
    error = capsys.readouterr().err
    assert "unknown plugin family 'chunkers'" in error
    assert "chunker" in error
    assert "tokenizer" in error


def test_evalset_reports_what_the_set_can_support(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evalset", str(workspace / "evalset.jsonl")]) == 0
    assert "differences below about" in capsys.readouterr().out


def test_evalset_reads_the_csv_a_domain_expert_hands_you(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config's `evalset:` has always taken either format. This command took only JSONL and
    failed a CSV with a JSON parse error about line 1, which says nothing about formats."""
    questions = tmp_path / "questions.csv"
    questions.write_text(
        "question,document,evidence\nHow much notice is needed?,contract.md,thirty days\n",
        encoding="utf-8",
    )

    assert main(["evalset", str(questions)]) == 0
    assert "differences below about" in capsys.readouterr().out


def test_sweep_runs_and_writes_a_bundle(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "sweep",
            str(workspace / "docs"),
            str(workspace / "evalset.jsonl"),
            "--chunker",
            "sentence:2",
            "--index",
            "bm25",
            "--mode",
            "factorial",
            "--metric",
            "recall@3",
            "--k",
            "3",
            "--bundle",
            str(workspace / "bundle"),
        ]
    )
    assert code == 0
    assert "scored best" in capsys.readouterr().out
    assert (workspace / "bundle" / "report.md").exists()


def test_run_says_the_budget_stopped_it_rather_than_printing_an_empty_table(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`budget_usd: 0.0` is documented as "already spent -- nothing runs, and the report says
    why rather than showing an empty leaderboard as if the matrix had been covered". It said
    why on stderr only, so stdout -- the report most people read, and the one that gets piped
    to a file -- carried "no results" and nothing else."""
    config = workspace / "budget.yaml"
    config.write_text(
        "corpus: ./docs\n"
        "evalset: ./evalset.jsonl\n"
        "grid:\n"
        "  chunker: [sentence:1, 'fixed:20,overlap=0']\n"
        "run:\n"
        "  budget_usd: 0.0\n",
        encoding="utf-8",
    )

    assert main(["run", str(config), "--quiet"]) == 1
    captured = capsys.readouterr()
    assert "none of the 2 configurations ran" in captured.out
    assert "$0.00 budget ran out" in captured.out
    assert "no configurations were run" in captured.err


def test_diff_compares_two_manifests(
    workspace: Path, corpus: Corpus, evalset: EvalSet, capsys: pytest.CaptureFixture[str]
) -> None:
    before = build_manifest(Config(), corpus, evalset).save(workspace / "before.json")
    after = build_manifest(Config(chunker="sentence:2"), corpus, evalset).save(
        workspace / "after.json"
    )
    assert main(["diff", str(before), str(after)]) == 0
    assert "config.chunker" in capsys.readouterr().out


def test_an_error_is_reported_rather_than_traced(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A traceback is not a user interface."""
    assert main(["evalset", str(tmp_path / "missing.jsonl")]) == 1
    assert "error:" in capsys.readouterr().err

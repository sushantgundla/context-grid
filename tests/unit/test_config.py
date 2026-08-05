"""The config file is the whole public interface for most users.

Somebody who never imports contextgrid still writes one of these, so a bad message here costs
more than a bad message anywhere else in the package. These tests are as much about what the
errors say as about what the parser accepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextgrid.config import ConfigError, ExperimentConfig, load, loads, render
from contextgrid.config.loader import build_cache, build_evalset, run, write_report
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor

MINIMAL = "corpus: ./docs\n"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_a_corpus_is_the_only_required_field(tmp_path: Path) -> None:
    config = loads(MINIMAL, base=tmp_path)
    assert config.corpus == (tmp_path / "docs").resolve()
    assert config.grid.chunker == ("recursive:512",)
    assert config.run.mode == "ofat"


def test_a_config_without_a_corpus_says_what_a_corpus_is() -> None:
    with pytest.raises(ConfigError, match="directory of documents"):
        loads("name: nope\n")


def test_json_is_accepted_whatever_the_file_is_called(tmp_path: Path) -> None:
    """The format follows the content. A .yaml file holding JSON is still valid JSON."""
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps({"corpus": "./docs", "run": {"k": 3}}), encoding="utf-8")
    assert load(path).run.k == 3


def test_yaml_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text("corpus: ./docs\nrun:\n  k: 7\n", encoding="utf-8")
    assert load(path).run.k == 7


def test_a_missing_file_names_the_path_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no config file at"):
        load(tmp_path / "absent.yaml")


def test_broken_yaml_keeps_pyyamls_line_number() -> None:
    with pytest.raises(ConfigError, match="line"):
        loads("corpus: ./docs\ngrid:\n  chunker: [a, b\n")


def test_a_top_level_list_is_rejected() -> None:
    with pytest.raises(ConfigError, match="mapping at the top level"):
        loads("- one\n- two\n")


def test_a_section_that_is_not_a_mapping_says_so() -> None:
    with pytest.raises(ConfigError, match="'grid' section must be a mapping"):
        loads("corpus: ./docs\ngrid: [chunker]\n")


def test_an_empty_section_is_the_defaults() -> None:
    """`run:` with nothing under it is how YAML spells "leave this alone"."""
    assert loads("corpus: ./docs\nrun:\n").run.k == 10


# ---------------------------------------------------------------------------
# typos
# ---------------------------------------------------------------------------


def test_a_misspelled_axis_is_rejected_and_guessed() -> None:
    """Running with defaults after a typo produces a leaderboard that answers a different
    question than the one asked, and nothing on screen says so."""
    with pytest.raises(ConfigError, match="Did you mean 'chunker'"):
        loads("corpus: ./docs\ngrid:\n  chunkers: [a]\n")


def test_a_misspelled_run_key_is_rejected_and_guessed() -> None:
    with pytest.raises(ConfigError, match="Did you mean 'headline'"):
        loads("corpus: ./docs\nrun:\n  headlines: recall@5\n")


def test_a_misspelled_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Did you mean 'evalset'"):
        loads("corpus: ./docs\nevalsets: ./q.jsonl\n")


def test_an_unguessable_key_still_lists_what_is_allowed() -> None:
    with pytest.raises(ConfigError, match=r"Known keys:.*chunker"):
        loads("corpus: ./docs\ngrid:\n  zzzzzz: [a]\n")


def test_a_misspelled_report_format_lists_the_real_ones() -> None:
    with pytest.raises(ConfigError, match="unknown report format"):
        loads("corpus: ./docs\nreport:\n  formats: [pdf]\n")


# ---------------------------------------------------------------------------
# one value or many
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["parser", "chunker", "embedder", "index"])
def test_every_axis_takes_a_bare_value(axis: str) -> None:
    config = loads(f"corpus: ./docs\ngrid:\n  {axis}: one\n")
    assert getattr(config.grid, axis) == ("one",)


@pytest.mark.parametrize("axis", ["parser", "chunker", "embedder", "index"])
def test_every_axis_takes_a_list(axis: str) -> None:
    config = loads(f"corpus: ./docs\ngrid:\n  {axis}: [one, two]\n")
    assert getattr(config.grid, axis) == ("one", "two")


def test_null_is_a_real_arm_on_the_axes_that_allow_it() -> None:
    """ "No reranker" is a configuration under test, not a missing value."""
    config = loads("corpus: ./docs\ngrid:\n  reranker: [null, lexical]\n")
    assert config.grid.reranker == (None, "lexical")


def test_null_on_an_axis_that_needs_a_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot contain an empty value"):
        loads("corpus: ./docs\ngrid:\n  chunker: [null]\n")


def test_candidates_must_be_whole_numbers() -> None:
    with pytest.raises(ConfigError, match="whole numbers"):
        loads("corpus: ./docs\ngrid:\n  candidates: [deep]\n")


def test_candidates_takes_a_bare_number() -> None:
    assert loads("corpus: ./docs\ngrid:\n  candidates: 30\n").grid.candidates == (30,)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_an_unknown_mode_lists_the_three_that_exist() -> None:
    with pytest.raises(ConfigError, match="'factorial', 'ofat' or 'staged'"):
        loads("corpus: ./docs\nrun:\n  mode: everything\n")


def test_a_headline_without_a_cutoff_is_rejected() -> None:
    """`recall` alone is ambiguous, and silently picking a k for the user decides the
    leaderboard's ordering on their behalf."""
    with pytest.raises(ConfigError, match="must name a cut-off"):
        loads("corpus: ./docs\nrun:\n  headline: recall\n")


def test_a_headline_with_a_word_cutoff_is_rejected() -> None:
    with pytest.raises(ConfigError, match="non-numeric cut-off"):
        loads("corpus: ./docs\nrun:\n  headline: recall@five\n")


def test_an_unknown_metric_lists_the_real_ones() -> None:
    with pytest.raises(ConfigError, match="Available: recall, precision"):
        loads("corpus: ./docs\nrun:\n  headline: f1@5\n")


def test_the_headline_cutoff_is_always_reported() -> None:
    """Sorting the leaderboard on a number that is not in the table would be absurd."""
    assert 7 in loads("corpus: ./docs\nrun:\n  headline: ndcg@7\n").run.ks


@pytest.mark.parametrize("value", ["0", "-1"])
def test_k_must_be_at_least_one(value: str) -> None:
    with pytest.raises(ConfigError, match=r"run\.k must be at least 1"):
        loads(f"corpus: ./docs\nrun:\n  k: {value}\n")


def test_an_unknown_resolution_policy_is_rejected() -> None:
    with pytest.raises(ConfigError, match="'coverage', 'iou' or 'containment'"):
        loads("corpus: ./docs\nrun:\n  resolution_policy: overlap\n")


@pytest.mark.parametrize("value", ["0", "1.5"])
def test_the_resolution_threshold_must_be_a_fraction(value: str) -> None:
    with pytest.raises(ConfigError, match="must be in"):
        loads(f"corpus: ./docs\nrun:\n  resolution_threshold: {value}\n")


def test_an_unknown_cache_is_rejected() -> None:
    with pytest.raises(ConfigError, match="'memory', 'disk' or 'none'"):
        loads("corpus: ./docs\nrun:\n  cache: redis\n")


def test_a_budget_that_is_not_a_number_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a number"):
        loads("corpus: ./docs\nrun:\n  budget_seconds: soon\n")


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_relative_paths_resolve_against_the_config_not_the_shell(tmp_path: Path) -> None:
    """The alternative is a config that only runs from one directory, which is the first thing
    anybody breaks."""
    (tmp_path / "nested").mkdir()
    path = tmp_path / "nested" / "experiment.yaml"
    path.write_text("corpus: ./docs\nreport:\n  out: ./results\n", encoding="utf-8")

    config = load(path)
    assert config.corpus == (tmp_path / "nested" / "docs").resolve()
    assert config.report.out == (tmp_path / "nested" / "results").resolve()


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    config = loads(f"corpus: {tmp_path}\n", base=Path("/somewhere/else"))
    assert config.corpus == tmp_path


def test_a_home_relative_path_is_expanded() -> None:
    assert not str(loads("corpus: ~/docs\n").corpus).startswith("~")


def test_missing_inputs_are_reported_before_anything_expensive(tmp_path: Path) -> None:
    config = loads(f"corpus: {tmp_path / 'absent'}\n")
    with pytest.raises(ConfigError, match="corpus not found"):
        config.validate_paths()


def test_a_missing_evalset_is_reported_too(tmp_path: Path) -> None:
    config = loads(f"corpus: {tmp_path}\nevalset: {tmp_path / 'absent.jsonl'}\n")
    with pytest.raises(ConfigError, match="eval set not found"):
        config.validate_paths()


def test_a_config_can_be_inspected_without_its_corpus_present() -> None:
    """Parsing and existence-checking are separate so `check` can validate a config written
    for a machine other than this one."""
    assert loads("corpus: /nowhere/at/all\n").describe()


def test_scoring_without_an_evalset_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="contextgrid evalset"):
        build_evalset(loads(f"corpus: {tmp_path}\n"))


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------


def test_env_vars_are_substituted(monkeypatch: pytest.MonkeyPatch) -> None:
    """So a config can reference a key without containing one -- a config with a secret in it
    ends up in version control, and then in a screenshot."""
    monkeypatch.setenv("CG_TEST_CORPUS", "/data/docs")
    assert loads("corpus: ${CG_TEST_CORPUS}\n").corpus == Path("/data/docs")


def test_an_unset_env_var_names_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CG_TEST_ABSENT", raising=False)
    with pytest.raises(ConfigError, match=r"\$\{CG_TEST_ABSENT\}"):
        loads("corpus: ${CG_TEST_ABSENT}\n")


def test_a_dollar_sign_that_is_not_a_variable_is_left_alone() -> None:
    assert loads("corpus: ./a$b\n").corpus.name == "a$b"


# ---------------------------------------------------------------------------
# describing before running
# ---------------------------------------------------------------------------


def test_describe_reports_the_matrix_and_what_will_actually_run() -> None:
    config = loads(
        "corpus: ./docs\n"
        "grid:\n"
        "  chunker: [a, b, c]\n"
        "  index: [dense, bm25]\n"
        "run:\n"
        "  mode: factorial\n"
    )
    text = config.describe()
    # The shape line joins the axes with a multiplication sign.
    assert "3 \u00d7 2" in text or "\u00d7 3 \u00d7" in text
    assert "factorial" in text
    assert "recall@5" in text


def test_impossible_combinations_are_counted_not_hidden() -> None:
    """`embedder: [tfidf, null]` with `index: [dense, bm25]` obviously means two configurations,
    but a factorial expansion also produces "dense with no vectors", which cannot run."""
    config = loads(
        "corpus: ./docs\n"
        "grid:\n"
        "  embedder: [tfidf, null]\n"
        "  index: [dense, bm25]\n"
        "run:\n"
        "  mode: factorial\n"
    )
    assert "skipped" in config.describe()


def test_a_config_survives_a_round_trip_through_its_own_dict() -> None:
    """`as_dict` feeds the report bundle, so it has to be readable back in."""
    original = loads(
        "name: round-trip\n"
        "corpus: /docs\n"
        "grid:\n"
        "  chunker: [a, b]\n"
        "  reranker: [null, lexical]\n"
        "run:\n"
        "  k: 4\n"
        "  headline: ndcg@4\n"
    )
    again = ExperimentConfig.from_mapping(original.as_dict(), base=Path("/"))
    assert again.grid == original.grid
    assert again.run == original.run
    assert again.name == "round-trip"


def test_the_name_defaults_to_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "chunker-sweep.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    assert load(path).name == "chunker-sweep"


# ---------------------------------------------------------------------------
# the starter file
# ---------------------------------------------------------------------------


def test_the_generated_starter_config_parses() -> None:
    """A template that does not load is worse than no template."""
    assert loads(render()).corpus.name == "documents"


def test_the_starter_config_only_offers_plugins_that_are_installed() -> None:
    """Every optional plugin is registered whether or not its package is present, so listing
    the registry wholesale would advertise chunkers that raise on first use. Somebody's first
    contact with the tool should not be an ImportError from a file the tool wrote for them."""
    from contextgrid.chunk import get_chunker

    for line in render().splitlines():
        if not line.strip().startswith("# also available:") or "recursive" not in line:
            continue
        for name in line.split("available:", 1)[1].split(","):
            if name.strip():
                get_chunker(name.strip())  # raises if the package is missing


def test_the_starter_config_does_not_list_a_plugin_it_already_chose() -> None:
    """`recursive:512` is on the chunker line; repeating `recursive` underneath reads as a
    second, different plugin."""
    for line in render().splitlines():
        if not (line.strip().startswith("# also available:") and "chonkie" in line):
            continue
        assert " recursive," not in line
        assert not line.rstrip().endswith(" recursive")


def test_comment_lines_in_the_starter_config_stay_readable() -> None:
    for line in render().splitlines():
        assert len(line) <= 100, line


def test_the_starter_config_carries_the_paths_it_was_given(tmp_path: Path) -> None:
    text = render(corpus="./my-docs", evalset="./my-questions.jsonl")
    config = loads(text, base=tmp_path)
    assert config.corpus == (tmp_path / "my-docs").resolve()
    assert config.evalset == (tmp_path / "my-questions.jsonl").resolve()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A corpus and an eval set small enough to run in a test, real enough to score."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text(
        "# Refunds\n\nRefunds are issued within 30 days of purchase.\n\n"
        "## Exceptions\n\nDigital goods are not refundable once downloaded.\n",
        encoding="utf-8",
    )
    (docs / "shipping.md").write_text(
        "# Shipping\n\nStandard shipping takes 5 to 7 business days.\n\n"
        "## Express\n\nExpress shipping arrives the next business day.\n",
        encoding="utf-8",
    )

    evalset = EvalSet(
        id="config-e2e",
        items=(
            EvalItem(
                id="q1",
                question="How long do refunds take?",
                anchors=(GoldAnchor(quote="within 30 days of purchase", source_id="refunds.md"),),
            ),
            EvalItem(
                id="q2",
                question="How fast is express shipping?",
                anchors=(GoldAnchor(quote="the next business day", source_id="shipping.md"),),
            ),
        ),
    )
    from contextgrid.evalset.io import write_jsonl

    write_jsonl(evalset, tmp_path / "questions.jsonl")
    return tmp_path


def test_one_config_file_runs_the_whole_experiment(workspace: Path) -> None:
    """The point of the package: a corpus, some questions, one file, a leaderboard."""
    path = workspace / "experiment.yaml"
    path.write_text(
        "name: e2e\n"
        "corpus: ./docs\n"
        "evalset: ./questions.jsonl\n"
        "grid:\n"
        "  chunker: [recursive:256, sentence:2]\n"
        "  index: bm25\n"
        "  embedder: null\n"
        "run:\n"
        "  k: 3\n"
        "  headline: recall@3\n"
        "report:\n"
        "  out: ./results\n"
        "  formats: [markdown, json, yaml, python]\n",
        encoding="utf-8",
    )

    config = load(path)
    results = run(config)

    assert len(results.runs) == 2
    assert results.best("recall@3") is not None

    written = {p.name for p in write_report(config, results)}
    assert {"report.md", "results.json", "manifest.json"} <= written
    # The config that produced the bundle travels with it. A bundle that cannot be re-run is a
    # screenshot.
    assert "experiment.yaml" in written
    assert (workspace / "results" / "experiment.yaml").read_text(encoding="utf-8")


def test_the_winning_config_is_written_as_runnable_python(workspace: Path) -> None:
    path = workspace / "experiment.yaml"
    path.write_text(
        "corpus: ./docs\n"
        "evalset: ./questions.jsonl\n"
        "grid:\n"
        "  index: bm25\n"
        "  embedder: null\n"
        "run:\n"
        "  k: 3\n"
        "  headline: recall@3\n"
        "report:\n"
        "  out: ./results\n"
        "  formats: [python]\n",
        encoding="utf-8",
    )
    config = load(path)
    write_report(config, run(config))

    source = (workspace / "results" / "use_winning_config.py").read_text(encoding="utf-8")
    compile(source, "use_winning_config.py", "exec")


def test_no_report_out_writes_nothing(workspace: Path) -> None:
    """A sweep run for the numbers on screen should not scatter files."""
    config = loads(
        f"corpus: {workspace / 'docs'}\n"
        f"evalset: {workspace / 'questions.jsonl'}\n"
        "grid:\n  index: bm25\n  embedder: null\n"
        "run:\n  k: 3\n  headline: recall@3\n"
    )
    assert write_report(config, run(config)) == []


def test_the_disk_cache_lands_beside_the_results(workspace: Path) -> None:
    config = loads(
        f"corpus: {workspace / 'docs'}\n"
        f"report:\n  out: {workspace / 'results'}\n"
        "run:\n  cache: disk\n"
    )
    cache = build_cache(config)
    assert ".contextgrid-cache" in str(getattr(cache, "root", ""))


def test_a_single_file_corpus_works(workspace: Path) -> None:
    """Pointing at one document is the smallest useful experiment, and people try it first."""
    config = loads(f"corpus: {workspace / 'docs' / 'refunds.md'}\n")
    from contextgrid.config.loader import build_corpus

    assert len(build_corpus(config).files) == 1


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def test_init_check_run_is_a_working_cycle(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The path a new user actually takes: generate a config, validate it, run it."""
    from contextgrid.cli import main

    path = workspace / "experiment.yaml"
    assert main(["init", str(path), "--corpus", "./docs", "--evalset", "./questions.jsonl"]) == 0

    # The template sweeps embedders by default, which needs an optional extra. Narrow it to
    # what a bare install can run, leaving the rest of the generated file untouched.
    text = path.read_text(encoding="utf-8")
    text = text.replace("  index: [dense, bm25, hybrid]", "  index: [bm25]")
    text = text.replace("  embedder: [tfidf, null]", "  embedder: [null]")
    text = text.replace("  k: 10", "  k: 3").replace("headline: recall@5", "headline: recall@3")
    path.write_text(text, encoding="utf-8")

    assert main(["check", str(path)]) == 0
    assert "config is valid" in capsys.readouterr().out

    assert main(["run", str(path), "--quiet"]) == 0
    assert (workspace / "results" / "report.md").exists()


def test_init_refuses_to_overwrite_without_being_told_to(workspace: Path) -> None:
    """Silently replacing a config somebody spent an afternoon tuning would be unforgivable."""
    from contextgrid.cli import main

    path = workspace / "experiment.yaml"
    path.write_text("corpus: ./docs\n", encoding="utf-8")

    assert main(["init", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == "corpus: ./docs\n"

    assert main(["init", str(path), "--force"]) == 0
    assert path.read_text(encoding="utf-8") != "corpus: ./docs\n"


def test_check_fails_on_a_config_whose_inputs_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Better a non-zero exit now than an hour into a sweep."""
    from contextgrid.cli import main

    path = tmp_path / "experiment.yaml"
    path.write_text("corpus: ./absent\nevalset: ./absent.jsonl\n", encoding="utf-8")

    assert main(["check", str(path)]) == 1
    assert "corpus not found" in capsys.readouterr().err


def test_check_flags_a_config_with_no_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from contextgrid.cli import main

    docs = tmp_path / "docs"
    docs.mkdir()
    path = tmp_path / "experiment.yaml"
    path.write_text(f"corpus: {docs}\n", encoding="utf-8")

    assert main(["check", str(path)]) == 1
    assert "nothing to score against" in capsys.readouterr().err


def test_check_prints_every_axis_so_the_matrix_is_visible(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from contextgrid.cli import main

    path = workspace / "experiment.yaml"
    path.write_text(
        "corpus: ./docs\nevalset: ./questions.jsonl\ngrid:\n  chunker: [a, b]\n", encoding="utf-8"
    )
    main(["check", str(path)])

    out = capsys.readouterr().out
    for axis in ("parser", "chunker", "embedder", "index", "transform", "reranker", "candidates"):
        assert axis in out


def test_a_dollar_budget_in_the_config_stops_a_real_sweep(workspace: Path) -> None:
    """End to end: `budget_usd:` in the YAML has to reach the runner. It did not before --
    the value was parsed, stored and written into the report bundle, and never checked."""
    path = workspace / "experiment.yaml"
    path.write_text(
        "corpus: ./docs\n"
        "evalset: ./questions.jsonl\n"
        "grid:\n"
        "  chunker: [recursive:128, recursive:256, sentence:2]\n"
        "  index: bm25\n"
        "  embedder: null\n"
        "run:\n"
        "  mode: factorial\n"
        "  k: 3\n"
        "  headline: recall@3\n"
        "  budget_usd: 0.0\n",
        encoding="utf-8",
    )

    results = run(load(path))

    # A zero budget is already spent, so nothing runs and the report says why rather than
    # presenting an empty leaderboard as if the matrix had been covered.
    assert results.runs == []
    assert any("budget ran out" in warning.message for warning in results.warnings)


def test_no_budget_runs_the_whole_matrix(workspace: Path) -> None:
    config = loads(
        f"corpus: {workspace / 'docs'}\n"
        f"evalset: {workspace / 'questions.jsonl'}\n"
        "grid:\n  chunker: [recursive:128, recursive:256]\n  index: bm25\n  embedder: null\n"
        "run:\n  mode: factorial\n  k: 3\n  headline: recall@3\n"
    )
    assert len(run(config).runs) == 2

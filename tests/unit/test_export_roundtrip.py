"""`use_winning_config.py` and `winning-config.yaml` must describe the same pipeline.

They did not. A sweep won by `parent-document:4 · markdown · recursive:96 ·
~relevance-feedback:3 · bm25 · lexical@20` exported a YAML file naming `ingestion` and
`retrieval`, and beside it a Python snippet naming neither -- so the code a reader
copy-pastes built plain chunking and plain search, a different pipeline from the one that
won, with nothing anywhere to say so.

The cause was a list of field names written out inside `config_to_python` that fell behind
the dataclass. So the test that matters is not "does the snippet mention ingestion": it is
"run the snippet and check you get the winner back", plus a check that fails the day somebody
adds a field to `Config` without this file covering it.
"""

from __future__ import annotations

import sys
import types
from dataclasses import fields
from typing import Any

import pytest
import yaml

from contextgrid.pipeline import Config
from contextgrid.report.export import config_to_python, config_to_yaml, results_to_markdown
from contextgrid.report.results import Results, RunResult

#: The configuration from the report that found the bug.
THE_REPORTED_WINNER = Config(
    ingestion="parent-document:4",
    parser="markdown",
    chunker="recursive:96",
    embedder=None,
    retrieval="relevance-feedback:3",
    index="bm25",
    reranker="lexical",
    candidates=20,
    k=3,
)

#: Every field set to something that is not its default. Kept here rather than built in a
#: test so the coverage check below can assert against the same object the round trip uses.
EVERY_FIELD_CHANGED = Config(
    ingestion="parent-document:4",
    parser="layout",
    chunker="recursive:96",
    embedder=None,
    transform="hyde",
    retrieval="relevance-feedback:3",
    index="bm25",
    reranker="lexical",
    candidates=20,
    k=3,
    generator="echo",
)


def _fields_at_default(config: Config) -> list[str]:
    defaults = Config()
    return [f.name for f in fields(Config) if getattr(config, f.name) == getattr(defaults, f.name)]


def _rebuild(config: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Run the exported snippet the way a reader would, and return the `Config` it built.

    The snippet ends by reading a corpus off disk and searching it, so `contextgrid` is
    swapped for a stub -- but the whole script is executed, not just the part with the
    configuration in it. An export that does not run is not an export.
    """
    snippet = config_to_python(config)
    stub = types.ModuleType("contextgrid")
    stub.__dict__.update({"Config": Config, "Corpus": _StubCorpus, "build": _stub_build})
    monkeypatch.setitem(sys.modules, "contextgrid", stub)

    namespace: dict[str, Any] = {}
    exec(compile(snippet, "use_winning_config.py", "exec"), namespace)
    rebuilt = namespace["config"]
    assert isinstance(rebuilt, Config)
    return rebuilt


class _StubCorpus:
    @staticmethod
    def from_dir(path: str) -> object:
        return object()


class _StubPipeline:
    def search(self, question: str) -> list[str]:
        return []


def _stub_build(config: Config, corpus: object) -> _StubPipeline:
    return _StubPipeline()


# -- the round trip ---------------------------------------------------------


def test_the_snippet_rebuilds_the_reported_winner_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug, as an assertion: `ingestion` and `retrieval` were silently dropped."""
    assert _rebuild(THE_REPORTED_WINNER, monkeypatch) == THE_REPORTED_WINNER


def test_the_snippet_rebuilds_a_config_with_every_field_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _rebuild(EVERY_FIELD_CHANGED, monkeypatch) == EVERY_FIELD_CHANGED


def test_the_snippet_rebuilds_a_config_that_changed_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other end of the rule: every field at its default exports no arguments at all."""
    assert _rebuild(Config(), monkeypatch) == Config()


# -- the guard against the same bug returning -------------------------------


def test_every_config_field_is_exercised_by_the_round_trip() -> None:
    """Fails the day a field is added to `Config` without a value here.

    Without this, a new field would sit at its default in `EVERY_FIELD_CHANGED`, the round
    trip would pass whether the export carried it or not, and the drift that caused the
    original bug would be invisible again.
    """
    assert _fields_at_default(EVERY_FIELD_CHANGED) == []


def test_the_snippet_names_every_field_that_is_not_at_its_default() -> None:
    snippet = config_to_python(EVERY_FIELD_CHANGED)
    missing = [f.name for f in fields(Config) if f"{f.name}=" not in snippet]
    assert missing == []


def test_as_dict_still_names_every_config_field() -> None:
    """`config_to_yaml` writes whatever `as_dict()` returns, so a field missing from `as_dict`
    disappears from `winning-config.yaml` exactly the way it disappeared from the snippet."""
    assert set(Config().as_dict()) == {f.name for f in fields(Config)}


def test_the_yaml_and_the_python_describe_the_same_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evaluator's complaint, checked directly: read both files back, compare."""
    from_yaml = Config(**yaml.safe_load(config_to_yaml(THE_REPORTED_WINNER)))
    assert from_yaml == _rebuild(THE_REPORTED_WINNER, monkeypatch)


def test_the_snippet_leaves_out_what_the_config_did_not_use() -> None:
    """Omission is still allowed, but only for a field the constructor puts back itself."""
    snippet = config_to_python(Config(chunker="structural:800"))
    assert "reranker" not in snippet
    assert "generator" not in snippet
    assert "chunker='structural:800'" in snippet


def test_the_snippet_names_the_whole_pipeline_in_a_comment() -> None:
    """So a config that changed one field still says which winner it is, next to a report
    and a leaderboard that identify configurations by that label."""
    assert THE_REPORTED_WINNER.label in config_to_python(THE_REPORTED_WINNER)


# -- the report title -------------------------------------------------------


@pytest.fixture
def one_result() -> Results:
    return Results(runs=[RunResult(config=Config(), metrics={"recall@5": 0.5}, scored_queries=10)])


def test_the_report_is_titled_after_the_experiment(one_result: Results) -> None:
    """A directory of experiments was a directory of identically titled reports."""
    report = results_to_markdown(one_result, name="support-tickets")
    assert report.startswith("# support-tickets — retrieval configuration comparison")


def test_the_report_takes_the_name_off_the_results_when_not_given(one_result: Results) -> None:
    one_result.meta["name"] = "api-docs"
    assert results_to_markdown(one_result).startswith("# api-docs — ")


def test_the_report_falls_back_to_the_generic_title(one_result: Results) -> None:
    assert results_to_markdown(one_result).startswith("# Retrieval configuration comparison")

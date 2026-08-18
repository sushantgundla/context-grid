"""Two things a bundle and a validation run said that were not so.

Both were found by installing 0.9.3 from PyPI and driving it from the documentation alone.

**The bundle that forgot its seed.** `contextgrid sweep --bundle` wrote `"seeds": {}` while
`contextgrid run` wrote `"seeds": {"run": 0}`, so `contextgrid diff` between the two reported
`seeds.run: 0 -> None` -- a change that never happened, from the one command whose whole job
is saying what changed. The sweep did use a seed. It just did not write it down.

The same command's output carried a second line of the same kind, `config.retrieval: 'simple'
-> None`, and that one is not a recording failure: `grid/matrix.py` says outright that the two
"do run the same search -- `get_retriever(None)` returns `SimpleRetrieval()`", and
`Config.label` already renders them as one configuration. Two spellings of one run are not a
difference, and a diff that calls them one sends somebody hunting for a regression between two
identical pipelines.

**The validation that failed and passed.** `contextgrid validate bench.json ./docs
--recall-at-10 0.90` printed "outside the 0.05 tolerance" and "anything left over is a problem
with our scoring", then exited 0. In CI that is a green build for a scorer that missed the
published number -- exactly the trap `/reference/cli` argues against for `run`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contextgrid.cli import main
from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
from contextgrid.corpus import Corpus
from contextgrid.evalset import write_jsonl
from contextgrid.pipeline import Config
from contextgrid.report import Manifest, build_manifest, diff, write_bundle
from tests.support import API_DOCS, CONTRACT

QUESTIONS = [
    ("q1", "How much notice to terminate for convenience?", "contract.md", "thirty days"),
    ("q2", "Which header carries the API key?", "api.md", "X-Api-Key"),
]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A corpus and an eval set on disk, the way the CLI is always handed them."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "contract.md").write_text(CONTRACT, encoding="utf-8")
    (docs / "api.md").write_text(API_DOCS, encoding="utf-8")
    write_jsonl(
        EvalSet(
            id="es",
            items=tuple(
                EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
                for i, q, s, t in QUESTIONS
            ),
        ),
        tmp_path / "evalset.jsonl",
    )
    return tmp_path


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_texts({"contract.md": CONTRACT, "api.md": API_DOCS})


@pytest.fixture
def evalset() -> EvalSet:
    return EvalSet(
        id="es",
        items=tuple(
            EvalItem(id=i, question=q, anchors=(GoldAnchor(source_id=s, quote=t),))
            for i, q, s, t in QUESTIONS
        ),
    )


def sweep_bundle(workspace: Path, *, into: str = "bundle") -> Path:
    """Run `contextgrid sweep --bundle` and hand back the directory it wrote."""
    target = workspace / into
    assert (
        main(
            [
                "sweep",
                str(workspace / "docs"),
                str(workspace / "evalset.jsonl"),
                "--bundle",
                str(target),
            ]
        )
        == 0
    )
    return target


def manifest_json(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the seed the bundle ran with
# ---------------------------------------------------------------------------


def test_a_sweep_bundle_records_the_seed_it_ran_with(workspace: Path) -> None:
    """`"seeds": {}` for a sweep that did have a seed. It ran with `Lab`'s default of 0."""
    assert manifest_json(sweep_bundle(workspace))["seeds"] == {"run": 0}


def test_a_sweep_bundle_and_a_run_bundle_do_not_differ_on_the_seed(workspace: Path) -> None:
    """The reported symptom: `contextgrid diff` naming a seed change between two runs that
    used the same seed, because only one of them wrote it down."""
    (workspace / "experiment.yaml").write_text(
        "corpus: ./docs\nevalset: ./evalset.jsonl\nreport:\n  out: ./results\n", encoding="utf-8"
    )
    assert main(["run", str(workspace / "experiment.yaml")]) == 0

    changes = diff(
        Manifest.load(workspace / "results" / "manifest.json"),
        Manifest.load(sweep_bundle(workspace) / "manifest.json"),
    )
    assert "seeds.run" not in changes, changes


def test_write_bundle_records_the_seed_when_it_builds_its_own_manifest(
    workspace: Path, tmp_path: Path
) -> None:
    """The second route to a manifest. `write_bundle` builds one from the paths when it is
    not handed one, and its docstring promises that manifest is the same "to the hash" as the
    one `contextgrid sweep --bundle` builds -- so it has to record the seed too."""
    import contextgrid as cg

    lab = cg.Lab(workspace / "docs", seed=7)
    lab.grid(chunker=["recursive:512", "sentence:2"])
    results = lab.run(cg.read_jsonl(workspace / "evalset.jsonl"))

    write_bundle(
        results,
        tmp_path / "written",
        corpus=workspace / "docs",
        evalset=workspace / "evalset.jsonl",
    )
    assert manifest_json(tmp_path / "written")["seeds"] == {"run": 7}


def test_a_real_seed_change_is_still_reported(corpus: Corpus, evalset: EvalSet) -> None:
    """The guard on the fix above. Recording the seed is only worth anything if a genuine
    change in it still reaches the diff."""
    changes = diff(
        build_manifest(Config(), corpus, evalset, seeds={"run": 0}),
        build_manifest(Config(), corpus, evalset, seeds={"run": 7}),
    )
    assert changes["seeds.run"] == (0, 7)


# ---------------------------------------------------------------------------
# two spellings of one configuration
# ---------------------------------------------------------------------------

#: Axis values that name the same run as leaving the axis out. Every one of them is folded by
#: `grid.matrix._fold`, which is where the claim that they are identical is made and tested;
#: `retrieval` is the exception it declines to fold, and it says why in the same breath as
#: saying the two run the same search.
SAME_RUN = [
    ("retrieval", "simple"),
    ("ingestion", "plain"),
    ("transform", "none"),
    ("reranker", "none"),
]


@pytest.mark.parametrize(("axis", "alias"), SAME_RUN)
def test_naming_the_default_out_loud_is_not_a_change(
    corpus: Corpus, evalset: EvalSet, axis: str, alias: str
) -> None:
    """`retrieval: simple` and no retrieval at all are one configuration under two names.
    A diff that calls that a change sends somebody looking for a regression between two
    identical pipelines."""
    changes = diff(
        build_manifest(Config(**{axis: alias}), corpus, evalset),
        build_manifest(Config(), corpus, evalset),
    )
    assert f"config.{axis}" not in changes, changes


@pytest.mark.parametrize(("axis", "alias"), SAME_RUN)
def test_two_spellings_of_one_run_hash_the_same(
    corpus: Corpus, evalset: EvalSet, axis: str, alias: str
) -> None:
    """ "Two runs with the same manifest hash must produce identical numbers" is the promise
    the module opens with. Read the other way, two runs that must produce identical numbers
    should not be told they are different -- and `matches()` and `diff` should agree, or the
    next reader gets one answer from the CLI and the other from the API."""
    named = build_manifest(Config(**{axis: alias}), corpus, evalset)
    silent = build_manifest(Config(), corpus, evalset)
    assert named.matches(silent)


def test_a_real_retrieval_change_is_still_a_change(corpus: Corpus, evalset: EvalSet) -> None:
    """The guard. `widened` is a different search from plain, so it stays a difference."""
    changes = diff(
        build_manifest(Config(retrieval="simple"), corpus, evalset),
        build_manifest(Config(retrieval="widened"), corpus, evalset),
    )
    assert changes["config.retrieval"] == ("simple", "widened")


def test_the_manifest_still_records_what_the_user_actually_wrote(
    corpus: Corpus, evalset: EvalSet
) -> None:
    """Folding is for comparing, not for recording. `manifest.json` is the record of the run,
    and rewriting `simple` to `null` in it would make the file disagree with the
    `winning-config.yaml` written beside it."""
    manifest = build_manifest(Config(retrieval="simple"), corpus, evalset)
    assert manifest.config["retrieval"] == "simple"
    assert manifest.to_dict()["config"]["retrieval"] == "simple"


# ---------------------------------------------------------------------------
# validate, and what a failed comparison is worth in CI
# ---------------------------------------------------------------------------

BENCH_TEXT = "Fees are payable within thirty days of the invoice date.\n"


@pytest.fixture
def benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """A LegalBench-RAG file and the documents its spans point into."""
    documents = tmp_path / "corpus"
    documents.mkdir()
    (documents / "contract.txt").write_text(BENCH_TEXT, encoding="utf-8")

    start = BENCH_TEXT.index("thirty days")
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "query": "When are fees payable?",
                        "snippets": [
                            {
                                "file_path": "contract.txt",
                                "span": [start, start + len("thirty days")],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path, documents


def test_validate_fails_when_it_misses_the_published_number(
    benchmark: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """It printed "outside the 0.05 tolerance" and exited 0, which in CI is a green build for
    a scorer that did not reproduce the benchmark."""
    bench, documents = benchmark
    exit_code = main(["validate", str(bench), str(documents), "--recall-at-10", "0.10"])

    assert "outside the 0.05 tolerance" in capsys.readouterr().out
    assert exit_code == 1


def test_validate_passes_when_it_reproduces_the_published_number(
    benchmark: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    bench, documents = benchmark
    exit_code = main(["validate", str(bench), str(documents), "--recall-at-10", "1.0"])

    assert "reproduces a benchmark it did not define" in capsys.readouterr().out
    assert exit_code == 0


def test_validate_passes_when_there_is_no_published_number_to_miss(
    benchmark: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing was claimed, so nothing was missed. `ValidationResult.agrees` is `False` for an
    empty reference, and reading it as a verdict would fail every run that just reports its
    own numbers."""
    bench, documents = benchmark
    exit_code = main(["validate", str(bench), str(documents)])

    assert "No published numbers were supplied" in capsys.readouterr().out
    assert exit_code == 0

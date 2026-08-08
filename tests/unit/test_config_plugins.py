"""Reaching your own plugins from a config file.

Found by handing the docs to an agent with no access to the source. It wrote a metric,
registered it, named it in a YAML config, ran the CLI, and got:

    error: unknown metric 'sharp_mrr' in run.headline.
    Available: hit_rate, map, mrr, ndcg, precision, recall

Everything it did was what the docs describe. `contextgrid run` simply starts a fresh process
that imports `contextgrid` and nothing else, so the `register` call in its own file never ran.
The plugin system worked from Python and was unreachable from the command line -- and the
config reference is exactly where the promise is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextgrid.config import ConfigError, loads
from contextgrid.score import METRICS, available_metrics

METRIC_SOURCE = '''
"""A user's own metric, in their own file."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from contextgrid.score import METRICS


@dataclass(frozen=True, slots=True)
class {class_name}:
    name: ClassVar[str] = "{metric}"
    version: ClassVar[str] = "1"

    def evaluate(self, judgements: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
        return 1.0 if any(judgements.get(c, 0) > 0 for c in ranked[:k]) else 0.0


if "{metric}" not in METRICS:
    METRICS.register("{metric}", doc="test-only")({class_name})
'''


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A directory laid out the way somebody's own project would be."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.md").write_text("# Refunds\n\nRefunds take 30 days.\n")
    return tmp_path


def _config(*, plugins: str, headline: str = "recall@5") -> str:
    return f"""
corpus: ./corpus
plugins:
{plugins}
run:
  headline: {headline}
"""


def _write_metric(workspace: Path, name: str, metric: str) -> Path:
    path = workspace / name
    path.write_text(METRIC_SOURCE.format(metric=metric, class_name="Custom"))
    return path


@pytest.fixture(autouse=True)
def _clean_registry() -> object:
    """Metrics registered by a test must not leak into the next one.

    `available_metrics()` is global, and a test that registers `plug_headline` would otherwise
    make a later test asserting it is *unknown* pass for the wrong reason.
    """
    before = set(METRICS.names())
    yield
    for name in set(METRICS.names()) - before:
        METRICS.unregister(name)


# ---------------------------------------------------------------------------
# the thing that was broken
# ---------------------------------------------------------------------------


def test_a_metric_from_a_plugin_file_can_be_the_headline(workspace: Path) -> None:
    """The exact failure, as a test: this config was rejected as a typo."""
    _write_metric(workspace, "my_metrics.py", "plug_headline")

    config = loads(
        _config(plugins="  - ./my_metrics.py", headline="plug_headline@5"), base=workspace
    )

    assert config.run.headline == "plug_headline@5"
    assert "plug_headline" in available_metrics()


def test_the_plugin_loads_before_the_headline_is_validated(workspace: Path) -> None:
    """Ordering is the whole feature. Parsing `run` is what checks `headline` against the
    registry, so a plugin loaded afterwards would be rejected by the validation it exists to
    satisfy -- and the failure would look exactly like a typo."""
    _write_metric(workspace, "late.py", "plug_ordering")
    assert "plug_ordering" not in available_metrics()

    loads(_config(plugins="  - ./late.py", headline="plug_ordering@5"), base=workspace)

    assert "plug_ordering" in available_metrics()


def test_a_path_is_resolved_against_the_config_not_the_working_directory(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that only works from one directory is not portable, and running one from
    somewhere else is the first thing anybody does."""
    _write_metric(workspace, "beside.py", "plug_relative")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loads(_config(plugins="  - ./beside.py", headline="plug_relative@5"), base=workspace)

    assert "plug_relative" in available_metrics()


def test_a_dotted_module_name_works_too(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`my_package.metrics`, not just `./file.py`."""
    _write_metric(workspace, "importable_plugin.py", "plug_dotted")
    monkeypatch.syspath_prepend(str(workspace))

    loads(_config(plugins="  - importable_plugin", headline="plug_dotted@5"), base=workspace)

    assert "plug_dotted" in available_metrics()


def test_one_plugin_does_not_need_a_list(workspace: Path) -> None:
    """`plugins: ./one.py` rather than a one-item list, the same shorthand every other key
    in this schema allows."""
    _write_metric(workspace, "one.py", "plug_single")

    config = loads(
        "corpus: ./corpus\nplugins: ./one.py\nrun:\n  headline: plug_single@5\n", base=workspace
    )

    assert config.plugins == (str(workspace / "one.py"),)


# ---------------------------------------------------------------------------
# when it goes wrong
# ---------------------------------------------------------------------------


def test_a_missing_file_says_which_file(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="no such file"):
        loads(_config(plugins="  - ./nope.py"), base=workspace)


def test_an_unimportable_module_suggests_the_file_form(workspace: Path) -> None:
    """The likeliest mistake is naming a file that is not on `sys.path`, so the error says how
    to load a file instead rather than only reporting the import failure."""
    with pytest.raises(ConfigError, match=r"my_plugins\.py"):
        loads(_config(plugins="  - definitely_not_installed"), base=workspace)


def test_a_plugin_that_raises_is_not_reported_as_missing(workspace: Path) -> None:
    """A module that was found and blew up is a different problem from one that was not found,
    and saying "could not import" would send somebody looking for a file that is right there."""
    (workspace / "broken.py").write_text('raise ValueError("boom")\n')

    with pytest.raises(ConfigError, match=r"raised ValueError.*boom"):
        loads(_config(plugins="  - ./broken.py"), base=workspace)


def test_a_directory_is_refused_with_advice(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="is a directory"):
        loads(_config(plugins="  - ./corpus"), base=workspace)


# ---------------------------------------------------------------------------
# the surrounding contract
# ---------------------------------------------------------------------------


def test_loading_the_same_config_twice_runs_the_file_once(workspace: Path) -> None:
    """Keyed by resolved path, so re-running a config in one process is a no-op rather than a
    second execution of everything in the file -- which for a registry means a duplicate-name
    error on something the user did nothing wrong to cause."""
    path = workspace / "counted.py"
    path.write_text(
        METRIC_SOURCE.format(metric="plug_once", class_name="Custom")
        + '\nimport pathlib\np = pathlib.Path(__file__).with_suffix(".count")\n'
        "p.write_text(str(int(p.read_text()) + 1 if p.exists() else 1))\n"
    )

    source = _config(plugins="  - ./counted.py", headline="plug_once@5")
    loads(source, base=workspace)
    loads(source, base=workspace)

    assert (workspace / "counted.count").read_text() == "1"


def test_what_was_loaded_is_recorded_for_the_manifest(workspace: Path) -> None:
    """A result produced with a custom metric cannot be reproduced, or even read, without
    knowing which code defined it."""
    _write_metric(workspace, "recorded.py", "plug_manifest")

    config = loads(_config(plugins="  - ./recorded.py", headline="plug_manifest@5"), base=workspace)

    assert config.plugins == (str(workspace / "recorded.py"),)
    assert config.as_dict()["plugins"] == [str(workspace / "recorded.py")]


def test_no_plugins_key_changes_nothing(workspace: Path) -> None:
    config = loads("corpus: ./corpus\n", base=workspace)
    assert config.plugins == ()


def test_the_module_name_is_stable_across_processes(workspace: Path) -> None:
    """It used to be `abs(hash(str(path)))`, and `PYTHONHASHSEED` randomises `hash()` per
    process. That works -- a module name only has to be unique within one process -- but the
    name shows up in tracebacks raised by the user's own plugin, so the same crash looked
    different on every run. Nothing caught it, because every test ran in one process.
    """
    import subprocess
    import sys as _sys

    path = _write_metric(workspace, "named.py", "plug_stable")
    program = (
        "from pathlib import Path;"
        "from contextgrid.config.plugins import _module_name_for;"
        f"print(_module_name_for(Path({str(path)!r})))"
    )
    runs = {
        subprocess.run(
            [_sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }

    assert len(runs) == 1, f"module name moved between processes: {runs}"
    # And it says which file it came from, so a traceback names something recognisable.
    assert "named" in runs.pop()


def test_two_plugin_files_with_the_same_name_do_not_collide(tmp_path: Path) -> None:
    """Both called `plugins.py`, in different directories. Keyed by full path, so the second
    is loaded rather than silently treated as already-imported."""
    from contextgrid.config.plugins import _module_name_for

    first = tmp_path / "a" / "plugins.py"
    second = tmp_path / "b" / "plugins.py"

    assert _module_name_for(first) != _module_name_for(second)

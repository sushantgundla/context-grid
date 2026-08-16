"""What `contextgrid plugins` lists, and whether the sentences it prints are true.

`cli.md:21` calls the command "List everything registered" and `plugins.md:5` calls it the
authoritative live list. It listed six of the twelve registries. The six it skipped were not
obscure ones: four of them -- `ingestion`, `transform`, `retrieval`, `generator` -- are
sweepable axes with their own tables in `plugins.md`, so `--family transform` answered
"unknown plugin family 'transform'" about an axis the config file accepts and the starter
template writes a comment about.

The other half of the promise is that the sentence next to each name is true. `usearch`
advertised `b1` storage, which is the one dtype it deliberately does not have.
"""

from __future__ import annotations

from typing import Any

import pytest

from contextgrid.cli import main
from contextgrid.cli.__main__ import _HEADINGS
from contextgrid.core.registry import Registry

#: Every registry in the package, by the name `--family` takes. Written out here rather than
#: imported from the command, so the test fails when a registry is added and the command is
#: not told about it -- which is exactly how six of these came to be missing.
ALL_FAMILIES = {
    "parser": "contextgrid.parse:PARSERS",
    "ingestion": "contextgrid.ingest:INGESTERS",
    "chunker": "contextgrid.chunk:CHUNKERS",
    "embedder": "contextgrid.embed:EMBEDDERS",
    "index": "contextgrid.index:INDEXES",
    "transform": "contextgrid.transform:TRANSFORMS",
    "retrieval": "contextgrid.retrieve:RETRIEVERS",
    "reranker": "contextgrid.rerank:RERANKERS",
    "generator": "contextgrid.generate:GENERATORS",
    "llm": "contextgrid.evalset.llm:LLMS",
    "metric": "contextgrid.score.base:METRICS",
    "tokenizer": "contextgrid.tokens:TOKENIZERS",
}


def registry_for(path: str) -> Registry[Any]:
    from importlib import import_module

    module, _, attribute = path.partition(":")
    registry: Registry[Any] = getattr(import_module(module), attribute)
    return registry


# ---------------------------------------------------------------------------
# every registry
# ---------------------------------------------------------------------------


def test_the_package_has_exactly_the_registries_this_test_knows_about() -> None:
    """The guard on the guard. A thirteenth registry added later should fail here, loudly,
    rather than quietly not being listed the way six of these were not."""
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "src" / "contextgrid"
    found = {
        match.group(1)
        for path in source.rglob("*.py")
        for match in re.finditer(r'Registry\(family="([^"]+)"\)', path.read_text(encoding="utf-8"))
    }
    assert found == set(ALL_FAMILIES), found.symmetric_difference(set(ALL_FAMILIES))


@pytest.mark.parametrize("family", sorted(ALL_FAMILIES))
def test_every_family_can_be_asked_for_by_name(
    family: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`contextgrid plugins --family transform` used to be an error about an axis that exists."""
    assert main(["plugins", "--family", family]) == 0
    assert capsys.readouterr().out.strip(), f"--family {family} printed nothing"


@pytest.mark.parametrize("family", sorted(ALL_FAMILIES))
def test_every_registered_plugin_appears_in_the_full_listing(
    family: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "List everything registered" -- so every name in every registry, with its description."""
    assert main(["plugins"]) == 0
    printed = capsys.readouterr().out

    for name, description in registry_for(ALL_FAMILIES[family]).describe().items():
        assert name in printed, f"{family} {name} is registered and not listed"
        assert description in printed, f"{family} {name} is listed without its description"


def test_every_family_prints_under_a_heading() -> None:
    """A heading nobody can search for makes a reference feel untrustworthy, which is why
    `index` is not printed as "indexs". The same care for the six that were missing."""
    assert set(_HEADINGS) == set(ALL_FAMILIES)
    assert _HEADINGS["index"] == "indexes"
    assert "ingestions" not in _HEADINGS.values()
    assert "retrievals" not in _HEADINGS.values()


def test_an_unknown_family_is_still_an_error_that_lists_the_real_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Printing nothing and exiting 0 reads as "no such plugins here", not "you typed a name
    this flag does not know"."""
    assert main(["plugins", "--family", "chunkers"]) == 1
    error = capsys.readouterr().err
    assert "unknown plugin family 'chunkers'" in error
    for family in ALL_FAMILIES:
        assert family in error


# ---------------------------------------------------------------------------
# the plugins that exist and cannot be registered
# ---------------------------------------------------------------------------


def test_the_model_backed_transforms_are_listed(capsys: pytest.CaptureFixture[str]) -> None:
    """`plugins.md:196`: `available_transforms()` "is what `contextgrid plugins` and the config
    template actually print". The template did; the command showed two arms of six."""
    from contextgrid.transform import available_transforms

    assert main(["plugins", "--family", "transform"]) == 0
    printed = capsys.readouterr().out
    for name in available_transforms():
        assert name in printed, name


def test_the_llm_generator_is_listed(capsys: pytest.CaptureFixture[str]) -> None:
    from contextgrid.generate import available_generators

    assert main(["plugins", "--family", "generator"]) == 0
    printed = capsys.readouterr().out
    for name in available_generators():
        assert name in printed, name


def test_a_plugin_that_needs_a_model_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    """They are absent from the registry because a transform built without a model is silently
    the identity. Listing them without that caveat would invite exactly that config."""
    assert main(["plugins", "--family", "transform"]) == 0
    printed = capsys.readouterr().out

    hyde = next(line for line in printed.splitlines() if line.strip().startswith("hyde"))
    assert "*" in hyde
    assert "* needs a model. Set `run.model` in your config to use it." in printed

    none = next(line for line in printed.splitlines() if line.strip().startswith("none"))
    assert "*" not in none, "`none` needs no model and must not be starred"


def test_the_descriptions_are_the_real_ones_not_placeholders(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Taken from the classes' own docstrings. Inventing a sentence here would put a second,
    drifting description of HyDE in a file that knows nothing about HyDE."""
    from contextgrid.transform import HyDE

    assert main(["plugins", "--family", "transform"]) == 0
    assert (HyDE.__doc__ or "").strip().splitlines()[0] in capsys.readouterr().out


# ---------------------------------------------------------------------------
# usearch and the dtype it does not have
# ---------------------------------------------------------------------------


def test_usearch_rejects_b1_and_names_the_dtypes_it_does_have() -> None:
    """Confirmed before the description was touched: `b1` is still refused, so the docs are
    the right half and the string was the wrong one."""
    from contextgrid.index import get_index
    from contextgrid.index.dense import IndexBuildError

    with pytest.raises(IndexBuildError) as caught:
        get_index("usearch:dtype=b1")

    assert "unknown usearch dtype 'b1'" in str(caught.value)
    assert "Choose one of: f32, f16, i8" in str(caught.value)


def test_the_usearch_description_names_only_dtypes_it_accepts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one line `plugins` prints about this index pointed at the one setting that fails."""
    from contextgrid.index.ann import USearchIndex

    assert main(["plugins", "--family", "index"]) == 0
    line = next(
        row for row in capsys.readouterr().out.splitlines() if row.strip().startswith("usearch")
    )

    assert "b1" not in line, line
    for dtype in USearchIndex.DTYPES:
        assert dtype in line, f"{dtype} is a real usearch dtype and is not advertised"


def test_the_usearch_description_cannot_drift_from_dtypes_again() -> None:
    """Keyed off `DTYPES` rather than off the literal string, so adding or removing a dtype
    fails here until the sentence the command prints is updated to match."""
    import re

    from contextgrid.index import INDEXES
    from contextgrid.index.ann import USearchIndex

    doc = INDEXES.describe()["usearch"]
    advertised = set(re.findall(r"\b(?:f32|f16|i8|b1|f64|i4)\b", doc))
    assert advertised == set(USearchIndex.DTYPES), advertised

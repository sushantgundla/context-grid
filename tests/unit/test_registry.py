"""Unit tests for plugin registration and spec parsing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from contextgrid.core.errors import ContextGridError, MissingExtraError
from contextgrid.core.registry import Registry, UnknownPluginError, _coerce
from contextgrid.parse import PARSERS


@dataclass(frozen=True)
class Widget:
    size: int = 10
    label: str = "plain"
    fancy: bool = False


@pytest.fixture
def registry() -> Registry[Widget]:
    reg: Registry[Widget] = Registry(family="widget")
    reg.register("basic", shorthand="size", doc="A basic widget.")(Widget)
    return reg


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_registering_and_creating(registry: Registry[Widget]) -> None:
    assert registry.names() == ["basic"]
    assert "basic" in registry
    assert len(registry) == 1
    assert registry.create("basic") == Widget()


def test_registering_the_same_name_twice_is_an_error(registry: Registry[Widget]) -> None:
    with pytest.raises(ContextGridError, match="already registered"):
        registry.register("basic")(Widget)


def test_unknown_name_lists_what_is_available(registry: Registry[Widget]) -> None:
    with pytest.raises(UnknownPluginError) as caught:
        registry.create("nope")
    message = str(caught.value)
    assert "no widget named 'nope'" in message
    assert "basic" in message


def test_unknown_plugin_error_is_a_key_error(registry: Registry[Widget]) -> None:
    with pytest.raises(KeyError):
        registry.registration("nope")


def test_the_decorator_returns_the_original(registry: Registry[Widget]) -> None:
    """So a class stays usable directly, not only through the registry."""
    reg: Registry[Widget] = Registry(family="widget")
    returned = reg.register("basic")(Widget)
    assert returned is Widget


def test_doc_defaults_to_the_first_docstring_line() -> None:
    reg: Registry[Widget] = Registry(family="widget")

    @reg.register("documented")
    def _make() -> Widget:
        """The first line.

        And more detail nobody needs in a listing.
        """
        return Widget()

    assert reg.describe()["documented"] == "The first line."


# ---------------------------------------------------------------------------
# spec strings
# ---------------------------------------------------------------------------


def test_bare_name(registry: Registry[Widget]) -> None:
    assert registry.parse_spec("basic") == ("basic", {})


def test_shorthand_value(registry: Registry[Widget]) -> None:
    assert registry.parse_spec("basic:512") == ("basic", {"size": 512})


def test_shorthand_plus_keywords(registry: Registry[Widget]) -> None:
    name, params = registry.parse_spec("basic:512,label=wide,fancy=true")
    assert name == "basic"
    assert params == {"size": 512, "label": "wide", "fancy": True}


def test_keywords_alone(registry: Registry[Widget]) -> None:
    assert registry.parse_spec("basic:label=wide") == ("basic", {"label": "wide"})


def test_whitespace_is_tolerated(registry: Registry[Widget]) -> None:
    assert registry.parse_spec(" basic : 512 , label = wide ") == (
        "basic",
        {"size": 512, "label": "wide"},
    )


def test_a_second_bare_value_is_rejected_with_a_useful_message(
    registry: Registry[Widget],
) -> None:
    with pytest.raises(ContextGridError, match="key=value"):
        registry.parse_spec("basic:512,wide")


def test_a_bare_value_without_a_shorthand_is_rejected() -> None:
    reg: Registry[Widget] = Registry(family="widget")
    reg.register("noshort")(Widget)
    with pytest.raises(ContextGridError, match="key=value"):
        reg.parse_spec("noshort:512")


def test_an_empty_name_is_rejected(registry: Registry[Widget]) -> None:
    with pytest.raises(ContextGridError, match="no name"):
        registry.parse_spec(":512")


def test_create_builds_from_a_spec(registry: Registry[Widget]) -> None:
    assert registry.create("basic:512,fancy=yes") == Widget(size=512, fancy=True)


def test_keyword_arguments_beat_the_spec(registry: Registry[Widget]) -> None:
    assert registry.create("basic:512", size=64) == Widget(size=64)


# ---------------------------------------------------------------------------
# value coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("512", 512),
        ("-3", -3),
        ("0.5", 0.5),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("none", None),
        ("null", None),
        ("", None),
        ("wide", "wide"),
        ("'quoted'", "quoted"),
        ('"quoted"', "quoted"),
        ("\\n\\n", "\\n\\n"),
    ],
)
def test_coercion(raw: str, expected: object) -> None:
    """Spec strings come from YAML and command lines, where everything is text. Plugins
    should receive the type they obviously meant."""
    assert _coerce(raw) == expected


# ---------------------------------------------------------------------------
# lazy plugins
# ---------------------------------------------------------------------------


def test_lazy_registration_does_not_import_at_registration_time() -> None:
    reg: Registry[Widget] = Registry(family="widget")
    reg.register_lazy(
        "absent",
        module="contextgrid.definitely_not_a_module",
        attr="Thing",
        extra="parse",
        package="somepackage",
    )
    assert "absent" in reg  # registering it did not try to import anything


def test_a_missing_extra_names_the_install_command() -> None:
    reg: Registry[Widget] = Registry(family="widget")
    reg.register_lazy(
        "absent",
        module="contextgrid.definitely_not_a_module",
        attr="Thing",
        extra="parse-ml",
        package="docling",
    )
    with pytest.raises(MissingExtraError) as caught:
        reg.create("absent")
    message = str(caught.value)
    assert 'pip install "context-grid[parse-ml]"' in message
    assert "docling" in message


def test_a_lazily_registered_parser_loads_once_its_extra_is_installed() -> None:
    """Registration is by module path, so nothing heavy is imported until it is asked for."""
    assert "pymupdf" in PARSERS
    assert PARSERS.create("pymupdf").name == "pymupdf"


def test_a_parser_whose_module_does_not_exist_yet_reports_its_extra() -> None:
    """Docling lands in a later milestone. Until then, asking for it should say which extra
    to install rather than raising ModuleNotFoundError from four frames down."""
    assert "docling" in PARSERS
    with pytest.raises(MissingExtraError, match=r"context-grid\[parse-ml\]"):
        PARSERS.create("docling")


def test_registry_is_iterable_over_registrations(registry: Registry[Widget]) -> None:
    entries = list(registry)
    assert [entry.name for entry in entries] == ["basic"]
    assert entries[0].family == "widget"
    assert entries[0].shorthand == "size"

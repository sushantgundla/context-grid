"""Name-based plugin registration.

Two things this has to get right.

**Nothing heavy is imported until it is asked for.** `import contextgrid` must not pull in a
PDF engine or an ONNX runtime. Plugins that need an optional dependency register lazily, by
module path, and are imported on first use.

**A missing dependency produces an instruction, not a traceback.** Asking for the `docling`
parser without the extra installed should say `pip install "context-grid[parse-ml]"`, not
raise `ModuleNotFoundError: No module named 'docling'` from four frames down.
"""

from __future__ import annotations

import importlib
import inspect
import types
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from contextgrid.core.errors import ContextGridError, MissingExtraError

T = TypeVar("T")

Factory = Callable[..., Any]


class SpecValueError(ContextGridError, ValueError):
    """A spec string gave a parameter a value of the wrong kind.

    Separate from `ContextGridError` so a caller reporting many problems at once can tell that
    this message already names its axis and its spec, and does not want them prefixed on again.
    """


class UnknownPluginError(ContextGridError, KeyError):
    """A plugin was asked for by a name nothing is registered under."""

    def __init__(self, family: str, name: str, known: list[str]) -> None:
        self.family = family
        self.name = name
        self.known = known
        options = ", ".join(sorted(known)) if known else "none registered"
        # KeyError's __str__ quotes its argument, so build the whole message as one string.
        super().__init__(f"no {family} named {name!r}. Available: {options}")

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


@dataclass(frozen=True, slots=True)
class Registration:
    """One registered plugin, and how to build it."""

    name: str
    family: str
    #: Set for in-tree plugins available immediately.
    factory: Factory | None = None
    #: Set for plugins imported on first use.
    module: str | None = None
    attr: str | None = None
    #: The optional extra that must be installed, e.g. "parse-ml".
    extra: str | None = None
    #: The third-party package the extra provides, for the error message.
    package: str | None = None
    #: Keyword this plugin accepts in shorthand form, so "recursive:512" means size=512.
    shorthand: str | None = None
    doc: str = ""

    def load(self) -> Factory:
        """Return the callable that builds this plugin, importing it if needed."""
        if self.factory is not None:
            return self.factory
        if self.module is None or self.attr is None:  # pragma: no cover - guarded at register
            raise ContextGridError(f"registration for {self.name!r} has no factory and no module")
        try:
            module = importlib.import_module(self.module)
        except ImportError as exc:
            if self.extra is None:
                raise
            raise MissingExtraError(
                f"The {self.name!r} {self.family}", self.extra, package=self.package
            ) from exc
        return getattr(module, self.attr)  # type: ignore[no-any-return]


@dataclass(slots=True)
class Registry(Generic[T]):
    """A named collection of plugins of one family."""

    family: str
    _entries: dict[str, Registration] = field(default_factory=dict)

    # -- registration --------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        shorthand: str | None = None,
        doc: str = "",
    ) -> Callable[[Factory], Factory]:
        """Decorator registering an in-tree plugin with no optional dependencies."""

        def decorate(factory: Factory) -> Factory:
            self._add(
                Registration(
                    name=name,
                    family=self.family,
                    factory=factory,
                    shorthand=shorthand,
                    doc=doc or (factory.__doc__ or "").strip().split("\n")[0],
                )
            )
            return factory

        return decorate

    def register_lazy(
        self,
        name: str,
        *,
        module: str,
        attr: str,
        extra: str,
        package: str | None = None,
        shorthand: str | None = None,
        doc: str = "",
    ) -> None:
        """Register a plugin that needs an optional dependency, imported on first use."""
        self._add(
            Registration(
                name=name,
                family=self.family,
                module=module,
                attr=attr,
                extra=extra,
                package=package,
                shorthand=shorthand,
                doc=doc,
            )
        )

    def _add(self, registration: Registration) -> None:
        if registration.name in self._entries:
            raise ContextGridError(
                f"a {self.family} named {registration.name!r} is already registered"
            )
        self._entries[registration.name] = registration

    def unregister(self, name: str) -> None:
        """Remove a registration, if there is one. A no-op otherwise.

        For tests that register a throwaway plugin into a real, shared registry (rather than
        a local one) and want it gone afterwards -- a plugin left registered leaks into every
        other test that lists or validates against this registry for the rest of the process,
        which is a real failure mode for a plugin that's registered only to prove it *fails*.
        """
        self._entries.pop(name, None)

    # -- lookup --------------------------------------------------------------

    def registration(self, name: str) -> Registration:
        try:
            return self._entries[name]
        except KeyError:
            raise UnknownPluginError(self.family, name, list(self._entries)) from None

    def create(self, spec: str, **overrides: Any) -> T:
        """Build a plugin from a spec string.

        A spec is a name, optionally followed by parameters:

            "recursive"                 -> defaults
            "recursive:512"             -> the plugin's shorthand parameter
            "recursive:512,overlap=64"  -> shorthand plus keywords
            "recursive:overlap=64"      -> keywords alone

        Keyword arguments passed here win over anything in the spec.
        """
        name, params = self.parse_spec(spec)
        params.update(overrides)
        return self.registration(name).load()(**params)  # type: ignore[no-any-return]

    def parse_spec(self, spec: str) -> tuple[str, dict[str, Any]]:
        """Split a spec string into a plugin name and its parameters."""
        name, tail = self._split_name(spec)
        if not name:
            raise ContextGridError(f"{spec!r} is not a valid {self.family} spec: no name")

        registration = self.registration(name)
        params: dict[str, Any] = {}
        for index, raw in enumerate(part.strip() for part in tail.split(",") if part.strip()):
            key, sep, value = raw.partition("=")
            if sep:
                params[key.strip()] = _coerce(value.strip())
            elif index == 0 and registration.shorthand:
                params[registration.shorthand] = _coerce(key.strip())
            else:
                raise ContextGridError(
                    f"{raw!r} in {spec!r} must be written as key=value"
                    + (
                        ""
                        if registration.shorthand is None
                        else f" (only the first value may be bare, meaning "
                        f"{registration.shorthand}={raw!r})"
                    )
                    + self._list_hint(raw)
                )

        self._check_param_types(spec, registration, params)
        return name, params

    def _list_hint(self, raw: str) -> str:
        """An extra clause for the case where the comma was meant to separate two plugins.

        `--chunker recursive:128,recursive:256` is the obvious guess and it parses as one spec
        with a stray second value, so the honest `must be written as key=value` answers a
        question the user was not asking. A comma-separated part that names a plugin of this
        family is not a mistyped parameter, it is a list -- and the fix is a different shape
        entirely, so say which one.
        """
        if self.name_in(raw) not in self._entries:
            return ""
        return (
            f". To sweep several {self.family}s, give one value per entry -- a YAML list in a "
            "config file, or the command-line flag repeated once per value"
        )

    def _check_param_types(
        self, spec: str, registration: Registration, params: dict[str, Any]
    ) -> None:
        """Refuse a parameter whose value is not the kind of thing the plugin takes.

        A spec string carries no types, so `_coerce` guesses from the text: `512` becomes an
        int and `banana` stays a str, because that is all it can know. The str then travelled
        all the way into the chunker and met `self.size // 8`, and what reached the user was
        `unsupported operand type(s) for //: 'str' and 'int'` -- a sentence about the inside of
        a class, from a tool whose premise is that nobody has to read its source.

        Every other way of getting a spec wrong already reads well. A bad name lists the real
        ones, a bad sign says `chunk size must be positive, got -5`. Only a bad *type* fell
        through, and it fell through in seven places, so the check belongs here rather than in
        seven `__post_init__` methods -- the eighth would be written without it.

        Two deliberate limits. Only in-tree plugins are checked, because reading a lazy
        plugin's annotations means importing it, and `parse_spec` is supposed to work on a
        machine where the optional package is not installed. And only parameters annotated
        purely with `int`, `float`, `str`, `bool` or `None` are judged; anything richer -- a
        tokenizer, an embedder, a tuple of separators -- is left to the plugin, which knows
        what it will accept and this does not.
        """
        if registration.factory is None or not params:
            return
        hints = _parameter_types(registration.factory)
        for key, value in params.items():
            allowed = _simple_types_in(hints.get(key))
            if allowed is None or _acceptable(value, allowed):
                continue
            raise SpecValueError(
                f"{self.family} {spec!r}: {key} must be {_wanted(allowed)}, got {value!r}"
            )

    def _split_name(self, spec: str) -> tuple[str, str]:
        """Separate the plugin name from its parameters.

        Splitting on the first colon is not enough, because plugin names are namespaced by the
        library they come from: `chonkie:recursive:512` is the plugin `chonkie:recursive` with
        `size=512`, not a plugin called `chonkie`. So take the *longest* registered name the
        spec starts with.

        Unregistered names fall back to the first segment, which keeps the "no chunker named
        'foo'" error pointing at the part the user actually got wrong.
        """
        candidate = spec.strip()
        if candidate in self._entries:
            return candidate, ""

        # Slice after the colon that ended the match, not after the name's own length -- the
        # two differ whenever there is whitespace around the colon.
        head = ""
        tail_from = 0
        for position, character in enumerate(candidate):
            if character != ":":
                continue
            prefix = candidate[:position].strip()
            if prefix in self._entries:
                head, tail_from = prefix, position + 1
        if head:
            return head, candidate[tail_from:]

        name, _, tail = candidate.partition(":")
        return name.strip(), tail

    def name_in(self, spec: str) -> str:
        """The plugin name a spec string asks for, without checking anything is registered.

        For validating a name on its own -- which is what a config file can be checked for on a
        machine that cannot run the sweep. `create` and `parse_spec` both go further than that:
        they read the parameters too, and a parameter is only meaningful to the plugin that
        takes it, which may need an extra that is not installed.
        """
        return self._split_name(spec)[0]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def describe(self) -> Mapping[str, str]:
        """Plugin name to one-line description, for `--help` and error messages."""
        return {name: entry.doc for name, entry in sorted(self._entries.items())}

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Registration]:
        return iter(self._entries.values())


#: The types a spec string can express, and therefore the only ones worth judging.
_SIMPLE: frozenset[type] = frozenset({int, float, str, bool, type(None)})

#: Resolved annotations per factory. `get_type_hints` re-evaluates every annotation in the
#: module's namespace, and `from __future__ import annotations` means it has to; doing that on
#: every spec in a sweep would be a measurable cost for an answer that cannot change.
_HINTS: dict[Any, Mapping[str, Any]] = {}


def _parameter_types(factory: Factory) -> Mapping[str, Any]:
    """The factory's parameter annotations, resolved to real types, cached, never raising.

    A plugin whose annotations cannot be resolved -- a forward reference to something only
    imported under `TYPE_CHECKING`, most likely -- gets no checking rather than a crash. Failing
    to validate is a much smaller problem than refusing to build.
    """
    try:
        return _HINTS[factory]
    except KeyError:
        pass
    except TypeError:  # pragma: no cover - an unhashable factory, which nothing here is
        return {}

    try:
        hints = dict(typing.get_type_hints(factory))
        wanted = set(inspect.signature(factory).parameters)
        resolved: Mapping[str, Any] = {k: v for k, v in hints.items() if k in wanted}
    except Exception:
        # Any resolution failure at all means "cannot judge this one", which is the safe
        # answer: not validating a parameter costs less than refusing to build a good plugin.
        resolved = {}

    _HINTS[factory] = resolved
    return resolved


def _simple_types_in(annotation: Any) -> frozenset[type] | None:
    """The set of plain types an annotation admits, or None when it is not that simple.

    `int` and `int | None` are answerable. `str | Tokenizer | None` and `tuple[str, ...]` are
    not, and returning None for them is what keeps this from second-guessing a plugin that
    takes a real object.
    """
    if annotation in _SIMPLE:
        return frozenset({annotation})
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = typing.get_args(annotation)
        if all(member in _SIMPLE for member in members):
            return frozenset(members)
    return None


def _acceptable(value: Any, allowed: frozenset[type]) -> bool:
    """Whether a coerced spec value fits. Numeric widening only, and only upwards.

    An `int` where a `float` is wanted is fine -- that is ordinary Python. A `float` where an
    `int` is wanted is not: `recursive:1.5` is a chunk size somebody meant to type differently,
    and silently truncating it would make the manifest disagree with the config file.
    """
    if value is None:
        return type(None) in allowed
    if isinstance(value, bool):
        return bool(allowed & {bool, int, float})
    if isinstance(value, int):
        return bool(allowed & {int, float, bool})
    if isinstance(value, float):
        return float in allowed
    if isinstance(value, str):
        return str in allowed
    return True  # pragma: no cover - `_coerce` produces nothing else


def _wanted(allowed: frozenset[type]) -> str:
    """What the parameter takes, in words, since the reader is not looking at the signature."""
    names = {int: "a whole number", float: "a number", str: "text", bool: "true or false"}
    kinds = [names[kind] for kind in (int, float, str, bool) if kind in allowed]
    if type(None) in allowed:
        kinds.append("none")
    if not kinds:  # pragma: no cover - an annotation of bare `None`, which nothing uses
        return "nothing"
    if len(kinds) == 1:
        return kinds[0]
    return " or ".join([", ".join(kinds[:-1]), kinds[-1]])


def _coerce(value: str) -> Any:
    """Turn a spec-string value into the type it obviously is.

    Spec strings come from YAML and command lines, where everything is text. `size=512`
    should reach the plugin as an int, not "512" -- otherwise every plugin has to defend
    itself against strings.
    """
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")

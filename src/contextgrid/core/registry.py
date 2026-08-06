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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from contextgrid.core.errors import ContextGridError, MissingExtraError

T = TypeVar("T")

Factory = Callable[..., Any]


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
                )
        return name, params

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

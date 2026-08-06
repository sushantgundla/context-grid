# The plugin registry

[`core/registry.py`](../../src/contextgrid/core/registry.py) is name-based plugin lookup,
shared by every plugin family (`CHUNKERS`, `PARSERS`, `RETRIEVERS`, `GENERATORS`, ...). Each
family creates its own `Registry[T]` instance — `CHUNKERS: Registry[Chunker] = Registry(family="chunker")`
in `chunk/__init__.py` is typical — and registers plugins into it at import time.

Two things it has to get right, stated directly in the module docstring:

1. **Nothing heavy is imported until it is asked for.** `import contextgrid` must not pull in a
   PDF engine or an ONNX runtime.
2. **A missing dependency produces an instruction, not a traceback.** Asking for `docling`
   without the extra installed should say `pip install "context-grid[parse-ml]"`, not
   `ModuleNotFoundError: No module named 'docling'` from four frames down.

Both are why `Registry` has two ways to register a plugin.

## `register` — eager

```python
CHUNKERS.register("fixed", shorthand="size", doc="Fixed-size token windows with overlap.")(
    FixedTokenChunker
)
```

A decorator. The factory (usually a class) is held directly on the `Registration` and built
immediately when asked for — used for in-tree plugins with no optional dependency, which is
every plugin `pip install context-grid` alone can already run.

## `register_lazy` — deferred

```python
CHUNKERS.register_lazy(
    "chonkie:recursive",
    module="contextgrid.chunk.chonkie",
    attr="ChonkieRecursiveChunker",
    extra="chunk",
    package="chonkie",
    shorthand="size",
    doc="Chonkie's recursive splitter. The head-to-head against ours.",
)
```

No factory is imported at registration time — only the name and *where to find it* are
recorded (`module`, `attr`). `Registration.load()` (`registry.py:62`) does the actual
`importlib.import_module(self.module)` the first time the plugin is asked for. If that import
fails, and `extra` is set, the `ImportError` is caught and re-raised as `MissingExtraError`
(`core/errors.py`), which names the exact `pip install "context-grid[chunk]"` command and, when
given, the real package name (`needs chonkie`) — because "chunk" the extra and "chonkie" the
package are not the same string, and a user who only sees the extra name still has to go
looking for what it installs.

This is why `import contextgrid` stays cheap regardless of how many third-party chunkers,
parsers and embedders get registered: none of their modules are touched until `.create()` or
`.load()` actually needs one.

## Spec strings: `create()` and `parse_spec()`

A spec string is the public, writable form of a configured plugin — see
[matrix.py's `matrix()` docstring](../../src/contextgrid/grid/matrix.py) for why this matters
more than it looks: **a `Config` axis takes a spec string, never a plugin instance**, because a
configuration has to be writable into a leaderboard row, a cache key, and a YAML file, and an
object is none of those. `matrix()` rejects a plugin instance early
(`_require_specs` in `grid/matrix.py`) specifically so the error appears at the point the
mistake was made, not as `TypeError: sequence item 2: expected str` from inside a report
formatter long after a sweep has run.

```
"recursive"                 -> RecursiveChunker()                    (defaults)
"recursive:512"             -> RecursiveChunker(size=512)            (shorthand)
"recursive:512,overlap=64"  -> RecursiveChunker(size=512, overlap=64) (shorthand + keywords)
"recursive:overlap=64"      -> RecursiveChunker(overlap=64)          (keywords alone)
```

`Registry.create(spec, **overrides)` calls `parse_spec(spec)` to get `(name, params)`, applies
`overrides` (which win over anything in the spec string — this is how the runner can pin a
value after parsing), then `self.registration(name).load()(**params)`.

`parse_spec` splits the spec into a name and a comma-separated tail. Each tail segment is
either `key=value`, or — only in the first position, and only if the registration declared a
`shorthand` — a bare value meaning `shorthand=value`. `_coerce()` turns that value into the type
it obviously is: `"true"/"yes"/"on"` → `True`, `"none"/"null"/""` → `None`, then tries `int`,
then `float`, else a stripped string. This exists because spec strings come from YAML and
command lines, where everything is text — without it, every plugin would have to defend its own
`__init__` against `size="512"` instead of `size=512`.

### Namespaced names: why `_split_name` isn't `spec.split(":", 1)`

`chonkie:recursive:512` must resolve to the plugin named `chonkie:recursive` with `size=512` —
**not** a plugin called `chonkie` with a first argument of `"recursive:512"`. Splitting on the
first colon breaks the moment a family has a namespaced name, which chunkers do (`chonkie:*`,
`langchain:*`).

`_split_name` (`registry.py:193`) handles this by taking the **longest registered name the spec
starts with**: it walks every colon in the string, and each time the prefix up to that colon is
a registered name, it remembers that as the current best match. So for
`"chonkie:recursive:512"` it tries `"chonkie"` (not registered — skip),
then `"chonkie:recursive"` (registered — record it), and stops there; the tail becomes
`"512"`. An unregistered name falls back to the text before the first colon, which is what
keeps `"no chunker named 'foo'"` pointing at the part the user actually got wrong rather than
the whole spec string.

## Errors

`UnknownPluginError(family, name, known)` — a `KeyError` subclass raised by
`registration(name)` when nothing is registered under that name. Lists every registered name
in the family so the error is also the discovery mechanism (`Available: fixed, langchain:character, ...`).

`ContextGridError` is raised by `_add()` if two plugins in the same family try to register
under the same name — a collision the registry refuses to resolve silently in either direction.

`MissingExtraError` — see above. It is an `ImportError` subclass, so code that already catches
`ImportError` around a plugin load still catches this.

## Read-only surface

`Registry.names()` (sorted list), `.describe()` (name → one-line doc, for `--help` and error
messages), `.__contains__`, `.__len__`, `.__iter__` (over `Registration`s, not instances — you
get the metadata, not a built plugin). None of these build anything, so listing what's
available never triggers a lazy import.

## `params_of`: how a plugin's config reaches the cache key

`pipeline.params_of(plugin)` (`pipeline.py:470`) is the other half of this system: given a
built plugin instance, it returns its dataclass fields (skipping private ones, i.e. anything
starting with `_`) as a dict, which is folded into `cache_key(...)` alongside the plugin's
`name` and `version`. This is why plugins in this codebase are dataclasses with public
parameters — `params_of` reads `dataclasses.fields()`, and a plugin that hides its
configuration in closures or non-dataclass attributes would cache-key identically regardless of
those settings, silently serving one configuration's cached output to another.

See [extending.md](extending.md) for registering a new plugin end to end, and
[protocols.md](protocols.md) for what each family's plugins must implement.

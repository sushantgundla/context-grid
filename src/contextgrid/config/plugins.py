"""Importing the user's own plugins before any name in the config is resolved.

Every axis in this package is a registry, and the docs have always said you can write your own
chunker, metric or embedder and name it in a config file. That was true from Python and false
from the command line, for a reason nobody would guess: `contextgrid run config.yaml` starts a
fresh process that imports `contextgrid` and nothing else. The `@METRICS.register` line in your
own module never ran, so the name it registers does not exist, and the config is rejected as a
typo -- `unknown metric 'sharp_mrr'. Available: hit_rate, map, mrr, ndcg, precision, recall`.

A `plugins:` list fixes it by naming the modules to import first:

```yaml
plugins:
  - my_metrics           # a module on sys.path
  - ./local_plugins.py   # or a file sitting beside the config

run:
  headline: sharp_mrr@5
```

**Order matters more than it looks.** These imports happen before `run` is parsed, because
parsing `run` is what validates `headline` against the registry. Loading plugins afterwards
would reject the very name the plugin file exists to provide.

**This executes code the config points at.** That is the entire feature -- registering a plugin
means running a `register` call -- but it does mean a config file is as trusted as a script.
Anything imported here can do whatever a Python module can do, so a `contextgrid.yaml` from
somewhere you would not run a script from should be read before it is run.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import zlib
from pathlib import Path
from typing import Any

from contextgrid.config.schema import ConfigError
from contextgrid.core.errors import MissingExtraError
from contextgrid.core.registry import Registration, Registry, UnknownPluginError


def load_plugins(specs: tuple[str, ...], *, base: Path) -> tuple[str, ...]:
    """Import each module so its `register` calls run. Returns what was loaded, in order.

    A spec is either a dotted module name on `sys.path` (`my_package.metrics`) or a path to a
    file (`./plugins.py`), resolved against the config file's own directory so the config stays
    portable between working directories.
    """
    loaded: list[str] = []
    for spec in specs:
        if _looks_like_a_path(spec):
            loaded.append(_load_file(spec, base=base))
        else:
            loaded.append(_load_module(spec))
    return tuple(loaded)


def _looks_like_a_path(spec: str) -> bool:
    """Tell `./plugins.py` from `my_package.metrics`.

    Both contain a dot, so the dot cannot be the test. A path is anything ending in `.py`, or
    anything with a separator or a leading `~` in it -- none of which can appear in a dotted
    module name.
    """
    return spec.endswith(".py") or "/" in spec or "\\" in spec or spec.startswith("~")


def _load_module(name: str) -> str:
    """Import a module by dotted name, the way an `import` statement would."""
    try:
        importlib.import_module(name)
    except ImportError as error:
        raise ConfigError(
            f"plugins: could not import {name!r} ({error}). It has to be importable from where "
            "you are running -- either installed, or on PYTHONPATH. To load a file instead, "
            "give a path: `./my_plugins.py`"
        ) from error
    except Exception as error:
        # The module was found and blew up while executing. That is the user's own code, and
        # saying "could not import" would send them looking for a missing file.
        raise ConfigError(
            f"plugins: {name!r} raised {type(error).__name__} while being imported: {error}"
        ) from error
    return name


def _module_name_for(path: Path) -> str:
    """The name a plugin file is imported under. Stable, readable, and keyed by full path.

    Keyed by the *full* path so two `plugins.py` files in different directories do not collide,
    and so re-running the same config in one process is a no-op rather than a second execution
    of everything in the file -- which for a registry means a duplicate-name error the user did
    nothing to cause.

    `zlib.crc32` rather than the builtin `hash()`, which `PYTHONHASHSEED` randomises per
    process. Within one process either works, since that is the only scope a module name has to
    be unique in. But this name is what appears in tracebacks from the user's own plugin code,
    and one that changes on every run makes two reports of the same crash look like two
    different crashes. The file's stem is kept in front for the same reason: a name like
    `contextgrid_plugin_my_metrics_66c5a80e` says which file raised, and a bare number does not.
    """
    stem = re.sub(r"\W", "_", path.stem) or "plugin"
    return f"contextgrid_plugin_{stem}_{zlib.crc32(str(path).encode()):08x}"


def _load_file(spec: str, *, base: Path) -> str:
    """Import a `.py` file by path, without it needing to be on `sys.path`."""
    path = Path(spec).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()

    if not path.exists():
        raise ConfigError(f"plugins: no such file: {path}")
    if path.is_dir():
        raise ConfigError(
            f"plugins: {path} is a directory. Name the file itself, or use a dotted module name."
        )

    module_name = _module_name_for(path)
    if module_name in sys.modules:
        return str(path)

    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ConfigError(f"plugins: {path} could not be loaded as a Python module")

    module = importlib.util.module_from_spec(module_spec)
    # Registered before executing, so a plugin file that imports itself -- or imports something
    # that imports it -- does not run twice.
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as error:
        del sys.modules[module_name]
        raise ConfigError(
            f"plugins: {path} raised {type(error).__name__} while being imported: {error}"
        ) from error
    return str(path)


# ---------------------------------------------------------------------------
# whether a named plugin can actually run here
# ---------------------------------------------------------------------------


def missing_extra(registry: Registry[Any], spec: str) -> MissingExtraError | None:
    """The error a spec would eventually raise for want of an optional package, or None.

    Building a plugin is supposed to be what proves it can run, and for most of them it is:
    `Registration.load()` imports the plugin's module, and if that module imports a package
    that is not installed it raises `MissingExtraError` naming the extra. `contextgrid check`
    builds one of everything precisely so that happens early.

    It does not always happen. A plugin whose module is *in this tree* imports fine no matter
    what is installed, because the third-party import sits inside the method that needs it --
    `MarkerParser` lives in `contextgrid.parse.layout` and does `import marker` in `parse()`,
    which is deliberate (importing Surya to build a parser object would make `contextgrid
    plugins` cost seconds). The consequence was that `check` built `marker` happily and said
    "config is valid.", and the sweep it green-lit died on the first document. `check` exists
    to catch exactly that, so construction cannot be the only evidence.

    So ask the registration instead of the plugin. It already records `extra` and `package` --
    that is where `Registration.load` gets the text of its own error, and it is the same thing
    `contextgrid init` consults to decide which plugins to write into a starter config. One
    fact, read the same way in all three places, rather than a second opinion that can drift
    from the first.

    Returns the error rather than raising it, because the caller checking a whole matrix wants
    to collect every problem and print them together, not stop at the first.
    """
    try:
        registration = registry.registration(registry.name_in(spec))
    except UnknownPluginError:
        # Not a plugin at all. `create` says so, and says it better -- it lists the names that
        # do exist. Reporting "unknown" twice, in two wordings, helps nobody.
        return None
    return extra_missing_for(registration)


def extra_missing_for(registration: Registration) -> MissingExtraError | None:
    """The same question, for a registration already in hand."""
    if registration.extra is None or registration.package is None:
        # Nothing optional to be missing, or nothing recorded to look for. `tei` needs a server
        # rather than a package, and no amount of `pip install` settles whether one is running.
        return None
    if _dependency_present(registration.package):
        return None
    return MissingExtraError(
        # `The marker parser`, matching word for word what `MarkerParser.parse` raises when the
        # sweep reaches a document -- so `check` and `run` report one failure, not two that
        # have to be recognised as the same thing.
        f"The {registration.name} {registration.family}",
        registration.extra,
        package=registration.package,
    )


def model_missing_for(spec: str, llm: object | None) -> Exception | None:
    """The error a model-backed ingestion strategy deserves for want of a model, or None.

    Three of the four axes that can call a model refuse to be built without one, and say so in
    the same sentence: `transform: hyde`, `retrieval: agentic` and `generator: llm` all fail
    `contextgrid check` naming `run.model` and listing the model-free alternatives. Ingestion
    was the fourth and did not. `ingestion: contextual` validated, ran, failed its model call
    once per chunk, fell back to indexing each chunk as written, and produced a leaderboard row
    labelled `contextual` that had made no model call at all -- a result somebody would act on.

    Ingestion cannot refuse in its own constructor the way the other three do: the model
    reaches a strategy through `IngestionContext` at ingest time, not through the factory, so
    `get_ingester("contextual")` is a legitimate call with no model in sight. So the question is
    asked here instead, where a configuration is being validated and the answer is knowable:
    this config names a paid strategy, and this config sets no model.

    Returns the error rather than raising it, for the same reason `missing_extra` does -- the
    caller is checking a whole matrix and wants every problem at once.

    The message itself comes from `ingest.base.needs_model_error`, which `pipeline.build` also
    raises. `check` and `run` refuse from here, before a document is read; the `cg.Lab` path
    has no config to inspect and refuses there. Two places, one sentence.
    """
    if llm is not None:
        return None

    from contextgrid.ingest import get_ingester
    from contextgrid.ingest.base import needs_a_model, needs_model_error

    try:
        strategy = get_ingester(spec)
    except Exception:
        # An unknown or malformed spec is reported by whatever builds it, in better words.
        return None
    if not needs_a_model(strategy):
        return None

    return needs_model_error(strategy.name)


def _dependency_present(package: str) -> bool:
    """Is this third-party package installed? Asked without importing it.

    Two questions, either of which is a yes, because neither alone is reliable and a false
    "missing" is much worse than a false "present": it would refuse to run a sweep that works.

    **Is a distribution of that name installed?** This is the name in `pip install`, and it is
    what the registration records. It has to be asked first because a distribution's name is
    frequently not its module's: `marker-pdf` imports as `marker`, `faiss-cpu` as `faiss`.

    **Failing that, is a module of that name importable?** For a package installed without
    metadata a distribution lookup finds nothing -- a source checkout on `PYTHONPATH`, or a
    vendored copy. `find_spec` locates the module without executing it, so this stays cheap
    and reads no documents, opens no sockets and downloads no model.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution(package)
        return True
    except PackageNotFoundError:
        pass
    except Exception:
        # A broken or half-written .dist-info should not be able to make `check` crash.
        pass
    return _importable(package.split(".")[0].replace("-", "_"))


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # `find_spec` imports parent packages, and raises ValueError for a module whose parent
        # is present but broken. Either way it is not usable here.
        return False

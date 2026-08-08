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

from contextgrid.config.schema import ConfigError


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

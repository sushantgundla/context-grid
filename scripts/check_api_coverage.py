#!/usr/bin/env python3
"""Check whether docs/ mentions every piece of contextgrid's public API.

Regenerate docs/COVERAGE.md with::

    .venv/bin/python scripts/check_api_coverage.py

The check is mechanical, not semantic: "documented" means the identifier's exact
name (or, for warning codes, its string value) appears somewhere under docs/ as a
whole word. It does not check that the surrounding prose actually explains the
thing -- only that the name is present at all. Treat every "documented" row here as
"at least mentioned", and treat "undocumented" rows as certain.

Five things are compared:

1. Every name in ``contextgrid.__all__`` and in each public subpackage's
   ``__all__``, against all of docs/.
2. Every public method, property and dataclass field on twelve classes users
   touch directly, against all of docs/.
3. Every settable (``init=True``) dataclass field of every plugin registered
   under a ``Registry`` -- the parameters that go into a spec string like
   ``"recursive:512,overlap=64"`` -- against all of docs/.
4. Every ``WarningCode`` member (checked both as ``CODE_NAME`` and as its
   ``.value`` string), against all of docs/.
5. Every key in ``RunConfig.KNOWN``, ``GridConfig``'s axes (``AXIS_ORDER``) and
   ``ReportConfig.KNOWN``, against ``docs/guide/configuration.md`` specifically.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SRC_DIR = ROOT / "src"
OUT_FILE = ROOT / "docs" / "COVERAGE.md"

sys.path.insert(0, str(SRC_DIR))

import contextgrid  # noqa: E402

# ---------------------------------------------------------------------------
# docs text
# ---------------------------------------------------------------------------


def load_docs() -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.rglob("*.md"))}


DOC_FILES = load_docs()
ALL_DOCS_TEXT = "\n".join(DOC_FILES.values())
CONFIG_DOC = DOCS_DIR / "guide" / "configuration.md"
CONFIG_TEXT = DOC_FILES.get(CONFIG_DOC, "")


def mentioned(name: str, text: str = ALL_DOCS_TEXT) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def files_mentioning(name: str) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return sorted(
        p.relative_to(ROOT).as_posix() for p, text in DOC_FILES.items() if pattern.search(text)
    )


# ---------------------------------------------------------------------------
# 1. exported names
# ---------------------------------------------------------------------------

# Subpackages that declare a public __all__. `core` and `score` deliberately have
# none -- their public surface is re-exported straight through the top-level
# contextgrid.__all__ instead, so they are not double-counted here.
SUBPACKAGES = [
    "assemble",
    "cache",
    "chunk",
    "cli",
    "config",
    "corpus",
    "cost",
    "diagnose",
    "embed",
    "evalset",
    "generate",
    "grid",
    "index",
    "ingest",
    "parse",
    "report",
    "rerank",
    "retrieve",
    "transform",
]


def check_exports() -> dict:
    out: dict = {"top_level": [], "subpackages": {}}
    for name in sorted(contextgrid.__all__):
        out["top_level"].append((name, mentioned(name)))
    for sp in SUBPACKAGES:
        mod = importlib.import_module(f"contextgrid.{sp}")
        names = sorted(getattr(mod, "__all__", []))
        out["subpackages"][sp] = [(n, mentioned(n)) for n in names]
    return out


# ---------------------------------------------------------------------------
# 2. main classes: public methods / properties / fields
# ---------------------------------------------------------------------------

# name -> (import path, attribute) for classes not in contextgrid.__all__
CLASS_SOURCES = {
    "Lab": ("contextgrid", "Lab"),
    "Results": ("contextgrid", "Results"),
    "RunResult": ("contextgrid", "RunResult"),
    "Corpus": ("contextgrid", "Corpus"),
    "EvalSet": ("contextgrid", "EvalSet"),
    "EvalItem": ("contextgrid", "EvalItem"),
    "Config": ("contextgrid", "Config"),
    "Matrix": ("contextgrid", "Matrix"),
    "Ingested": ("contextgrid.ingest", "Ingested"),
    "CompositeScore": ("contextgrid.report", "CompositeScore"),
    "EmbeddingQuality": ("contextgrid.embed", "EmbeddingQuality"),
    "RetrievalTrace": ("contextgrid.retrieve", "RetrievalTrace"),
}


def public_members(cls: type) -> list[tuple[str, str]]:
    """Public methods, properties and dataclass fields defined on cls or a base."""
    members: list[tuple[str, str]] = []
    seen: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in seen:
                continue
            if isinstance(value, property):
                members.append((name, "property"))
                seen.add(name)
            elif inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod)):
                members.append((name, "method"))
                seen.add(name)
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if not f.name.startswith("_") and f.name not in seen:
                members.append((f.name, "field"))
                seen.add(f.name)
    return sorted(members)


def check_classes() -> dict:
    out = {}
    for class_name, (module_path, attr) in CLASS_SOURCES.items():
        mod = importlib.import_module(module_path)
        cls = getattr(mod, attr)
        members = public_members(cls)
        out[class_name] = [(name, kind, mentioned(name)) for name, kind in members]
    return out


# ---------------------------------------------------------------------------
# 3. plugin parameters
# ---------------------------------------------------------------------------

REGISTRY_PATHS = [
    ("contextgrid.tokens", "TOKENIZERS"),
    ("contextgrid.chunk", "CHUNKERS"),
    ("contextgrid.embed", "EMBEDDERS"),
    ("contextgrid.ingest", "INGESTERS"),
    ("contextgrid.parse", "PARSERS"),
    ("contextgrid.evalset.llm", "LLMS"),
    ("contextgrid.generate", "GENERATORS"),
    ("contextgrid.rerank", "RERANKERS"),
    ("contextgrid.index", "INDEXES"),
    ("contextgrid.retrieve", "RETRIEVERS"),
    ("contextgrid.transform.query", "TRANSFORMS"),
]


def settable_params(factory) -> list[str]:
    """Parameters a spec string could set: init=True dataclass fields, else the
    factory's non-self, non-private call signature parameters."""
    if dataclasses.is_dataclass(factory):
        return sorted(
            f.name for f in dataclasses.fields(factory) if f.init and not f.name.startswith("_")
        )
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return []
    return sorted(
        p.name
        for p in sig.parameters.values()
        if p.name not in ("self", "cls")
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and not p.name.startswith("_")
    )


def check_plugins() -> dict:
    out: dict = {}
    for module_path, registry_attr in REGISTRY_PATHS:
        mod = importlib.import_module(module_path)
        registry = getattr(mod, registry_attr)
        family_entries = []
        for registration in registry:
            try:
                factory = registration.load()
            except ImportError:
                family_entries.append(
                    {"plugin": registration.name, "skipped": "optional dependency not installed"}
                )
                continue
            params = settable_params(factory)
            family_entries.append(
                {
                    "plugin": registration.name,
                    "params": [(p, mentioned(p)) for p in params],
                }
            )
        out[registry_attr] = family_entries
    return out


# ---------------------------------------------------------------------------
# 4. warning codes
# ---------------------------------------------------------------------------


def check_warning_codes() -> list[tuple[str, str, bool]]:
    rows = []
    for member in contextgrid.WarningCode:
        seen = mentioned(member.name) or mentioned(member.value)
        rows.append((member.name, member.value, seen))
    return rows


# ---------------------------------------------------------------------------
# 5. config keys, against docs/guide/configuration.md specifically
# ---------------------------------------------------------------------------


def check_config_keys() -> dict:
    from contextgrid.config.schema import GridConfig, ReportConfig, RunConfig
    from contextgrid.grid.matrix import AXIS_ORDER

    assert GridConfig  # keys come from AXIS_ORDER, not a KNOWN tuple of its own
    return {
        "run": [(k, mentioned(k, CONFIG_TEXT)) for k in RunConfig.KNOWN],
        "grid": [(k, mentioned(k, CONFIG_TEXT)) for k in AXIS_ORDER],
        "report": [(k, mentioned(k, CONFIG_TEXT)) for k in ReportConfig.KNOWN],
    }


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------


def render(exports, classes, plugins, warning_codes, config_keys) -> str:
    lines: list[str] = []
    w = lines.append

    w("# API documentation coverage")
    w("")
    w(
        "Generated by `scripts/check_api_coverage.py`. Regenerate with "
        "`.venv/bin/python scripts/check_api_coverage.py` after any change to the public "
        "API or to docs/. Do not hand-edit this file."
    )
    w("")
    w(
        '"Documented" means the exact name (or a `WarningCode`\'s `.value` string) appears '
        "as a whole word somewhere under docs/. It is a presence check, not a quality check: "
        "a name mentioned once in passing counts the same as one with a worked example."
    )
    w("")

    # --- 1. exports ---
    w("## 1. Exported names vs docs/")
    w("")
    total = len(exports["top_level"])
    documented = sum(1 for _, ok in exports["top_level"] if ok)
    w(f"`contextgrid.__all__`: **{documented} of {total}** documented.")
    w("")
    missing_top = [n for n, ok in exports["top_level"] if not ok]
    if missing_top:
        w("Undocumented top-level exports:")
        w("")
        for n in missing_top:
            w(f"- `{n}`")
        w("")
    else:
        w("All top-level exports are mentioned somewhere in docs/.")
        w("")

    w("Per-subpackage `__all__` (each name is also part of the public API surface):")
    w("")
    w("| Subpackage | Documented | Total | Undocumented |")
    w("|---|---|---|---|")
    sub_missing_total = 0
    sub_total = 0
    for sp, entries in exports["subpackages"].items():
        d = sum(1 for _, ok in entries if ok)
        t = len(entries)
        sub_total += t
        missing = [n for n, ok in entries if not ok]
        sub_missing_total += len(missing)
        missing_str = ", ".join(f"`{n}`" for n in missing) if missing else "—"
        w(f"| `{sp}` | {d} | {t} | {missing_str} |")
    w("")
    w(
        f"Subpackage totals: **{sub_total - sub_missing_total} of {sub_total}** documented "
        f"across {len(exports['subpackages'])} subpackages."
    )
    w("")
    w(
        "`contextgrid.core` and `contextgrid.score` have no `__all__` of their own; their "
        "public names reach users only through the top-level `contextgrid.__all__` above, "
        "so they are not counted twice here."
    )
    w("")

    # --- 2. classes ---
    w("## 2. Public methods, properties and fields of the twelve main classes")
    w("")
    w("| Class | Documented | Total | Undocumented members |")
    w("|---|---|---|---|")
    class_documented_total = 0
    class_total = 0
    for class_name, entries in classes.items():
        d = sum(1 for _, _, ok in entries if ok)
        t = len(entries)
        class_documented_total += d
        class_total += t
        missing = [f"`{name}` ({kind})" for name, kind, ok in entries if not ok]
        missing_str = ", ".join(missing) if missing else "—"
        w(f"| `{class_name}` | {d} | {t} | {missing_str} |")
    w("")
    w(f"Totals: **{class_documented_total} of {class_total}** members documented.")
    w("")
    w(
        "`Ingested`, `CompositeScore`, `EmbeddingQuality` and `RetrievalTrace` are not in "
        "`contextgrid.__all__` -- they are only importable via their subpackage "
        "(`contextgrid.ingest.Ingested`, `contextgrid.report.CompositeScore`, "
        "`contextgrid.embed.EmbeddingQuality`, `contextgrid.retrieve.RetrievalTrace`). They "
        "are still public (listed in that subpackage's `__all__`), just one import away from "
        "the other eight."
    )
    w("")

    # --- 3. plugin params ---
    w("## 3. Plugin parameters (spec-string fields) vs docs/")
    w("")
    w("| Registry | Plugin | Documented | Total | Undocumented params |")
    w("|---|---|---|---|---|")
    plugin_doc_total = 0
    plugin_param_total = 0
    skipped = []
    for registry_name, entries in plugins.items():
        for entry in entries:
            if "skipped" in entry:
                skipped.append((registry_name, entry["plugin"], entry["skipped"]))
                continue
            params = entry["params"]
            d = sum(1 for _, ok in params if ok)
            t = len(params)
            plugin_doc_total += d
            plugin_param_total += t
            missing = (
                ", ".join(f"`{p}`" for p, ok in params if not ok)
                if any(not ok for _, ok in params)
                else "—"
            )
            w(f"| `{registry_name}` | `{entry['plugin']}` | {d} | {t} | {missing} |")
    w("")
    w(f"Totals: **{plugin_doc_total} of {plugin_param_total}** plugin parameters documented.")
    w("")
    w(
        "Matching is by parameter name alone, not by which plugin it belongs to: two "
        "plugins that both take `size` are both marked documented as soon as `size` "
        "appears anywhere in docs/, even if only one of them is actually covered. Treat "
        "the undocumented list as certain and the documented list as an upper bound."
    )
    w("")
    if skipped:
        w("Skipped (optional dependency not installed, parameters not inspected):")
        w("")
        for registry_name, plugin_name, reason in skipped:
            w(f"- `{registry_name}` / `{plugin_name}`: {reason}")
        w("")

    # --- 4. warning codes ---
    w("## 4. WarningCode members vs docs/")
    w("")
    d = sum(1 for _, _, ok in warning_codes if ok)
    t = len(warning_codes)
    w(f"**{d} of {t}** `WarningCode` members documented (checked by name or by `.value`).")
    w("")
    missing = [(name, value) for name, value, ok in warning_codes if not ok]
    if missing:
        w("Undocumented:")
        w("")
        for name, value in missing:
            w(f"- `WarningCode.{name}` (`{value!r}`)")
        w("")
    else:
        w("All warning codes are mentioned somewhere in docs/.")
        w("")

    # --- 5. config keys ---
    w("## 5. Config keys vs docs/guide/configuration.md")
    w("")
    w("| Section | Documented | Total | Undocumented keys |")
    w("|---|---|---|---|")
    config_doc_total = 0
    config_key_total = 0
    for section, entries in config_keys.items():
        d = sum(1 for _, ok in entries if ok)
        t = len(entries)
        config_doc_total += d
        config_key_total += t
        missing = ", ".join(f"`{k}`" for k, ok in entries if not ok) or "—"
        w(f"| `{section}` | {d} | {t} | {missing} |")
    w("")
    w(f"Totals: **{config_doc_total} of {config_key_total}** config keys documented.")
    w("")
    w(
        "`GridConfig` has no `KNOWN` tuple of its own; its valid keys are "
        "`contextgrid.grid.matrix.AXIS_ORDER`, which is what `grid.from_mapping` actually "
        "validates against, and what is checked here."
    )
    w("")

    return "\n".join(lines) + "\n"


def main() -> None:
    exports = check_exports()
    classes = check_classes()
    plugins = check_plugins()
    warning_codes = check_warning_codes()
    config_keys = check_config_keys()
    OUT_FILE.write_text(render(exports, classes, plugins, warning_codes, config_keys))
    print(f"wrote {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

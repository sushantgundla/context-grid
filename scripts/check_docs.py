#!/usr/bin/env python3
"""Run every runnable code example in docs/**/*.md and report which ones fail.

Docs rot silently once they are frozen against a codebase that keeps moving. This walks
every Markdown file, pulls out the fenced ```python, ```bash and ```sh blocks, actually
runs them, and says which ones broke -- rather than trusting whoever wrote the page last.

Two design decisions, stated here because both were judgment calls:

**Shared state, per file, per language.** The docs are written as a growing REPL
transcript -- a later block imports nothing because an earlier block in the same file
already did. So every python block in one Markdown file shares one namespace, executed
top to bottom, in document order. Bash blocks in one file share one temporary working
directory the same way (so a block that `mkdir`s something can be used by a later block),
but each runs as its own `bash` subprocess -- shell variables and `cd` do not carry over,
only the filesystem does. A failing block does not stop the ones after it, in either
language, so one bad example never hides everything past it in the report.

**Opt-out, not opt-in.** A fence is skipped only if its info string carries the `no-run`
token, e.g. ```python no-run: needs a live TEI server```. Everything after `no-run`
(stripped of a leading `:` or `-`) is kept as the reason and printed in the summary. A
`no-run` with nothing after it still works, but is called out as reasonless so it does not
become a silent, unexplained skip.

Doctest-style blocks (containing `>>> `) are run through the standard library's `doctest`
parser and checked output-for-output, including `Traceback` blocks. Plain blocks (no
`>>> `) are just executed; success means no exception, there is no expected output to
compare against.
"""

from __future__ import annotations

import argparse
import contextlib
import doctest
import io
import os
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"

RUNNABLE_LANGS = {"python", "bash", "sh"}

#: Generous on purpose: a few doc examples load real ML models (docling, marker) on first
#: use, which can legitimately take over a minute. A short timeout doesn't just fail those
#: blocks, it can interrupt inference mid-flight and get misreported as an unrelated
#: exception from whatever library's own error handling catches it -- worse than slow.
PYTHON_TIMEOUT_SECONDS = 120
BASH_TIMEOUT_SECONDS = 20

FENCE_RE = re.compile(
    r"^```[ \t]*(?P<info>[^\n]*)\n(?P<body>.*?)\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """One fenced code block, and where it came from."""

    path: Path
    fence_line: int  # 1-indexed line number of the opening ``` line
    lang: str
    info: str
    body: str
    skip: bool = False
    skip_reason: str | None = None


@dataclass
class Result:
    block: Block
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str | None = None


def extract_blocks(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8")
    blocks: list[Block] = []
    for match in FENCE_RE.finditer(text):
        info = match.group("info").strip()
        tokens = info.split()
        lang = tokens[0].lower() if tokens else ""
        if lang not in RUNNABLE_LANGS:
            continue
        fence_line = text.count("\n", 0, match.start()) + 1
        skip, reason = _parse_no_run(info)
        blocks.append(
            Block(
                path=path,
                fence_line=fence_line,
                lang=lang,
                info=info,
                body=match.group("body"),
                skip=skip,
                skip_reason=reason,
            )
        )
    return blocks


def _parse_no_run(info: str) -> tuple[bool, str | None]:
    """Look for the `no-run` opt-out token in a fence info string.

    ```python no-run: needs a live TEI server``` -> (True, "needs a live TEI server")
    ```python no-run```                          -> (True, "(no reason given)")
    ```python```                                 -> (False, None)
    """
    match = re.search(r"\bno-run\b[:\-]?\s*(?P<reason>.*)$", info, re.IGNORECASE)
    if match is None:
        return False, None
    reason = match.group("reason").strip()
    return True, reason or "(no reason given)"


def _looks_like_doctest(body: str) -> bool:
    return any(
        line.lstrip().startswith(">>> ") or line.strip() == ">>>" for line in body.splitlines()
    )


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------


class _BlockTimeoutError(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _BlockTimeoutError()


@contextlib.contextmanager
def _time_limit(seconds: int):
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# running python
# ---------------------------------------------------------------------------


def _exc_last_line(exc: BaseException) -> str:
    formatted = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return formatted.splitlines()[-1] if formatted else repr(exc)


def _short(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def run_doctest_block(block: Block, globs: dict, results: list[Result]) -> None:
    parser = doctest.DocTestParser()
    try:
        examples = parser.get_examples(block.body)
    except ValueError as exc:
        results.append(Result(block, "FAIL", f"could not parse as a doctest: {exc}"))
        return

    if not examples:
        results.append(Result(block, "FAIL", "contains '>>>' but no parseable examples"))
        return

    checker = doctest.OutputChecker()
    failed = False
    detail: str | None = None

    for example in examples:
        buf = io.StringIO()
        try:
            with _time_limit(PYTHON_TIMEOUT_SECONDS), contextlib.redirect_stdout(buf):
                exec(compile(example.source, "<doctest>", "single", dont_inherit=True), globs)
        except _BlockTimeoutError:
            failed = True
            detail = detail or f"timed out after {PYTHON_TIMEOUT_SECONDS}s"
            continue
        except BaseException as exc:
            got = "".join(traceback.format_exception_only(type(exc), exc))
            if example.exc_msg is not None and checker.check_output(
                example.exc_msg, got, doctest.ELLIPSIS
            ):
                continue  # an exception was expected here, and this is the one we got
            failed = True
            detail = detail or f"raised {_exc_last_line(exc)}"
            continue

        got = buf.getvalue()
        if example.exc_msg is not None:
            failed = True
            detail = (
                detail or f"expected an exception ({_short(example.exc_msg)}) but none was raised"
            )
            continue
        if not checker.check_output(example.want, got, doctest.ELLIPSIS):
            failed = True
            detail = detail or f"expected {_short(example.want)!r}, got {_short(got)!r}"

    results.append(Result(block, "FAIL" if failed else "PASS", detail))


def run_plain_python_block(block: Block, globs: dict, results: list[Result]) -> None:
    buf = io.StringIO()
    try:
        with _time_limit(PYTHON_TIMEOUT_SECONDS), contextlib.redirect_stdout(buf):
            exec(compile(block.body, str(block.path), "exec", dont_inherit=True), globs)
    except _BlockTimeoutError:
        results.append(Result(block, "FAIL", f"timed out after {PYTHON_TIMEOUT_SECONDS}s"))
        return
    except BaseException as exc:
        if os.environ.get("CHECK_DOCS_DEBUG"):
            traceback.print_exc()
        results.append(Result(block, "FAIL", f"raised {_exc_last_line(exc)}"))
        return
    results.append(Result(block, "PASS", None))


# ---------------------------------------------------------------------------
# running bash / sh
# ---------------------------------------------------------------------------


def run_bash_block(block: Block, cwd: Path, env: dict, results: list[Result]) -> None:
    try:
        proc = subprocess.run(
            ["bash", "-c", block.body],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        results.append(Result(block, "FAIL", f"timed out after {BASH_TIMEOUT_SECONDS}s"))
        return

    if proc.returncode != 0:
        stderr_lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
        last = stderr_lines[-1] if stderr_lines else "(no stderr)"
        results.append(Result(block, "FAIL", f"exit {proc.returncode}: {_short(last)}"))
        return
    results.append(Result(block, "PASS", None))


# ---------------------------------------------------------------------------
# per-file orchestration
# ---------------------------------------------------------------------------


def check_file(path: Path, venv_python: Path) -> list[Result]:
    blocks = extract_blocks(path)
    if not blocks:
        return []

    results: list[Result] = []
    py_blocks = [b for b in blocks if b.lang == "python"]
    sh_blocks = [b for b in blocks if b.lang in ("bash", "sh")]

    with tempfile.TemporaryDirectory(prefix="check-docs-") as tmp_name:
        tmp = Path(tmp_name)

        # -- python: one shared namespace per file, cwd inside the temp dir --
        # A handful of docs legitimately do `sys.path.insert(0, "tests")` to reach this
        # repo's own test fixtures (e.g. tests/pdf_fixtures.py), the same way a reader
        # running the snippet from the repo root would. Symlinking tests/ into the temp
        # dir honours that without giving up the temp-dir isolation for everything else.
        tests_dir = REPO_ROOT / "tests"
        if tests_dir.is_dir():
            os.symlink(tests_dir, tmp / "tests")

        old_cwd = os.getcwd()
        os.chdir(tmp)
        # A real `python script.py` run puts the script's own directory on sys.path
        # automatically; exec() does not. Add it so cwd-relative imports (including into
        # the symlinked tests/ above) behave the way a reader's terminal would.
        sys.path.insert(0, str(tmp))
        modules_before = set(sys.modules)
        try:
            globs: dict = {"__name__": "__doc_example__"}
            for block in py_blocks:
                if block.skip:
                    results.append(Result(block, "SKIP", block.skip_reason))
                    continue
                if _looks_like_doctest(block.body):
                    run_doctest_block(block, globs, results)
                else:
                    run_plain_python_block(block, globs, results)
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(tmp))
            # Don't let a module imported *from this file's temp dir* (now about to be
            # deleted) linger in the cache under a name a later file might reuse -- e.g.
            # `tests/pdf_fixtures.py` reached via the symlink above. Only that: deleting
            # sys.modules entries for ordinary packages (numpy, torch, ...) and re-importing
            # them is a different thing from a fresh process starting up, and at least one
            # of them (numpy's C extension) explicitly refuses a second init with
            # "cannot load module more than once per process" -- so those must stay put.
            for name in set(sys.modules) - modules_before:
                module = sys.modules.get(name)
                module_file = getattr(module, "__file__", None)
                module_path = getattr(module, "__path__", None)
                under_tmp = (module_file and str(tmp) in module_file) or (
                    module_path and any(str(tmp) in p for p in module_path)
                )
                if under_tmp:
                    del sys.modules[name]

        # -- bash/sh: one shared temp dir per file, each block its own subprocess --
        # Docs often write `.venv/bin/python ...` (relative to the repo root, as a reader
        # would run it) rather than a bare `python`. Symlinking the real .venv into the
        # temp dir makes that literal path resolve without copying the whole environment.
        env = dict(os.environ)
        venv_bin = str(venv_python.parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
        if sh_blocks:
            os.symlink(venv_python.parent.parent, tmp / ".venv")
        for block in sh_blocks:
            if block.skip:
                results.append(Result(block, "SKIP", block.skip_reason))
                continue
            run_bash_block(block, tmp, env, results)

    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="doc files to check (default: every .md file under docs/)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="also print PASS lines, not just FAIL/SKIP"
    )
    args = parser.parse_args(argv)

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print(f"error: {venv_python} not found. Set up .venv before running this.", file=sys.stderr)
        return 2
    if (
        str(sys.executable) != str(venv_python)
        and Path(sys.executable).resolve() != venv_python.resolve()
    ):
        print(
            f"error: run this with {venv_python}, not {sys.executable} "
            "(python examples need the project's dependencies). Use scripts/check-docs.sh.",
            file=sys.stderr,
        )
        return 2

    if args.paths:
        files = sorted(Path(p).resolve() for p in args.paths)
    else:
        files = sorted(DEFAULT_DOCS_DIR.rglob("*.md"))

    all_results: list[Result] = []
    for file_path in files:
        all_results.extend(check_file(file_path, venv_python))

    all_results.sort(key=lambda r: (str(r.block.path), r.block.fence_line))

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    for result in all_results:
        if result.status == "PASS" and not args.verbose:
            continue
        line = (
            f"{rel(result.block.path)}:{result.block.fence_line}  "
            f"{result.status:<4}  {result.block.lang}"
        )
        if result.detail:
            line += f"  -- {result.detail}"
        print(line)

    n_pass = sum(1 for r in all_results if r.status == "PASS")
    n_fail = sum(1 for r in all_results if r.status == "FAIL")
    n_skip = sum(1 for r in all_results if r.status == "SKIP")

    if n_skip:
        print(f"\n{n_skip} skipped, and why:")
        for result in all_results:
            if result.status != "SKIP":
                continue
            reasonless = result.detail == "(no reason given)"
            flag = "  [NO REASON GIVEN]" if reasonless else ""
            print(f"  {rel(result.block.path)}:{result.block.fence_line}  {result.detail}{flag}")

    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped ({len(all_results)} total)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

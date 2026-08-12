#!/usr/bin/env python3
"""Prove the drive's fixture corpus is sound before blaming the tool for anything.

Every anchor in the eval set claims a quote appears in a named document. If one does not,
every downstream score is wrong for a reason that has nothing to do with context-grid --
and a drive that skips this check reports fixture typos as tool bugs.

Run this first, from the skill's data directory or with --data pointing at it:

    python scripts/verify_fixtures.py --data .claude/skills/docs-e2e-drive/data

Exit 0 means the fixture is sound and any later mismatch is the tool's. Exit 1 means fix
the fixture first.

Reads no source code and imports nothing from contextgrid, on purpose: this must stay
true even when the tool is broken.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Quotes are matched twice. EXACT is what a user gets by copying from a rendered page.
# NORMALISED collapses runs of whitespace, which is what a user gets by copying a
# sentence that a Markdown file soft-wraps across two lines -- extremely common, and the
# case `nw13` exists to exercise. A quote that only matches NORMALISED is a valid
# fixture, not a broken one, so it is reported separately rather than failed.
EXACT, NORMALISED, MISSING = "exact", "normalised", "missing"


def _squash(text: str) -> str:
    return " ".join(text.split())


def classify(quote: str, document: str) -> str:
    if quote in document:
        return EXACT
    if _squash(quote) in _squash(document):
        return NORMALISED
    return MISSING


def anchors_from_csv(path: Path) -> list[tuple[str, str, str]]:
    """Yield (item_id, source_id, quote) from the loose-column CSV an SME would hand over."""
    # These alias lists mirror the ones documented in docs/guide/evalsets.md. They are
    # duplicated here rather than imported so this script keeps working when the tool
    # cannot even be imported.
    id_keys = ("id", "question_id", "qid")
    doc_keys = ("source_id", "document", "doc", "doc_id", "file", "filename")
    quote_keys = ("quote", "evidence", "answer_span", "context", "passage")

    def pick(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for key in keys:
            for actual, value in row.items():
                if actual and actual.strip().lower() == key:
                    return (value or "").strip()
        return ""

    out: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            out.append((pick(row, id_keys) or f"row{index}", pick(row, doc_keys), pick(row, quote_keys)))
    return out


def anchors_from_jsonl(path: Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "_evalset" in record:  # the header line carries identity, not a question
            continue
        for anchor in record.get("anchors", ()):
            out.append((record.get("id", "?"), anchor.get("source_id", ""), anchor.get("quote", "")))
        if not record.get("anchors"):
            out.append((record.get("id", "?"), "", ""))
    return out


def check(data: Path) -> int:
    documents = data / "documents"
    if not documents.is_dir():
        print(f"error: no documents directory at {documents}", file=sys.stderr)
        return 1

    sources = {path.name: path.read_text(encoding="utf-8") for path in sorted(documents.iterdir()) if path.is_file()}
    print(f"{len(sources)} documents under {documents}")

    failures = 0
    for name, loader in (("questions.csv", anchors_from_csv), ("questions.jsonl", anchors_from_jsonl)):
        path = data / name
        if not path.exists():
            print(f"  {name}: not present, skipped")
            continue
        rows = loader(path)
        counts = {EXACT: 0, NORMALISED: 0, MISSING: 0, "no-evidence": 0}
        for item_id, source_id, quote in rows:
            if not quote or not source_id:
                counts["no-evidence"] += 1
                continue
            if source_id not in sources:
                print(f"  {name}: {item_id} names {source_id!r}, which is not in the corpus")
                failures += 1
                continue
            verdict = classify(quote, sources[source_id])
            counts[verdict] += 1
            if verdict is MISSING:
                print(f"  {name}: {item_id} quote not found in {source_id}: {quote[:60]!r}")
                failures += 1
            elif verdict is NORMALISED:
                print(f"  {name}: {item_id} matches {source_id} only after collapsing whitespace (deliberate)")
        print(
            f"  {name}: {len(rows)} anchors -- "
            f"{counts[EXACT]} exact, {counts[NORMALISED]} whitespace-normalised, "
            f"{counts['no-evidence']} with no evidence (deliberate), {counts[MISSING]} broken"
        )

    if failures:
        print(f"\n{failures} broken anchor(s). Fix the fixture before drawing any conclusion about the tool.")
        return 1
    print("\nFixture is sound. Any mismatch from here is the tool's, not the data's.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="the skill's data directory (default: the one beside this script)",
    )
    return check(parser.parse_args().data)


if __name__ == "__main__":
    raise SystemExit(main())

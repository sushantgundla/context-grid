"""The run manifest.

"Reproducible" is a claim until something makes it checkable. The manifest is a hash-pinned
record of everything that could change a number: the corpus contents, the eval set and its
version, every plugin and its version, every parameter, the resolution policy, the library
versions, the seeds.

Two properties follow, and both are load-bearing.

**Two runs with the same manifest hash must produce identical numbers.** If they do not,
something outside the manifest is affecting results, and that is a bug worth finding.

**When a metric drops, diff the manifest against the last passing run.** The changed line is
the suspect. That turns regression triage from an investigation into a comparison, and it is
the thing everybody describes and nobody ships.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextgrid.core.evalset import EvalSet
from contextgrid.corpus import Corpus
from contextgrid.pipeline import Config
from contextgrid.score.resolve import SpanResolver


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to reproduce one run, and nothing that changes between runs.

    Deliberately excludes wall-clock timings, costs and the results themselves. A manifest
    that changed every run could not be compared with another one, which is its whole job.
    """

    config: dict[str, Any]
    corpus_hash: str
    corpus_files: int
    evalset_id: str
    evalset_version: int
    evalset_hash: str
    resolution: dict[str, Any]
    versions: dict[str, str]
    seeds: dict[str, int] = field(default_factory=dict)
    #: Recorded but excluded from the hash: useful context, not an input to the result.
    created_at: str = ""
    notes: str = ""

    def hash(self) -> str:
        """A hash of everything that could change a number.

        `created_at` and `notes` are left out on purpose. Including them would make every
        run's manifest unique, which would defeat the point.
        """
        payload = {
            "config": self.config,
            "corpus_hash": self.corpus_hash,
            "evalset_id": self.evalset_id,
            "evalset_version": self.evalset_version,
            "evalset_hash": self.evalset_hash,
            "resolution": self.resolution,
            "versions": self.versions,
            "seeds": self.seeds,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.hash()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.hash(),
            "config": self.config,
            "corpus_hash": self.corpus_hash,
            "corpus_files": self.corpus_files,
            "evalset_id": self.evalset_id,
            "evalset_version": self.evalset_version,
            "evalset_hash": self.evalset_hash,
            "resolution": self.resolution,
            "versions": self.versions,
            "seeds": self.seeds,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls(
            config=data["config"],
            corpus_hash=data["corpus_hash"],
            corpus_files=data.get("corpus_files", 0),
            evalset_id=data["evalset_id"],
            evalset_version=data["evalset_version"],
            evalset_hash=data["evalset_hash"],
            resolution=data["resolution"],
            versions=data["versions"],
            seeds=data.get("seeds", {}),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
        )

    def matches(self, other: Manifest) -> bool:
        return self.hash() == other.hash()


def build_manifest(
    config: Config,
    corpus: Corpus,
    evalset: EvalSet,
    *,
    resolver: SpanResolver | None = None,
    seeds: dict[str, int] | None = None,
    notes: str = "",
) -> Manifest:
    """Record everything about a run that could change its numbers."""
    policy = resolver or SpanResolver()
    return Manifest(
        config=config.as_dict(),
        corpus_hash=corpus.content_hash(),
        corpus_files=len(corpus),
        evalset_id=evalset.id,
        evalset_version=evalset.version,
        evalset_hash=evalset_hash(evalset),
        resolution={"policy": policy.policy.value, "threshold": policy.threshold},
        versions=_versions(),
        seeds=seeds or {},
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )


def evalset_hash(evalset: EvalSet) -> str:
    """A hash of the questions and their evidence.

    The version number alone is not enough: somebody can edit an eval set without bumping it,
    and then two runs claim the same ground truth while using different questions.
    """
    digest = hashlib.sha256()
    for item in sorted(evalset, key=lambda i: i.id):
        digest.update(item.id.encode("utf-8"))
        digest.update(item.question.encode("utf-8"))
        for anchor in item.anchors:
            digest.update(f"{anchor.source_id}:{anchor.quote}:{anchor.grade}".encode())
        for gold in item.gold:
            digest.update(
                f"{gold.span.doc_id}:{gold.span.start}:{gold.span.end}:{gold.grade}".encode()
            )
    return digest.hexdigest()


def diff(before: Manifest, after: Manifest) -> dict[str, tuple[Any, Any]]:
    """Everything that changed between two runs.

    The regression-triage tool. When a metric drops, this names the suspect instead of
    leaving somebody to reconstruct what was different from memory.
    """
    changes: dict[str, tuple[Any, Any]] = {}

    for key, old, new in [
        ("corpus_hash", before.corpus_hash, after.corpus_hash),
        ("evalset_id", before.evalset_id, after.evalset_id),
        ("evalset_version", before.evalset_version, after.evalset_version),
        ("evalset_hash", before.evalset_hash, after.evalset_hash),
    ]:
        if old != new:
            changes[key] = (old, new)

    for section in ("config", "resolution", "versions", "seeds"):
        old_section = getattr(before, section)
        new_section = getattr(after, section)
        for key in sorted(set(old_section) | set(new_section)):
            old_value = old_section.get(key)
            new_value = new_section.get(key)
            if old_value != new_value:
                changes[f"{section}.{key}"] = (old_value, new_value)

    return changes


def explain_diff(before: Manifest, after: Manifest) -> str:
    """The difference between two runs, in plain English."""
    changes = diff(before, after)
    if not changes:
        return (
            "Nothing in the manifest changed, so these two runs should have produced "
            "identical numbers. If they did not, something outside the manifest is affecting "
            "results and that is worth finding."
        )

    lines = [f"{len(changes)} thing(s) changed between these runs:"]
    for key, (old, new) in changes.items():
        lines.append(f"  {key}: {old!r} -> {new!r}")

    if "corpus_hash" in changes:
        lines.append(
            "The corpus itself is different, so nothing else in this list can be blamed for "
            "a change in the numbers until that is accounted for."
        )
    elif "evalset_hash" in changes:
        lines.append(
            "The eval set is different, so the two runs were not measured against the same "
            "ground truth and are not directly comparable."
        )

    return "\n".join(lines)


def _versions() -> dict[str, str]:
    """Everything installed that could change a result."""
    from contextgrid import __version__

    versions = {
        "contextgrid": __version__,
        "python": platform.python_version(),
        "platform": platform.system().lower(),
    }

    for package in ("numpy", "pymupdf", "pdfplumber", "openai", "anthropic"):
        module = sys.modules.get(package)
        version = getattr(module, "__version__", None) if module else None
        if version:
            versions[package] = str(version)

    return versions

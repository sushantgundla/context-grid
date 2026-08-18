"""The run manifest.

"Reproducible" is a claim until something makes it checkable. The manifest is a hash-pinned
record of everything that could change a number: the corpus contents, the eval set and its
version, every plugin and its version, every parameter, the resolution policy, the library
versions, the seeds.

Two properties follow, and both are load-bearing.

**Two runs with the same manifest hash must produce identical numbers.** If they do not,
something outside the manifest is affecting results, and that is a bug worth finding. Read the
other way round, two runs that must produce identical numbers should not be told they differ:
an axis value that names the same run as leaving the axis out is folded before anything is
compared, so `retrieval: simple` and no retrieval at all are one manifest, not two.

**When a metric drops, diff the manifest against the last passing run.** The changed line is
the suspect. That turns regression triage from an investigation into a comparison, and it is
the thing everybody describes and nobody ships.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
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
            # Normalised, so that two spellings of one configuration hash the same. The
            # stored `config` is left exactly as the run wrote it; only the comparison is
            # folded. See `_same_run`.
            "config": _same_run(self.config),
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


#: Axis values that name the same run as leaving the axis out.
#:
#: Every one of these is folded by `grid.matrix._fold`, which is where the claim that they are
#: identical lives. `retrieval: "simple"` is the one `_fold` declines to fold, and it says why
#: in the same breath as saying the two are the same run: folding it would make `simple`
#: unusable as an explicit baseline name, which is how it is mostly written. That is a good
#: reason to keep both spellings in a sweep and no reason at all to call them a difference.
_SAME_AS_UNSET: dict[str, str] = {
    "ingestion": "plain",
    "transform": "none",
    "reranker": "none",
    "retrieval": "simple",
}


def _same_run(config: Mapping[str, Any]) -> dict[str, Any]:
    """One configuration written one way, whichever way it was spelled.

    For comparing only. The manifest keeps recording what the user actually wrote, because
    `manifest.json` is the record of the run and rewriting `simple` to `null` in it would make
    the file disagree with the `winning-config.yaml` written beside it.

    The symptom this exists for: `contextgrid run` over a starter config naming
    `retrieval: [simple]`, and `contextgrid sweep` over the same corpus, wrote manifests that
    diffed as `config.retrieval: 'simple' -> None`. Both ran `SimpleRetrieval`. `Config.label`
    already renders them as one row -- `if self.retrieval and self.retrieval != "simple"` --
    so the leaderboard, the report and the bundle filename all agreed the two were one
    configuration, and `diff` alone said otherwise.
    """
    normalised = dict(config)
    for axis, alias in _SAME_AS_UNSET.items():
        if normalised.get(axis) == alias:
            normalised[axis] = None

    # Candidate depth is only read when something reranks the candidates, so without a
    # reranker it cannot change a number. `_fold` resets it for the same reason, and leaving
    # it out here would let `candidates: 80` with no reranker read as a difference from
    # `candidates: 50` with no reranker -- two runs that do the identical search.
    if normalised.get("reranker") is None:
        normalised["candidates"] = 50

    return normalised


#: The prefix `Results.manifest_note` stamps on a bundle whose sweep did not finish.
PARTIAL_PREFIX = "PARTIAL RUN"


def is_partial(manifest: Manifest) -> bool:
    """Whether this manifest's bundle was written by a sweep that stopped early."""
    return manifest.notes.startswith(PARTIAL_PREFIX)


def diff(before: Manifest, after: Manifest) -> dict[str, tuple[Any, Any]]:
    """Everything that changed between two runs.

    The regression-triage tool. When a metric drops, this names the suspect instead of
    leaving somebody to reconstruct what was different from memory.

    `notes` is compared like anything else, and it is the one field here that is not a
    setting. It carries the `PARTIAL RUN` stamp, and leaving it out meant a complete sweep and
    a sweep the budget cut off after one configuration compared as identical -- `explain_diff`
    then said the two "should have produced identical numbers", about a run that never
    measured most of its matrix. The most important difference between two bundles was the one
    difference this function could not see.
    """
    changes: dict[str, tuple[Any, Any]] = {}

    for key, old, new in [
        ("corpus_hash", before.corpus_hash, after.corpus_hash),
        ("evalset_id", before.evalset_id, after.evalset_id),
        ("evalset_version", before.evalset_version, after.evalset_version),
        ("evalset_hash", before.evalset_hash, after.evalset_hash),
        ("notes", before.notes, after.notes),
    ]:
        if old != new:
            changes[key] = (old, new)

    for section in ("config", "resolution", "versions", "seeds"):
        old_section = getattr(before, section)
        new_section = getattr(after, section)
        # Whether an axis changed is decided on the folded values, the same way `hash()` folds
        # them, so `matches()` and `diff` cannot give two answers about one pair of manifests
        # -- one from the API and the other from `contextgrid diff`.
        #
        # What gets *printed* is what each manifest actually recorded. Deciding and reporting
        # on the folded values would swap the user's own word for the fold's: somebody who
        # swept `retrieval: simple` against `widened` would read
        # `config.retrieval: None -> 'widened'` and go looking for the run that set it to
        # null. There isn't one.
        old_compared = _same_run(old_section) if section == "config" else old_section
        new_compared = _same_run(new_section) if section == "config" else new_section
        for key in sorted(set(old_compared) | set(new_compared)):
            if old_compared.get(key) != new_compared.get(key):
                changes[f"{section}.{key}"] = (old_section.get(key), new_section.get(key))

    return changes


def explain_diff(before: Manifest, after: Manifest) -> str:
    """The difference between two runs, in plain English.

    Opens with the partial-run warning whenever there is one, ahead of the change list,
    because it changes what every line under it is worth. Comparing a finished sweep against
    one that ran a third of its matrix is not a comparison of two experiments, and the reader
    has to know that before they read which axis moved.
    """
    partial = [
        f"the {side} run" for side, m in (("before", before), ("after", after)) if is_partial(m)
    ]
    warning: list[str] = []
    if partial:
        both = len(partial) == 2
        warning = [
            f"**{' and '.join(partial).capitalize()} did not finish.** "
            f"{'Both bundles were' if both else 'That bundle was'} written by a sweep that "
            f"stopped early, so {'neither describes' if both else 'it does not describe'} the "
            "whole matrix. Anything below is a comparison of the winning configurations these "
            "two runs happened to reach, not of the two experiments.",
            "",
        ]
        for side, manifest in (("before", before), ("after", after)):
            if is_partial(manifest):
                warning.insert(-1, f"  {side}: {manifest.notes}")

    changes = diff(before, after)
    if warning and set(changes) <= {"notes"}:
        # Nothing else moved, so there is no change list worth printing under the warning --
        # but "nothing is different" would be exactly the wrong sentence to end on here.
        return "\n".join(
            [
                *warning,
                "Nothing else in these two manifests is different: same winning configuration, "
                "corpus, eval set, resolution policy, versions and seeds.",
            ]
        )

    if not changes:
        # The old wording here said "these two runs should have produced identical numbers",
        # which claims more than a manifest can support. A manifest records the configuration
        # that won, not the sweep that found it, so a 7-configuration `ofat` run and a
        # 27-configuration `factorial` run over the same grid write identical manifests when
        # they pick the same winner. A reader told those runs were the same goes hunting for
        # nondeterminism that is not there. Nothing in the manifest distinguishes them --
        # there is no sweep mode and no configuration count in it -- so the honest fix is to
        # say what was actually compared.
        return (
            "Nothing in these two manifests is different: same winning configuration, corpus, "
            "eval set, resolution policy, versions and seeds. On that evidence the two runs "
            "should have produced identical numbers for this one configuration.\n"
            "\n"
            "That is not the same as the two runs being identical. A manifest records the "
            "configuration that won, not the sweep that found it, so two runs over different "
            "grids -- or over the same grid in different modes -- can both end up here.\n"
            "\n"
            "If this configuration did score differently in the two runs, something outside "
            "the manifest is affecting results and that is worth finding."
        )

    # `notes` is prose, and it is already spelled out in full in the warning above. Repeating
    # a two-sentence PARTIAL RUN stamp inside a list of `axis: old -> new` lines makes the
    # short lines around it unreadable.
    listed = {key: value for key, value in changes.items() if key != "notes"}
    lines = [*warning, f"{len(listed)} thing(s) changed between these runs:"]
    for key, (old, new) in listed.items():
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

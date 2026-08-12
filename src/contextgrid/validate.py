"""Checking the scorer against a published benchmark.

Every number this package produces depends on the span resolver being right, and the resolver
is the one part with no external reference to check against -- nobody else stores ground truth
as character spans, so there is nothing to compare with.

Except LegalBench-RAG, which does exactly that. Its gold is expert-annotated character spans
in known documents, which is the same decision this package makes, so the whole chain can be
run against it and the results compared with the published ones.

That is the difference between "trust me" and "check me", and it costs almost nothing once the
importer exists. If our numbers diverge from the paper's, the problem is ours.

The benchmark is not vendored -- it is large and not ours to redistribute. Point this at a
local copy and it does the rest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextgrid.core.documents import MediaType, SourceFile
from contextgrid.core.errors import ContextGridError
from contextgrid.core.evalset import EvalItem, EvalSet
from contextgrid.corpus import Corpus
from contextgrid.evalset.io import describe_skipped, read_legalbench_rag
from contextgrid.grid.runner import Runner
from contextgrid.pipeline import Config
from contextgrid.score.resolve import SpanResolver


class ValidationError(ContextGridError, ValueError):
    """A validation run could not be set up."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """What we scored, what the paper reported, and whether that is close enough."""

    name: str
    metrics: dict[str, float]
    reference: dict[str, float] = field(default_factory=dict)
    tolerance: float = 0.05
    questions: int = 0
    resolved: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution_rate(self) -> float:
        """How much of the benchmark's gold we could locate at all.

        The first thing to check. A low rate means the corpus text does not match what the
        annotations point into, which invalidates everything downstream -- and it is a
        loading problem, not a retrieval result.
        """
        return self.resolved / self.questions if self.questions else 0.0

    def deviations(self) -> dict[str, float]:
        return {
            metric: self.metrics.get(metric, 0.0) - value
            for metric, value in self.reference.items()
        }

    @property
    def agrees(self) -> bool:
        """True when every compared metric is within tolerance of the published one."""
        if not self.reference:
            return False
        return all(abs(delta) <= self.tolerance for delta in self.deviations().values())

    def report(self) -> str:
        """The comparison, as a table somebody can check."""
        lines = [f"# Validation against {self.name}", ""]
        lines.append(
            f"Resolved {self.resolved} of {self.questions} questions "
            f"({self.resolution_rate:.0%}) to character spans in the corpus."
        )
        lines.append("")

        if not self.reference:
            lines += ["No published numbers were supplied, so this run only reports its own:", ""]
            for metric, value in sorted(self.metrics.items()):
                lines.append(f"- {metric}: {value:.3f}")
            return "\n".join(lines)

        lines += ["| Metric | Ours | Published | Delta |", "|---|---:|---:|---:|"]
        for metric, published in sorted(self.reference.items()):
            ours = self.metrics.get(metric, 0.0)
            lines.append(f"| {metric} | {ours:.3f} | {published:.3f} | {ours - published:+.3f} |")
        lines.append("")

        if self.agrees:
            lines.append(
                f"Every metric is within {self.tolerance:.2f} of the published value, so the "
                "scoring chain reproduces a benchmark it did not define."
            )
        else:
            worst = max(self.deviations().items(), key=lambda item: abs(item[1]))
            lines.append(
                f"**{worst[0]} differs by {worst[1]:+.3f}, outside the {self.tolerance:.2f} "
                "tolerance.** A difference in retrieval configuration explains some of this; "
                "anything left over is a problem with our scoring, not with the benchmark."
            )
        return "\n".join(lines)


def load_benchmark(
    benchmark_path: str | Path,
    corpus_path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[Corpus, EvalSet]:
    """Load a LegalBench-RAG benchmark file and the documents its spans point into.

    The spans are offsets into the raw files, so the corpus is loaded verbatim and read by
    the plain-text parser. Anything that reflows the text would move every offset and turn a
    validation run into a comparison of two different things.
    """
    benchmark = Path(benchmark_path).expanduser()
    documents = Path(corpus_path).expanduser()
    if not benchmark.exists():
        raise ValidationError(f"no benchmark file at {benchmark}")
    if not documents.is_dir():
        raise ValidationError(f"no corpus directory at {documents}")

    evalset = read_legalbench_rag(benchmark)
    if limit is not None:
        evalset = evalset.with_items(evalset.items[:limit])

    # Three different failures that used to share one message, and the shared one was
    # "none of the 0 documents the benchmark refers to were found" -- self-contradicting,
    # and it sent people to check a corpus path when the problem was in the benchmark file.
    if not len(evalset):
        raise ValidationError(_nothing_loaded(benchmark, evalset))

    wanted = {span.doc_id for item in evalset for span in item.gold_spans}
    if not wanted:
        raise ValidationError(_nothing_to_point_at(benchmark, evalset))

    files: list[SourceFile] = []
    for doc_id in sorted(wanted):
        path = documents / doc_id
        if not path.exists():
            continue
        files.append(
            SourceFile(
                id=doc_id,
                media_type=MediaType.TEXT,
                path=str(path),
                raw=path.read_bytes(),
            )
        )

    if not files:
        raise ValidationError(
            f"none of the {len(wanted)} documents the benchmark refers to were found under "
            f"{documents}. The spans are offsets into those exact files, so the corpus has "
            "to be the one the annotations were made against."
        )

    return Corpus(files=tuple(files), name="legalbench-rag"), evalset


def _nothing_loaded(benchmark: Path, evalset: EvalSet) -> str:
    """The benchmark file gave us no questions at all.

    Nothing to do with the corpus, so the message does not mention it. The old one did, and
    sent people off to check a directory that was perfectly fine.
    """
    in_file = int(evalset.meta.get("tests_in_file", 0) or 0)
    if not in_file:
        return (
            f"{benchmark} contains no tests. Expected a JSON object with a `tests` array, or "
            "a bare array of tests, with at least one test in it."
        )
    word = "test" if in_file == 1 else "tests"
    return (
        f"{benchmark} contains no usable tests: all {in_file} {word} in it were skipped. "
        f"{describe_skipped(evalset)} A test needs a non-empty `query` to be loaded at all."
    )


def _nothing_to_point_at(benchmark: Path, evalset: EvalSet) -> str:
    """Questions loaded, but not one of them carries a span to check.

    A different problem from an empty file and a different one again from a corpus that does
    not have the documents, so it gets its own message. Without gold there is nothing to
    resolve, so the run cannot start -- and the reason is almost always that every snippet
    was dropped for a reason worth printing.
    """
    one = len(evalset) == 1
    word = "test" if one else "tests"
    between = "in it" if one else "between them"
    notes = describe_skipped(evalset)
    detail = (
        f" {notes}"
        if notes
        else f" {'It does' if one else 'None of them'} not carry a `snippets` array with a "
        "`file_path` and a `span`."
    )
    return (
        f"{benchmark} loaded {len(evalset)} {word}, but not one usable gold span {between}, "
        f"so there is nothing to check against the corpus.{detail}"
    )


def validate(
    corpus: Corpus,
    evalset: EvalSet,
    *,
    config: Config | None = None,
    reference: Mapping[str, float] | None = None,
    tolerance: float = 0.05,
    name: str = "LegalBench-RAG",
) -> ValidationResult:
    """Score a benchmark with our chain and compare against its published numbers.

    The default configuration is deliberately plain -- the text parser, recursive chunking,
    BM25 -- because the point is to check the *scorer*, not to win the benchmark. A clever
    configuration that beat the paper would prove nothing about whether our metrics are right.
    """
    settings = config or Config(
        parser="text", chunker="recursive:512,overlap=64", embedder=None, index="bm25", k=10
    )

    runner = Runner(corpus=corpus, headline="recall@10")
    result = runner.run_one(settings, evalset)

    return ValidationResult(
        name=name,
        metrics=dict(result.metrics),
        reference=dict(reference or {}),
        tolerance=tolerance,
        questions=len(evalset),
        resolved=result.scored_queries,
        config=settings.as_dict(),
    )


def self_check(corpus: Corpus, evalset: EvalSet) -> dict[str, Any]:
    """Check the span resolver against the benchmark's own annotations, with no retrieval.

    Narrower than a full validation run and more diagnostic. It asks one question: do the
    benchmark's gold spans point at text that exists in the corpus we loaded? If they do not,
    nothing downstream can be trusted and the cause is loading rather than retrieval.

    A snippet naming a file that is not in the corpus is counted separately and left out of
    the rate, because it is a different problem with a different fix. "This document is
    missing" means add the file; "these offsets miss the text" means the corpus is the wrong
    edition of documents you already have. Charging one against the other produced advice
    that was simply wrong -- a four-test benchmark with one absent document reported that the
    corpus was "almost certainly not the one the annotations were made against", when in fact
    three of its three loadable spans were exact. Missing files are reported by name, and
    loudly, so nothing is skipped in silence.

    Anything the import itself dropped -- a snippet with no `file_path`, a span with one
    offset -- is reported here too, on a run that otherwise succeeds. Those drops are
    correct and documented; saying nothing about them is not, because a file whose every
    snippet was discarded scores exactly like a file that was never annotated.
    """
    from contextgrid.parse import get_parser

    parser = get_parser("text")
    parses = {source.id: parser.parse(source) for source in corpus}

    checked = 0
    valid = 0
    empty = 0
    missing = 0
    missing_files: set[str] = set()
    for item in evalset:
        for gold in item.gold:
            parsed = parses.get(gold.doc_id)
            if parsed is None:
                missing += 1
                missing_files.add(gold.doc_id)
                continue
            checked += 1
            if not parsed.document.contains_span(gold.span):
                continue
            valid += 1
            if not parsed.document.slice(gold.span).strip():
                empty += 1

    absent = sorted(missing_files)
    skipped = describe_skipped(evalset)
    return {
        "spans_checked": checked,
        "spans_in_range": valid,
        "spans_empty": empty,
        "spans_missing_file": missing,
        "missing_files": absent,
        "skipped_on_import": skipped,
        "in_range_rate": valid / checked if checked else 0.0,
        "verdict": _self_check_verdict(checked, valid, empty, missing, absent, skipped),
    }


def _self_check_verdict(
    checked: int,
    valid: int,
    empty: int,
    missing: int = 0,
    missing_files: Sequence[str] = (),
    skipped: str = "",
) -> str:
    notice = f"{skipped} " if skipped else ""
    notice += _missing_files_notice(missing, missing_files)

    if checked == 0 and missing:
        return (
            f"{notice}Every gold span in the benchmark points at a file that is not in the "
            "corpus, so there is nothing left to check."
        )
    if checked == 0:
        return "the benchmark contains no gold spans to check"

    rate = valid / checked
    if rate < 0.95:
        return (
            f"{notice}only {rate:.0%} of the benchmark's spans fall inside the documents as "
            "loaded. The corpus is almost certainly not the one the annotations were made "
            "against, and no number from this run means anything until that is fixed."
        )
    if empty:
        return (
            f"{notice}{empty} spans point at whitespace. Either the annotations are off by a "
            "little or the files were re-encoded after they were made."
        )
    return (
        f"{notice}{valid} of {checked} spans point at real text in the documents as loaded, "
        "so the benchmark and the corpus agree and the scoring chain has something valid to "
        "run on."
    )


def _missing_files_notice(missing: int, missing_files: Sequence[str]) -> str:
    """The line that goes in front of every verdict when documents are absent.

    In front, not appended, because the CLI prints the verdict and then decides whether to
    carry on -- and a skipped document has to be read before the number it is not part of.
    """
    if not missing:
        return ""
    names = ", ".join(missing_files[:5])
    if len(missing_files) > 5:
        names += f", and {len(missing_files) - 5} more"
    span_word = "span" if missing == 1 else "spans"
    one_file = len(missing_files) == 1
    file_phrase = "1 file that is not" if one_file else f"{len(missing_files)} files that are not"
    return (
        f"Skipped {missing} {span_word} pointing at {file_phrase} in the corpus: {names}. "
        f"Add {'it' if one_file else 'them'}, or drop those tests -- they are not counted "
        "either way. Of the rest: "
    )


def resolution_report(
    evalset: EvalSet, chunks: Sequence[Any], resolver: SpanResolver | None = None
) -> dict[str, Any]:
    """How much of the gold survived being resolved to chunks, and how.

    Separates the two ways a validation run can go wrong: gold that does not match the
    corpus, and gold that matches but falls between chunk boundaries.
    """
    policy = resolver or SpanResolver()
    resolutions, _ = policy.resolve(evalset, chunks)

    reachable = sum(1 for resolution in resolutions.values() if resolution.labels)
    split = sum(
        1 for resolution in resolutions.values() for gold in resolution.per_gold if gold.is_split
    )

    return {
        "questions": len(evalset),
        "resolved": reachable,
        "split_gold": split,
        "resolution_rate": reachable / len(evalset) if len(evalset) else 0.0,
        "policy": policy.policy.value,
        "threshold": policy.threshold,
    }


def items_without_gold(evalset: EvalSet) -> list[EvalItem]:
    """Benchmark questions that carry no spans at all. Worth counting before running."""
    return [item for item in evalset if not item.gold]

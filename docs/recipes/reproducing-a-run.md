# Reproducing a run

## The question

"Reproducible" is a claim until something makes it checkable. Two questions this recipe answers
for real, on disk: does rerunning the same config actually give the same numbers, and when a
number *does* change, can you name exactly what caused it?

## The setup

A tiny corpus and eval set — the same shape as [getting-started.md](../guide/getting-started.md)
— and two configs that differ in exactly one place, the chunker:

```
documents/refunds.md      "Refunds are issued within 45 days of purchase..."
documents/shipping.md     "Express orders arrive the next business day..."
questions.jsonl           3 questions, quoted evidence in each doc
config-a.yaml             chunker: [recursive:256]
config-b.yaml             chunker: [sentence:2]
```

## The command

```bash
.venv/bin/contextgrid run config-a.yaml
.venv/bin/contextgrid run config-b.yaml
cat results-a/manifest.json
```

## The real output

```json
{
  "manifest_hash": "0954f1472ad505b07dc3616c90359106927c43a3f0f055f0e4b8a26f1d4e8cb9",
  "config": {
    "ingestion": null, "parser": "markdown", "chunker": "recursive:256",
    "embedder": "tfidf", "index": "dense", "transform": null,
    "retrieval": null, "reranker": null, "k": 5, "candidates": 50
  },
  "corpus_hash": "ee52e863fe5961642bf9ece5e1c8de10e83fcb12cbc06758da98f6c4761dd64b",
  "corpus_files": 2,
  "evalset_id": "policy-questions",
  "evalset_version": 1,
  "evalset_hash": "e33ddae2a06fe4ea14a13f05bf4820d8b0dbbb17ad0592aad7cdf4d3583a93df",
  "resolution": {"policy": "coverage", "threshold": 0.5},
  "versions": {"contextgrid": "0.9.0", "python": "3.13.5", "platform": "darwin", "numpy": "2.4.6"},
  "seeds": {"run": 0},
  "created_at": "2026-08-06T04:27:36+00:00",
  "notes": ""
}
```

## What's in the manifest, and why each piece is there

`Manifest.hash()` covers everything that could change a number: the config, the corpus's content
hash, the eval set's id *and* a hash of its actual questions and evidence (not just the version —
someone can edit an eval set without bumping the version number, and the hash catches that
anyway), the resolution policy, and every installed library version that could matter (`numpy`,
and `pymupdf`/`pdfplumber`/`openai`/`anthropic` when present). `created_at` and `notes` are
recorded but deliberately **excluded** from the hash — including them would make every run's
manifest unique by definition, which defeats the entire point of a hash you can compare.

## Proving reproducibility, not just claiming it

Rerun `config-a.yaml` into a second output directory and compare hashes:

```bash
.venv/bin/contextgrid run config-a.yaml --quiet   # (into ./results-a2, config copied and edited)
python3 -c "
import json
a  = json.load(open('results-a/manifest.json'))['manifest_hash']
a2 = json.load(open('results-a2/manifest.json'))['manifest_hash']
print('same hash:', a == a2)
"
```

```
same hash: True
```

That's the property that makes "reproducible" a checkable claim rather than a hope: two runs of
the same config against the same corpus produced **the same hash**, and — because context-grid's
core scoring is deterministic — the same numbers. If a rerun ever disagrees with its own
manifest hash, that's a real bug worth finding, not noise to average away.

## `contextgrid diff`: naming the suspect when a number changes

This is the part that turns regression triage from an investigation into a comparison. Two
manifests in, a plain-English answer out — `contextgrid diff <before> <after>`:

**Only the chunker changed:**

```bash
.venv/bin/contextgrid diff results-a/manifest.json results-b/manifest.json
```
```
1 thing(s) changed between these runs:
  config.chunker: 'recursive:256' -> 'sentence:2'
```

**The corpus itself changed** (edited `refunds.md`'s notice period from 30 to 45 days, and the
eval set's gold quote to match):

```bash
.venv/bin/contextgrid diff results-a/manifest.json results-c/manifest.json
```
```
2 thing(s) changed between these runs:
  corpus_hash: 'ee52e863fe5961642bf9ece5e1c8de10e83fcb12cbc06758da98f6c4761dd64b' -> '1c016dc787665cc7733aa875083c8f848cdef39ac45597d385a9d984de37f9b9'
  evalset_hash: 'a688bfc2a351f2c6b30fbdae0efd45c0aa11cd0e8406caad067d2fbc86d0940d' -> 'c896f5ec2bee6c866c54165e94d7ef6b24e7de9cbddac1e22d6d108272d4ad74'
The corpus itself is different, so nothing else in this list can be blamed for a change in the numbers until that is accounted for.
```

That last line matters: `explain_diff` (in `contextgrid/report/manifest.py`) deliberately calls
out `corpus_hash` and `evalset_hash` changes as disqualifying — if the ground truth moved, every
other line in the diff is a red herring until the ground-truth change is accounted for first.

## Seeds: what `run.seed` actually controls, proven not asserted

`run.seed` is recorded into the manifest — `"seeds": {"run": N}` — and it drives every resample
a sweep does: the bootstrap confidence interval on each leaderboard row
(`RunResult.interval()`) and the paired significance test between the top two configurations
(`Results.is_the_winner_real()`, `Results.significance()`). Neither takes a hidden `seed=0` of
its own any more — both fall back to the seed the run itself was configured with.

**Same config, same seed, run twice:**

```bash
.venv/bin/contextgrid run config-x.yaml --quiet   # run.seed: 7, ./results-x
.venv/bin/contextgrid run config-y.yaml --quiet   # identical config, run.seed: 7, ./results-y
diff out-x.txt out-y.txt
```

```
(no output -- the two runs' printed leaderboard, summary and significance verdict are
byte-identical, down to the confidence interval)

seed-demo: 1 × 1 × 2 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = 2 on paper, 2 to run in factorial mode, scored on recall@5

configuration                            recall@5   p95 ms     $/1k
--------------------------------------------------------------------
markdown · recursive:256 · tfidf · dense    1.000      0.1   0.0000
markdown · sentence:2 · tfidf · dense       1.000      0.0   0.0000

markdown · recursive:256 · tfidf · dense scored best on recall@5 at 1.000, across 2 configurations
on 3 questions. markdown · recursive:256 · tfidf · dense and markdown · sentence:2 · tfidf · dense
are not distinguishable on this eval set (n=3). ...

wrote 3 files to ./results-x
```

```python
import json
print(json.load(open("results-x/manifest.json"))["seeds"])   # {"run": 7}
```

That's the reproducibility claim, checked rather than assumed: two independent `contextgrid run`
processes, same config, same `seed: 7`, produced the identical significance verdict — and the
manifest's recorded seed (7) is the one that was actually used.

**On this 3-question corpus the two configs simply tie, so there's nothing for the seed to
flip.** To see the seed actually change the *call*, not just reproduce it, construct a case
where the paired test sits right on the `alpha=0.05` boundary — the situation the seed has to
control for the manifest's claim to mean anything:

```python
from contextgrid.report.results import Results, RunResult
from contextgrid.pipeline import Config

left = {f"q{i}": 1.0 for i in range(24)}
right = {**left, **{f"q{i}": 0.0 for i in range(5)}}   # a small, real, paired gap

left_run = RunResult(config=Config(chunker="recursive:512"), per_query=left)
right_run = RunResult(config=Config(chunker="sentence:3"), per_query=right)

for seed in (0, 0, 1):
    verdict = Results(runs=[left_run, right_run], seed=seed).is_the_winner_real("recall@5")
    print(f"seed={seed}  distinguishable={verdict.distinguishable}  p={verdict.p_value:.4f}")
```

```
seed=0  distinguishable=True  p=0.0500
seed=0  distinguishable=True  p=0.0500
seed=1  distinguishable=False  p=0.0625
```

Two calls with `seed=0` agree exactly (reproducibility), and `seed=1` on the *identical* data
gives a different answer (the seed is real, not decorative) — because a paired randomisation
test at 2,000 resamples has its own Monte Carlo noise, and a comparison this close to the
boundary is exactly where that noise can decide the verdict. That is also the honest reading of
"not distinguishable" results elsewhere in these recipes: a gap reported as borderline is worth
rerunning with the manifest's own seed before trusting either side of the call.

## What would change the answer

- **A model-backed stage** (`agentic` retrieval, a `transform`, an LLM-judged generation metric)
  makes reproducibility genuinely harder: the manifest records the model name and the config, but
  not the provider's own version drift — a hosted model can change behavior under a fixed name.
  `versions` records the *library* versions (litellm, etc.), which is the part that's actually
  pinnable.
- **A PDF parser that isn't process-isolated.** `docs/adoption-backlog.md` documents a real case
  — `pymupdf4llm` carrying hidden state across documents *within one process* — where two runs
  of the identical manifest could still disagree, because the non-determinism lived outside
  anything the manifest could see. That's exactly the "if they did not, something outside the
  manifest is affecting results" case `explain_diff`'s own message warns about, and it's why that
  parser now runs each document in its own subprocess.
- **A per-plugin seed that isn't `run.seed`.** A few axis values carry their *own* seed as a
  spec parameter — `hash:512,seed=3` (the hashing salt), `quantized:pq,seed=3` (k-means training)
  — and neither currently reads `run.seed` at all; each defaults to `0` independently, the same
  way significance and the confidence interval used to. Worth knowing plainly: this is the same
  class of gap `run.seed` just got fixed for, one level down, in axes that build the index rather
  than in the layer that reads the results. It still lands in the manifest, under `config.*`
  rather than `seeds.run`, and `contextgrid diff` will name it like any other config change.

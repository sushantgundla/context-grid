# Generation

Everything before this dimension is scored on whether the *right passages came back*. This one
asks the question they were retrieved *for*: is the answer any good, and is it actually
supported by what was retrieved? Those are different failures, and conflating them is how a
retrieval problem gets misdiagnosed as a prompting problem for a fortnight — a configuration can
retrieve perfectly and still generate a confident falsehood, or retrieve badly and be saved by a
model that says it doesn't know.

Retrieval stays the default view of this tool for a reason: generation noise swamps retrieval
signal, so a sweep judged purely on answer quality mostly measures the generator, not the
retriever. But retrieval is a means, not the goal — a tool that never checks whether its gains
survive to the answer is asking to be trusted about the one thing it never measured. That check
is the **lift** question: does +0.10 recall@5 turn into a better answer, or does the generator
find it either way?

Set it with `grid.generator` — the tenth axis, last in `AXIS_ORDER`, because it operates on
whatever reranking produced. `None` (`null` in YAML) is the default and means no generation at
all: no assembly, no model call, no cost, exactly what every config meant before this axis
existed. Set it to `extractive` or `llm` and `Runner.run_one` answers every question, scores
the answers, and — when `run.model` is set and `deepeval` is installed — folds `faithfulness`
and `answer_relevancy` into the same metrics dict every other axis reports into, which is what
lets `DIMENSION_METRICS["generation"]` in
[`report/composite.py`](../../src/contextgrid/report/composite.py) find them. See
[rerankers](rerankers.md) for what builds the context this layer receives, and
[metrics](../scoring/metrics.md) for how retrieval-side scores fold into a leaderboard.

## The axis itself

`Config.generator: str | None = None` is the last field on `Config`, for the same reason
`ingestion` is: `Config("markdown", "recursive:512,overlap=64", "tfidf", "dense")` is public
API, and a new field ahead of an existing one would silently shift every positional argument
anybody has already written. It shows up in `Config.label` only when set:

```python
>>> from contextgrid.pipeline import Config
>>> Config().label
'markdown · recursive:512 · tfidf · dense'
>>> Config(generator="extractive").label
'markdown · recursive:512 · tfidf · dense · ->extractive'
```

`extractive` needs no model and is in the ordinary plugin registry. `llm` is not, for the same
reason `hyde` and the rest of [transforms'](transforms.md) `MODEL_BACKED` are not: a generator
built with no model would have nothing to generate with, and a config that looks like it is
testing an LLM generator while testing nothing is worse than an error.

```python
>>> from contextgrid.generate import get_generator, available_generators, MODEL_BACKED
>>> available_generators()
('extractive', 'llm')
>>> MODEL_BACKED
('llm',)
>>> get_generator("llm")
Traceback (most recent call last):
    ...
contextgrid.evalset.llm.LLMError: the 'llm' generator needs a model. Pass one to `get_generator`, or use one of the model-free generators: extractive
```

`None` is different from every other axis's `None`. `NoTransform` and `NoReranker` are real
plugin instances — the identity, still built, still run. Generation has nothing to be the
identity *of*, so `get_generator(None)` returns `None` outright, and the pipeline never
assembles a context or calls a generator at all:

```python
>>> get_generator(None) is None
True
```

In a config file, the model comes from `run.model`, same as everywhere else — one name, one
key, one price, shared with transforms, agentic retrieval, LLM-backed ingestion, and the
generation judge below:

```yaml
grid:
  reranker: [null, lexical]
  generator: [null, extractive, llm]

run:
  model: openai:gpt-4o-mini
```

It sweeps like any other axis:

```python
>>> from contextgrid.grid import matrix
>>> configs = matrix(chunker=["a", "b"], generator=["extractive", "llm"]).expand("factorial")
>>> len(configs)
4
>>> sorted({c.generator for c in configs})
['extractive', 'llm']
```

`generator` is last in `AXIS_ORDER` (`contextgrid.grid.matrix`), after `candidates`, because it
runs on whatever reranking produced — the same reasoning that puts `reranker` after `retrieval`
and `retrieval` after `transform`.

### The most expensive axis on the grid

A model call per question, for the `llm` generator, on every configuration a sweep tries. Set
`run.budget_usd` or `run.budget_seconds` or the runner says so up front, the same warning an
unbounded `AgenticRetrieval` sweep already gets:

```python
>>> from contextgrid.grid import matrix
>>> from contextgrid.grid.runner import Budget, _warn_if_unbounded
>>> grid = matrix(generator="llm")
>>> _warn_if_unbounded(grid, Budget())
>>> grid.meta["unbounded_model_calls"]
'llm'
>>> grid.meta.clear()
>>> _warn_if_unbounded(grid, Budget(usd=5.0))
>>> "unbounded_model_calls" in grid.meta
False
```

### What actually happens per question

`BuiltPipeline.answer` is retrieval's `search` handed on to assembly and generation — it takes
chunk ids `search` already found, not a fresh round trip through the index:

```python
>>> from contextgrid.core.documents import MediaType
>>> from contextgrid.corpus import Corpus
>>> from contextgrid.evalset.llm import RecordingLLM
>>> from contextgrid.pipeline import build
>>> doc = "Either party may terminate this agreement for convenience by giving thirty days written notice."
>>> corpus = Corpus.from_texts({"contract.md": doc}, media_type=MediaType.MARKDOWN)
>>> llm = RecordingLLM(replies=["Thirty days written notice is required [1]."])
>>> pipeline = build(Config(generator="llm"), corpus, llm=llm)
>>> question = "How much notice is needed to terminate for convenience?"
>>> chunk_ids = pipeline.search(question)
>>> answer, context = pipeline.answer(question, chunk_ids)
>>> answer.text
'Thirty days written notice is required [1].'
```

`Runner.run_one` does this for every question in the eval set, scores each answer, and folds
the result into the same `metrics` dict every other axis reports into -- see
[the DeepEval section](#deepeval-backed-generation-metrics) below for the full, tested
end-to-end example, including where `faithfulness` and `answer_relevancy` come from.

## From retrieved chunks to a prompt: `ContextAssembler`

Nothing else in the grid sweeps the layer between "the retriever returned these chunks" and
"the model answered." `contextgrid.assemble.context.ContextAssembler` does three things to that
gap, each one a real, measurable decision:

**Order** (`ordering: Ordering`). Long-context models attend to the start and end of their
context far more reliably than the middle — a positional bias baked into how rotary embeddings
decay. Putting the best evidence at the edges costs nothing and measurably changes answers.

| value | effect |
|---|---|
| `Ordering.RELEVANCE` (default) | best first — the obvious choice, and it buries later evidence in the middle |
| `Ordering.ENDS` | best first, second-best last, working inward — the "lost in the middle" mitigation |
| `Ordering.DOCUMENT` | original reading order, ignoring rank — preserves narrative flow |
| `Ordering.REVERSED` | worst first, best last — puts the strongest evidence closest to the question when the question follows the context |

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.span import Span
>>> from contextgrid.assemble.context import ContextAssembler, Ordering
>>> chunks = [   # already ranked best to worst by retrieval
...     Chunk(id=f"doc:{i}", span=Span("doc", i * 100, i * 100 + len(t)), text=t)
...     for i, t in enumerate([
...         "Best evidence.", "Second best evidence.", "Third best evidence.", "Weakest evidence.",
...     ])
... ]
>>> ordered = ContextAssembler(ordering=Ordering.ENDS).assemble(chunks)
>>> [c.id for c in ordered.chunks]
['doc:0', 'doc:2', 'doc:3', 'doc:1']
```

**Budget** (`budget_tokens: int | None`). `k` is a poor proxy for what the generator actually
pays for — five structural chunks can be four times the text of five sentence windows. When a
budget is set, chunks are kept in rank order until it runs out; the *last* chunk that would
overflow it is dropped whole rather than truncated, because half a passage reads to the model
like a complete one, and a model handed half an answer will confidently answer with half of it:

```python
>>> tight = ContextAssembler(budget_tokens=5).assemble(chunks)
>>> tight.used, tight.dropped
(1, 3)
```

**Deduplication** (`deduplicate: bool = True`). Overlapping chunks pay twice for the same
sentence — a top-5 that's three copies of one paragraph fills the window with one fact.
`ContextAssembler` drops any chunk whose text is already fully covered by a higher-ranked one,
and records the character count it removed:

```python
>>> overlapping = [
...     Chunk(id="doc:0", span=Span("doc", 0, 30), text="Refunds are issued within thirty days."),
...     Chunk(id="doc:1", span=Span("doc", 5, 20), text="issued within"),  # fully inside doc:0
... ]
>>> deduped = ContextAssembler().assemble(overlapping)
>>> deduped.used, deduped.duplicate_characters
(1, 15)
```

None of this changes what was *retrieved* — it can't rescue a bad retriever — but it routinely
decides whether a good one produces the right answer. Other fields worth knowing: `include_source`
(default `True`) prefixes each chunk with `[1] doc-id` so the model has something to cite;
`separator` (default `"\n\n---\n\n"`) joins the rendered chunks; `tokens_sent()` reports what a
given retrieval will cost the generator, independent of `k`.

## Generators

```python
>>> from contextgrid.generate import LLMGenerator, ExtractiveGenerator
```

| name | spec | needs a model | what it returns |
|---|---|---|---|
| `extractive` | `extractive` or `extractive:3` | no | the top passage, verbatim |
| `llm` | `llm` (via `grid.generator` + `run.model`, or `get_generator("llm", llm)` directly) | yes | a generated answer, with citations parsed out |

### `extractive` — `contextgrid.generate.ExtractiveGenerator`

Returns the highest-ranked passage verbatim, trimmed to its first few sentences. Not a
generator in any useful sense — that's the point. It's the *ceiling retrieval alone can reach*.
Scoring answer quality against it separates "the retriever found the evidence" from "the
generator did something useful with it," which is exactly the distinction the lift chart needs.

| parameter | default | meaning |
|---|---|---|
| `sentences` | `2` | how many leading sentences of the top chunk to return |

### `llm` — `contextgrid.generate.LLMGenerator`

Answers with a model, using `DEFAULT_PROMPT` — itself a sweepable axis, since prompt changes
routinely beat retrieval changes. That's an uncomfortable result and a useful one: worth knowing
before a quarter goes into an embedding migration that a better prompt would have matched.

| parameter | default | meaning |
|---|---|---|
| `llm` | — | required |
| `prompt` | `DEFAULT_PROMPT` | must contain `{context}` and `{question}` |
| `max_tokens` | `400` | cap on the answer |

`DEFAULT_PROMPT` tells the model to cite passages as `[1]`, `[2]`, ... and to say plainly when
the passages don't contain the answer rather than guess. `Answer.citations` is parsed straight
out of the reply with a `\[(\d+)\]` regex.

```python
>>> from contextgrid.core.documents import Chunk
>>> from contextgrid.core.span import Span
>>> from contextgrid.assemble.context import ContextAssembler
>>> from contextgrid.evalset.llm import RecordingLLM
>>> from contextgrid.generate import LLMGenerator
>>> chunks = [Chunk(id="doc:0", span=Span("doc", 0, 51), text="Refunds are issued within thirty days of purchase.")]
>>> ctx = ContextAssembler().assemble(chunks)
>>> gen = LLMGenerator(llm=RecordingLLM(replies=["Refunds take thirty days [1]."]))
>>> answer = gen.answer("How long do refunds take?", ctx)
>>> answer.text, answer.citations
('Refunds take thirty days [1].', (1,))
```

## Scoring an answer without a second model: `score_answer`

`contextgrid.generate.score_answer` is deliberately lexical rather than LLM-judged. An LLM
judge is more sensitive, but it puts a second model with its own unmeasured biases into a tool
whose entire premise is that unmeasured assumptions are the problem. These checks are coarser,
and they're checkable:

- **`groundedness`** — the fraction of the answer's content words that also appear in the
  context it was given. Words in the answer but not the context are either invention or general
  knowledge; both are reasons to trust the answer less.
- **`citation_accuracy`** — the fraction of cited passage numbers (`[1]`, `[2]`, ...) that were
  actually in the context. `None` if the answer cited nothing.
- **`evidence_overlap`** — overlap between the answer's words and the *gold* evidence's words,
  when gold chunks are available.
- **`abstained`** — whether `Answer.is_abstention` matched a refusal phrase ("I don't know",
  "not enough information," ...). Deliberately broad: a false positive mislabels one abstention,
  a false negative hides the failure mode entirely.
- **`should_have_abstained`** — true when the item's gold is empty or the context has no
  chunks at all.
- **`abstention_correct`** — `abstained == should_have_abstained`. Scored as a success either
  way: a system that declines when the corpus genuinely can't support an answer is behaving
  correctly, and marking that a zero teaches exactly the wrong lesson.

```python
>>> from contextgrid.core.evalset import EvalItem, GoldSpan
>>> from contextgrid.generate import score_answer, Answer, GenerationReport
>>> item = EvalItem(id="q1", question="How long do refunds take?", gold=(GoldSpan(chunks[0].span),))
>>> score = score_answer(item, answer, ctx)
>>> round(score.groundedness, 2), score.citation_accuracy
(0.8, 1.0)
```

`GenerationReport` aggregates scores across a whole eval set. `confident_when_it_should_not_be`
is the failure worth naming: the list of question ids the corpus could not support, which the
model answered anyway.

```python
>>> unanswerable = EvalItem(id="q2", question="What is the CEO's phone number?")  # no gold
>>> declined = Answer(text="The passages do not contain the answer.")
>>> s2 = score_answer(unanswerable, declined, ctx)  # correctly declines: abstention_correct
>>>
>>> overconfident = EvalItem(id="q3", question="What is the CEO's phone number?")  # no gold either
>>> guessed = Answer(text="The CEO's number is 555-0100.")
>>> s3 = score_answer(overconfident, guessed, ctx)  # answers anyway: the failure worth naming
>>>
>>> report = GenerationReport(scores=[score, s2, s3], generator="llm")
>>> report.metrics()
{'groundedness': 0.26666666666666666, 'citation_accuracy': 1.0, 'evidence_overlap': 0.0, 'abstention_accuracy': 0.6666666666666666}
>>> report.confident_when_it_should_not_be
['q3']
>>> report.summary()
"llm answered 3 questions. 27% of the average answer's words came from the context it was given. 1 question(s) had no supporting evidence in the retrieved context and were answered anyway. That is worse than a lower score with a refusal, and no retrieval metric shows it."
```

### `lift` — did the retrieval gain survive to the answer?

`contextgrid.generate.lift(retrieval_score, answer_score, baseline_answer)` is the question this
whole project implicitly promises to answer and that nothing else in the field plots:

```python
>>> from contextgrid.generate import lift
>>> lift(retrieval_score=0.62, answer_score=0.71, baseline_answer=0.70)
'Retrieval scored 0.620 and answer quality rose +0.010. The retrieval gain survived to the answer.'
```

Three outcomes: unchanged (`|gain| < 0.01`, the generator was finding the answer either way, so
the retrieval gain bought nothing), risen (the gain survived), or fallen (better retrieval
producing *worse* answers — usually a sign of more context rather than better context, worth
checking character precision on before trusting the retrieval number at all).

## DeepEval-backed generation metrics

`contextgrid.generate.GenerationJudge` scores generated answers using
[DeepEval](https://github.com/confident-ai/deepeval), for the questions the lexical checks
above can't reach — is this answer actually faithful to the retrieved text, and is it relevant
to the question at all. Needs `pip install "context-grid[judge]"` (`deepeval>=4.0`).

### Automatic, through the `generator` axis

`Runner.run_one` builds this judge itself — `Runner._generation_judge()` — whenever `run.model`
is set and `deepeval` imports cleanly, and scores every generated answer with it alongside the
lexical checks above. No wiring beyond the config:

```python
>>> from contextgrid.core.documents import MediaType
>>> from contextgrid.core.evalset import EvalItem, EvalSet, GoldAnchor
>>> from contextgrid.corpus import Corpus
>>> from contextgrid.evalset.llm import RecordingLLM
>>> from contextgrid.grid.runner import Runner
>>> from contextgrid.pipeline import Config
>>> doc = "Either party may terminate this agreement for convenience by giving thirty days written notice."
>>> corpus = Corpus.from_texts({"contract.md": doc}, media_type=MediaType.MARKDOWN)
>>> evalset = EvalSet(id="es", items=(
...     EvalItem(
...         id="q1",
...         question="How much notice is needed to terminate for convenience?",
...         anchors=(GoldAnchor(source_id="contract.md", quote="thirty days"),),
...     ),
... ))
>>> judge_json = ('{"truths": ["Thirty days notice is required."], '
...     '"claims": ["Thirty days notice is required."], '
...     '"statements": ["Thirty days notice is required."], '
...     '"verdicts": [{"verdict": "yes", "reason": "supported"}], "reason": "ok"}')
>>> llm = RecordingLLM(replies=["Thirty days written notice is required [1]."], default=judge_json)
>>> runner = Runner(corpus=corpus, headline="recall@5", llm=llm)
>>> result = runner.run_one(Config(generator="llm"), evalset)
>>> result.metrics["faithfulness"], result.metrics["answer_relevancy"]
(1.0, 1.0)
```

The first scripted reply answers the question; every call after that -- the judge's -- gets the
`default` reply, which is DeepEval's own trick for testing without a key
(`tests/unit/test_generation_metrics.py`'s `ScriptedJudge` does the same thing). Those two
metrics are exactly what `report/composite.py` needs:

```python
>>> from contextgrid.report.composite import composite
>>> composite(result.metrics).dimensions
('generation', 'retrieval')
```

A missing `[judge]` extra, or no `run.model` at all, degrades gracefully: generation still
runs, the lexical scores (`groundedness`, `citation_accuracy`, ...) still land in `metrics`,
and `faithfulness`/`answer_relevancy` are simply absent -- `composite()` then reports
`generation` as `missing` rather than guessing.

### Why DeepEval and not four in-house prompts

Writing four prompts is the easy part; agreeing what "faithful" *means* is the hard part, and
DeepEval's definitions are ones a reader can look up and argue with independently of this
project. Four metric names that mean something published beat four that mean whatever this
package decided on its own.

### The four metrics, chosen because they fail differently

```python
>>> from contextgrid.generate import available_generation_metrics
>>> available_generation_metrics()
('answer_relevancy', 'contextual_recall', 'contextual_relevancy', 'faithfulness')
```

| metric | DeepEval class | needs a reference answer | catches |
|---|---|---|---|
| `faithfulness` | `FaithfulnessMetric` | no | the hallucination check — is every claim in the answer supported by what was retrieved. The only one of the four usable on a corpus nobody has written answers for. |
| `answer_relevancy` | `AnswerRelevancyMetric` | no | does the answer address the question, rather than being true and beside the point |
| `contextual_relevancy` | `ContextualRelevancyMetric` | no | were the retrieved passages relevant to the question — a generation-time view of a *retrieval* failure, and the one that tells you which half of the pipeline to go fix |
| `contextual_recall` | `ContextualRecallMetric` | **yes** | did the retrieved passages contain what the reference answer needed |

Not all fifty metrics DeepEval offers — these four, deliberately, because between them they
catch a wrong answer, an unsupported answer, an evasive answer, and a retrieval problem wearing
a generation problem's clothes. `contextual_recall` is skipped (recorded in `JudgedAnswer.failed`,
not raised) on any item with no reference answer, rather than silently scoring zero — zero would
read as "the context contained nothing useful," a claim about the retriever that isn't what
actually happened; the truth is nobody wrote a reference for that question.

### One model, one key, one budget

DeepEval reaches for its own OpenAI configuration by default. Left alone, that's a second,
unpriced model call landing in the middle of a package whose whole argument is that cost belongs
on the chart. `GenerationJudge` instead wraps whatever model `run.model` already chose — the
same model doing generation — so `openai:gpt-4o-mini` in the YAML is the judge too, its calls
are counted, and `run.budget_usd` still means what it says.

The judge is never the model under test unless you explicitly point it at itself: a model
grading its own answers scores them generously, and the effect is largest on exactly the answers
most worth doubting.

```yaml
run:
  model: openai:gpt-4o-mini   # the generator, and — via GenerationJudge — the judge
```

| parameter | default | meaning |
|---|---|---|
| `llm` | — | required; wrapped via `build_judge` into what DeepEval expects |
| `metrics` | `("faithfulness", "answer_relevancy")` | which of the four to run |
| `threshold` | `0.5` | DeepEval's pass/fail cutoff per metric |

`async_mode` is always off. DeepEval defaults to running its judges concurrently, which is
faster but makes the order of model calls non-deterministic — and a sweep whose numbers move
between identical runs is a sweep nobody can trust to compare anything against.

### Running it with no key and no network

`tests/unit/test_generation_metrics.py` scripts a fake judge that answers DeepEval's prompts in
the shape it expects — real DeepEval code, real scores, nothing on the network:

```python
>>> from contextgrid.generate import GenerationJudge
>>>
>>> class ScriptedJudge:
...     name = "scripted"
...     def __init__(self):
...         self.calls = 0
...     def complete(self, prompt, *, max_tokens=512):
...         self.calls += 1
...         return ('{"truths": ["Refunds take 30 days."], "claims": ["Refunds take 30 days."],'
...                 ' "statements": ["Refunds take 30 days."],'
...                 ' "verdicts": [{"verdict": "yes", "reason": "supported"}], "reason": "ok"}')
...
>>> judge_llm = ScriptedJudge()
>>> judge = GenerationJudge(llm=judge_llm, metrics=("faithfulness", "answer_relevancy"))
>>> result = judge.score(
...     query_id="q1",
...     question="How long do refunds take?",
...     answer="Refunds take thirty days.",
...     contexts=["Refunds are issued within thirty days of purchase."],
... )
>>> result.scores
{'faithfulness': 1.0, 'answer_relevancy': 1.0}
>>> result.model_calls, judge_llm.calls
(7, 7)
```

`result.model_calls` and the judge's own call count agree — `GenerationJudge` counts every
call made while scoring one answer, because a judge grading a thousand answers is a real
expense this package refuses to leave off the chart. A metric that raises (a judge model
refusing an awkward question, a malformed reply) is recorded in `result.failed[name]` and
skipped, not raised — one bad question must not discard the other nine hundred answers the judge
graded perfectly well.

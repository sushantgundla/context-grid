# Measuring honestly

How to capture what the tool actually did, and how to tell a real disagreement from an
artefact of how you looked. Every trap below has produced a wrong finding in a real drive.

- [The capture traps](#the-capture-traps)
- [Judging a disagreement](#judging-a-disagreement)
- [Staying out of the repo](#staying-out-of-the-repo)
- [Working alongside other drives](#working-alongside-other-drives)

## The capture traps

### Piping reorders stdout against stderr

A pipe block-buffers stdout; stderr stays unbuffered. Commands that write a summary to one
and errors to the other arrive in a different order than a terminal shows.

```bash
contextgrid check both.yaml 2>&1 | head      # errors first -- an artefact
script -q /dev/null contextgrid check both.yaml   # what a terminal shows
```

A whole finding was once raised and withdrawn on this. **Never claim an ordering bug from
piped output.**

If `script` is unavailable in the environment — some sandboxes return
`script: tcgetattr/ioctl: Operation not supported on socket` — then make **no ordering claim
at all** in the report, and say so in the coverage section. Silence is correct; a guess is not.

### `$?` after a pipeline is the wrong exit code

```bash
contextgrid check m.yaml 2>&1 | tail -2 ; echo $?   # tail's exit code. Always 0.
contextgrid check m.yaml >/dev/null 2>&1 ; echo $?  # the tool's exit code
```

This produced a "still broken" report on a bug that was already fixed. Run the command bare
when the exit code is the claim.

### A warm cache hides a fix, and a stale one invents a bug

Content-addressed caches key on a version field. If a fix changes what a component produces
without bumping that field, a cached run serves the old output and the fix looks absent.
Conversely, numbers captured while another process was mid-edit are meaningless.

When a number moves between runs, before reporting it: re-run in a clean directory, and
confirm nothing else is writing to the tree.

### Determinism must be tested across processes

Anything built on Python's `hash()` is salted per process and stable *within* one. A check
that runs in a single process passes against the bug.

```bash
for s in 0 1 2 12345; do PYTHONHASHSEED=$s <command>; done
```

Four identical outputs, or it is not deterministic.

## Judging a disagreement

**A finding is a disagreement between the docs and the tool. It is not automatically a code
bug.** Roughly a fifth of real findings were resolved by changing the documentation, because
the documented behaviour was the worse behaviour.

Ask which half is wrong before assuming:

| Signal | Suggests |
|---|---|
| The tool's behaviour is safer, cheaper, or more useful than the promise | The doc is the wrong half |
| Another page already describes the tool's actual behaviour correctly | The doc is the wrong half, and the pages disagree with each other |
| The promise is specific and the behaviour is an obvious omission | The code is the wrong half |
| The message says something nobody checked ("holds no files at all") | The code is the wrong half, always |

Report the disagreement and your reading of which half is wrong. Do not fix either.

### Numbers versus shape

A doc example whose **numbers** differ is usually cosmetic — the corpus differs. A doc example
whose **shape, column names, file names, API names or exit code** differs is a real finding.

Exception: a number the docs present as reproducible, or a number that moves between two runs
on the same corpus. That is a finding regardless of size.

## Staying out of the repo

Doc snippets create files. Run every snippet and every command from a scratch directory, never
the repo root:

```bash
cp -r .claude/skills/docs-e2e-drive/data /tmp/drive && cd /tmp/drive
```

A previous drive left `documents/`, `questions.csv` and `questions.jsonl` in the repo root and
could not say for certain which session had done it. Anything the drive creates belongs outside
the checkout.

## Working alongside other drives

When more than one drive runs at once, each takes a lane and writes to its own report file.
Two drives covering the same surface waste effort; two drives reading each other's reports stop
being independent.

Do not read another drive's report, and do not read `docs-e2e-report*.md` from an earlier round.
Knowing where the last drive looked is precisely the knowledge that stops you looking elsewhere.

Expect transient failures from other sessions mid-edit — a doctest that fails once and passes a
minute later is someone else's write, not a finding. Re-run before reporting it.

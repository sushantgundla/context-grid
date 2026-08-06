# Recipes

Worked examples, each one run for real and pasted verbatim — the question, the config, the
command, the actual output, and what it means. Start with [getting-started](../guide/getting-started.md)
if you haven't run a sweep yet; these assume you have.

- [choose-a-chunker.md](choose-a-chunker.md) — sweep the chunker axis on a real corpus and read
  the result honestly, including when three differently-named chunkers turn out to be the same
  thing.
- [choose-an-embedder.md](choose-an-embedder.md) — compare embedders with an eval set, and — the
  more common situation — with none at all, using `contextgrid.embed.assess` on the vectors alone.
- [is-agentic-worth-it.md](is-agentic-worth-it.md) — `simple`, `decomposed` and `agentic` on one
  axis, and pricing the model calls the built-in cost column doesn't catch.
- [local-only.md](local-only.md) — a full sweep with no API key: what's free out of the box, and
  a real local model through TEI, with the exact `docker run` commands.
- [without-an-evalset.md](without-an-evalset.md) — what a corpus and its embeddings can tell you
  with zero questions, and the hard line where that stops and a real eval set becomes necessary.
- [reproducing-a-run.md](reproducing-a-run.md) — the manifest, what's in the hash and what's
  deliberately left out, `contextgrid diff`, and what seeds are and aren't pinned today.

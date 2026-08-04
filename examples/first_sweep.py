"""A complete sweep, in the smallest honest example.

Run it with `python examples/first_sweep.py`. No models are downloaded and nothing touches
the network -- the whole thing runs on the two documents defined below.
"""

from __future__ import annotations

import contextgrid as cg

CONTRACT = """\
# Master Services Agreement

## 1. Term

This agreement begins on the Effective Date and continues for twelve months.

## 2. Termination

Either party may terminate this agreement for convenience by giving thirty days
written notice. A party may terminate immediately if the other commits a material
breach and fails to remedy it within fifteen days of written notice.

## 3. Fees

| Service | Monthly fee |
|---|---|
| Standard | $1,200 |
| Premium | $3,400 |
"""

API_DOCS = """\
# Widget API

## Authentication

Send your key in the `X-Api-Key` header. Requests without a valid key return 401.

## Endpoints

### GET /widgets/{id}

Returns one widget. Returns 404 when the id is unknown.
"""

# Ground truth is authored as *quoted evidence*, not as chunk ids or character offsets.
# That is what lets the same eval set be re-resolved against every parser on the grid.
QUESTIONS = [
    ("q1", "How much notice is needed to terminate for convenience?", "contract.md", "thirty days"),
    (
        "q2",
        "What happens after a material breach?",
        "contract.md",
        "fifteen days of written notice",
    ),
    ("q3", "What is the Premium monthly fee?", "contract.md", "$3,400"),
    ("q4", "Which header carries the API key?", "api.md", "X-Api-Key"),
    ("q5", "What does GET /widgets return for an unknown id?", "api.md", "Returns 404"),
]


def main() -> None:
    lab = cg.Lab({"contract.md": CONTRACT, "api.md": API_DOCS}, machine_usd_per_hour=0.10)

    profile = lab.fingerprint()
    print(profile.summary())
    for hint in profile.hints():
        print(f"  - {hint}")

    evalset = cg.EvalSet(
        id="demo",
        items=tuple(
            cg.EvalItem(
                id=item_id,
                question=question,
                anchors=(cg.GoldAnchor(source_id=source, quote=quote),),
            )
            for item_id, question, source, quote in QUESTIONS
        ),
    )

    lab.grid(
        chunker=["fixed:24,overlap=0", "sentence:2", "structural:80,min_size=16"],
        embedder=["tfidf", "hash:256", "length"],
        index=["dense", "bm25", "hybrid"],
        k=3,
    )
    print(f"\n{lab.matrix.shape()} on paper, {lab.estimate('factorial')['configurations']} to run")

    results = lab.run(evalset, mode="factorial", headline="recall@3")

    print(f"\n{'configuration':56} {'R@3':>6} {'nDCG@3':>7} {'chunks':>7}")
    for row in results.leaderboard("recall@3", extra=["ndcg@3"])[:10]:
        print(f"{row['config']:56} {row['recall@3']:6.3f} {row['ndcg@3']:7.3f} {row['chunks']:7}")

    print("\nWhich decision mattered:")
    for axis in ("chunker", "embedder", "index"):
        effect = results.axis_effect(axis, "recall@3")
        best = max(effect, key=lambda value: effect[value])
        spread = max(effect.values()) - min(effect.values())
        print(f"  {axis:9} best {best!r} (+{spread:.3f} over the worst value)")

    print(f"\n{results.summary('recall@3')}")
    print(f"\nCache: {results.cache_summary}")


if __name__ == "__main__":
    main()

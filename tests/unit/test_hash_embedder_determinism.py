"""The `hash` embedder must give the same vectors in every process, forever.

README.md line 5 promises "ranked, reproducible results", and the hashing embedder is the one
that ships in the box -- the one every quickstart, every doctest and every offline example
reaches for. If it moves between runs then so does every score computed on top of it.

The trap this file exists to avoid: a determinism test that runs inside a single process
passes against the bug. Python salts `hash()` on a `str` once per interpreter, so within one
process the broken embedder looks perfectly stable. It only misbehaves when compared against
*another* process. So these tests spawn real subprocesses with different `PYTHONHASHSEED`
values and compare what comes back.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from contextgrid.embed.local import HashEmbedder

#: Deliberately varied: repeated words (log term-frequency), punctuation, casing, non-ASCII and
#: a digit, so a change to the tokenizer or the digest shows up here rather than passing by.
TEXTS = (
    "the quick brown fox jumps over the lazy dog",
    "Rétention: the invoice was paid 30 days late, the invoice was disputed.",
    "hash hash hash hash collide",
    "",
)

#: `PYTHONHASHSEED=0` disables the randomisation; the others each pick a different salt. A run
#: that only ever used 0 would agree with itself and prove nothing.
HASH_SEEDS = ("0", "1", "2", "12345")

_CHILD = """
import hashlib, sys
import numpy as np
from contextgrid.embed.local import HashEmbedder

spec = eval(sys.argv[1])
texts = eval(sys.argv[2])
vectors = HashEmbedder(**spec).embed_documents(texts).vectors
packed = np.ascontiguousarray(vectors, dtype=np.float32).tobytes()
sys.stdout.write(hashlib.sha256(packed).hexdigest())
"""


def _checksum_in_subprocess(spec: dict[str, int], *, hash_seed: str) -> str:
    """Embed `TEXTS` in a fresh interpreter and return a checksum of the vectors.

    `sys.executable` keeps the child on the same virtualenv as the test. `PYTHONHASHSEED` is
    set in the child's environment rather than mutated in this one, because Python reads it at
    interpreter start-up -- setting it here would change nothing at all. The rest of the
    environment is inherited so the child can still find the package however this one did.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, repr(spec), repr(TEXTS)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    return completed.stdout.strip()


@pytest.mark.parametrize(
    "spec",
    [
        {},  # the documented defaults: dimensions=256, seed=0
        {"dimensions": 512},
        {"dimensions": 512, "seed": 3},
        {"dimensions": 64, "seed": -7},
    ],
    ids=["defaults", "dim512", "dim512-seed3", "dim64-negative-seed"],
)
def test_vectors_are_identical_across_processes(spec: dict[str, int]) -> None:
    """Same spec, same text, four interpreters with four different hash salts, one answer."""
    checksums = {seed: _checksum_in_subprocess(spec, hash_seed=seed) for seed in HASH_SEEDS}
    distinct = set(checksums.values())
    assert len(distinct) == 1, (
        f"HashEmbedder({spec}) produced {len(distinct)} different vector sets across "
        f"PYTHONHASHSEED values: {checksums}. It is using a process-salted hash."
    )


def test_seed_actually_changes_the_vectors() -> None:
    """`seed=` has to pin the output, which means it also has to *move* it.

    A "fix" that ignored the seed would sail through the test above.
    """
    checksums = {
        seed: _checksum_in_subprocess({"dimensions": 512, "seed": seed}, hash_seed="0")
        for seed in (0, 3, 4)
    }
    assert len(set(checksums.values())) == 3, f"seed= did not change the vectors: {checksums}"


def test_documented_defaults() -> None:
    """docs/dimensions/embedders.md documents `dimensions: int = 256`, `seed: int = 0`."""
    embedder = HashEmbedder()
    assert embedder.dimensions == 256
    assert embedder.seed == 0


def test_same_word_lands_in_the_same_place_regardless_of_neighbours() -> None:
    """The hashing trick's whole contract: a word's bucket depends on the word, nothing else."""
    embedder = HashEmbedder(dimensions=512)
    alone = embedder.embed_documents(["axolotl"]).vectors[0]
    with_others = embedder.embed_documents(["axolotl narwhal"]).vectors[0]
    (bucket,) = np.nonzero(alone)
    assert with_others[bucket] != 0.0


def test_vectors_are_unit_length_as_the_protocol_claims() -> None:
    """`normalised = True` is read by the index to treat dot product and cosine as the same."""
    vectors = HashEmbedder(dimensions=512).embed_documents(list(TEXTS)).vectors
    norms = np.linalg.norm(vectors, axis=1)
    # The empty string has no words, so its row is all zeros and `normalise` leaves it be.
    assert np.allclose(norms[:-1], 1.0, atol=1e-5)
    assert norms[-1] == 0.0


def test_signs_are_not_all_the_same() -> None:
    """Both signs must appear, or every document drifts towards one corner of the space.

    This is what keeps anisotropy low. If a future change to the digest accidentally made the
    sign constant, nothing else in the suite would notice -- the vectors would still be stable,
    still unit length, and quietly much worse.

    Asserted one side at a time on purpose. All-positive and all-negative are different faults
    with different causes -- a sign bit that is stuck high is not the same bug as one stuck low
    -- and a single combined assertion would report either as the same anonymous failure.
    """
    embedder = HashEmbedder(dimensions=4096)
    words = " ".join(f"word{index}" for index in range(2000))
    values = embedder.embed_documents([words]).vectors[0]
    assert (values > 0).any(), "every non-zero weight is negative: the sign bit is stuck low"
    assert (values < 0).any(), "every non-zero weight is positive: the sign bit is stuck high"

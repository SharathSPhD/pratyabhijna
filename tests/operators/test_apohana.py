"""apohana invariants: pure numpy operator, no LM."""
from __future__ import annotations

import numpy as np
import pytest

from pce.operators.apohana import apohana
from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint


@pytest.fixture(scope="module")
def embed() -> Embedder:
    return Embedder()


@pytest.mark.real_model
def test_apohana_no_avoid_returns_positive_alignment(embed: Embedder) -> None:
    q = embed.encode("a small bird sings at dawn")
    target = embed.encode("a robin chirps in the morning light")
    distractor = embed.encode("locomotive engineering thermodynamics")
    constraint = Constraint(text="bird at dawn", embedding=q)
    cands = (
        Candidate(seed=0, sampler={"tau": 1.0}, tokens=(1,), text="robin", logp=-0.1, embedding=target),
        Candidate(seed=1, sampler={"tau": 1.0}, tokens=(2,), text="train", logp=-0.1, embedding=distractor),
    )
    scores = apohana(cands, constraint, embed=embed)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]


@pytest.mark.real_model
def test_apohana_with_must_avoid_punishes_overlap(embed: Embedder) -> None:
    q = embed.encode("a small bird sings at dawn")
    constraint_no = Constraint(text="bird at dawn", embedding=q)
    constraint_with = Constraint(
        text="bird at dawn",
        embedding=q,
        must_avoid=("a tiny robin chirps in the morning light",),
    )
    overlap = embed.encode("a tiny robin chirps in the morning light")
    other = embed.encode("eagle soars over the canyon")
    cands = (
        Candidate(seed=0, sampler={"tau": 1.0}, tokens=(1,), text="overlap", logp=-0.1, embedding=overlap),
        Candidate(seed=1, sampler={"tau": 1.0}, tokens=(2,), text="other", logp=-0.1, embedding=other),
    )
    no_avoid = apohana(cands, constraint_no, embed=embed)
    with_avoid = apohana(cands, constraint_with, embed=embed)
    assert with_avoid[0] < no_avoid[0]


def test_apohana_normalize_clamps_to_unit_interval() -> None:
    """ADR-002 normalize=True min-max-shifts apoha into [0, 1]."""
    from pce.operators.apohana import _shift_apoha

    raw = np.array([-2.0, 0.0, 1.0, 0.5], dtype=np.float32)
    shifted = _shift_apoha(raw)
    assert float(shifted.min()) == 0.0
    assert float(shifted.max()) == 1.0
    # idempotent on already-shifted input
    twice = _shift_apoha(shifted)
    assert np.allclose(shifted, twice)


def test_apohana_normalize_constant_returns_half() -> None:
    """All-equal apoha collapses to 0.5 so pseudo-counts stay symmetric."""
    from pce.operators.apohana import _shift_apoha

    raw = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    shifted = _shift_apoha(raw)
    assert np.allclose(shifted, 0.5)


def test_apohana_empty_returns_empty() -> None:
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    constraint = Constraint(text="x", embedding=q)

    class FakeEmbed:
        def encode(self, items: list[str]):  # type: ignore[no-untyped-def]
            return np.zeros((len(items), 8), dtype=np.float32)

        def cosine(self, a, b):  # pragma: no cover
            return float(np.dot(a, b))

    out = apohana((), constraint, embed=FakeEmbed())  # type: ignore[arg-type]
    assert out.shape == (0,)

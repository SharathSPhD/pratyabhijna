"""ananda scorer invariants."""
from __future__ import annotations

import numpy as np
import pytest

from pce.operators.ananda import ananda
from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint


@pytest.fixture(scope="module")
def embed() -> Embedder:
    return Embedder()


def _cand(text: str, emb: np.ndarray) -> Candidate:
    return Candidate(seed=0, sampler={"tau": 1.0}, tokens=(1, 2), text=text, logp=-0.1, embedding=emb)


@pytest.mark.real_model
def test_ananda_in_unit_interval(embed: Embedder) -> None:
    q = embed.encode("morning birdsong")
    c = _cand("a robin sings at sunrise", embed.encode("a robin sings at sunrise"))
    constraint = Constraint(text="morning birdsong", embedding=q)
    s = ananda(c, constraint=constraint, embed=embed)
    assert 0.0 <= s <= 1.0


@pytest.mark.real_model
def test_ananda_aligned_outscores_unrelated(embed: Embedder) -> None:
    q = embed.encode("morning birdsong")
    constraint = Constraint(text="morning birdsong", embedding=q)
    aligned = _cand(
        "a tiny finch trills as the sun first crests the maple line",
        embed.encode("a tiny finch trills as the sun first crests the maple line"),
    )
    unrelated = _cand(
        "tax brackets in OECD nations are usually piecewise-linear",
        embed.encode("tax brackets in OECD nations are usually piecewise-linear"),
    )
    s_aligned = ananda(aligned, constraint=constraint, embed=embed)
    s_unrel = ananda(unrelated, constraint=constraint, embed=embed)
    assert s_aligned > s_unrel


@pytest.mark.real_model
def test_ananda_empty_text_returns_zero(embed: Embedder) -> None:
    q = embed.encode("anything")
    constraint = Constraint(text="anything", embedding=q)
    cand = _cand("   ", embed.encode("   "))
    assert ananda(cand, constraint=constraint, embed=embed) == 0.0

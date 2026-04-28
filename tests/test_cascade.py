"""End-to-end cascade smoke test."""
from __future__ import annotations

import pytest

from pce.cascade import run_cascade
from pce.substrate.embed import Embedder
from pce.substrate.lm import LocalLM
from pce.types import Constraint


@pytest.fixture(scope="module")
def lm() -> LocalLM:
    return LocalLM()


@pytest.fixture(scope="module")
def embed() -> Embedder:
    return Embedder()


@pytest.mark.real_model
@pytest.mark.slow
def test_cascade_end_to_end(lm: LocalLM, embed: Embedder) -> None:
    q = embed.encode("a haiku about autumn leaves")
    constraint = Constraint(
        text="a haiku about autumn leaves",
        embedding=q,
        must_avoid=("a busy city street",),
    )
    state = run_cascade(
        prompt="Compose a short poem.\n",
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=4,
        max_tokens=20,
        base_seed=42,
        retrieval_set=["raindrops on a tin roof"],
        aspects=["leaves spinning in wind", "the smell of decay"],
    )
    assert state.surface is not None and state.surface != ""
    assert state.selected is not None
    assert len(state.candidates) == 4
    assert state.posterior.shape == (4,)
    assert "delta_F" in state.audit
    assert "ananda_scores" in state.audit

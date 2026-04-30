"""End-to-end cascade smoke test."""
from __future__ import annotations

import pytest

from pce.cascade import run_cascade
from pce.operators.iccha import k_runtime_for
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
    K_rt = k_runtime_for(K_eff=4, cit_temperature=1.0)
    assert len(state.candidates) == K_rt
    assert state.posterior.shape == (K_rt,)
    assert "delta_F" in state.audit
    assert "ananda_scores" in state.audit

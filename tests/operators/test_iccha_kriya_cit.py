"""iccha + kriya + cit (LM-touching, slow)."""
from __future__ import annotations

import pytest

from pce.operators.cit import cit
from pce.operators.iccha import iccha
from pce.operators.kriya import kriya
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
def test_cit_returns_candidate_with_text(lm: LocalLM) -> None:
    out = cit("Write one English sentence about rain:\n", lm=lm, max_tokens=16, seed=7)
    assert out.text != ""
    assert "tau" in out.sampler


@pytest.mark.real_model
@pytest.mark.slow
def test_iccha_K_distinct_samplers(lm: LocalLM, embed: Embedder) -> None:
    q = embed.encode("a haiku about autumn leaves")
    constraint = Constraint(text="a haiku about autumn leaves", embedding=q)
    cands = iccha(
        "Compose a short poem.\n",
        constraint,
        lm=lm,
        K=4,
        max_tokens=24,
        base_seed=11,
    )
    assert len(cands) == 4
    taus = {c.sampler["tau"] for c in cands}
    assert len(taus) >= 3


@pytest.mark.real_model
@pytest.mark.slow
def test_kriya_verbatim_is_identity(lm: LocalLM) -> None:
    out = cit("Write one English sentence about wind:\n", lm=lm, max_tokens=12, seed=3)
    assert kriya(out, render_mode="verbatim") == out.text


def test_kriya_claude_polish_uses_renderer() -> None:
    import numpy as np

    from pce.types import Candidate

    cand = Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=(1,),
        text="raw text",
        logp=-0.1,
        embedding=np.zeros(4, dtype=np.float32),
    )
    out = kriya(cand, render_mode="claude_polish", claude_renderer=lambda t: f"polished: {t}")
    assert out == "polished: raw text"

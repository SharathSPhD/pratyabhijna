"""kriya: surface enaction (verbatim, polish, claude_polish)."""
from __future__ import annotations

import numpy as np
import pytest

from pce.operators.cit import cit
from pce.operators.kriya import kriya
from pce.substrate.lm import LocalLM
from pce.types import Candidate


@pytest.fixture(scope="module")
def lm() -> LocalLM:
    return LocalLM()


@pytest.mark.real_model
@pytest.mark.slow
def test_kriya_verbatim_is_identity(lm: LocalLM) -> None:
    out = cit("Write one English sentence about wind:\n", lm=lm, max_tokens=12, seed=3)
    assert kriya(out, render_mode="verbatim") == out.text


def test_kriya_claude_polish_uses_renderer() -> None:
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


def test_kriya_polish_without_lm_raises() -> None:
    cand = Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=(1,),
        text="raw text",
        logp=-0.1,
        embedding=np.zeros(4, dtype=np.float32),
    )
    try:
        kriya(cand, render_mode="polish")
    except ValueError:
        return
    raise AssertionError("expected ValueError for polish without lm")


def test_kriya_claude_polish_without_renderer_raises() -> None:
    cand = Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=(1,),
        text="raw text",
        logp=-0.1,
        embedding=np.zeros(4, dtype=np.float32),
    )
    try:
        kriya(cand, render_mode="claude_polish")
    except ValueError:
        return
    raise AssertionError("expected ValueError for claude_polish without renderer")


def test_kriya_unknown_mode_raises() -> None:
    cand = Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=(1,),
        text="raw text",
        logp=-0.1,
        embedding=np.zeros(4, dtype=np.float32),
    )
    try:
        kriya(cand, render_mode="bogus")  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")

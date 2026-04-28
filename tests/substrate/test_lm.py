"""Real-model tests for the local-LM substrate.

Loads Qwen2-1.5B-Instruct once per session (cached). Marked `real_model + slow`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pce.substrate.lm import LMConfig, LocalLM

pytestmark = [pytest.mark.real_model, pytest.mark.slow]


@pytest.fixture(scope="module")
def lm() -> LocalLM:
    return LocalLM(LMConfig(dtype="float32"))


def test_lm_loads(lm: LocalLM) -> None:
    info = lm.report()
    assert info["model_id"]
    assert info["vocab_size"] > 1000


def test_generate_returns_candidate(lm: LocalLM) -> None:
    cand = lm.generate("The capital of France is", max_tokens=8, sampler={"tau": 0.1}, seed=0)
    assert cand.text  # non-empty
    assert len(cand.tokens) > 0
    assert cand.embedding.shape == (384,)
    assert cand.logp <= 0.0


def test_low_temperature_is_near_greedy(lm: LocalLM) -> None:
    """At τ→0 the first sampled token must match the unconstrained argmax."""
    prompt = "The capital of France is"
    expected = lm.argmax_next(prompt)
    cand = lm.generate(prompt, max_tokens=1, sampler={"tau": 0.001, "top_p": 1.0, "top_k": 0}, seed=0)
    assert cand.tokens[0] == expected


def test_higher_temperature_higher_entropy(lm: LocalLM) -> None:
    """Per-step Shannon entropy must be monotonic in τ on average."""
    prompt = "Once upon a time, in a land far away"
    e_low = lm.entropy_at(prompt, tau=0.5)
    e_med = lm.entropy_at(prompt, tau=1.0)
    e_high = lm.entropy_at(prompt, tau=2.0)
    assert e_low < e_med < e_high


def test_generate_seed_reproducible(lm: LocalLM) -> None:
    a = lm.generate("Write a poem about a cat.", max_tokens=12, sampler={"tau": 1.2}, seed=7)
    b = lm.generate("Write a poem about a cat.", max_tokens=12, sampler={"tau": 1.2}, seed=7)
    assert a.tokens == b.tokens
    assert a.text == b.text


def test_generate_different_seeds_differ(lm: LocalLM) -> None:
    a = lm.generate("Write a poem about a cat.", max_tokens=12, sampler={"tau": 1.2}, seed=1)
    b = lm.generate("Write a poem about a cat.", max_tokens=12, sampler={"tau": 1.2}, seed=2)
    assert a.tokens != b.tokens or a.text != b.text


def test_logp_is_finite_and_negative(lm: LocalLM) -> None:
    cand = lm.generate("The sun rose", max_tokens=6, sampler={"tau": 0.8}, seed=0)
    assert np.isfinite(cand.logp)
    assert cand.logp <= 0.0


def test_invalid_tau_raises(lm: LocalLM) -> None:
    with pytest.raises(ValueError):
        lm.generate("test", max_tokens=2, sampler={"tau": 0.0}, seed=0)

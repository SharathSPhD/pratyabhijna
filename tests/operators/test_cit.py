"""cit operator: deterministic-on-seed wrapper around LocalLM.generate."""
from __future__ import annotations

import pytest

from pce.operators.cit import cit
from pce.substrate.lm import LocalLM


@pytest.fixture(scope="module")
def lm() -> LocalLM:
    return LocalLM()


@pytest.mark.real_model
@pytest.mark.slow
def test_cit_returns_candidate_with_text(lm: LocalLM) -> None:
    out = cit("Write one English sentence about rain:\n", lm=lm, max_tokens=16, seed=7)
    assert out.text != ""
    assert "tau" in out.sampler


@pytest.mark.real_model
@pytest.mark.slow
def test_cit_seed_is_deterministic(lm: LocalLM) -> None:
    a = cit("Write a single line about the moon:\n", lm=lm, max_tokens=12, seed=21)
    b = cit("Write a single line about the moon:\n", lm=lm, max_tokens=12, seed=21)
    assert a.tokens == b.tokens


def test_cit_zero_temperature_raises() -> None:
    class FakeLM:
        def generate(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

    try:
        cit("hello", lm=FakeLM(), temperature=0.0)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for temperature=0")


def test_cit_zero_max_tokens_raises() -> None:
    class FakeLM:
        def generate(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

    try:
        cit("hello", lm=FakeLM(), max_tokens=0)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for max_tokens=0")

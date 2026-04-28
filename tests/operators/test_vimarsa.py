"""vimarśa detector invariants."""
from __future__ import annotations

import pytest

from pce.operators.vimarsa import vimarsa
from pce.substrate.embed import Embedder


@pytest.fixture(scope="module")
def embed() -> Embedder:
    return Embedder()


@pytest.mark.real_model
def test_vimarsa_high_novelty_and_aspects_fires(embed: Embedder) -> None:
    surface = "the river is a clock and a clock is a river"
    aspects = [
        "a flowing river that measures time",
        "a clock face that ripples like water",
    ]
    retrieval_set = [
        "the cat sat on the mat",
        "two plus two equals four",
    ]
    event, novelty, _ = vimarsa(
        prompt="describe time",
        surface=surface,
        embed=embed,
        retrieval_set=retrieval_set,
        aspects=aspects,
        ananda_score=0.8,
        # No trajectory => switching gate auto-pass.
        iccha_apoha_trajectory=None,
        aspect_cosine_hit=0.30,
    )
    assert novelty >= 0.30
    assert event is True


@pytest.mark.real_model
def test_vimarsa_low_aesthetic_blocks_event(embed: Embedder) -> None:
    surface = "the river is a clock and a clock is a river"
    aspects = ["a flowing river that measures time", "a clock face that ripples like water"]
    event, _, _ = vimarsa(
        prompt="x",
        surface=surface,
        embed=embed,
        retrieval_set=["a totally unrelated sentence"],
        aspects=aspects,
        ananda_score=0.10,  # below the 0.40 floor
        iccha_apoha_trajectory=None,
        aspect_cosine_hit=0.30,
    )
    assert event is False


def test_vimarsa_empty_surface_returns_false() -> None:
    class FakeEmbed:
        def encode(self, x):  # pragma: no cover
            raise AssertionError("should not be called for empty surface")

        def cosine(self, a, b):  # pragma: no cover
            return 0.0

    out = vimarsa(
        prompt="x",
        surface="   ",
        embed=FakeEmbed(),  # type: ignore[arg-type]
        retrieval_set=[],
        aspects=[],
        ananda_score=1.0,
    )
    assert len(out) == 3
    event, novelty, diag = out
    assert event is False
    assert novelty == 0.0
    assert diag.get("empty_surface") == 1.0


@pytest.mark.real_model
def test_vimarsa_one_point_trajectory_does_not_block_event(embed: Embedder) -> None:
    """v0.2 ADR-003: switching gate is N/A when trajectory is None or short.

    The v0.1 cascade always passed a one-point trajectory which then failed
    the hardcoded `switching >= 2` floor (P0-2 in the adversarial review).
    """
    surface = "the river is a clock and a clock is a river"
    aspects = [
        "a flowing river that measures time",
        "a clock face that ripples like water",
    ]
    out = vimarsa(
        prompt="describe time",
        surface=surface,
        embed=embed,
        retrieval_set=["the cat sat on the mat"],
        aspects=aspects,
        ananda_score=0.8,
        iccha_apoha_trajectory=None,
        aspect_cosine_hit=0.30,
    )
    assert len(out) == 3
    event, _, diag = out
    assert event is True
    assert diag["switching_gate"] == 0.0  # explicitly N/A


@pytest.mark.real_model
def test_vimarsa_return_brief_emits_revision_brief(embed: Embedder) -> None:
    """v0.2 ADR-003: with return_brief=True vimarsa emits a brief for revision."""
    surface = "the river is a clock"
    aspects = ["a flowing river that measures time", "a clock face that ripples like water"]
    out = vimarsa(
        prompt="describe time",
        surface=surface,
        embed=embed,
        retrieval_set=["a totally unrelated sentence"],
        aspects=aspects,
        ananda_score=0.8,
        iccha_apoha_trajectory=None,
        aspect_cosine_hit=0.30,
        return_brief=True,
    )
    assert len(out) == 4
    _, _, _, brief = out
    assert isinstance(brief, str) and len(brief.strip()) > 0


def test_vimarsa_no_aspects_emits_generic_brief() -> None:
    """v0.2: when domain has no aspects (poetry_gen, AUT) we emit the generic brief."""
    from pce.operators.vimarsa import GENERIC_BRIEF

    class FakeEmbed:
        def encode(self, x):  # type: ignore[no-untyped-def]
            import numpy as np
            if isinstance(x, str):
                return np.zeros(4, dtype=np.float32)
            return np.zeros((len(x), 4), dtype=np.float32)

    out = vimarsa(
        prompt="x",
        surface="hello world",
        embed=FakeEmbed(),  # type: ignore[arg-type]
        retrieval_set=[],
        aspects=[],
        ananda_score=0.5,
        iccha_apoha_trajectory=None,
        return_brief=True,
    )
    assert len(out) == 4
    _, _, _, brief = out
    assert brief == GENERIC_BRIEF

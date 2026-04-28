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

    event, novelty, diag = vimarsa(
        prompt="x",
        surface="   ",
        embed=FakeEmbed(),  # type: ignore[arg-type]
        retrieval_set=[],
        aspects=[],
        ananda_score=1.0,
    )
    assert event is False
    assert novelty == 0.0
    assert diag.get("empty_surface") == 1.0

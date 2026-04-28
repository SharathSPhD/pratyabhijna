"""Real-model tests for the embedding substrate.

Asserts the deterministic L2-normalized cosine geometry the operators rely on.
"""
from __future__ import annotations

import numpy as np
import pytest

from pce.substrate.embed import Embedder

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_dim_is_384(embedder: Embedder) -> None:
    assert embedder.dim == 384


def test_encode_single_returns_1d(embedder: Embedder) -> None:
    v = embedder.encode("a cat sat on a mat")
    assert isinstance(v, np.ndarray)
    assert v.shape == (384,)
    assert v.dtype == np.float32


def test_encode_list_returns_2d(embedder: Embedder) -> None:
    arr = embedder.encode(["a cat sat on a mat", "the kitten is on the rug"])
    assert arr.shape == (2, 384)


def test_encode_is_l2_normalized(embedder: Embedder) -> None:
    v = embedder.encode("test sentence")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_encode_is_deterministic(embedder: Embedder) -> None:
    a = embedder.encode("the quick brown fox jumps over the lazy dog")
    b = embedder.encode("the quick brown fox jumps over the lazy dog")
    assert np.allclose(a, b, atol=1e-6)


def test_cosine_paraphrase_high(embedder: Embedder) -> None:
    a = embedder.encode("A cat sits on a mat.")
    b = embedder.encode("A cat is sitting on a mat.")
    assert embedder.cosine(a, b) > 0.85


def test_cosine_unrelated_low(embedder: Embedder) -> None:
    a = embedder.encode("A cat sits on a mat.")
    b = embedder.encode("Quantum chromodynamics describes the strong force.")
    assert embedder.cosine(a, b) < 0.30


def test_cosine_shape_check(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.cosine(
            np.zeros(10, dtype=np.float32), np.zeros(20, dtype=np.float32)
        )


def test_encode_rejects_empty(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.encode([])

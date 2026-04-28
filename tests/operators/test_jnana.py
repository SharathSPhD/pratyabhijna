"""jñāna BMR invariants: pure numpy operator."""
from __future__ import annotations

import numpy as np

from pce.operators.jnana import jnana, log_beta


def _stub_candidates(K: int) -> tuple:
    class C:
        def __init__(self, i: int) -> None:
            self.i = i
    return tuple(C(i) for i in range(K))


def test_log_beta_matches_log_dirichlet_normalizer() -> None:
    import math

    a = np.array([1.5, 2.0, 0.7], dtype=np.float64)
    expected = float(
        sum(math.lgamma(float(x)) for x in a) - math.lgamma(float(a.sum()))
    )
    assert abs(log_beta(a) - expected) < 1e-9


def test_jnana_selects_dominant_evidence() -> None:
    K = 4
    cands = _stub_candidates(K)
    apoha = np.array([0.2, 0.1, 0.9, 0.0], dtype=np.float32)
    anan = np.array([0.3, 0.2, 0.95, 0.1], dtype=np.float32)
    sel, delta, post = jnana(cands, apoha, anan)
    assert sel == 2
    assert post.shape == (K,)
    assert post.argmax() == 2
    assert delta > -1e9


def test_jnana_uniform_evidence_picks_some_index() -> None:
    K = 4
    cands = _stub_candidates(K)
    apoha = np.zeros(K, dtype=np.float32)
    anan = np.zeros(K, dtype=np.float32)
    sel, delta, post = jnana(cands, apoha, anan)
    assert 0 <= sel < K
    assert post.shape == (K,)


def test_jnana_K_lt_2_raises() -> None:
    cands = _stub_candidates(1)
    apoha = np.array([0.0], dtype=np.float32)
    anan = np.array([0.0], dtype=np.float32)
    try:
        jnana(cands, apoha, anan)
    except ValueError:
        return
    raise AssertionError("expected ValueError for K<2")


def test_jnana_shape_mismatch_raises() -> None:
    cands = _stub_candidates(3)
    bad_apoha = np.array([0.0, 0.0], dtype=np.float32)
    anan = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    try:
        jnana(cands, bad_apoha, anan)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched shapes")

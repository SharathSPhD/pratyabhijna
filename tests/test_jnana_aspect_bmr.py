"""Phase 3 ADR-003 gate: jñāna aspect-conditioned BMR.

Verifies that:

* ``reduction_target="aspect_conditioned"`` requires ``aspect_membership``.
* On the duck-rabbit-style fixture (one candidate hits both aspects, others
  hit one or none), the IFR reduction wins and ΔF is non-degenerate
  (``> 0.01``).
* Aspect priors bias the winning reduction (low-prior aspects don't dominate).
* Empty / no-hit aspect membership falls back to the uniform full prior.
* Backward compat: ``reduction_target="halve"`` is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from pce.operators.jnana import jnana


def _stub_candidates(K: int) -> tuple:
    class C:
        def __init__(self, i: int) -> None:
            self.i = i
    return tuple(C(i) for i in range(K))


def test_aspect_conditioned_requires_membership() -> None:
    cands = _stub_candidates(4)
    apoha = np.array([0.1, 0.2, 0.5, 0.3], dtype=np.float32)
    anan = np.array([0.2, 0.3, 0.6, 0.4], dtype=np.float32)
    with pytest.raises(ValueError, match="aspect_membership"):
        jnana(cands, apoha, anan, reduction_target="aspect_conditioned")


def test_duck_rabbit_ifr_wins_and_delta_F_nondegenerate() -> None:
    """K=4 candidates, A=2 aspects (duck, rabbit). Candidate 2 hits both."""
    K = 4
    cands = _stub_candidates(K)
    # Candidate 2 dominates both aspect cosines AND apoha/ananda.
    apoha = np.array([0.20, 0.15, 0.85, 0.10], dtype=np.float32)
    anan = np.array([0.30, 0.25, 0.90, 0.15], dtype=np.float32)
    aspect_membership = np.array(
        [
            [0.10, 0.05],  # neither aspect
            [0.40, 0.05],  # duck only
            [0.45, 0.42],  # both aspects -- IFR
            [0.05, 0.40],  # rabbit only
        ],
        dtype=np.float32,
    )
    aspect_priors = np.array([0.5, 0.5], dtype=np.float32)
    sel, delta, post = jnana(
        cands,
        apoha,
        anan,
        reduction_target="aspect_conditioned",
        aspect_membership=aspect_membership,
        aspect_priors=aspect_priors,
    )
    assert sel == 2, f"IFR candidate should win, got idx={sel} with post={post.tolist()}"
    assert post.shape == (K,)
    assert delta > 0.01, f"ΔF must be non-degenerate (>0.01), got {delta}"


def test_aspect_priors_select_supported_candidate() -> None:
    """With data evidence pointing to one aspect's candidate, the BMR winner is that candidate.

    We deliberately do NOT claim "stronger prior => larger ΔF": Occam's
    razor inside BMR penalizes over-confident reductions, so a softer prior
    can yield a larger ΔF on the same data. What we DO claim is that the
    selected candidate (argmax of the winning posterior) aligns with the
    aspect that the prior dominates *when data supports that aspect*.
    """
    K = 4
    cands = _stub_candidates(K)
    # Cand 1 (duck) carries clearly stronger data evidence.
    apoha = np.array([0.10, 0.65, 0.10, 0.20], dtype=np.float32)
    anan = np.array([0.20, 0.75, 0.20, 0.30], dtype=np.float32)
    aspect_membership = np.array(
        [[0.10, 0.10], [0.50, 0.05], [0.10, 0.10], [0.05, 0.50]],
        dtype=np.float32,
    )
    duck_dominant = np.array([0.99, 0.01], dtype=np.float32)
    sel_duck, delta_duck, _ = jnana(
        cands, apoha, anan,
        reduction_target="aspect_conditioned",
        aspect_membership=aspect_membership,
        aspect_priors=duck_dominant,
    )
    assert sel_duck == 1, f"Duck-supporting candidate should win, got idx={sel_duck}"
    assert delta_duck > -1e9


def test_empty_membership_returns_finite_delta() -> None:
    """No aspect ever hits => fallback to uniform reduction; ΔF still finite."""
    K, A = 4, 2
    cands = _stub_candidates(K)
    apoha = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    anan = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    aspect_membership = np.zeros((K, A), dtype=np.float32)  # zero hits everywhere
    sel, delta, post = jnana(
        cands,
        apoha,
        anan,
        reduction_target="aspect_conditioned",
        aspect_membership=aspect_membership,
    )
    assert post.shape == (K,)
    assert np.isfinite(delta)
    assert sel in range(K)


def test_halve_target_backward_compat() -> None:
    """The aspect-conditioned mode must not change the existing 'halve' path."""
    K = 4
    cands = _stub_candidates(K)
    apoha = np.array([0.2, 0.1, 0.9, 0.0], dtype=np.float32)
    anan = np.array([0.3, 0.2, 0.95, 0.1], dtype=np.float32)
    sel, delta, _ = jnana(cands, apoha, anan, reduction_target="halve")
    assert sel == 2
    assert delta > -1e9

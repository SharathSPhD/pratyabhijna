"""v0.4 Phase 2 (ADR-001) gate: best-of-K candidate width as cit_temperature mechanism.

Verifies the pure functions that translate ``cit_temperature`` into runtime
candidate width (``k_runtime_for``) and into deterministic prompt-level
perturbations (``perturbation_idx`` / ``perturbation_for``). The end-to-end
n-gram-entropy probe lives in ``scripts/prove_gate.py``; this unit test
locks the math so the prove-gate has a stable foundation to build on.
"""
from __future__ import annotations

import pytest

from pce.operators.iccha import (
    K_MAX,
    K_MIN,
    PERTURBATION_TABLE,
    k_runtime_for,
    perturbation_for,
    perturbation_idx,
)


def test_k_runtime_monotonic_in_cit_temperature() -> None:
    """K_runtime(0.0) <= K_runtime(0.5) <= K_runtime(1.0) at K_eff=4."""
    K_eff = 4
    k_low = k_runtime_for(K_eff, 0.0)
    k_mid = k_runtime_for(K_eff, 0.5)
    k_high = k_runtime_for(K_eff, 1.0)
    assert k_low <= k_mid <= k_high
    # Strict at the endpoints when K_eff is large enough.
    assert k_low < k_high


def test_k_runtime_neutral_at_one_third() -> None:
    """At cit_temperature = 1/3, K_runtime == K_eff (the formula's neutral point).

    Per ADR-001: ``K_runtime = round(K_eff * (0.5 + 1.5 * cit_temperature))``.
    Solving ``0.5 + 1.5 * t = 1.0`` gives ``t = 1/3``, so cit_temperature = 1/3
    is the only setting where K_runtime exactly equals K_eff for every K_eff
    in [K_MIN, K_MAX].
    """
    t = 1.0 / 3.0
    for K_eff in (2, 4, 6, 8):
        assert k_runtime_for(K_eff, t) == K_eff


def test_k_runtime_clipped_to_bounds() -> None:
    """K_runtime is always in [K_MIN, K_MAX]."""
    # At cit_temperature = 0.0 with K_eff = 1, raw rounds to 0; clipped to K_MIN.
    assert k_runtime_for(1, 0.0) >= K_MIN
    # At cit_temperature = 1.0 with K_eff = 100, raw = 200; clipped to K_MAX.
    assert k_runtime_for(100, 1.0) <= K_MAX


def test_k_runtime_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        k_runtime_for(0, 0.5)
    with pytest.raises(ValueError):
        k_runtime_for(-1, 0.5)
    with pytest.raises(ValueError):
        k_runtime_for(4, -0.1)


def test_perturbation_idx_deterministic() -> None:
    """Same (seed, k) -> same index. Different k -> different index."""
    assert perturbation_idx(0, 0) == perturbation_idx(0, 0)
    assert perturbation_idx(0, 0) != perturbation_idx(0, 1)
    assert perturbation_idx(0, 0) == 0  # identity at (0, 0) preserves v0.3 prompt


def test_perturbation_idx_modular() -> None:
    """The index wraps mod 8 in both seed and k."""
    assert perturbation_idx(8, 0) == perturbation_idx(0, 0)
    assert perturbation_idx(0, 8) == perturbation_idx(0, 0)
    assert perturbation_idx(3, 5) == (3 + 5) % 8


def test_perturbation_table_has_eight_entries() -> None:
    """The frozen table is exactly 8 entries; index 0 is the identity (empty string)."""
    assert len(PERTURBATION_TABLE) == 8
    assert PERTURBATION_TABLE[0] == ""


def test_perturbation_for_returns_table_entry() -> None:
    for seed in range(4):
        for k in range(4):
            idx = perturbation_idx(seed, k)
            assert perturbation_for(seed, k) == PERTURBATION_TABLE[idx]


def test_k_runtime_for_default_K_eff_4() -> None:
    """The v0.4 pilot uses K_eff=4; verify the canonical width sequence.

    Formula (ADR-001): ``K_runtime = round(K_eff * (0.5 + 1.5 * cit_temp))``.

    * cit_temperature=0.0 -> 0.5 * 4 = 2.0 -> K_runtime=2
    * cit_temperature=1/3 -> 1.0 * 4 = 4.0 -> K_runtime=4 (neutral point)
    * cit_temperature=0.5 -> 1.25 * 4 = 5.0 -> K_runtime=5
    * cit_temperature=1.0 -> 2.0 * 4 = 8.0 -> K_runtime=8 (broader exploration)
    """
    assert k_runtime_for(4, 0.0) == 2
    assert k_runtime_for(4, 1.0 / 3.0) == 4
    assert k_runtime_for(4, 0.5) == 5
    assert k_runtime_for(4, 1.0) == 8

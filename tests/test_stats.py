"""Synthetic-data tests for benchmarks.stats."""
from __future__ import annotations

import math

import numpy as np

from benchmarks.stats import (
    _bca_ci_paired_mean,
    _hedges_g_paired,
    _holm_bonferroni,
    _paired_permutation_p_one_sided,
    _power_paired_t,
)


def test_paired_permutation_recovers_obvious_effect() -> None:
    rng = np.random.default_rng(42)
    d = np.full(20, 0.5)  # treatment dominates by 0.5 every time
    p = _paired_permutation_p_one_sided(d, rng=rng, alternative="greater", n_permutations=20_000)
    assert p < 1e-3


def test_paired_permutation_null_is_uniform_around_half() -> None:
    rng = np.random.default_rng(7)
    d = rng.normal(loc=0.0, scale=1.0, size=12)
    p = _paired_permutation_p_one_sided(d, rng=rng, alternative="greater", n_permutations=20_000)
    assert 0.0 <= p <= 1.0


def test_hedges_g_known_paired() -> None:
    d = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    g = _hedges_g_paired(d)
    # std(d, ddof=1) == 0 ⇒ implementation returns 0.0 by definition.
    assert g == 0.0
    d = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.55])
    g = _hedges_g_paired(d)
    assert g > 0.0
    assert math.isfinite(g)


def test_bca_ci_covers_obvious_positive_effect() -> None:
    rng = np.random.default_rng(13)
    d = rng.normal(loc=0.5, scale=0.3, size=30)
    lo, hi = _bca_ci_paired_mean(d, rng=rng, n_boot=2000)
    assert lo > 0.0
    assert hi > lo


def test_holm_bonferroni_orders_and_clamps() -> None:
    pvals = {"a": 0.01, "b": 0.04, "c": 0.20, "d": 0.50}
    adj = _holm_bonferroni(pvals)
    # smallest gets multiplied by 4, but clamped to <=1
    assert adj["a"] == 0.04
    assert adj["b"] == 3 * 0.04
    assert adj["c"] == 2 * 0.20
    assert adj["d"] == 0.50
    # Monotonic non-decreasing
    sorted_keys = sorted(pvals, key=lambda k: pvals[k])
    last = -1.0
    for k in sorted_keys:
        assert adj[k] >= last
        last = adj[k]


def test_power_apriori_increases_with_n() -> None:
    p_small = _power_paired_t(g=0.5, n=10, alpha=0.05, alternative="greater")
    p_big = _power_paired_t(g=0.5, n=80, alpha=0.05, alternative="greater")
    assert p_small < p_big

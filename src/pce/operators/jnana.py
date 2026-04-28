"""`jñāna` - Bayesian Model Reduction posterior selection.

Implements the categorical-Dirichlet BMR equations from
[docs/research-extended.md §2.3](../../../docs/research-extended.md#23-reducer-pseudocode-used-as-jnana-core)
in log-space using `scipy.special.gammaln`.

The full prior is constructed from a uniform Dirichlet `Dir(1,1,...,1)` of
length K updated with pseudo-counts `α_k = 1 + λ_a · ananda_k + λ_p · max(0, apoha_k)`
to give the *full-model posterior* `a_post`. K reduced priors are then
enumerated; ΔF for each is computed via the log-Beta form.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

EPS = 1e-7
ReductionTarget = Literal["halve", "single", "custom"]


def _log_beta(a: npt.NDArray[np.float64]) -> float:
    """Log of multivariate Beta normalizer = sum_i lgamma(a_i) - lgamma(sum_i a_i)."""
    return float(np.sum(gammaln(a)) - gammaln(np.sum(a)))


def _delta_F(
    a_post: npt.NDArray[np.float64],
    full_prior: npt.NDArray[np.float64],
    reduced_prior: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64]]:
    tilde_a_post = a_post + reduced_prior - full_prior
    if np.any(tilde_a_post <= 0):
        return float("-inf"), reduced_prior.copy()
    delta = (
        _log_beta(tilde_a_post)
        - _log_beta(a_post)
        + _log_beta(full_prior)
        - _log_beta(reduced_prior)
    )
    return float(delta), tilde_a_post


def _enumerate_reductions(
    K: int, target: ReductionTarget, pseudo_counts: npt.NDArray[np.float64],
    custom: list[npt.NDArray[np.float64]] | None,
) -> list[npt.NDArray[np.float64]]:
    flat = np.ones((K,), dtype=np.float64)
    out: list[npt.NDArray[np.float64]] = []
    if target == "single":
        for k in range(K):
            r = np.full((K,), EPS, dtype=np.float64)
            r[k] = 1.0
            out.append(r)
        return out
    if target == "halve":
        # Each reduction keeps the top-half by pseudo-count.
        order = np.argsort(-pseudo_counts)  # descending
        half = max(1, K // 2)
        keep_top = set(order[:half].tolist())
        r1 = np.full((K,), EPS, dtype=np.float64)
        for k in keep_top:
            r1[k] = 1.0
        out.append(r1)
        # And one reduction that keeps each single candidate (single-survivor).
        for k in range(K):
            r = np.full((K,), EPS, dtype=np.float64)
            r[k] = 1.0
            out.append(r)
        # And the unchanged full-prior baseline.
        out.append(flat.copy())
        return out
    if target == "custom":
        if not custom:
            raise ValueError("jnana: reduction_target='custom' requires explicit `custom` priors")
        return list(custom)
    raise ValueError(f"unknown reduction_target: {target}")


def jnana(
    candidates: tuple[object, ...],
    apoha_scores: npt.NDArray[np.float32],
    ananda_scores: npt.NDArray[np.float32],
    *,
    reduction_target: ReductionTarget = "halve",
    custom_priors: list[npt.NDArray[np.float64]] | None = None,
    lambda_a: float = 2.0,
    lambda_p: float = 2.0,
) -> tuple[int, float, npt.NDArray[np.float32]]:
    """Returns (selected_index, best_delta_F, posterior).

    `selected_index` is the candidate that survives the winning reduction
    (i.e. the argmax of the reduced posterior).
    """
    K = len(candidates)
    if K < 2:
        raise ValueError(f"jnana: need K >= 2 candidates, got {K}")
    if apoha_scores.shape != (K,) or ananda_scores.shape != (K,):
        raise ValueError(
            f"jnana: score shapes must be ({K},), got apoha={apoha_scores.shape} ananda={ananda_scores.shape}"
        )
    apoha = apoha_scores.astype(np.float64)
    anan = ananda_scores.astype(np.float64)
    pseudo = 1.0 + float(lambda_a) * anan + float(lambda_p) * np.clip(apoha, 0.0, None)

    full_prior = np.ones((K,), dtype=np.float64)
    a_post = full_prior + pseudo - 1.0  # i.e. = pseudo

    reductions = _enumerate_reductions(K, reduction_target, pseudo, custom_priors)
    best_delta = float("-inf")
    best_post: npt.NDArray[np.float64] = full_prior / full_prior.sum()
    for r in reductions:
        delta, tilde = _delta_F(a_post, full_prior, r)
        if delta > best_delta:
            best_delta = delta
            best_post = tilde / max(tilde.sum(), 1e-30)

    sel = int(np.argmax(best_post))
    return sel, float(best_delta), best_post.astype(np.float32)


def log_beta(a: npt.NDArray[np.float64]) -> float:
    """Public re-export of `_log_beta` for tests."""
    return _log_beta(a)

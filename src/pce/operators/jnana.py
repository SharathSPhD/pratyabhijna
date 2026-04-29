"""`jñāna` - Bayesian Model Reduction posterior selection.

Implements the categorical-Dirichlet BMR equations from
[docs/research-extended.md §2.3](../../../docs/research-extended.md#23-reducer-pseudocode-used-as-jnana-core)
in log-space using `scipy.special.gammaln`.

The full prior is constructed from a uniform Dirichlet `Dir(1,1,...,1)` of
length K updated with pseudo-counts
`α_k = 1 + λ_a · ananda_k + λ_p · shift(apoha_k)` (v0.2: ADR-002), where
`shift` is a min-max normalization into `[0, 1]`. K reduced priors are then
enumerated; ΔF for each is computed via the log-Beta form.

v0.2 change: the v0.1 `np.clip(apoha, 0., None)` discarded negative apoha
evidence (P1-4 in the adversarial review). The min-max shift in `apohana`
preserves the relative ordering of must-avoid penalties so candidates close
to the avoid set lose posterior mass instead of tying with neutral
candidates.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from pce.operators.apohana import _shift_apoha

EPS = 1e-7
# v0.3 ADR-003: cosine threshold for "candidate satisfies aspect i" used by
# the aspect-conditioned reductions. Tuned against the duck-rabbit fixture
# where MiniLM avg-pooling gives ~0.30-0.45 cosine on hits and <0.20 on
# misses. The same floor is used by vimarsa for aspect counting; we rebind
# here rather than importing vimarsa to keep the operator boundary clean.
ASPECT_HIT_FLOOR = 0.30
ReductionTarget = Literal["halve", "single", "custom", "aspect_conditioned"]


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


def _enumerate_aspect_conditioned_reductions(
    K: int,
    aspect_membership: npt.NDArray[np.float32],
    aspect_priors: npt.NDArray[np.float32] | None,
    *,
    boost: float = 4.0,
) -> list[npt.NDArray[np.float64]]:
    """Enumerate one reduction per non-empty aspect subset (S_i).

    ``aspect_membership`` is a (K, A) matrix of cosine similarities between
    candidate ``k`` and aspect ``i``; values in roughly ``[-1, 1]``. We
    binarize at ``aspect_membership >= ASPECT_HIT_FLOOR`` to get the
    aspect→candidate support sets. Each non-empty support set becomes one
    reduced prior that **boosts** the in-support concentration parameters
    relative to the uniform full prior.

    Per ADR-003 we use a *soft* reduction (boost concentration to ``1+boost``
    for in-support, keep ``1.0`` for out-of-support, and weight the boost by
    the aspect prior) rather than a *hard* reduction (EPS for out-of-support)
    because hard reductions blow up ``log Z(reduced_prior)`` via
    ``lgamma(EPS) ≈ 16`` and make ΔF deeply negative on every IFR reduction
    even when the surface satisfies the aspect. The soft form keeps the BMR
    posterior contrast informative (positive ΔF on real aspect coverage).

    We also include:

    * The intersection of all aspect supports (the IFR reduction the cascade
      hopes will win on duck-rabbit-like items), with a slightly stronger
      boost ``boost * 1.5``.
    * The unchanged full prior so jñāna can "decline to reduce" when no
      aspect is satisfied -- this is the calibration baseline against which
      ΔF is measured.

    Returns a list of length-K float64 priors, all entries ``>= 1.0`` so
    ``lgamma`` stays bounded.
    """
    if aspect_membership.shape[0] != K:
        raise ValueError(
            f"jnana: aspect_membership rows must equal K={K}, got {aspect_membership.shape}"
        )
    A = int(aspect_membership.shape[1])
    out: list[npt.NDArray[np.float64]] = []
    if aspect_priors is None or aspect_priors.shape[0] != A or float(aspect_priors.sum()) <= 0.0:
        priors = np.ones((A,), dtype=np.float64) / max(A, 1)
    else:
        priors = np.asarray(aspect_priors, dtype=np.float64)
        priors = priors / float(priors.sum() + 1e-30)
    hit = (aspect_membership >= ASPECT_HIT_FLOOR).astype(np.float64)  # (K, A)
    base = float(boost)
    # Per-aspect support reductions: boost in-support, leave others at 1.
    any_support = False
    for i in range(A):
        support = hit[:, i] > 0.5
        if not bool(support.any()):
            continue
        any_support = True
        r = np.ones((K,), dtype=np.float64)
        r[support] = 1.0 + base * float(priors[i] * A)  # rescale so total prior mass tracks A
        out.append(r)
    # IFR reduction: candidates that hit every aspect, boosted harder.
    if A > 0:
        full_support = hit.sum(axis=1) >= float(A)
        if bool(full_support.any()):
            any_support = True
            r = np.ones((K,), dtype=np.float64)
            r[full_support] = 1.0 + base * 1.5
            out.append(r)
    # Always include the unchanged full prior so jnana can "decline to reduce".
    out.append(np.ones((K,), dtype=np.float64))
    if not any_support and len(out) == 1:
        # No aspect ever hit; the only entry is the uniform reduction.
        # Append a zero-boost copy so the loop in jnana has at least 2 candidates.
        out.append(np.ones((K,), dtype=np.float64))
    return out


def _enumerate_reductions(
    K: int, target: ReductionTarget, pseudo_counts: npt.NDArray[np.float64],
    custom: list[npt.NDArray[np.float64]] | None,
    aspect_membership: npt.NDArray[np.float32] | None = None,
    aspect_priors: npt.NDArray[np.float32] | None = None,
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
    if target == "aspect_conditioned":
        if aspect_membership is None:
            raise ValueError(
                "jnana: reduction_target='aspect_conditioned' requires `aspect_membership`"
            )
        return _enumerate_aspect_conditioned_reductions(K, aspect_membership, aspect_priors)
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
    aspect_membership: npt.NDArray[np.float32] | None = None,
    aspect_priors: npt.NDArray[np.float32] | None = None,
) -> tuple[int, float, npt.NDArray[np.float32]]:
    """Returns (selected_index, best_delta_F, posterior).

    `selected_index` is the candidate that survives the winning reduction
    (i.e. the argmax of the reduced posterior).

    v0.3 (ADR-003): when ``reduction_target="aspect_conditioned"`` the BMR
    enumerates one reduction per aspect support set (and one IFR reduction
    over the full intersection). ``aspect_membership`` is a (K, A) matrix
    of aspect-cosine scores per candidate, and ``aspect_priors`` is an
    optional (A,) prior over aspects (typically the storehouse-attended
    aspect mass returned by :class:`pce.active_inference.HopfieldStore`).
    The non-degenerate ΔF this produces is what the active-inference
    uplift gate measures.
    """
    K = len(candidates)
    if K < 2:
        raise ValueError(f"jnana: need K >= 2 candidates, got {K}")
    if apoha_scores.shape != (K,) or ananda_scores.shape != (K,):
        raise ValueError(
            f"jnana: score shapes must be ({K},), got apoha={apoha_scores.shape} ananda={ananda_scores.shape}"
        )
    apoha = apoha_scores.astype(np.float32)
    anan = ananda_scores.astype(np.float64)
    apoha_shift = _shift_apoha(apoha).astype(np.float64)
    pseudo = 1.0 + float(lambda_a) * anan + float(lambda_p) * apoha_shift

    full_prior = np.ones((K,), dtype=np.float64)
    a_post = full_prior + pseudo - 1.0  # i.e. = pseudo

    reductions = _enumerate_reductions(
        K, reduction_target, pseudo, custom_priors,
        aspect_membership=aspect_membership,
        aspect_priors=aspect_priors,
    )
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

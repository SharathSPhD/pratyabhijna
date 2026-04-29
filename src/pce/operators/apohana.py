"""`apohana-śakti` - contrastive exclusion (anti-non-X scoring).

For each candidate, return cos(c_k, q) - max_n cos(c_k, n) where q is the
constraint embedding and n ranges over the constraint's `must_avoid`
exemplars.

v0.2 (ADR-002): adds an opt-in `normalize` flag that min-max-shifts the raw
contrastive score into `[0, 1]` over the K candidates of the current call.
The shift is idempotent on inputs already in `[0, 1]`. Default
`normalize=False` preserves backward compatibility for direct callers; the
two-pass cascade in `pce.cascade.run_cascade` passes `normalize=True`.

v0.3 (ADR-004): adds an opt-in ``hopfield`` warm-start. When the cascade
threads in a per-domain :class:`pce.active_inference.HopfieldStore`, each
candidate's apoha score is augmented by ``hopfield_weight * att_k`` where
``att_k`` is the Hopfield softmax mass of the candidate against the stored
patterns. This biases jñāna toward candidates that resemble previously
consolidated successful surfaces in the same domain. Default
``hopfield=None`` reproduces v0.2 exactly.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pce.active_inference.hopfield import HopfieldStore
from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint


def _shift_apoha(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Min-max shift K raw apoha scores into [0, 1]; idempotent on [0, 1] input.

    Used by both `apohana(normalize=True)` and `jnana` (which applies the shift
    unconditionally on the input it receives, per ADR-002). The shift collapses
    to a constant 0.5 vector when all K candidates have identical apoha so the
    pseudo-counts stay symmetric in that degenerate case.
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    arr = np.asarray(x, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def apohana(
    candidates: tuple[Candidate, ...],
    constraint: Constraint,
    *,
    embed: Embedder,
    normalize: bool = False,
    hopfield: HopfieldStore | None = None,
    hopfield_weight: float = 0.25,
) -> npt.NDArray[np.float32]:
    if not candidates:
        return np.zeros((0,), dtype=np.float32)
    cand_emb = np.stack([c.embedding for c in candidates], axis=0).astype(np.float32)
    q = constraint.embedding
    pos = cand_emb @ q  # shape (K,)
    if not constraint.must_avoid:
        raw = pos.astype(np.float32)
    else:
        avoid_emb: np.ndarray = embed.encode(list(constraint.must_avoid))  # type: ignore[type-arg]
        if avoid_emb.ndim == 1:
            avoid_mat = avoid_emb[None, :]
        else:
            avoid_mat = avoid_emb
        neg = cand_emb @ avoid_mat.T  # shape (K, n_avoid)
        neg_max = neg.max(axis=1)
        raw = (pos - neg_max).astype(np.float32)
    if hopfield is not None and hopfield.n_patterns > 0:
        # Warm-start: per-candidate softmax mass under the Hopfield query.
        # We query the storehouse with each candidate's embedding and use the
        # MAX of stored-pattern cosine as the per-candidate retrieval signal,
        # then min-max normalize across K so the warm-start is on a comparable
        # scale to apoha. This stays bounded and never erases the apoha signal.
        bonus = np.zeros((cand_emb.shape[0],), dtype=np.float32)
        for k in range(cand_emb.shape[0]):
            res = hopfield.query(cand_emb[k])
            # Use the max cosine to any stored pattern, derived from energy =
            # -1/beta * logsumexp(beta * sims). We approximate by the softmax-
            # weighted retrieval norm: ||retrieved|| since patterns are unit norm.
            norm = float(np.linalg.norm(res.retrieved))
            bonus[k] = min(1.0, max(0.0, norm))
        bonus_norm = _shift_apoha(bonus)
        raw = (raw + np.float32(hopfield_weight) * bonus_norm).astype(np.float32)
    if normalize:
        return _shift_apoha(raw)
    return raw

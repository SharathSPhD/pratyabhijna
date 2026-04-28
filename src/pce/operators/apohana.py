"""`apohana-śakti` - contrastive exclusion (anti-non-X scoring).

For each candidate, return cos(c_k, q) - max_n cos(c_k, n) where q is the
constraint embedding and n ranges over the constraint's `must_avoid`
exemplars.

v0.2 (ADR-002): adds an opt-in `normalize` flag that min-max-shifts the raw
contrastive score into `[0, 1]` over the K candidates of the current call.
The shift is idempotent on inputs already in `[0, 1]`. Default
`normalize=False` preserves backward compatibility for direct callers; the
two-pass cascade in `pce.cascade.run_cascade` passes `normalize=True`.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

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
    if normalize:
        return _shift_apoha(raw)
    return raw

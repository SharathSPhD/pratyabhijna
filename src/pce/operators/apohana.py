"""`apohana-śakti` - contrastive exclusion (anti-non-X scoring).

For each candidate, return cos(c_k, q) - max_n cos(c_k, n) where q is the
constraint embedding and n ranges over the constraint's `must_avoid`
exemplars.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint


def apohana(
    candidates: tuple[Candidate, ...],
    constraint: Constraint,
    *,
    embed: Embedder,
) -> npt.NDArray[np.float32]:
    if not candidates:
        return np.zeros((0,), dtype=np.float32)
    cand_emb = np.stack([c.embedding for c in candidates], axis=0).astype(np.float32)
    q = constraint.embedding
    pos = cand_emb @ q  # shape (K,)
    if not constraint.must_avoid:
        return pos.astype(np.float32)
    avoid_emb: np.ndarray = embed.encode(list(constraint.must_avoid))  # type: ignore[type-arg]
    if avoid_emb.ndim == 1:  # single avoid - shape (D,)
        avoid_mat = avoid_emb[None, :]
    else:
        avoid_mat = avoid_emb
    neg = cand_emb @ avoid_mat.T  # shape (K, n_avoid)
    neg_max = neg.max(axis=1)
    out: npt.NDArray[np.float32] = (pos - neg_max).astype(np.float32)
    return out

"""Modern continuous Hopfield store -- the storehouse (ālaya-vijñāna) layer.

The store owns a per-domain matrix of unit-norm aspect/surface embeddings.
``query`` returns the softmax-attended retrieval over the matrix (Ramsauer et
al. 2020, *Hopfield Networks Is All You Need*) so callers can use it as a
warm-start prior. ``write`` appends a new pattern (REM mode) or replaces the
nearest existing pattern (SWS / slow-consolidation mode); ``persist`` writes
the matrix to ``audit/storehouse/<domain>.npz``.

Only the v0.3 cascade calls into this store: :func:`pce.operators.apohana.apohana`
queries it for warm-start aspect mass and :func:`pce.operators.vimarsa.consolidate`
writes back the committed surface at the end of every prompt. The storehouse
is reset between domains for the benchmark so prompts within a domain share
context but cross-domain runs stay independent.

Mathematically the lookup is

.. math::

    p_i = \\text{softmax}(\\beta\\, X x)_i, \\quad \\hat{x} = X^\\top p

where ``X`` is the (N, D) row-normalized pattern matrix, ``x`` is the unit
query, and ``β`` is an inverse-temperature (default 8.0). The retrieval
"energy" we expose to apohana is the negative log-sum-exp value
``-1/β · logsumexp(β X x)``: lower is more confidently retrieved (closer to
a stored aspect), higher is more diffuse / novel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

WriteMode = Literal["rem", "sws"]
EmbeddingArray = npt.NDArray[np.float32]


def _l2_normalize(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Row-wise L2 normalize (1D treated as a single row)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr) + 1e-12)
        out: npt.NDArray[np.float32] = (arr / n).astype(np.float32)
        return out
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    out2: npt.NDArray[np.float32] = (arr / norms).astype(np.float32)
    return out2


@dataclass(frozen=True)
class HopfieldQueryResult:
    """The result of one :meth:`HopfieldStore.query` call."""

    n_patterns: int
    """Number of stored patterns at query time (0 if the store is empty)."""

    energy: float
    """The Hopfield retrieval energy, ``-1/β · logsumexp(β · X x)``.

    Smaller (more negative) values indicate confident retrieval; ``+inf``
    means the store is empty so no warm-start is available.
    """

    attention: npt.NDArray[np.float32]
    """The softmax over stored patterns; shape ``(N,)``. Empty array if N=0."""

    retrieved: EmbeddingArray
    """The convex combination of stored patterns under ``attention``;
    shape ``(D,)``. Equal to the query when N=0 (zero-warm-start fallback)."""

    aspect_priors: npt.NDArray[np.float32]
    """One scalar per *aspect* in the per-query aspect set, giving the mass
    that the storehouse softmax assigns to patterns labelled with that
    aspect. Used by jñāna BMR aspect-conditioned reductions and by apohana
    as a warm-start mixture weight. Shape ``(A,)`` where A is the number
    of aspects supplied to :meth:`query`. Empty when no aspects supplied."""


class HopfieldStore:
    """Per-domain modern Hopfield storehouse.

    The store is created either fresh (``HopfieldStore(domain="poetry_gen")``)
    or from a saved snapshot (``HopfieldStore.load(path, domain=...)``). Use
    :meth:`query` from operators that need a warm-start prior, :meth:`write`
    from :func:`pce.operators.vimarsa.consolidate`, and :meth:`persist` from
    benchmark drivers that want cross-prompt continuity within a domain.

    Persistence format: ``audit/storehouse/<domain>.npz`` with keys
    ``patterns`` (float32 (N, D)), ``aspect_labels`` (object array length N),
    and ``meta`` (json-encoded dict with ``beta``, ``capacity``).
    """

    def __init__(
        self,
        *,
        domain: str,
        beta: float = 8.0,
        capacity: int = 256,
        sws_replace_threshold: float = 0.92,
    ) -> None:
        if not domain:
            raise ValueError("HopfieldStore: domain must be a non-empty string")
        if beta <= 0:
            raise ValueError(f"HopfieldStore: beta must be > 0, got {beta}")
        if capacity <= 0:
            raise ValueError(f"HopfieldStore: capacity must be > 0, got {capacity}")
        if not 0.0 < sws_replace_threshold <= 1.0:
            raise ValueError(
                "HopfieldStore: sws_replace_threshold must be in (0, 1], "
                f"got {sws_replace_threshold}"
            )
        self.domain = str(domain)
        self.beta = float(beta)
        self.capacity = int(capacity)
        self.sws_replace_threshold = float(sws_replace_threshold)
        self._patterns: EmbeddingArray = np.zeros((0, 0), dtype=np.float32)
        self._labels: list[str] = []

    @property
    def n_patterns(self) -> int:
        return int(self._patterns.shape[0])

    @property
    def dim(self) -> int:
        return int(self._patterns.shape[1]) if self._patterns.size else 0

    def query(
        self,
        query_vec: npt.NDArray[np.float32],
        *,
        aspect_labels: list[str] | None = None,
    ) -> HopfieldQueryResult:
        """Softmax-attend over stored patterns.

        ``aspect_labels`` is the ordered list of aspect names whose per-aspect
        mass should be returned in :attr:`HopfieldQueryResult.aspect_priors`.
        Each entry sums the softmax mass of every stored pattern whose label
        equals that aspect.
        """
        q = _l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        labels = list(aspect_labels or [])
        if self.n_patterns == 0:
            return HopfieldQueryResult(
                n_patterns=0,
                energy=float("inf"),
                attention=np.zeros((0,), dtype=np.float32),
                retrieved=q.copy(),
                aspect_priors=np.zeros((len(labels),), dtype=np.float32),
            )
        scores = self._patterns @ q  # (N,)
        # Energy = -1/beta * logsumexp(beta * scores) (Ramsauer 2020 eq. 1).
        m = float(np.max(scores))
        lse = float(m + np.log(np.exp(self.beta * (scores - m)).sum() + 1e-30) / 1.0)
        energy = -lse / self.beta
        # Softmax attention.
        att_logits = self.beta * scores
        att_logits -= float(att_logits.max())
        att = np.exp(att_logits, dtype=np.float64)
        att /= att.sum() + 1e-30
        att32 = att.astype(np.float32)
        retrieved = (att32 @ self._patterns).astype(np.float32)
        if labels:
            priors = np.zeros((len(labels),), dtype=np.float32)
            for i, lab in enumerate(labels):
                mass = float(sum(float(att32[j]) for j, p in enumerate(self._labels) if p == lab))
                priors[i] = mass
        else:
            priors = np.zeros((0,), dtype=np.float32)
        return HopfieldQueryResult(
            n_patterns=self.n_patterns,
            energy=float(energy),
            attention=att32,
            retrieved=retrieved,
            aspect_priors=priors,
        )

    def write(
        self,
        embedding: npt.NDArray[np.float32],
        *,
        label: str,
        mode: WriteMode = "rem",
    ) -> None:
        """Add (REM) or consolidate (SWS) one pattern.

        REM appends. SWS finds the nearest stored pattern by cosine and, if
        cosine ≥ ``sws_replace_threshold``, averages with it instead of
        appending; below the threshold it falls through to REM. Capacity is
        enforced FIFO: if writing would exceed capacity the oldest pattern
        is dropped.
        """
        if mode not in ("rem", "sws"):
            raise ValueError(f"HopfieldStore.write: unknown mode={mode!r}")
        emb = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if self.dim and emb.shape[0] != self.dim:
            raise ValueError(
                f"HopfieldStore.write: embedding dim {emb.shape[0]} != store dim {self.dim}"
            )
        if self.n_patterns == 0:
            self._patterns = emb[None, :].copy()
            self._labels = [str(label)]
            return
        if mode == "sws":
            sims = self._patterns @ emb
            j_best = int(np.argmax(sims))
            if float(sims[j_best]) >= self.sws_replace_threshold:
                merged = _l2_normalize(0.5 * (self._patterns[j_best] + emb))
                self._patterns[j_best] = merged
                # Keep the existing label (slow consolidation does not relabel).
                return
        # REM (or SWS fall-through): append, drop oldest if at capacity.
        if self.n_patterns >= self.capacity:
            self._patterns = self._patterns[1:]
            self._labels = self._labels[1:]
        self._patterns = np.vstack([self._patterns, emb[None, :]]).astype(np.float32)
        self._labels.append(str(label))

    def persist(self, root: Path | str = Path("audit/storehouse")) -> Path:
        """Write the matrix + labels + meta to ``root/<domain>.npz``."""
        root_p = Path(root)
        root_p.mkdir(parents=True, exist_ok=True)
        out = root_p / f"{self.domain}.npz"
        np.savez(
            out,
            patterns=self._patterns,
            aspect_labels=np.array(self._labels, dtype=object),
            meta=np.array({"beta": self.beta, "capacity": self.capacity}, dtype=object),
        )
        return out

    @classmethod
    def load(cls, path: Path | str, *, domain: str | None = None) -> HopfieldStore:
        """Restore a store from a snapshot written by :meth:`persist`."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"HopfieldStore.load: {p} does not exist")
        with np.load(p, allow_pickle=True) as data:
            patterns = np.asarray(data["patterns"], dtype=np.float32)
            labels = [str(x) for x in data["aspect_labels"].tolist()]
            meta_obj = data["meta"].item()
        beta = float(meta_obj.get("beta", 8.0)) if isinstance(meta_obj, dict) else 8.0
        capacity = int(meta_obj.get("capacity", 256)) if isinstance(meta_obj, dict) else 256
        store = cls(domain=str(domain or p.stem), beta=beta, capacity=capacity)
        store._patterns = patterns
        store._labels = labels
        return store

    def reset(self) -> None:
        """Clear all stored patterns. Used between domains in the benchmark."""
        self._patterns = np.zeros((0, 0), dtype=np.float32)
        self._labels = []

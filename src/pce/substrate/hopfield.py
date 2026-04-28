"""Hopfield-style attractor store for the ālayavijñāna substrate.

A continuous Hopfield network (Krotov–Hopfield modern attractor; Ramsauer
2020) over L2-normalized embedding vectors. The store keeps all patterns ever
written and recalls the closest attractor by iterative softmax descent over
the inner-product energy:

    E(x) = -lse(beta * X x)

with `lse` the log-sum-exp and X the matrix of stored patterns. The recall
rule is the corresponding gradient step:

    x_{t+1} = X^T softmax(beta * X x_t)

This is exact for normalized vectors and gives the storage capacity discussed
in Ramsauer 2020 / research-extended.md §3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class HopfieldStore:
    """Continuous (modern) Hopfield network for embedding-shaped patterns."""

    dim: int
    beta: float = 8.0
    patterns: list[npt.NDArray[np.float32]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.patterns is None:
            self.patterns = []
        if self.dim <= 0:
            raise ValueError("HopfieldStore.dim must be positive")
        if self.beta <= 0:
            raise ValueError("HopfieldStore.beta must be positive")

    @property
    def n_patterns(self) -> int:
        return len(self.patterns)

    def store(self, pattern: npt.NDArray[np.float32]) -> None:
        if pattern.shape != (self.dim,):
            raise ValueError(f"pattern shape {pattern.shape} != ({self.dim},)")
        normalized = pattern / (np.linalg.norm(pattern) + 1e-12)
        self.patterns.append(normalized.astype(np.float32))

    def _matrix(self) -> npt.NDArray[np.float32]:
        if not self.patterns:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self.patterns, axis=0).astype(np.float32)

    def recall(
        self,
        cue: npt.NDArray[np.float32],
        *,
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> npt.NDArray[np.float32]:
        if not self.patterns:
            return cue.astype(np.float32) / (np.linalg.norm(cue) + 1e-12)
        x = cue.astype(np.float32) / (np.linalg.norm(cue) + 1e-12)
        X = self._matrix()
        prev = x
        for _ in range(int(max_iter)):
            scores = self.beta * (X @ x)
            scores -= scores.max()  # logsumexp stability
            weights = np.exp(scores)
            weights /= weights.sum() + 1e-30
            x = (X.T @ weights).astype(np.float32)
            x = x / (np.linalg.norm(x) + 1e-12)
            if float(np.linalg.norm(x - prev)) < tol:
                break
            prev = x
        return x

    def consolidate_sws(
        self,
        traces: list[npt.NDArray[np.float32]],
        *,
        n_centroids: int = 4,
        n_iter: int = 25,
        seed: int = 0,
    ) -> list[npt.NDArray[np.float32]]:
        """SWS-like schema abstraction: k-means on `traces`, return centroids.

        Each centroid is then *stored* back into the Hopfield network, simulating
        the SWS-style abstraction-and-replay pattern.
        """
        if not traces:
            return []
        rng = np.random.default_rng(int(seed))
        X = np.stack([t / (np.linalg.norm(t) + 1e-12) for t in traces], axis=0).astype(np.float32)
        k = int(min(n_centroids, len(traces)))
        # k-means++ init.
        idx = [int(rng.integers(0, len(traces)))]
        for _ in range(1, k):
            # Cosine distance to nearest centroid so far. Clip to [0, 2] to absorb
            # the tiny negative-cosine drift that arises when rounded unit vectors
            # produce dot-products slightly above 1.
            dists = np.clip(
                np.min(1.0 - X @ X[idx].T, axis=1).astype(np.float64),
                0.0,
                2.0,
            )
            total = float(dists.sum())
            if total <= 1e-12:
                # All remaining traces are degenerate copies of the seeds; fall
                # back to uniform sampling so we still draw k distinct indices.
                probs = np.ones(len(traces), dtype=np.float64) / float(len(traces))
            else:
                probs = (dists / total).astype(np.float64)
            idx.append(int(rng.choice(len(traces), p=probs)))
        centroids = X[idx].copy()
        for _ in range(int(n_iter)):
            sims = X @ centroids.T  # (n, k)
            assign = np.argmax(sims, axis=1)
            new_centroids = centroids.copy()
            for c in range(k):
                members = X[assign == c]
                if len(members) > 0:
                    new = members.mean(axis=0)
                    new_centroids[c] = new / (np.linalg.norm(new) + 1e-12)
            if np.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids
        out = [c.astype(np.float32) for c in centroids]
        for c in out:
            self.store(c)
        return out

    def consolidate_rem(
        self,
        *,
        n_steps: int = 100,
        temperature: float = 1.5,
        seed: int = 0,
    ) -> list[npt.NDArray[np.float32]]:
        """REM-like cross-basin replay: random walk between attractors at high temperature.

        Implements a Metropolis-style hop between stored basins by perturbing
        a random pattern with isotropic Gaussian noise and recalling at a
        higher beta-effective (temperature). Returns the trajectory.
        """
        if not self.patterns:
            return []
        rng = np.random.default_rng(int(seed))
        beta_orig = self.beta
        try:
            self.beta = max(beta_orig / float(temperature), 0.5)
            x = self.patterns[int(rng.integers(0, self.n_patterns))].copy()
            traj: list[npt.NDArray[np.float32]] = []
            for _ in range(int(n_steps)):
                noise = rng.normal(0.0, 0.1 * float(temperature), size=self.dim).astype(np.float32)
                x = x + noise
                x = self.recall(x, max_iter=20)
                traj.append(x.copy())
            return traj
        finally:
            self.beta = beta_orig

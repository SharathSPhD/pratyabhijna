"""Hopfield store invariants - pure numpy, no real-model marker required."""
from __future__ import annotations

import numpy as np
import pytest

from pce.substrate.hopfield import HopfieldStore


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)


def test_construction_and_store(rng: np.random.Generator) -> None:
    h = HopfieldStore(dim=16)
    assert h.n_patterns == 0
    p = _normalize(rng.normal(size=16).astype(np.float32))
    h.store(p)
    assert h.n_patterns == 1


def test_store_rejects_wrong_shape() -> None:
    h = HopfieldStore(dim=16)
    with pytest.raises(ValueError):
        h.store(np.zeros(15, dtype=np.float32))


def test_recall_returns_stored_when_cue_close(rng: np.random.Generator) -> None:
    h = HopfieldStore(dim=32, beta=12.0)
    patterns = [_normalize(rng.normal(size=32).astype(np.float32)) for _ in range(4)]
    for p in patterns:
        h.store(p)
    target = patterns[2]
    cue = _normalize(target + 0.05 * rng.normal(size=32).astype(np.float32))
    out = h.recall(cue)
    assert float(np.dot(out, target)) > 0.95


def test_recall_idempotent(rng: np.random.Generator) -> None:
    h = HopfieldStore(dim=32, beta=10.0)
    p = _normalize(rng.normal(size=32).astype(np.float32))
    h.store(p)
    out1 = h.recall(p)
    out2 = h.recall(out1)
    assert np.allclose(out1, out2, atol=1e-5)


def test_consolidate_sws_returns_centroids(rng: np.random.Generator) -> None:
    h = HopfieldStore(dim=32, beta=8.0)
    cluster_a = [_normalize(np.array([1.0] * 16 + [0.0] * 16, dtype=np.float32)
                            + 0.05 * rng.normal(size=32).astype(np.float32))
                 for _ in range(8)]
    cluster_b = [_normalize(np.array([0.0] * 16 + [1.0] * 16, dtype=np.float32)
                            + 0.05 * rng.normal(size=32).astype(np.float32))
                 for _ in range(8)]
    centroids = h.consolidate_sws(cluster_a + cluster_b, n_centroids=2, n_iter=30, seed=0)
    assert len(centroids) == 2
    cosines = [float(np.dot(c, _normalize(np.array([1.0] * 16 + [0.0] * 16, dtype=np.float32))))
               for c in centroids]
    assert max(cosines) > 0.95
    assert min(cosines) < 0.20


def test_consolidate_rem_walks_basins(rng: np.random.Generator) -> None:
    h = HopfieldStore(dim=32, beta=10.0)
    for _ in range(6):
        h.store(_normalize(rng.normal(size=32).astype(np.float32)))
    traj = h.consolidate_rem(n_steps=20, temperature=1.5, seed=0)
    assert len(traj) == 20
    norms = [float(np.linalg.norm(x)) for x in traj]
    assert all(abs(n - 1.0) < 1e-4 for n in norms)


def test_recall_with_empty_store_returns_normalized_cue(rng: np.random.Generator) -> None:
    h = HopfieldStore(dim=8)
    cue = rng.normal(size=8).astype(np.float32)
    out = h.recall(cue)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_invalid_construction() -> None:
    with pytest.raises(ValueError):
        HopfieldStore(dim=0)
    with pytest.raises(ValueError):
        HopfieldStore(dim=8, beta=0.0)

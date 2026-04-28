"""SWS / REM consolidation cycles."""
from __future__ import annotations

import numpy as np

from pce.consolidation.sleep import is_consolidated, run_rem, run_sleep_cycle, run_sws
from pce.substrate.hopfield import HopfieldStore


def _orthogonal_pair(dim: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=dim).astype(np.float32)
    b = rng.normal(size=dim).astype(np.float32)
    a /= np.linalg.norm(a) + 1e-12
    b -= float(np.dot(a, b)) * a
    b /= np.linalg.norm(b) + 1e-12
    return a, b


def test_sws_creates_centroids() -> None:
    rng = np.random.default_rng(0)
    store = HopfieldStore(dim=32)
    traces = []
    a, b = _orthogonal_pair(32)
    for _ in range(20):
        n = rng.normal(scale=0.05, size=32).astype(np.float32)
        traces.append((a + n).astype(np.float32))
    for _ in range(20):
        n = rng.normal(scale=0.05, size=32).astype(np.float32)
        traces.append((b + n).astype(np.float32))
    centroids = run_sws(store, traces, n_centroids=2, n_iter=20, seed=0)
    assert len(centroids) == 2
    assert store.n_patterns >= 2
    cos_a = max(float(np.dot(c, a)) for c in centroids)
    cos_b = max(float(np.dot(c, b)) for c in centroids)
    assert cos_a > 0.85 and cos_b > 0.85


def test_rem_returns_trajectory_when_patterns_exist() -> None:
    store = HopfieldStore(dim=16)
    a, b = _orthogonal_pair(16)
    store.store(a)
    store.store(b)
    traj = run_rem(store, n_steps=20, temperature=1.5, seed=1)
    assert len(traj) == 20
    for x in traj:
        assert x.shape == (16,)


def test_full_sleep_cycle_records_diagnostics() -> None:
    rng = np.random.default_rng(1)
    store = HopfieldStore(dim=32)
    a, b = _orthogonal_pair(32)
    traces = []
    for _ in range(10):
        traces.append((a + rng.normal(scale=0.03, size=32).astype(np.float32)).astype(np.float32))
    for _ in range(10):
        traces.append((b + rng.normal(scale=0.03, size=32).astype(np.float32)).astype(np.float32))
    diag = run_sleep_cycle(store, traces, sws_centroids=2, sws_iter=20, rem_steps=10, seed=2)
    assert diag["n_sws_centroids"] == 2
    assert diag["n_rem_steps"] == 10
    assert diag["n_patterns_after"] >= diag["n_patterns_before"] + 2


def test_is_consolidated_after_storage() -> None:
    store = HopfieldStore(dim=16)
    a, b = _orthogonal_pair(16)
    store.store(a)
    assert is_consolidated(store, a, threshold=0.95)
    # b is orthogonal to a -> recall returns a, dot ~ 0
    assert not is_consolidated(store, b, threshold=0.95)

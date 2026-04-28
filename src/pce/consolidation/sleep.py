"""SWS / REM consolidation cycles for the Hopfield store.

`run_sws` performs the deep-sleep abstraction step (k-means over recent traces).
`run_rem` performs the REM Metropolis replay step. Together they produce a
post-consolidation HopfieldStore whose patterns include both the originals and
the abstracted/replayed centroids.

Background: see [docs/research-extended.md §6](../../../docs/research-extended.md#6-sleepconsolidation-loops).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pce.substrate.hopfield import HopfieldStore


def run_sws(
    store: HopfieldStore,
    traces: list[npt.NDArray[np.float32]],
    *,
    n_centroids: int = 4,
    n_iter: int = 25,
    seed: int = 0,
) -> list[npt.NDArray[np.float32]]:
    """Slow-wave consolidation: cluster recent traces, store centroids."""
    return store.consolidate_sws(traces, n_centroids=n_centroids, n_iter=n_iter, seed=seed)


def run_rem(
    store: HopfieldStore,
    *,
    n_steps: int = 100,
    temperature: float = 1.5,
    seed: int = 0,
) -> list[npt.NDArray[np.float32]]:
    """REM consolidation: stochastic recall trajectory at higher temperature."""
    return store.consolidate_rem(n_steps=n_steps, temperature=temperature, seed=seed)


def run_sleep_cycle(
    store: HopfieldStore,
    traces: list[npt.NDArray[np.float32]],
    *,
    sws_centroids: int = 4,
    sws_iter: int = 25,
    rem_steps: int = 100,
    rem_temperature: float = 1.5,
    seed: int = 0,
) -> dict[str, int]:
    """Run one SWS pass, then one REM pass. Return diagnostics."""
    n_before = store.n_patterns
    centroids = run_sws(store, traces, n_centroids=sws_centroids, n_iter=sws_iter, seed=seed)
    rem_traj = run_rem(store, n_steps=rem_steps, temperature=rem_temperature, seed=seed + 1)
    return {
        "n_patterns_before": int(n_before),
        "n_patterns_after": int(store.n_patterns),
        "n_sws_centroids": int(len(centroids)),
        "n_rem_steps": int(len(rem_traj)),
    }


def is_consolidated(
    store: HopfieldStore,
    pattern: npt.NDArray[np.float32],
    *,
    threshold: float = 0.92,
) -> bool:
    """Probe: does `pattern` recall to itself in the store?"""
    if store.n_patterns == 0:
        return False
    recalled = store.recall(pattern)
    rec_norm = recalled / (np.linalg.norm(recalled) + 1e-12)
    pat_norm = pattern / (np.linalg.norm(pattern) + 1e-12)
    return float(np.dot(rec_norm, pat_norm)) >= float(threshold)

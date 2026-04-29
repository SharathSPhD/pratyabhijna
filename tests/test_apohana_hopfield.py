"""Phase 3 ADR-004 gate: HopfieldStore + apohana warm-start integration.

Verifies that:

* Empty store: query returns ``+inf`` energy, retrieval == query, attention
  has length 0.
* REM write appends; SWS write merges nearby patterns; capacity enforced FIFO.
* Aspect priors come back as a per-aspect mass vector.
* ``apohana(hopfield=store)`` shifts scores toward candidates that look like
  stored patterns; ``apohana(hopfield=None)`` reproduces v0.2 byte-for-byte.
* persist/load round-trip preserves ``patterns``, ``labels``, ``meta``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from pce.active_inference.hopfield import HopfieldStore
from pce.operators.apohana import apohana
from pce.types import Candidate, Constraint


def _unit(v: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32))


def _make_candidate(seed: int, embedding: np.ndarray) -> Candidate:
    return Candidate(
        seed=seed,
        sampler={"tau": 0.9},
        tokens=(),
        text=f"cand{seed}",
        logp=0.0,
        embedding=_unit(embedding),
    )


class _ConstantEmbedder:
    """Embedder stub that returns the row corresponding to the input string.

    Used only for apohana's `must_avoid` recall path; unit tests below either
    use no must_avoid or supply explicit avoid embeddings. We keep the stub
    minimal so the test does not depend on a real model.
    """

    def __init__(self, lookup: dict[str, np.ndarray]) -> None:
        self.lookup = {k: _unit(v) for k, v in lookup.items()}

    def encode(self, x: str | list[str]) -> np.ndarray:
        if isinstance(x, str):
            return self.lookup[x]
        return np.stack([self.lookup[s] for s in x], axis=0).astype(np.float32)


def test_empty_store_query_returns_infinity_energy() -> None:
    store = HopfieldStore(domain="poetry_gen")
    q = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    res = store.query(q)
    assert res.n_patterns == 0
    assert res.energy == float("inf")
    assert res.attention.shape == (0,)
    np.testing.assert_allclose(res.retrieved, q, atol=1e-6)


def test_rem_write_then_query_returns_close_pattern() -> None:
    store = HopfieldStore(domain="poetry_gen", beta=10.0)
    p1 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    p2 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    store.write(p1, label="duck", mode="rem")
    store.write(p2, label="rabbit", mode="rem")
    res = store.query(p1, aspect_labels=["duck", "rabbit"])
    assert res.n_patterns == 2
    assert res.aspect_priors.shape == (2,)
    assert res.aspect_priors[0] > res.aspect_priors[1]
    assert res.energy < float("inf")


def test_sws_consolidation_merges_near_duplicates() -> None:
    store = HopfieldStore(domain="poetry_gen", sws_replace_threshold=0.90)
    p = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    store.write(p, label="duck", mode="rem")
    near_p = _unit(np.array([0.99, 0.05, 0.0], dtype=np.float32))
    store.write(near_p, label="duck_near", mode="sws")
    assert store.n_patterns == 1, "SWS should have merged the near-duplicate"


def test_capacity_enforces_fifo() -> None:
    store = HopfieldStore(domain="poetry_gen", capacity=2)
    p1 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    p2 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    p3 = _unit(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    store.write(p1, label="a")
    store.write(p2, label="b")
    store.write(p3, label="c")
    assert store.n_patterns == 2
    np.testing.assert_allclose(store._patterns[0], p2, atol=1e-6)
    np.testing.assert_allclose(store._patterns[1], p3, atol=1e-6)


def test_apohana_with_empty_hopfield_matches_apohana_without() -> None:
    """Empty Hopfield must be a no-op so v0.2 callers are unaffected."""
    constraint_emb = _unit(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    constraint = Constraint(text="vivid", embedding=constraint_emb)
    cands = (
        _make_candidate(0, np.array([1.0, 0.5, 0.0], dtype=np.float32)),
        _make_candidate(1, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        _make_candidate(2, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    )
    embedder = _ConstantEmbedder({})
    base = apohana(cands, constraint, embed=embedder)
    empty_store = HopfieldStore(domain="poetry_gen")
    with_empty = apohana(cands, constraint, embed=embedder, hopfield=empty_store)
    np.testing.assert_allclose(base, with_empty, atol=1e-6)


def test_apohana_with_populated_hopfield_biases_toward_stored() -> None:
    constraint_emb = _unit(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    constraint = Constraint(text="vivid", embedding=constraint_emb)
    cand_a = _make_candidate(0, np.array([1.0, 0.5, 0.0], dtype=np.float32))
    cand_b = _make_candidate(1, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    cands = (cand_a, cand_b)
    embedder = _ConstantEmbedder({})
    store = HopfieldStore(domain="poetry_gen", beta=8.0)
    # Store a pattern that is much closer to cand_a than cand_b.
    store.write(cand_a.embedding, label="duck", mode="rem")
    base = apohana(cands, constraint, embed=embedder)
    biased = apohana(cands, constraint, embed=embedder, hopfield=store, hopfield_weight=0.5)
    # cand_a should be lifted *relative* to cand_b by the warm-start.
    delta_a = float(biased[0] - base[0])
    delta_b = float(biased[1] - base[1])
    assert delta_a >= delta_b, (
        f"Hopfield warm-start should favor stored pattern (a): {delta_a=} {delta_b=}"
    )


def test_persist_load_round_trip(tmp_path: Path) -> None:
    store = HopfieldStore(domain="poetry_gen", beta=12.0, capacity=64)
    p1 = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    p2 = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    store.write(p1, label="duck", mode="rem")
    store.write(p2, label="rabbit", mode="rem")
    out = store.persist(root=tmp_path)
    assert out.exists()
    loaded = HopfieldStore.load(out)
    assert loaded.n_patterns == 2
    assert loaded._labels == ["duck", "rabbit"]
    assert loaded.beta == 12.0
    assert loaded.capacity == 64


def test_audit_to_json_safe() -> None:
    """The HopfieldQueryResult fields must round-trip through ``json.dumps``."""
    store = HopfieldStore(domain="poetry_gen")
    p = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    store.write(p, label="duck", mode="rem")
    res = store.query(p, aspect_labels=["duck", "rabbit"])
    payload = {
        "n_patterns": res.n_patterns,
        "energy": res.energy,
        "attention": res.attention.tolist(),
        "aspect_priors": res.aspect_priors.tolist(),
    }
    s = json.dumps(payload, allow_nan=False)
    assert "duck" not in s  # we did not include labels in this snapshot
    assert "n_patterns" in s

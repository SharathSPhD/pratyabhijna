"""Two-pass cascade tests with a mock LMProtocol.

ADR-003 promises:

- ``state.surface`` is the *revision* (not the draft).
- ``state.surface_draft`` and ``state.surface_revision`` are both populated.
- ``state.vimarsa_brief`` is non-empty.
- ``bypass_vimarsa=True`` collapses to a single pass and ``state.surface ==
  state.surface_draft``.
- The revision pass uses a *different* base seed from the draft pass so the
  candidates differ even when the substrate is deterministic.

The mock LM tracks how many times it was called and returns deterministic
text keyed off the seed - enough to verify two-pass-always semantics
without paying for a real LM call. Real-model end-to-end coverage lives in
``tests/test_cascade.py`` (slow / real_model).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pce.cascade import _REVISION_SEED_OFFSET, run_cascade
from pce.substrate.embed import Embedder
from pce.substrate.lm_protocol import LMProtocol
from pce.types import Candidate, Constraint


class _FakeEmbed(Embedder):
    """Deterministic embedder whose vectors are seeded by string hash."""

    def __init__(self) -> None:
        self.model_id = "fake-embedder"
        self.dim = 16

    def encode(self, texts):  # type: ignore[no-untyped-def, override]
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts], axis=0)

    def _vec(self, t: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class _FakeLM:
    """Mock LMProtocol: returns text whose content depends on prompt + seed."""

    name = "fake-lm"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, prompt: str, *, max_tokens: int, sampler: dict[str, float], seed: int
    ) -> Candidate:
        self.calls.append(
            {"prompt": prompt[:200], "seed": int(seed), "sampler": dict(sampler)}
        )
        # Distinguishable text per (prompt-prefix, seed) so the cascade can
        # detect revision-vs-draft differences.
        if "Reviser brief" in prompt:
            text = f"REVISION-{seed}: refined surface with new aspect."
        else:
            text = f"DRAFT-{seed}: bare surface response."
        emb = np.random.default_rng(seed * 31 + 7).standard_normal(16).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        return Candidate(
            seed=int(seed),
            sampler=dict(sampler),
            tokens=(int(seed),),
            text=text,
            logp=-1.0,
            embedding=emb,
        )

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "n_calls": len(self.calls)}


def _make_lm_protocol_compliant(lm: _FakeLM) -> LMProtocol:
    """Help mypy: assert protocol conformance at runtime."""
    assert isinstance(lm, LMProtocol)
    return lm


@pytest.fixture
def fake_embed() -> Embedder:
    return _FakeEmbed()


@pytest.fixture
def fake_lm() -> _FakeLM:
    return _FakeLM()


def _constraint(embed: Embedder) -> Constraint:
    q = embed.encode("a vivid response with two named aspects")
    return Constraint(
        text="a vivid response",
        embedding=q,
        must_avoid=("a boring single-aspect statement",),
    )


def test_two_pass_default_returns_revision_as_surface(
    fake_lm: _FakeLM, fake_embed: Embedder
) -> None:
    state = run_cascade(
        prompt="Compose a short response.",
        constraint=_constraint(fake_embed),
        lm=_make_lm_protocol_compliant(fake_lm),
        embed=fake_embed,
        K=3,
        max_tokens=32,
        base_seed=0,
        retrieval_set=["unrelated retrieval entry"],
        aspects=["aspect one", "aspect two"],
    )
    assert state.surface is not None
    assert state.surface_draft is not None
    assert state.surface_revision is not None
    assert state.surface == state.surface_revision
    assert state.surface_revision != state.surface_draft
    assert state.vimarsa_brief is not None and len(state.vimarsa_brief) > 0
    assert state.audit["two_pass"] is True
    assert state.audit["revision_differs_from_draft"] is True


def test_two_pass_uses_distinct_seeds_for_revision(
    fake_lm: _FakeLM, fake_embed: Embedder
) -> None:
    K = 3
    base_seed = 100
    run_cascade(
        prompt="Compose a short response.",
        constraint=_constraint(fake_embed),
        lm=_make_lm_protocol_compliant(fake_lm),
        embed=fake_embed,
        K=K,
        max_tokens=32,
        base_seed=base_seed,
        retrieval_set=[],
        aspects=["aspect one", "aspect two"],
    )
    seeds = [c["seed"] for c in fake_lm.calls]
    assert len(seeds) == 2 * K
    draft_seeds = sorted(seeds[:K])
    rev_seeds = sorted(seeds[K:])
    assert draft_seeds == [base_seed + k for k in range(K)]
    assert rev_seeds == [base_seed + _REVISION_SEED_OFFSET + k for k in range(K)]
    # Disjoint so the revision pass explores a different sampler subspace.
    assert set(draft_seeds).isdisjoint(set(rev_seeds))


def test_bypass_vimarsa_returns_draft_as_surface(
    fake_lm: _FakeLM, fake_embed: Embedder
) -> None:
    state = run_cascade(
        prompt="Compose a short response.",
        constraint=_constraint(fake_embed),
        lm=_make_lm_protocol_compliant(fake_lm),
        embed=fake_embed,
        K=3,
        max_tokens=32,
        base_seed=0,
        aspects=["aspect one"],
        bypass_vimarsa=True,
    )
    assert state.surface == state.surface_draft
    assert state.surface_revision is None
    assert state.audit["two_pass"] is False
    assert state.audit["bypassed"] is True


def test_two_pass_calls_lm_2K_times(
    fake_lm: _FakeLM, fake_embed: Embedder
) -> None:
    """Cost validation: two-pass-always doubles iccha calls."""
    K = 4
    run_cascade(
        prompt="Compose.",
        constraint=_constraint(fake_embed),
        lm=_make_lm_protocol_compliant(fake_lm),
        embed=fake_embed,
        K=K,
        max_tokens=16,
        base_seed=42,
    )
    assert len(fake_lm.calls) == 2 * K


def test_bypass_calls_lm_K_times(
    fake_lm: _FakeLM, fake_embed: Embedder
) -> None:
    K = 4
    run_cascade(
        prompt="Compose.",
        constraint=_constraint(fake_embed),
        lm=_make_lm_protocol_compliant(fake_lm),
        embed=fake_embed,
        K=K,
        max_tokens=16,
        base_seed=42,
        bypass_vimarsa=True,
    )
    assert len(fake_lm.calls) == K

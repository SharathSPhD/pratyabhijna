"""v0.4 Phase 2 (ADR-003) gate: FE budget hard gate inside ``run_cascade``.

Verifies that:

1. A budget-starved fixture (``initial_bits=-3.0``, ``abort_threshold=-2.0``)
   triggers ``revision_skipped_reason="fe_budget_underwater"`` and commits
   the draft.
2. A healthy budget fixture runs both passes exactly as v0.3 would.
3. Both branches populate ``state.audit["fe_budget_underwater"]`` and
   ``state.audit["budget_ledger"]`` so the abort is observable on every
   cascade row regardless of outcome.
4. ``commit_policy`` does NOT override a budget abort (the two-tier
   hierarchy: budget gate -> commit policy).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pce.active_inference.budget import FreeEnergyBudget
from pce.cascade import run_cascade
from pce.substrate.embed import Embedder
from pce.substrate.lm_protocol import LMProtocol
from pce.types import Candidate, Constraint


class _FakeEmbed(Embedder):
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
    name = "fake-lm"
    supports_logprobs = True
    supports_score = False
    supports_entropy = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, prompt: str, *, max_tokens: int, sampler: dict[str, float], seed: int
    ) -> Candidate:
        self.calls.append(
            {"prompt": prompt[:200], "seed": int(seed), "sampler": dict(sampler)}
        )
        if "Reviser brief" in prompt:
            text = f"REVISION-{seed}: refined surface."
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

    def length_proxy_logp(self, candidate: Candidate) -> float:
        return float(candidate.logp)


def _protocol(lm: _FakeLM) -> LMProtocol:
    assert isinstance(lm, LMProtocol)
    return lm


@pytest.fixture()
def embed() -> Embedder:
    return _FakeEmbed()


@pytest.fixture()
def lm() -> _FakeLM:
    return _FakeLM()


def _constraint(embed: Embedder) -> Constraint:
    q = embed.encode("a vivid response with two named aspects")
    return Constraint(
        text="a vivid response",
        embedding=q,
        must_avoid=("a boring single-aspect statement",),
    )


def _run(
    *, lm: _FakeLM, embed: Embedder, budget: FreeEnergyBudget, commit_policy: str
):
    return run_cascade(
        prompt="Compose a short response.",
        constraint=_constraint(embed),
        lm=_protocol(lm),
        embed=embed,
        K=3,
        max_tokens=32,
        base_seed=0,
        aspects=["aspect one", "aspect two"],
        commit_policy=commit_policy,  # type: ignore[arg-type]
        budget=budget,
    )


def test_fe_budget_underwater_aborts_revision_and_commits_draft(
    lm: _FakeLM, embed: Embedder
) -> None:
    """When the ledger is already below ``abort_threshold`` before the revision
    pass, the cascade commits the draft and emits the v0.4 abort reason."""
    starved = FreeEnergyBudget(initial_bits=-3.0, abort_threshold=-2.0)
    state = _run(lm=lm, embed=embed, budget=starved, commit_policy="event_gated")

    assert state.committed == "draft"
    assert state.surface == state.surface_draft
    assert state.surface_revision is None
    assert state.audit["revision_skipped"] is True
    assert state.audit["revision_skipped_reason"] == "fe_budget_underwater"
    assert state.audit["fe_budget_underwater"] is True
    assert state.audit["two_pass"] is False
    # Only the K draft calls happened; no revision-pass calls.
    assert all("Reviser brief" not in c["prompt"] for c in lm.calls)


def test_fe_budget_healthy_runs_both_passes(lm: _FakeLM, embed: Embedder) -> None:
    """With the v0.3 default ``initial_bits=4.0`` the cascade always reaches the revision pass."""
    healthy = FreeEnergyBudget()
    state = _run(lm=lm, embed=embed, budget=healthy, commit_policy="always_revise")

    assert state.audit["two_pass"] is True
    assert state.audit["revision_skipped"] is False
    assert state.audit["fe_budget_underwater"] is False
    assert state.surface_revision is not None
    assert state.committed == "revision"
    # K draft calls + K revision calls.
    revision_calls = [c for c in lm.calls if "Reviser brief" in c["prompt"]]
    assert len(revision_calls) > 0


def test_fe_budget_audit_present_on_both_branches(
    lm: _FakeLM, embed: Embedder
) -> None:
    """``state.audit['budget_ledger']`` is populated on abort AND non-abort rows."""
    # Healthy branch.
    state_h = _run(
        lm=_FakeLM(),
        embed=embed,
        budget=FreeEnergyBudget(),
        commit_policy="always_revise",
    )
    assert "budget_ledger" in state_h.audit
    assert isinstance(state_h.audit["budget_ledger"], dict)
    assert "balance_bits" in state_h.audit["budget_ledger"]

    # Aborted branch.
    state_a = _run(
        lm=_FakeLM(),
        embed=embed,
        budget=FreeEnergyBudget(initial_bits=-5.0),
        commit_policy="event_gated",
    )
    assert "budget_ledger" in state_a.audit
    assert state_a.audit["fe_budget_underwater"] is True


def test_commit_policy_does_not_override_budget_abort(
    lm: _FakeLM, embed: Embedder
) -> None:
    """``commit_policy='always_revise'`` must NOT bypass an FE-budget abort.

    The two-tier hierarchy mandates: budget gate (generation) -> commit
    policy (selection). If the budget says no revision, no revision is
    generated and the policy can only commit the draft.
    """
    starved = FreeEnergyBudget(initial_bits=-5.0)
    state = _run(lm=lm, embed=embed, budget=starved, commit_policy="always_revise")
    assert state.committed == "draft"
    assert state.surface_revision is None
    assert state.audit["revision_skipped_reason"] == "fe_budget_underwater"

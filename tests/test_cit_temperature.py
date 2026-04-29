"""Phase 3 ADR-003 gate: cit_temperature plumbing through iccha.

Verifies that:

* ``cit_temperature=1.0`` (default) reproduces the v0.2 parity sampler tau
  exactly: every candidate has ``sampler["tau"] == 0.9``.
* ``cit_temperature=2.0`` doubles the effective parity tau.
* ``cit_temperature`` is recorded on every candidate's ``sampler`` dict so
  the audit log captures it.
* Negative or zero ``cit_temperature`` raises ``ValueError``.
* In ``sampler_grid_mode="grid"`` ``cit_temperature`` does NOT modulate the
  explore-exploit grid (we only modulate the parity sampler).
"""
from __future__ import annotations

import numpy as np
import pytest

from pce.operators.iccha import PARITY_BASE_TAU, iccha
from pce.types import Candidate, Constraint


class _StubLM:
    """Deterministic-on-seed stub: text encodes seed + sampler tau for assertion."""

    name = "stub"
    supports_logprobs = False
    supports_score = False
    supports_entropy = False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampler: dict[str, float],
        seed: int,
    ) -> Candidate:
        emb = np.zeros((4,), dtype=np.float32)
        emb[seed % 4] = 1.0
        return Candidate(
            seed=int(seed),
            sampler=dict(sampler),
            tokens=tuple(range(max_tokens)),
            text=f"{prompt[:8]}|tau={sampler['tau']:.3f}|seed={seed}",
            logp=-1.0,
            embedding=emb,
        )

    def length_proxy_logp(self, text: str) -> float:
        return -float(len(text))

    def report(self) -> dict[str, object]:
        return {"name": self.name}


def _constraint() -> Constraint:
    emb = np.zeros((4,), dtype=np.float32)
    emb[0] = 1.0
    return Constraint(text="vivid", embedding=emb)


def test_default_cit_temperature_reproduces_parity_sampler() -> None:
    cands = iccha("hello world", _constraint(), lm=_StubLM(), K=4, max_tokens=8)
    assert len(cands) == 4
    for c in cands:
        assert c.sampler["tau"] == pytest.approx(PARITY_BASE_TAU * 1.0)
        assert c.sampler["cit_temperature"] == pytest.approx(1.0)


def test_cit_temperature_modulates_tau() -> None:
    cands_hot = iccha(
        "hello world", _constraint(), lm=_StubLM(), K=2, max_tokens=8, cit_temperature=2.0
    )
    cands_cold = iccha(
        "hello world", _constraint(), lm=_StubLM(), K=2, max_tokens=8, cit_temperature=0.5
    )
    for c in cands_hot:
        assert c.sampler["tau"] == pytest.approx(PARITY_BASE_TAU * 2.0)
        assert c.sampler["cit_temperature"] == pytest.approx(2.0)
    for c in cands_cold:
        assert c.sampler["tau"] == pytest.approx(PARITY_BASE_TAU * 0.5)
        assert c.sampler["cit_temperature"] == pytest.approx(0.5)


def test_cit_temperature_recorded_on_audit() -> None:
    cands = iccha(
        "hello world", _constraint(), lm=_StubLM(), K=2, max_tokens=8, cit_temperature=1.5
    )
    audits = [c.to_audit() for c in cands]
    for a in audits:
        assert a["sampler"]["cit_temperature"] == pytest.approx(1.5)


def test_zero_cit_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="cit_temperature"):
        iccha("hello", _constraint(), lm=_StubLM(), K=2, max_tokens=4, cit_temperature=0.0)


def test_negative_cit_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="cit_temperature"):
        iccha("hello", _constraint(), lm=_StubLM(), K=2, max_tokens=4, cit_temperature=-0.5)


def test_grid_mode_ignores_cit_temperature() -> None:
    """Explore-exploit grid is unmodulated -- only parity sampler is."""
    grid = (
        {"tau": 0.4, "top_p": 0.92, "top_k": 30.0},
        {"tau": 1.5, "top_p": 0.98, "top_k": 100.0},
    )
    cands = iccha(
        "hello world",
        _constraint(),
        lm=_StubLM(),
        K=2,
        max_tokens=8,
        sampler_grid_mode="grid",
        sampler_grid=grid,
        cit_temperature=2.0,
    )
    assert cands[0].sampler["tau"] == pytest.approx(0.4)
    assert cands[1].sampler["tau"] == pytest.approx(1.5)

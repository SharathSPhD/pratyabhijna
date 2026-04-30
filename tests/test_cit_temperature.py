"""v0.3 + v0.4 cit_temperature plumbing through iccha.

v0.4 (ADR-001) introduces ``cit_temperature_mechanism``; the default is
``"best_of_k"``:

* In best-of-K mode ``cit_temperature`` modulates ``K_runtime``, not tau:
  ``tau`` stays at ``PARITY_BASE_TAU`` and the candidate count expands /
  contracts as a function of cit_temperature.
* In ``"parity_tau"`` mode (v0.3 backward-compat) ``cit_temperature``
  multiplies the parity sampler's ``tau`` and ``K_runtime == K_eff``.
* In ``"off"`` mode ``cit_temperature`` is recorded but does not enter
  generation.

These tests cover both modes plus the audit invariants (``cit_temperature``
and ``K_runtime`` always recorded on every candidate's sampler dict) and
the negative inputs (``cit_temperature < 0`` rejected).
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


def test_default_best_of_k_keeps_tau_at_parity_base() -> None:
    """v0.4 default mechanism is best-of-K: tau stays at PARITY_BASE_TAU."""
    cands = iccha("hello world", _constraint(), lm=_StubLM(), K=4, max_tokens=8)
    # cit_temperature=1.0 default -> K_runtime = round(4 * 2.0) = 8.
    assert len(cands) == 8
    for c in cands:
        assert c.sampler["tau"] == pytest.approx(PARITY_BASE_TAU)
        assert c.sampler["cit_temperature"] == pytest.approx(1.0)
        assert c.sampler["K_eff"] == pytest.approx(4.0)
        assert c.sampler["K_runtime"] == pytest.approx(8.0)


def test_parity_tau_mechanism_modulates_tau() -> None:
    """v0.3 backward-compat: ``mechanism='parity_tau'`` multiplies tau by cit_temp."""
    cands_hot = iccha(
        "hello world",
        _constraint(),
        lm=_StubLM(),
        K=2,
        max_tokens=8,
        cit_temperature=2.0,
        cit_temperature_mechanism="parity_tau",
    )
    cands_cold = iccha(
        "hello world",
        _constraint(),
        lm=_StubLM(),
        K=2,
        max_tokens=8,
        cit_temperature=0.5,
        cit_temperature_mechanism="parity_tau",
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
        assert "K_runtime" in a["sampler"]
        assert "perturbation_idx" in a["sampler"]


def test_zero_cit_temperature_allowed_in_v0_4() -> None:
    """v0.4 (ADR-001): cit_temperature=0.0 is the canonical 'concentrated' setting.

    K_runtime = round(K_eff * (0.5 + 1.5 * 0)) = round(0.5 * K_eff). The v0.3
    invariant ``cit_temperature > 0`` was tied to the parity_tau mechanism
    (where ``tau = 0`` is degenerate). The best-of-K mechanism has no such
    degeneracy at zero, so v0.4 relaxes the floor to ``cit_temperature >= 0``.
    """
    cands = iccha(
        "hello", _constraint(), lm=_StubLM(), K=4, max_tokens=4, cit_temperature=0.0
    )
    assert len(cands) >= 1


def test_negative_cit_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="cit_temperature"):
        iccha("hello", _constraint(), lm=_StubLM(), K=2, max_tokens=4, cit_temperature=-0.5)


def test_grid_mode_preserves_v0_3_semantics() -> None:
    """Grid mode keeps v0.3 behavior under v0.4: K_runtime == K_eff, taus from grid.

    v0.4 (ADR-001) confines best-of-K width expansion to the parity sampler
    path. Grid mode is the legacy v0.1 explore-exploit ladder and does NOT
    expand K_runtime under any cit_temperature_mechanism. This test guards
    that invariant under both ``best_of_k`` (default) and ``parity_tau``.
    """
    grid = (
        {"tau": 0.4, "top_p": 0.92, "top_k": 30.0},
        {"tau": 1.5, "top_p": 0.98, "top_k": 100.0},
    )
    for mechanism in ("best_of_k", "parity_tau"):
        cands = iccha(
            "hello world",
            _constraint(),
            lm=_StubLM(),
            K=2,
            max_tokens=8,
            sampler_grid_mode="grid",
            sampler_grid=grid,
            cit_temperature=2.0,
            cit_temperature_mechanism=mechanism,  # type: ignore[arg-type]
        )
        assert len(cands) == 2, f"grid mode under {mechanism} should keep K_runtime=K_eff"
        assert cands[0].sampler["tau"] == pytest.approx(0.4)
        assert cands[1].sampler["tau"] == pytest.approx(1.5)

"""Deterministic in-memory ``LMProtocol`` / embedder used by the v0.4 budget probe.

These fakes back :meth:`pce.substrate.integrity.IntegrityProbe.probe_budget_abort`.
They are intentionally tiny and dependency-free so the probe runs in any
environment, costs $0, and produces a deterministic cascade trajectory.

These types are not part of the public API; tests should import them from
this module if they need the same probe-style fakes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint


class ProbeFakeEmbed(Embedder):
    """Hash-seeded deterministic embedder. 16-D unit vectors."""

    def __init__(self) -> None:
        self.model_id = "probe-fake-embedder"
        self.dim = 16

    def encode(self, texts):  # type: ignore[no-untyped-def]
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts], axis=0)

    def _vec(self, t: str) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def cosine(self, a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
        return float(np.dot(a, b))


class ProbeFakeLM:
    """Deterministic ``LMProtocol`` fake. Records every call for assertions."""

    name = "probe-fake-lm"
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
        text = (
            f"REVISION-{seed}: refined surface."
            if "Reviser brief" in prompt
            else f"DRAFT-{seed}: bare surface response."
        )
        emb = (
            np.random.default_rng(seed * 31 + 7)
            .standard_normal(16)
            .astype(np.float32)
        )
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


def probe_fake_constraint(embed: Embedder) -> Constraint:
    """Fixed Constraint used by the budget-abort probe."""
    q = embed.encode("a vivid response with two named aspects")
    return Constraint(
        text="a vivid response",
        embedding=q,
        must_avoid=("a boring single-aspect statement",),
    )

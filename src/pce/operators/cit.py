"""`cit` - luminous-ground generative prior.

`cit` is the cascade's bottom layer: a temperature-scheduled token sampler over
the local LM. The operator is a thin, deterministic-on-seed wrapper around
`LocalLM.generate` that exists so the cascade can address the substrate by its
Pratyabhijna name and so the operator-spec invariants live in one place.

Semantics (see [docs/operator-spec.md §1](../../../docs/operator-spec.md#1-cit--luminous-ground-generative-prior)).
"""
from __future__ import annotations

from pce.substrate.lm import LocalLM
from pce.types import Candidate


def cit(
    prompt: str,
    *,
    lm: LocalLM,
    temperature: float = 1.0,
    max_tokens: int = 64,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int = 0,
) -> Candidate:
    """Sample one continuation from the LM under (temperature, top_p, top_k, seed)."""
    if temperature <= 0:
        raise ValueError(f"cit: temperature must be > 0, got {temperature}")
    if max_tokens <= 0:
        raise ValueError(f"cit: max_tokens must be > 0, got {max_tokens}")
    return lm.generate(
        prompt,
        max_tokens=int(max_tokens),
        sampler={"tau": float(temperature), "top_p": float(top_p), "top_k": float(top_k)},
        seed=int(seed),
    )

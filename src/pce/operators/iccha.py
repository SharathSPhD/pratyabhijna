"""`icchā` - pre-cognitive directional vector.

Emit K candidate continuations under K different sampler tuples drawn from a
sampler-grid that spans exploit -> explore. None are committed; selection
happens downstream in `jñāna`.

Default sampler grid mixes (low τ, narrow top-p, high top-k) for exploitation
with (high τ, broad top-p, no top-k cap) for exploration.
"""
from __future__ import annotations

from pce.operators.cit import cit
from pce.substrate.lm import LocalLM
from pce.types import Candidate, Constraint

DEFAULT_SAMPLER_GRID: tuple[dict[str, float], ...] = (
    {"tau": 0.40, "top_p": 0.92, "top_k": 30.0},
    {"tau": 0.60, "top_p": 0.94, "top_k": 40.0},
    {"tau": 0.80, "top_p": 0.95, "top_k": 50.0},
    {"tau": 0.95, "top_p": 0.95, "top_k": 50.0},
    {"tau": 1.10, "top_p": 0.96, "top_k": 60.0},
    {"tau": 1.30, "top_p": 0.97, "top_k": 80.0},
    {"tau": 1.50, "top_p": 0.98, "top_k": 100.0},
    {"tau": 1.80, "top_p": 0.99, "top_k": 0.0},  # no top-k cap
)


def _build_prompt(prompt: str, constraint: Constraint) -> str:
    """Wrap user prompt with constraint cue. The local LM is small, so we keep this short."""
    if not prompt.strip():
        raise ValueError("icchā: prompt must be non-empty")
    return (
        f"{prompt}\n"
        f"[Constraint: {constraint.text}]\n"
        f"Output:\n"
    )


def iccha(
    prompt: str,
    constraint: Constraint,
    *,
    lm: LocalLM,
    K: int = 8,
    sampler_grid: tuple[dict[str, float], ...] | None = None,
    base_seed: int = 0,
    max_tokens: int = 64,
) -> tuple[Candidate, ...]:
    """Generate K candidate continuations under K distinct samplers."""
    if K <= 0:
        raise ValueError(f"icchā: K must be > 0, got {K}")
    grid = sampler_grid or DEFAULT_SAMPLER_GRID
    if len(grid) < K:
        raise ValueError(
            f"icchā: sampler_grid has {len(grid)} entries, need at least K={K}"
        )
    full_prompt = _build_prompt(prompt, constraint)
    candidates: list[Candidate] = []
    for k in range(K):
        spec = grid[k]
        cand = cit(
            full_prompt,
            lm=lm,
            temperature=float(spec.get("tau", 1.0)),
            top_p=float(spec.get("top_p", 0.95)),
            top_k=int(spec.get("top_k", 50)),
            max_tokens=max_tokens,
            seed=int(base_seed) + k,
        )
        candidates.append(cand)
    return tuple(candidates)

"""`icchā` - pre-cognitive directional vector.

Emit K candidate continuations under K samplers. v0.2 introduces two parity
knobs (ADR-005) so the cascade arm sees the *same prompt* and *same sampler
distribution* as the bare arm; per-K diversity comes from the seed only:

* ``prompt_mode="verbatim"`` (default) returns the user prompt unchanged;
  the constraint is communicated downstream only via the ``Constraint``
  object that ``apohana`` and ``vimarsa`` consume. ``"constraint_suffix"``
  preserves the v0.1 behavior of appending
  ``"Write a response that is <constraint>."`` for direct callers that want
  the legacy prompt.
* ``sampler_grid_mode="parity"`` (default) ignores ``sampler_grid`` and
  emits K identical samplers ``{"tau": 0.9, "top_p": 0.95, "top_k": 50}`` -
  exactly what the bare baseline uses. ``"grid"`` falls back to the v0.1
  explore-exploit grid (or a caller-supplied ``sampler_grid``).

The substrate is now ``LMProtocol`` so cascade can run against either
``LocalLM`` (Qwen2-1.5B) or ``HaikuLM`` (Anthropic Haiku via ``claude``
CLI). Both implementations honor ``seed`` so per-K candidate diversity in
parity mode is purely seed-driven.
"""
from __future__ import annotations

from typing import Literal

from pce.operators.cit import cit
from pce.substrate.lm_protocol import LMProtocol
from pce.types import Candidate, Constraint

# v0.1 explore-exploit grid kept for backward compatibility under
# `sampler_grid_mode="grid"` and direct callers passing `sampler_grid=`.
DEFAULT_SAMPLER_GRID: tuple[dict[str, float], ...] = (
    {"tau": 0.40, "top_p": 0.92, "top_k": 30.0},
    {"tau": 0.60, "top_p": 0.94, "top_k": 40.0},
    {"tau": 0.80, "top_p": 0.95, "top_k": 50.0},
    {"tau": 0.95, "top_p": 0.95, "top_k": 50.0},
    {"tau": 1.10, "top_p": 0.96, "top_k": 60.0},
    {"tau": 1.30, "top_p": 0.97, "top_k": 80.0},
    {"tau": 1.50, "top_p": 0.98, "top_k": 100.0},
    {"tau": 1.80, "top_p": 0.99, "top_k": 0.0},
)

# v0.2 parity grid: K copies of the bare baseline's sampler. Different *seeds*
# across K give the candidate diversity, not different temperatures. This
# makes (cascade - bare) a clean architectural ablation per ADR-005.
PARITY_SAMPLER: dict[str, float] = {"tau": 0.9, "top_p": 0.95, "top_k": 50.0}

PromptMode = Literal["verbatim", "constraint_suffix"]
SamplerGridMode = Literal["parity", "grid"]


def _build_prompt(
    prompt: str, constraint: Constraint, *, prompt_mode: PromptMode = "verbatim"
) -> str:
    """Wrap user prompt with the constraint cue inlined as natural prose.

    v0.2: the default ``prompt_mode`` is ``"verbatim"``: the prompt is
    returned unchanged so the cascade arm sees the same string as the bare
    arm. ``"constraint_suffix"`` preserves the v0.1 behavior used by direct
    callers and the legacy ``sampler_grid_mode="grid"`` path; small models
    (Qwen2-1.5B) sometimes echo a literal ``[Constraint: ...]`` bracket so
    we use natural prose for the suffix.
    """
    if not prompt.strip():
        raise ValueError("icchā: prompt must be non-empty")
    if prompt_mode == "verbatim":
        return prompt
    if prompt_mode == "constraint_suffix":
        return f"{prompt.rstrip()}\nWrite a response that is {constraint.text}.\n\n"
    raise ValueError(f"icchā: unknown prompt_mode={prompt_mode!r}")


def iccha(
    prompt: str,
    constraint: Constraint,
    *,
    lm: LMProtocol,
    K: int = 8,
    sampler_grid: tuple[dict[str, float], ...] | None = None,
    sampler_grid_mode: SamplerGridMode = "parity",
    prompt_mode: PromptMode = "verbatim",
    base_seed: int = 0,
    max_tokens: int = 64,
) -> tuple[Candidate, ...]:
    """Generate K candidate continuations.

    In ``sampler_grid_mode="parity"`` (default) all K samplers are identical
    to the bare baseline; per-K diversity comes from ``seed = base_seed + k``.
    In ``"grid"`` mode the sampler grid spans an explore-exploit ladder.
    """
    if K <= 0:
        raise ValueError(f"icchā: K must be > 0, got {K}")
    if sampler_grid_mode == "parity":
        grid: tuple[dict[str, float], ...] = tuple(dict(PARITY_SAMPLER) for _ in range(K))
    elif sampler_grid_mode == "grid":
        grid = sampler_grid or DEFAULT_SAMPLER_GRID
        if len(grid) < K:
            raise ValueError(
                f"icchā: sampler_grid has {len(grid)} entries, need at least K={K}"
            )
    else:
        raise ValueError(f"icchā: unknown sampler_grid_mode={sampler_grid_mode!r}")
    full_prompt = _build_prompt(prompt, constraint, prompt_mode=prompt_mode)
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

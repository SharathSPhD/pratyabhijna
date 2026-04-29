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
#
# v0.3 (ADR-003): the parity sampler's ``tau`` is multiplicatively modulated
# by the cascade's ``cit_temperature`` so the cit operator can re-temper the
# K-fan-out posterior. ``effective_tau = base_tau * cit_temperature``. The
# default ``base_tau = 0.9`` matches v0.2 exactly when ``cit_temperature == 1``.
PARITY_BASE_TAU: float = 0.9
PARITY_SAMPLER: dict[str, float] = {"tau": PARITY_BASE_TAU, "top_p": 0.95, "top_k": 50.0}

# v0.4 (ADR-001): the Haiku CLI substrate does not expose token-level sampler
# flags, so ``cit_temperature`` cannot causally modulate token probabilities
# under the OAuth-only constraint. v0.4 makes ``cit_temperature`` causal via
# *best-of-K candidate width*: the runtime number of candidates is a function
# of ``cit_temperature`` and each candidate gets a deterministic prompt-level
# diversity perturbation drawn from a frozen 8-element table indexed by the
# seed. Posterior selection inside ``iccha`` (parity tau, ananda, jnana) is
# unchanged.
#
# At ``cit_temperature = 0.0``, ``K_runtime = round(0.5 * K_eff)`` (concentrated
# around the central reading). At ``cit_temperature = 1.0``, ``K_runtime =
# round(2.0 * K_eff)`` (broader exploration). The formula's neutral point
# (``K_runtime == K_eff``) is at ``cit_temperature = 1/3``; the v0.4 pilot
# default ``cit_temperature = 0.5`` deliberately sits slightly above neutral
# so the proxy probes a wider candidate front than the v0.3 default did.
K_MIN: int = 2
K_MAX: int = 16

# Frozen 8-element prompt-perturbation table. Each entry is prepended to the
# prompt as a single-line nudge. Index is computed as ``(seed % 8 + i) % 8``.
# Index 0 (identity) preserves the v0.3 prompt verbatim so K_runtime == K_eff
# at cit_temperature == 0.5 produces the same surface as v0.3 for the central
# candidate.
PERTURBATION_TABLE: tuple[str, ...] = (
    "",
    "Lean toward the literal interpretation. ",
    "Lean toward the figurative interpretation. ",
    "Foreground the unusual aspect of the prompt. ",
    "Foreground the conventional aspect of the prompt. ",
    "Use one specific concrete image. ",
    "Use one abstract or universal frame. ",
    "Reframe in one alternative perspective. ",
)

PromptMode = Literal["verbatim", "constraint_suffix"]
SamplerGridMode = Literal["parity", "grid"]


def k_runtime_for(K_eff: int, cit_temperature: float) -> int:
    """Compute v0.4 best-of-K width as a function of ``cit_temperature``.

    Per ADR-001:

        K_runtime = clip(round(K_eff * (0.5 + 1.5 * cit_temperature)), K_MIN, K_MAX)

    Pure function so it can be tested independently of an LM substrate.
    """
    if K_eff <= 0:
        raise ValueError(f"icchā: K_eff must be > 0, got {K_eff}")
    if cit_temperature < 0:
        raise ValueError(f"icchā: cit_temperature must be >= 0, got {cit_temperature}")
    raw = round(float(K_eff) * (0.5 + 1.5 * float(cit_temperature)))
    return max(K_MIN, min(K_MAX, int(raw)))


def perturbation_idx(seed: int, k: int) -> int:
    """Deterministic perturbation index for candidate ``k`` of an item with ``seed``."""
    return (int(seed) % 8 + int(k)) % 8


def perturbation_for(seed: int, k: int) -> str:
    """Look up the prompt-level perturbation for candidate ``k`` of seed ``seed``."""
    return PERTURBATION_TABLE[perturbation_idx(seed, k)]


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
    cit_temperature: float = 1.0,
    cit_temperature_mechanism: Literal["best_of_k", "parity_tau", "off"] = "best_of_k",
) -> tuple[Candidate, ...]:
    """Generate candidate continuations.

    The caller supplies a *nominal* ``K`` (``K_eff``); the runtime candidate
    count depends on ``cit_temperature`` under the v0.4 best-of-K mechanism
    (ADR-001). At ``cit_temperature = 0.5`` the runtime count equals
    ``K_eff``; at ``cit_temperature = 1.0`` it doubles; at ``0.0`` it halves.

    ``cit_temperature_mechanism`` selects how ``cit_temperature`` enters:

    * ``"best_of_k"`` (default, v0.4 ADR-001): runtime K width scales with
      ``cit_temperature``; each candidate gets a deterministic prompt-level
      perturbation indexed by ``(seed % 8, i % 8)``; parity ``tau`` stays at
      ``PARITY_BASE_TAU = 0.9``. This is the headline causal mechanism for
      Haiku CLI substrates that do not expose token-level sampler flags.
    * ``"parity_tau"`` (v0.3 behavior): no width modulation; parity ``tau``
      is multiplied by ``cit_temperature``. Kept for backward compatibility
      and for substrates where token-level temperature *is* honored
      (LocalLM, SDK).
    * ``"off"``: ``cit_temperature`` is recorded on the audit but does not
      enter generation. Used by the prove-gate as a control.

    Per :class:`Candidate.sampler` the audit log preserves
    ``cit_temperature``, ``K_eff``, ``K_runtime``, ``perturbation_idx``, and
    ``cit_temperature_mechanism``.
    """
    if K <= 0:
        raise ValueError(f"icchā: K must be > 0, got {K}")
    if cit_temperature < 0:
        raise ValueError(f"icchā: cit_temperature must be >= 0, got {cit_temperature}")

    K_eff = int(K)
    # v0.4 (ADR-001): width-expansion via cit_temperature is the production
    # mechanism. It applies only to the parity sampler path (the default and
    # the only path used by the cascade). Grid mode is the v0.1 legacy
    # explore-exploit ladder; it preserves v0.3 semantics (K_runtime == K_eff
    # and "grid too small raises") so caller contracts do not silently change.
    if cit_temperature_mechanism == "best_of_k" and sampler_grid_mode == "parity":
        K_runtime = k_runtime_for(K_eff, cit_temperature)
    else:
        K_runtime = K_eff

    if sampler_grid_mode == "parity":
        if cit_temperature_mechanism == "parity_tau":
            modulated_tau = float(PARITY_BASE_TAU) * float(cit_temperature)
        else:
            # best_of_k or off: parity tau stays at the v0.2 base so width is
            # the only causal handle on output diversity.
            modulated_tau = float(PARITY_BASE_TAU)
        parity = dict(PARITY_SAMPLER)
        parity["tau"] = modulated_tau
        grid: tuple[dict[str, float], ...] = tuple(dict(parity) for _ in range(K_runtime))
    elif sampler_grid_mode == "grid":
        grid = sampler_grid or DEFAULT_SAMPLER_GRID
        if len(grid) < K_runtime:
            raise ValueError(
                f"icchā: sampler_grid has {len(grid)} entries, need at least K={K_runtime}"
            )
    else:
        raise ValueError(f"icchā: unknown sampler_grid_mode={sampler_grid_mode!r}")

    full_prompt = _build_prompt(prompt, constraint, prompt_mode=prompt_mode)
    candidates: list[Candidate] = []
    for k in range(K_runtime):
        spec = grid[k]
        if cit_temperature_mechanism == "best_of_k":
            p_idx = perturbation_idx(int(base_seed), k)
            perturbation = PERTURBATION_TABLE[p_idx]
            generation_prompt = (perturbation + full_prompt) if perturbation else full_prompt
        else:
            p_idx = 0
            generation_prompt = full_prompt
        cand = cit(
            generation_prompt,
            lm=lm,
            temperature=float(spec.get("tau", 1.0)),
            top_p=float(spec.get("top_p", 0.95)),
            top_k=int(spec.get("top_k", 50)),
            max_tokens=max_tokens,
            seed=int(base_seed) + k,
        )
        new_sampler = dict(cand.sampler)
        new_sampler["cit_temperature"] = float(cit_temperature)
        new_sampler["K_eff"] = float(K_eff)
        new_sampler["K_runtime"] = float(K_runtime)
        new_sampler["perturbation_idx"] = float(p_idx)
        new_sampler["cit_temperature_mechanism"] = float(
            {"best_of_k": 1.0, "parity_tau": 2.0, "off": 0.0}[cit_temperature_mechanism]
        )
        cand = Candidate(
            seed=cand.seed,
            sampler=new_sampler,
            tokens=cand.tokens,
            text=cand.text,
            logp=cand.logp,
            embedding=cand.embedding,
        )
        candidates.append(cand)
    return tuple(candidates)

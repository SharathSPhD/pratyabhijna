# ADR-005 (v0.2) — Prompt and sampler parity between bare and cascade arms

Status: Accepted (frozen during planning round 3, TRIZ five-pack).
Date: 2026-04-28.
Related TRIZ card: [docs/triz/C5-determinism-vs-creativity.md](../../triz/C5-determinism-vs-creativity.md).

## Context

The adversarial review's P1-3 finding: in v0.1, `iccha._build_prompt` appends `"Write a response that is <constraint>."` to the user prompt, while `local_bare` sends the user prompt verbatim. P1-1 + P1-3 together: the cascade arm uses an explore-exploit sampler grid (`tau=0.40` to `tau=1.80`) while the bare arm uses a single fixed sampler (`tau=0.9`).

These asymmetries make `local_cascade` vs `local_bare` (the only same-substrate contrast available in v0.1) not a clean architectural ablation. Any score difference is at least partially attributable to the prompt and sampler change rather than the cascade itself.

## Decision

Two parameters are added to `iccha`, with parity defaults:

```python
def iccha(
    prompt: str,
    constraint: Constraint,
    *,
    lm: LMProtocol,
    K: int = 4,
    sampler_grid: tuple[dict[str, float], ...] | None = None,
    sampler_grid_mode: Literal["parity", "grid"] = "parity",
    prompt_mode: Literal["constraint_suffix", "verbatim"] = "verbatim",
    base_seed: int = 0,
    max_tokens: int = 200,
) -> tuple[Candidate, ...]:
    ...
```

- `prompt_mode="verbatim"` (default): `_build_prompt` returns the prompt unchanged. The constraint is communicated to the cascade only via the `Constraint` object (used by `apohana` and `vimarsa`). Result: bare and cascade see *identical* prompts.
- `prompt_mode="constraint_suffix"`: legacy v0.1 behavior, kept for backward compatibility.
- `sampler_grid_mode="parity"` (default): all K samplers are `{"tau": 0.9, "top_p": 0.95, "top_k": 50}`. Different *seeds* across K give candidate diversity; the sampler is identical to bare's. Result: any sampler-driven distribution shift between bare and cascade is gone.
- `sampler_grid_mode="grid"`: legacy v0.1 explore-exploit grid, kept for backward compatibility.

`run_cascade` passes `prompt_mode="verbatim"` and `sampler_grid_mode="parity"` by default.

The benchmark `local_bare` arm is unchanged: it still calls `lm.generate(prompt, max_tokens=200, sampler={"tau": 0.9, "top_p": 0.95, "top_k": 50}, seed=seed)`. Now the cascade's per-K samples are identical to that bare call (modulo seed), so the cascade's contribution is purely the operator chain (`iccha -> apohana -> jnana -> kriya -> vimarsa -> revision`).

## Consequences

Positive:

- `local_cascade - local_bare` is a clean architectural ablation, attributable to the cascade rather than to prompt/sampler drift.
- Same logic for `haiku_cascade - haiku_bare`.
- Per-K candidate diversity comes from `seed = base_seed + k`, which `LocalLM` honors via `torch.Generator(device="cpu").manual_seed(seed)` and `HaikuLM` honors via a per-call seed prefix in the prompt nonce.

Negative:

- The cascade loses the v0.1 explore-exploit grid by default; `iccha` may produce K=4 less-distinct candidates at `tau=0.9`. Mitigation: the prove-gate explicitly checks that `iccha(parity)` produces K=4 distinct candidates on the duck-rabbit and AUT brick prompts. If candidates collapse, the prove-gate fails and the cascade temperature is bumped.
- Some users may want the explore-exploit grid for pure brainstorming use cases; available via `sampler_grid_mode="grid"`.

## Alternatives considered

- *Two parity ablations* (cascade_no_suffix, cascade_matched_sampler): rejected because the user's frozen scope is four arms, not six. We collapse the parity dimension into the cascade's default behavior so the four-arm pilot already tests it.
- *Force prompt_mode and sampler_grid_mode to be required positional args*: rejected to keep `iccha` callable from MCP tools and notebooks without churn.

## Implementation pointers

- `src/pce/operators/iccha.py` — add `prompt_mode` + `sampler_grid_mode` kwargs; `_build_prompt(prompt_mode="verbatim")` returns prompt unchanged.
- `src/pce/cascade.py` — pass `prompt_mode="verbatim"` and `sampler_grid_mode="parity"` to `iccha`.
- `tests/operators/test_iccha.py` — assert `prompt_mode="verbatim"` returns identical prompt; assert `sampler_grid_mode="parity"` produces K samplers all equal.
- `scripts/prove_gate.py` — assert K=4 candidates are *distinct* (e.g. mean pairwise edit distance > 5 chars) on prove-gate cases.

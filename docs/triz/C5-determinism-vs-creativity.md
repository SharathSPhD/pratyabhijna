# TRIZ Card C5 — Determinism vs. creativity (sampler asymmetry)

## Contradiction

For the cascade to be *measurably* better than bare, the bare arm and the cascade arm must be sampled identically — same temperature, same top-p, same prompt — so that any difference in score is attributable to the cascade rather than to a sampler change. v0.1 breaks this: bare uses `tau=0.9`, cascade's `iccha` starts at `tau=0.40` and walks an explore-exploit grid, and `iccha` adds a "Write a response that is <constraint>." suffix that bare never sees. The contradiction is between (a) measurement accuracy (apples-to-apples) and (b) cascade expressiveness (the explore-exploit grid is part of why we want the cascade).

- Improving parameter: **28 — Measurement accuracy** (cascade vs. bare is a clean ablation).
- Worsening parameter: **35 — Adaptability or versatility** (the cascade loses the explore-exploit grid that gives `jnana` something to choose from).

## Matrix lookup

`lookup_matrix(28, 35)` -> recommended principles `[13, 35, 2]`.

## Selected principles

### Principle 3 — Local Quality (primary, off-matrix)

> Make different parts of the object carry out different functions; let each part operate at the optimum for its task.

PCE mapping: per-operator sampler profiles. `iccha` operates in `parity` mode by default — same sampler as bare arm, K=4, identical prompt — so the apples-to-apples ablation is clean. The explore-exploit grid is opt-in via `iccha(sampler_grid_mode="grid")` for users who want the v0.1 expressive cascade.

### Principle 13 — The Other Way Around (matrix-recommended primary)

> Invert the usual approach.

PCE mapping: instead of having `iccha` modify the prompt and the bare arm leave it raw, have `iccha._build_prompt(prompt_mode="verbatim")` be the default and let the bare arm and the cascade share the *same* prompt. The constraint is communicated to the cascade via the `Constraint` object (used by `apohana` and `vimarsa`) rather than being injected into the prompt text.

### Principle 2 — Taking Out (supporting)

> Extract a disturbing element so the core system stays stable.

PCE mapping: the constraint suffix is a disturbing element relative to bare. We extract it from `_build_prompt` and place it in the `vimarsa` brief, where it functions as the "missing aspect" hint passed to the revision call.

## Adopted resolution

- `iccha._build_prompt(prompt_mode: Literal["constraint_suffix", "verbatim"] = "verbatim")` — Principle 13 + 2.
- `iccha(sampler_grid_mode: Literal["parity", "grid"] = "parity")` — Principle 3.
- Cascade uses `parity` + `verbatim` by default. Bare arm uses identical sampler params: `tau=0.9, top_p=0.95, top_k=50, max_tokens=200`.
- Pilot ablation: spot-check that `iccha(parity)` returns K=4 distinct candidates (not collapsed to identical text) under the parity sampler at K=4 across the prove-gate cases.
- ADR: [docs/adr/v0.2/ADR-005-prompt-and-sampler-parity.md](../adr/v0.2/ADR-005-prompt-and-sampler-parity.md).

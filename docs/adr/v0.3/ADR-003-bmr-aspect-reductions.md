# ADR-003 (v0.3) — BMR aspect-conditioned reductions

Status: Accepted (frozen during planning round 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.3/C3-active-inference-vs-cli.md](../../triz/v0.3/C3-active-inference-vs-cli.md).

## Context

In v0.2, `jnana` enumerated three reduction families:

- `single`: keep one candidate per reduction.
- `halve`: keep top half by pseudo-count.
- `custom`: caller-supplied priors.

The reductions were *not aspect-conditioned*: they enumerated candidates, not hypotheses about which aspects the surface satisfies. The result was that the winning reduction's `delta_F` was either trivially positive (when one candidate dominated by ananda) or numerically degenerate (when ananda was flat). The v0.2 review found `delta_F` "uninformative" and demoted the active-inference half of the claim.

## Decision

Rewrite `_enumerate_reductions` in [src/pce/operators/jnana.py](../../../src/pce/operators/jnana.py) to enumerate **aspect-conditioned reductions**: each reduction asserts a hypothesis "the surface satisfies aspect subset `S_i`". Concretely:

- Inputs to `jnana` gain an optional `aspect_strengths: np.ndarray[K, A] | None = None` matrix where rows index candidates and columns index aspects (`A = len(aspects)`). Entries are computed by `apohana` from the candidate embedding's inner product with each aspect's prototype (or its Hopfield warm-start; ADR-004).
- For each non-empty subset `S_i ⊆ {1..A}` (or the top-`min(2^A, 16)` subsets by score when `A` is large), build a reduced prior over candidates:

  ```text
  reduced_prior[k] = (1 + λ_a · ananda[k] + λ_p · shifted_apoha[k])
                     · exp(γ · sum_{a ∈ S_i} aspect_strengths[k, a])
  ```

  with `γ = 1.0` by default. The reduction then runs the standard BMR `delta_F` calculation in log-space.
- `delta_F` reports the *winning* reduction's score relative to the unconditioned full prior. When the surface actually covers must-have aspects, the reduction "S_i = all must-have aspects" wins with positive `delta_F`. When coverage is poor, that reduction's `delta_F` is small or negative; a smaller subset wins, signaling a weaker hypothesis.
- Fallback for AUT-style domains with no aspect dictionary: enumerate a single uniform reduction per top-pseudocount candidate (essentially the v0.2 `halve` behavior). `delta_F` may then be modest but not numerically degenerate because `aspect_strengths` is absent and `γ = 0`.

Output signature unchanged: `(selected_index, best_delta_F, posterior)`. Contract: `best_delta_F` MUST be informative (`|best_delta_F| > 0.01`) on the duck-rabbit prove-gate fixture, where the aspect dictionary explicitly names the duck->rabbit perceptual flip.

## Consequences

Positive:

- BMR `delta_F` is informative: it reflects whether the surface covers must-have aspects relative to the prior, not whether one candidate has higher ananda.
- Vimarsa can now use `delta_F_draft` as a real evidence point (ADR-002), so the event-gated commit is grounded in active-inference signal.
- Hopfield warm-start (ADR-004) flows through `aspect_strengths` so the BMR is already non-degenerate on item 1.

Negative:

- Computational cost: enumerating subsets is `O(2^A)`. Mitigation: cap at `min(2^A, 16)` reductions and rank by greedy heuristic; document the cap.
- Aspect dictionaries must be reasonably small (typically `A <= 6`) for clean enumeration. Larger dictionaries fall back to top-K subsets by greedy aspect coverage.

## Implementation files (forecast)

- [src/pce/operators/jnana.py](../../../src/pce/operators/jnana.py) — rewrite `_enumerate_reductions`; new internal `_aspect_conditioned_reductions`; new arg `aspect_strengths`.
- [src/pce/operators/apohana.py](../../../src/pce/operators/apohana.py) — return `aspect_strengths` matrix alongside `apoha`.
- [src/pce/cascade.py](../../../src/pce/cascade.py) — thread `aspect_strengths` from `apohana` through `jnana`.
- [tests/test_jnana_aspect_bmr.py](../../../tests/test_jnana_aspect_bmr.py) — new test file: unit tests for aspect-conditioned reductions including (a) informative `delta_F` on a synthetic aspect-coverage scenario, (b) AUT fallback with no aspect dict.

## Acceptance gate (Phase 3)

- `tests/test_jnana_aspect_bmr.py` passes.
- On the duck-rabbit textual prove-gate fixture, `|delta_F| > 0.01` for at least one cascade pass.
- BMR fallback path covers the AUT case (no aspect dictionary) without numerical errors.

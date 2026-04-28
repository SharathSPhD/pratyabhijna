# ADR-003 — BMR implementation

* Status: accepted
* Date: 2026-04-28

## Context

`jñāna` selects between the K candidates produced by `icchā` via Bayesian Model Reduction. Two implementation options:

* (1) Use `pymdp` v1's JAX-first Agent and run BMR through it.
* (2) Implement the categorical-Dirichlet BMR equations directly using `scipy.special.gammaln`, leaving `pymdp` for full active-inference state-estimation experiments only.

Public `pymdp` documentation (as of v1.0.0, March 2026) does not ship a dedicated SPM-style `spm_MDP_log_evidence` BMR facade; the exposed surface is around `Agent.infer_states`, `Agent.infer_policies`, `calc_vfe`, and the rollout primitives. A custom Dirichlet-conjugate BMR layer is straightforward (see [docs/research-extended.md §2.3](research-extended.md#23-reducer-pseudocode-used-as-jñāna-core)) and is K-small (K ≤ 32) so vectorization is trivial.

JAX on macOS Apple Silicon has historically been finicky around BLAS/LAPACK (`metal` plugin maturity is improving but still differs from x86). We don't want our BMR step to be the source of CI/dev-machine fragility.

## Decision

Implement BMR directly in `src/pce/operators/jnana.py` over numpy float64 using `scipy.special.gammaln`. Use `pymdp.legacy` (numpy backend) as a *reference* for testing equivalence on a 3-state toy POMDP, but not as the production path.

`pymdp` JAX-Agent will be optionally importable for `tests/integration/test_pymdp_equivalence.py`, gated behind a `[pymdp_jax]` extra in `pyproject.toml`.

## Consequences

* Zero JAX-on-macOS dependency for the engine to function. CI does not need a working JAX install.
* The BMR layer is auditable line-by-line — every term in ΔF maps to a `gammaln` call.
* Tests assert numerical equivalence (`abs(delta_F_ours - delta_F_pymdp_legacy) < 1e-3`) on a fixed Dirichlet stress set.
* Future maintainers replacing this with pymdp's JAX BMR (when it ships) need only swap the implementation behind the operator boundary; tests stay valid.

## Rejected alternatives

* Pure `pymdp.legacy` numpy-Agent reuse: forces our state space into pymdp's POMDP factorization which is overkill for K=8 candidate selection.
* Custom JAX implementation: introduces JAX into the engine's runtime path without offsetting benefit.

## Verification

`tests/operators/test_jnana.py` Phase 5 asserts:

* `delta_F` finite for K∈[2, 32] on 100 random Dirichlet seeds.
* `selected_index` is in the top half by pseudo-count for `reduction_target="halve"`.
* For `reduction_target="single"` the posterior has a single entry > 0.9 and the rest < 0.05/(K-1).
* Numerical agreement with `pymdp.legacy` toy on 3-candidate Dirichlet stress (skip if pymdp_jax extra not installed): `|ΔF_ours - ΔF_pymdp| < 1e-3`.

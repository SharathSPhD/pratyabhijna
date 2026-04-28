# ADR-004 — benchmark statistics protocol

* Status: accepted
* Date: 2026-04-28

## Context

Phase 9 runs n_total = 70 paired observations across 4 domains (poetry_gen=20, poetry_interp=20, aut=15, sci_creativity=15). The user requirement is "statistically rigorous: A/B, p-value, power analysis." That implies a multi-test setup over six pre-registered hypotheses. We need to commit to *one* set of methods now so the analysis is not p-hacked.

Three issues to settle:

1. Parametric vs non-parametric primary test;
2. Multiple-comparisons correction;
3. Effect-size measure and CI estimator.

## Decision

### Primary tests (per Hi)

For each Hi the per-prompt paired score-delta `δ_i = score_pce - score_nopce`:

* **Paired permutation test** as the primary test (`scipy.stats.permutation_test`, `permutation_type="samples"`, `n_resamples=10_000`, `alternative="greater"` for directional Hi).
* **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`, `alternative="greater"`) as a non-parametric backup.
* **Hedges' g** as the effect size with small-sample correction:
  \(g = J(n) \cdot (\bar\delta / s_\delta)\), \(J(n) = 1 - \frac{3}{4n - 9}\).
* **BCa bootstrap CI** for the paired-mean delta, 10,000 resamples, `scipy.stats.bootstrap` with `method='BCa'`.

Permutation is preferred over t-test as primary because the score distributions (Likert + composites) are not guaranteed normal and we have small n per domain.

### Multiple-comparisons correction

Holm-Bonferroni across {H1, H2, H3, H4} (the four primary domain hypotheses). H5 (composite) and H6 (within-PCE event vs no-event) are reported as single tests — H5 because it is derived from H1-H4 z-scores and is one number; H6 because it is a different population (within-condition).

Reported quantities per Hi:

* raw `p` (paired permutation, one-sided);
* Holm-corrected `p_holm` (for H1-H4 only);
* Wilcoxon `p_wilcoxon`;
* Hedges' `g`;
* BCa 95% CI `[lo, hi]`;
* a-priori power at g=0.5 and `n_domain`;
* retrospective power at the observed g.

### Power analysis

A-priori: under the assumed effect size `g=0.5` (medium) at α=0.05 one-sided, paired t-test power. Per `statsmodels.stats.power.tt_solve_power`, `n_domain=15` gives `power ≈ 0.66`; `n_domain=20` gives `power ≈ 0.78`. We are deliberately under-powered on `aut` and `sci_creativity` (n=15) and target meta-analytic aggregation via H5 to get to the 0.80 threshold for the composite.

Retrospective power is computed at the observed Hedges' g.

## Consequences

* The pre-registered protocol is locked here. Any deviation in Phase 9 must be flagged in the paper as exploratory.
* The Phase-9 stats module (`benchmarks/stats.py`) implements every quantity above; the Phase-9 audit JSON contains one record per Hi with all eight reported numbers.
* If a domain falls short on n (e.g., a prompt fails and we get n=14), we DO NOT impute; we report the observed n and adjust degrees of freedom.
* Negative-result obligation: if Hi rejects (`p_holm > 0.05` AND CI overlaps zero), we report it in the abstract as such.

## Rejected alternatives

* Bayesian estimation of effect size (BEST / `pymc`): better in principle but adds modeling assumptions and reviewer complexity for v0.1.0. Deferred to a future ADR.
* Bonferroni (uncorrected): too conservative at 4 tests.
* Welch's t-test: assumes near-normal scores; permutation is robust without that assumption.
* Cohen's d (without small-sample correction): biases the effect size upward at n=15.

## Verification

`tests/benchmarks/test_stats.py` asserts:

* On a synthetic null dataset (matched random pairs with mean delta = 0), the paired permutation false-positive rate at α=0.05 is in [0.04, 0.06] over 1000 simulations.
* On a synthetic alternative dataset (g=0.5, n=15), the paired permutation power matches the analytical estimate within 5%.
* `holm_correct([0.001, 0.04, 0.10, 0.30])` returns the expected adjusted values.
* BCa CI on a known-distribution sample matches `scipy.stats.bootstrap` on the same input.

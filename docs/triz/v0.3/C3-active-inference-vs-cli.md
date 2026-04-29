# C3 — Active inference rigor vs CLI black-box

## Contradiction

Active inference needs informative free-energy signals (non-degenerate `delta_F`, calibrated logprobs, posterior over candidates). The `claude --print` CLI is a black box: it returns generated text and a JSON envelope with token counts and cost, but no logprobs, no token-level scores, no entropies. Yet we cannot use the SDK to obtain them (per C2 / user constraint).

- **If we admit `delta_F` is uninformative**, the active-inference half of the claim collapses to "we ran some embeddings."
- **If we demand logprobs**, the CLI cannot supply them; we'd need an SDK path; rejected by C2.

## Improving / Worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 28 — Measurement accuracy | Monitoring precision / metric granularity (here: informativeness of the active-inference signal) |
| Worsening | 36 — Device complexity | Architectural complexity / coupling (here: how richly we depend on the substrate's interface) |

## Matrix lookup

`lookup_matrix(28, 36) -> {27, 35, 10, 34}`.

- **27 — Cheap Short-living**: ephemeral, disposable observations.
- **35 — Parameter Changes**: shift the system into a more favorable regime by changing flexibility, concentration, or rate.
- **10 — Preliminary Action**: do work in advance so the critical path stays short.
- **34 — Discarding and Recovering**: let parts that have served their purpose disappear.

## Ideal Final Result (IFR)

> The active-inference signal (`delta_F`, free-energy budget) is non-degenerate and load-bearing using *only* what the substrate already exposes — generated text, embeddings of that text against an aspect dictionary, token counts, and per-call cost. No logprobs needed. The cascade behaves measurably differently when `delta_F` exceeds threshold than when it does not.

## Attractor-flow divergent ideation

1. **Demand logprobs from the CLI** -> blocked by C2.
2. **Bypass active inference; ship pure prompting** -> abandons the central claim.
3. **Make BMR reductions aspect-conditioned** so the winning reduction reports informative `delta_F` when the surface covers must-have aspects -> the embedding geometry already gives us per-aspect strength, no logprobs needed; *kept*.
4. **Build a free-energy budget** that pays / earns F based on `delta_F` + embedding-distance error + token-cost (the only things CLI exposes) -> a real ledger that gates revision; *kept*.
5. **Calibrated length-proxy logp** on `HaikuLM` (bytes / tokens) advertised honestly via `supports_logprobs=False` so callers cannot mistake it for real logprobs -> truth-in-advertising; *kept*.
6. **Hopfield warm-start** for aspect priors so the BMR reductions are seeded by the storehouse, not by uniform priors -> non-degenerate from item 1; *kept*.

## Selected resolution

Apply principles **35 (Parameter Changes)**, **10 (Preliminary Action)**, **27 (Cheap Short-living)**, and **34 (Discarding and Recovering)**:

- **Parameter Changes**: the substrate exposes only text + token counts; we change *what we measure* — embedding-aspect inner products, BMR over aspect-conditioned reductions, free-energy as a ledger over those — so the same black-box substrate yields informative signal.
- **Preliminary Action**: Hopfield/storehouse pre-computes aspect priors per domain (warm start) so each cascade run begins with non-uniform priors and `delta_F` becomes informative on the first item.
- **Cheap Short-living**: the per-item free-energy ledger is ephemeral; reset between items so observations stay independent.
- **Discarding and Recovering**: when the ledger goes underwater, the shadow revision aborts and the draft is committed by default — discarding the wasteful pass while preserving the audit row.

Implementation contracts: see [ADR-003 — BMR aspect-conditioned reductions](../../adr/v0.3/ADR-003-bmr-aspect-reductions.md) and [ADR-005 — free-energy budget](../../adr/v0.3/ADR-005-free-energy-budget.md).

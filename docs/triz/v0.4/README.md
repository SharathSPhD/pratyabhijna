# TRIZ contradictions for v0.4 (mechanism study)

The v0.3 adversarial review showed that:

- the cascade *generated* better shadow revisions on 15/20 items (mean Δ = +0.0458) but committed only 3 of them;
- three of the v0.3 "active inference" mechanisms (FE budget, `cit_temperature`, `vimarsa.switching_ok`) were audit-only on the Haiku CLI path;
- the SPEC said fixed-effects for H5 while `stats.py` and the paper used random-effects;
- the smoke run could not distinguish a 429 quota failure from a real implementation bug.

v0.4 closes those gaps as a focused mechanism study (Experiments A + B + C from the review). Four contradictions show up immediately when those goals collide with the v0.3 hard constraints:

| Card | Contradiction | Resolved by |
|------|---------------|-------------|
| [C1 — Theory purity vs measurable utility](C1-theory-purity-vs-measurable-utility.md) | Event-gated `vimarsa` is theoretically pure but discards revisions a regression model would correctly accept. | [ADR-002 — LearnedGate](../../adr/v0.4/ADR-002-learned-gate.md) |
| [C2 — OAuth-only substrate vs causal `cit_temperature`](C2-oauth-only-vs-cit-temperature.md) | The substrate constraint forbids the Anthropic SDK; `claude --print` does not expose token sampler flags; `cit_temperature` therefore had no causal handle on Haiku output in v0.3. | [ADR-001 — Best-of-K candidate width](../../adr/v0.4/ADR-001-best-of-k-cit-temperature.md) |
| [C3 — Free-energy budget as audit ledger vs decision authority](C3-budget-ledger-vs-authority.md) | v0.3 wired the budget but never gave it the authority to abort revision; the paper claimed it gated decisions. | [ADR-003 — FE budget gating](../../adr/v0.4/ADR-003-fe-budget-gating.md) |
| [C4 — Construct-validity vs cost cap](C4-judge-cost-vs-cap.md) | Local proxy composites are cheap but suspect; an LLM judge is more credible but costs ≥ 5× per call. | [ADR-004 — Sonnet judge protocol](../../adr/v0.4/ADR-004-sonnet-judge-protocol.md) |

Two more decisions surfaced in planning that don't map onto a contradiction matrix entry but still need ADR closure: hypothesis re-registration (H5 fixed-effects, H8 split) and rate-limit error surfacing. Captured in [ADR-005](../../adr/v0.4/ADR-005-hypothesis-re-registration.md) and [ADR-006](../../adr/v0.4/ADR-006-haiku-rate-limit-error.md).

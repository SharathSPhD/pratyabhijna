# PCE v0.3 — TRIZ contradiction cards

These five cards capture the v0.3 contradictions surfaced by [docs/reviews/2026-04-29-adversarial-v0.2-review.md](../../reviews/2026-04-29-adversarial-v0.2-review.md). Each card cites a TRIZ matrix lookup, the principles it surfaced, an Ideal Final Result (IFR), an attractor-flow divergent ideation pass, the selected resolution, and the corresponding ADR.

| ID | Contradiction | Improving / Worsening | Principles | ADR |
|----|---------------|------------------------|-----------|-----|
| C1 | [Fairness vs depth (matched-budget vs architectural contribution)](C1-fairness-vs-depth.md) | 28 Measurement accuracy / 21 Power | 3, 6, 32 | [ADR-001](../../adr/v0.3/ADR-001-clean-haiku-cli.md), [ADR-002](../../adr/v0.3/ADR-002-event-gated-shadow-revision.md) |
| C2 | [Clean substrate vs OAuth dependency (no API key)](C2-clean-substrate-vs-oauth.md) | 27 Reliability / 33 Ease of operation | 27, 17, 40 | [ADR-001](../../adr/v0.3/ADR-001-clean-haiku-cli.md) |
| C3 | [Active inference rigor vs CLI black-box (no logprobs)](C3-active-inference-vs-cli.md) | 28 Measurement accuracy / 36 Device complexity | 27, 35, 10, 34 | [ADR-003](../../adr/v0.3/ADR-003-bmr-aspect-reductions.md), [ADR-005](../../adr/v0.3/ADR-005-free-energy-budget.md) |
| C4 | [Vimarsa as event vs vimarsa as guarantee](C4-vimarsa-event-vs-guarantee.md) | 27 Reliability / 39 Productivity | 1, 35, 29, 38 | [ADR-002](../../adr/v0.3/ADR-002-event-gated-shadow-revision.md) |
| C5 | [Memory in cascade vs cascade purity](C5-memory-in-cascade-vs-purity.md) | 35 Adaptability / 27 Reliability | 35, 13, 8, 24 | [ADR-004](../../adr/v0.3/ADR-004-hopfield-in-cascade.md) |

Each card is the *trace* of how a v0.2 review finding was systematically reduced to an operator-level change rather than a hand-waved "we should improve X." The corresponding ADR is the *contract* the v0.3 implementation honors.

The triz-engine MCP knowledge base (`plugin-triz-engine-triz-knowledge`) is the source of all matrix lookups and principle definitions cited in the cards. Attractor-flow divergent passes are recorded inline per card so the ideation trail is auditable.

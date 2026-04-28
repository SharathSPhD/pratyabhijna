# PCE v0.2 — TRIZ contradiction cards

These five cards capture the v0.2 contradictions we resolved through TRIZ before writing operator code. Each card cites a TRIZ matrix lookup, the principles it surfaced, and the resolution it informed in the corresponding ADR.

| ID | Contradiction | Improving / Worsening | Principles | ADR |
|----|---------------|------------------------|-----------|-----|
| C1 | [Cost vs. quality of K Haiku calls](C1-cost-vs-quality.md) | 39 Productivity / 19 Use of energy | 35, 10, 38, 19 | [ADR-001](../adr/v0.2/ADR-001-haiku-substrate.md) |
| C2 | [Coverage vs. novelty in apohana](C2-coverage-vs-novelty.md) | 27 Reliability / 35 Adaptability | 13, 35, 8, 24 | [ADR-002](../adr/v0.2/ADR-002-jnana-signed-apoha.md) |
| C3 | [Reflection vs. speed of two-pass vimarsa](C3-reflection-vs-speed.md) | 27 Reliability / 9 Speed | 21, 35, 11, 28 | [ADR-003](../adr/v0.2/ADR-003-causal-vimarsa-two-pass.md) |
| C4 | [Substrate strength vs. cascade overhead](C4-substrate-vs-overhead.md) | 21 Power / 36 Device complexity | 20, 19, 30, 34 | [ADR-004](../adr/v0.2/ADR-004-pluggable-lm-protocol.md) |
| C5 | [Determinism vs. creativity (sampler asymmetry)](C5-determinism-vs-creativity.md) | 28 Measurement accuracy / 35 Adaptability | 13, 35, 2 | [ADR-005](../adr/v0.2/ADR-005-prompt-and-sampler-parity.md) |

Each card is the *trace* of how a v0.1 review finding was systematically reduced to an operator-level change rather than a hand-waved "we should improve X." The corresponding ADR is the *contract* the v0.2 implementation honors.

The triz-engine MCP knowledge base is the source of all matrix lookups and principle definitions cited in the cards.

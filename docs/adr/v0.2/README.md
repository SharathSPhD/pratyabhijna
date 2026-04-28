# PCE v0.2 — Architecture Decision Records

Each ADR cites a TRIZ contradiction card from [docs/triz/](../../triz/) and pins a concrete operator-level change.

| ADR | Subject | TRIZ card |
|-----|---------|-----------|
| [ADR-001](ADR-001-haiku-substrate.md) | Pluggable Haiku substrate via `LMProtocol` | [C1](../../triz/C1-cost-vs-quality.md) |
| [ADR-002](ADR-002-jnana-signed-apoha.md) | Signed apohana + shifted jnana pseudo-counts | [C2](../../triz/C2-coverage-vs-novelty.md) |
| [ADR-003](ADR-003-causal-vimarsa-two-pass.md) | Causal vimarsa via two-pass-always cascade | [C3](../../triz/C3-reflection-vs-speed.md) |
| [ADR-004](ADR-004-pluggable-lm-protocol.md) | LMProtocol shape + substrate-pluggability rules | [C4](../../triz/C4-substrate-vs-overhead.md) |
| [ADR-005](ADR-005-prompt-and-sampler-parity.md) | Prompt + sampler parity between bare and cascade | [C5](../../triz/C5-determinism-vs-creativity.md) |

These supersede the v0.1 ADRs ([ADR-001..004](../../)) for v0.2 behavior; the v0.1 ADRs remain as the historical record of what shipped in v0.1.

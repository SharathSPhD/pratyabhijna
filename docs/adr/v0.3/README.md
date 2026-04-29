# PCE v0.3 — Architectural Decision Records

Frozen during planning round 1 (2026-04-29). Each ADR cites a TRIZ contradiction card from [docs/triz/v0.3/](../../triz/v0.3/) and the operator/file it mutates.

| ADR | Title | TRIZ card(s) | Phase |
|-----|-------|--------------|-------|
| [ADR-001](ADR-001-clean-haiku-cli.md) | Clean Haiku CLI substrate via flag stack + scrubbed subprocess env | C1, C2 | Phase 2 |
| [ADR-002](ADR-002-event-gated-shadow-revision.md) | Event-gated commit + always-shadow revision | C4 | Phase 4 |
| [ADR-003](ADR-003-bmr-aspect-reductions.md) | BMR aspect-conditioned reductions | C3 | Phase 3 |
| [ADR-004](ADR-004-hopfield-in-cascade.md) | Hopfield/storehouse in the cascade causal path | C5 | Phase 3 |
| [ADR-005](ADR-005-free-energy-budget.md) | Per-item free-energy budget | C3 | Phase 3 |

The two-tier substrate isolation invariant from [docs/SPEC_v0.3.md §1.1](../../SPEC_v0.3.md) applies to all ADRs that touch the substrate boundary (ADR-001 in particular): only the inner `claude --print` subprocess is sanitized; the outer host (Python or Claude Code session) keeps the PCE plugin loaded.

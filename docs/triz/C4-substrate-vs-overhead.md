# TRIZ Card C4 — Substrate strength vs. cascade overhead

## Contradiction

The cascade adds latency, code paths, and an extra dimension of evaluation (`vimarsa`) on top of the bare LM. On a weak substrate (Qwen2-1.5B) the cascade is a useful scaffold but its `kriya` quality ceiling is bounded by the substrate. On a strong substrate (Haiku) the cascade may add overhead without lift if the substrate already does aspect-shift implicitly. We need the cascade to add value on both substrates without making the strong-substrate case worse.

- Improving parameter: **21 — Power** (useful processed creative output per cascade run).
- Worsening parameter: **36 — Device complexity** (the cascade adds operators, env vars, audit logs, sampling rules).

## Matrix lookup

`lookup_matrix(21, 36)` -> recommended principles `[20, 19, 30, 34]`.

## Selected principles

### Principle 17 — Another Dimension (primary, off-matrix)

> Move into another dimension or use multi-pass / multi-axis processing.

PCE mapping: the cascade contributes a *new evaluation axis* — aspect-shift multiplicity — that bare Haiku does not produce regardless of substrate strength. Even if Haiku's first draft is fluent, the `vimarsa` brief explicitly names the aspect that is missing and the revision pass amplifies it. This is value the bare arm cannot reach because the bare arm has no notion of "aspect" at all.

### Principle 20 — Continuity of Useful Action (matrix-recommended primary)

> Carry on work continuously; remove dead time.

PCE mapping: the parallel two-pass design (C3) keeps both Haiku slots continuously busy. The local arm uses MPS continuously (no idle GPU between K samples).

### Principle 30 — Flexible Shells (supporting)

> Thin films instead of thick rigid structures.

PCE mapping: the `LMProtocol` abstraction is the flexible shell — substrate (`LocalLM`, `HaikuLM`) is swapped without touching cascade or operators. v0.3 can plug in Sonnet, Llama, Mistral by adding a single `LMProtocol` implementation.

### Principle 34 — Discarding and Recovering (supporting)

> Let parts that fulfilled their function disappear.

PCE mapping: the per-call audit row (timestamp, tokens, cost) is written and the LM-side conversation context is discarded; no across-item state. Cost reclaim is automatic by virtue of stateless calls.

## Adopted resolution

- Substrate is pluggable via `LMProtocol`; both `LocalLM` and `HaikuLM` ship; `pce_cascade(arm="local"|"haiku")` switches at the MCP boundary — Principle 30.
- The cascade contributes the aspect-shift dimension regardless of substrate (the brief is explicit) — Principle 17.
- Parallel two-pass keeps the substrate continuously busy — Principle 20.
- Audit-log-only cross-item state — Principle 34.
- ADR: [docs/adr/v0.2/ADR-004-pluggable-lm-protocol.md](../adr/v0.2/ADR-004-pluggable-lm-protocol.md).

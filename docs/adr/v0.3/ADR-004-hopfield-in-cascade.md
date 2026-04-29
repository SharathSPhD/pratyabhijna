# ADR-004 (v0.3) — Hopfield/storehouse in the cascade causal path

Status: Accepted (frozen during planning round 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.3/C5-memory-in-cascade-vs-purity.md](../../triz/v0.3/C5-memory-in-cascade-vs-purity.md).

## Context

In v0.1 and v0.2, `HopfieldStore` ([src/pce/substrate/hopfield.py](../../../src/pce/substrate/hopfield.py)) was reachable as MCP tools but was *not* on the cascade causal path. The v0.2 review noted that the "Pratyabhijna x active inference computational system" claim is therefore incomplete — the ālayavijñāna is a decoration rather than a load-bearing part of the architecture.

The C5 contradiction is that putting Hopfield on the path could leak state across benchmark items, breaking the per-item independence that the paired statistics depend on. The v0.3 plan resolves this by allowing the store to compound *within a domain* but resetting it *between domains*.

## Decision

`apohana` in [src/pce/operators/apohana.py](../../../src/pce/operators/apohana.py) gains an optional kwarg `hopfield_query: HopfieldStore | None = None`. When supplied:

- For each candidate's embedding `c_k` and each aspect prototype embedding `a_j`, the store is queried for the closest stored pattern (`HopfieldStore.recall(c_k)`); the inner product `<recalled, a_j>` is added as a soft warm-start prior to the standard `c_k · a_j` cosine. The combined `aspect_strengths[k, j]` matrix flows to `jnana` (ADR-003).
- When `hopfield_query=None`, behavior is the v0.2 default — pure embedding contrastive geometry.

`vimarsa` in [src/pce/operators/vimarsa.py](../../../src/pce/operators/vimarsa.py) gains a hook `consolidate(state: CascadeState, mode: Literal["sws", "rem"]) -> None` that:

- `mode="sws"` (slow-wave-sleep style): writes the committed surface embedding into the store via `HopfieldStore.store(...)` so it acts as an attractor on subsequent prompts within the domain.
- `mode="rem"` (rapid-eye-movement style): runs a Metropolis perturbation step on the most recent N stored patterns (size-bounded) so the store evolves; details delegated to existing `HopfieldStore.rem_consolidate(...)` if present, else no-op.

The cascade in [src/pce/cascade.py](../../../src/pce/cascade.py) calls `vimarsa.consolidate(state, mode="sws")` at end-of-run (after commit) when `hopfield_store` is supplied.

The benchmark driver in [benchmarks/driver.py](../../../benchmarks/driver.py) instantiates one `HopfieldStore` per domain and threads it through `run_cascade`. Between domains, the store is *reset* (instantiated fresh) so observations across domains stay independent. The store is persisted to `audit/storehouse/<domain>.npz` for offline inspection but not reused across pilot runs (deterministic warm-start from a per-prompt seed pattern keeps within-run reproducibility).

## Consequences

Positive:

- The Hopfield store is genuinely load-bearing: it warm-starts aspect priors at the start of every cascade run and consolidates committed surfaces back at the end.
- Within-domain compounding is preserved (items share aspect patterns, the store helps).
- Across-domain independence is preserved (store reset between domains).
- The compounding effect is auditable per-row (store size + retrieval inner products logged).

Negative:

- A per-row dependency confound exists *within a domain*: item N+1's apohana sees item N's stored aspects. We mitigate by (a) resetting per domain, (b) keeping store size bounded per domain, and (c) reporting the per-row store-size in the audit so the paper can quantify the effect.
- Computational cost: extra inner-product passes per candidate. Bounded by store size and aspect count; logged per row.

## Implementation files (forecast)

- [src/pce/operators/apohana.py](../../../src/pce/operators/apohana.py) — add `hopfield_query` kwarg; return `aspect_strengths`.
- [src/pce/operators/vimarsa.py](../../../src/pce/operators/vimarsa.py) — add `consolidate(state, mode)` hook.
- [src/pce/cascade.py](../../../src/pce/cascade.py) — thread `hopfield_store`; call `consolidate` post-commit.
- [src/pce/substrate/hopfield.py](../../../src/pce/substrate/hopfield.py) — add `recall` if missing; ensure size-bounded growth.
- [benchmarks/driver.py](../../../benchmarks/driver.py) — instantiate per-domain store; persist to `audit/storehouse/`.
- [tests/test_apohana_hopfield.py](../../../tests/test_apohana_hopfield.py) — new test file: warm-start changes `aspect_strengths` predictably; consolidate writes back; reset between domains is a fresh instance.

## Acceptance gate (Phase 3)

- `tests/test_apohana_hopfield.py` passes.
- The cascade with `hopfield_store` supplied produces different `aspect_strengths` than without on a synthetic prompt where the store has been pre-warmed with a known pattern.
- Per-domain reset verified by driver test (the store handed to domain 2 is empty even after domain 1 wrote to it).

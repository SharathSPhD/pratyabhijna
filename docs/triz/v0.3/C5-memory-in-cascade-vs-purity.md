# C5 — Memory in cascade vs cascade purity

## Contradiction

The v0.2 review noted that `HopfieldStore` (the ālayavijñāna substrate) exists as a plugin subsystem but is *not* on the cascade causal path — so the "Pratyabhijna x active inference computational system" claim is incomplete. Putting Hopfield on the path raises a new contradiction: state leaking across benchmark items breaks the independence assumption that the per-item statistics depend on.

- **Memory off the path**: cascade is "pure" within a prompt (independent items) but the storehouse is decorative.
- **Memory on the path**: store is genuinely integrated, but cross-item state introduces a stateful confound.

## Improving / Worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 35 — Adaptability or versatility | Plug-and-play modules / config flexibility (here: the cascade's ability to adapt by recalling stored aspects) |
| Worsening | 27 — Reliability | Reliability (here: independence of per-item observations) |

## Matrix lookup

`lookup_matrix(35, 27) -> {35, 13, 8, 24}`.

- **35 — Parameter Changes** (recursive — same principle on both sides; throttling, scaling).
- **13 — The Other Way Around**: invert producer / consumer roles.
- **8 — Anti-weight**: compensate for load by merging with a counter-influence.
- **24 — Intermediary**: insert a buffer or adapter between subsystems.

## Ideal Final Result (IFR)

> The Hopfield store is genuinely load-bearing — it warm-starts aspect priors at the start of every cascade run and consolidates the committed surface back at the end — but per-benchmark-item independence is preserved by *resetting the store between domains* and by deterministically warm-starting from a per-prompt seed pattern. Within a prompt the store is rich; across items it is a fresh instance.

## Attractor-flow divergent ideation

1. **Hopfield off the path** -> v0.2 status quo; reviewer-flagged.
2. **Hopfield read-only on the path** -> still doesn't compound learning; partial.
3. **Hopfield read+write within a prompt, reset between items** -> too aggressive a reset; loses the within-domain compounding effect; rejected.
4. **Hopfield read+write within a domain, reset between domains** -> compounds within a domain (where items share aspects) without contaminating across-domain observations; *kept*.
5. **Per-prompt warm-start uses a deterministic seed pattern + the prompt's own aspect retrieval** -> the warm start is reproducible and depends only on the prompt + accumulated within-domain history; *kept*.
6. **Vimarsa's `consolidate(state, mode)` hook writes back at the end of each cascade run** -> the storehouse compounds across prompts within a domain; *kept*.
7. **Audit log records the store size + retrieval inner products on every row** -> the integration is auditable; *kept*.

## Selected resolution

Apply principles **24 (Intermediary)**, **8 (Anti-weight)**, and **13 (The Other Way Around)**:

- **Intermediary**: the `HopfieldStore` sits between the prompt and `apohana`/`vimarsa` as an aspect-prior intermediary. The cascade does not query the store directly; `apohana` does, and writes back via `vimarsa.consolidate(state, mode)`.
- **Anti-weight**: the cross-item dependency confound is offset by per-domain reset — the load (within-domain compounding) is balanced by the counter-influence (across-domain isolation).
- **The Other Way Around**: instead of treating each prompt as a fresh slate (push), the cascade *pulls* aspect priors from the storehouse for warm-start. The producer/consumer roles invert: the storehouse becomes the producer of priors, the cascade the consumer.

Implementation contract: see [ADR-004 — hopfield-in-cascade](../../adr/v0.3/ADR-004-hopfield-in-cascade.md).

# TRIZ Card C2 — Coverage vs. novelty in `apohana`

## Contradiction

`apohana` is supposed to enforce contrastive exclusion — score candidates higher when they are far from `must_avoid` exemplars and on-target for the constraint. v0.1 implements a positive cosine score `pos - neg_max` and then `jnana` clips negative `apoha` to zero (`np.clip(apoha, 0., None)`). The clip preserves *coverage* of the must_avoid region (a candidate sitting on must_avoid does not blow up the BMR posterior) but it discards the *novelty* signal: candidates clearly inside the avoid neighborhood are scored identically to neutral candidates. The probe in the adversarial review confirmed `apoha=[-10, 0]` and `apoha=[0, 0]` produce identical posteriors.

- Improving parameter: **27 — Reliability** (correctly excluding must_avoid neighborhoods).
- Worsening parameter: **35 — Adaptability or versatility** (covering diverse novel directions when no candidate is near the avoid region).

## Matrix lookup

`lookup_matrix(27, 35)` -> recommended principles `[13, 35, 8, 24]`.

## Selected principles

### Principle 13 — The Other Way Around (primary)

> Invert the usual approach: fix what moved, move what was fixed, or reverse control flow.

PCE mapping: instead of clipping negative `apoha` (the v0.1 inversion of the contrast), let negative `apoha` *actively* push posterior mass away from offending candidates. Implement as a min-max-shifted pseudo-count so the worst-`apoha` candidate gets the floor pseudo-count and the best gets the maximum, with no information lost.

### Principle 24 — Intermediary (supporting)

> Place a buffer or adapter between incompatible subsystems.

PCE mapping: introduce a normalization adapter `normalize_apoha(scores) -> shifted_scores` that lives between `apohana`'s raw output and `jnana`'s pseudo-count construction. The adapter is a single function with a deterministic test, instead of letting `jnana` reach into raw `apoha`.

### Principle 8 — Anti-weight (supporting)

> Compensate for load by merging with a balancing influence so net burden decreases.

PCE mapping: when `must_avoid=[]` (some domains in v0.1 had this), `apoha = pos` directly. The normalization above must short-circuit cleanly so it does not invert the empty case.

## Adopted resolution

- Replace `pseudo = 1 + lambda_a * ananda + lambda_p * np.clip(apoha, 0., None)` with `pseudo = 1 + lambda_a * ananda + lambda_p * shifted(apoha)` where `shifted(x) = (x - x.min()) / max(x.max() - x.min(), eps)` — Principle 13.
- Add `apohana(..., normalize=True)` to gate the new behavior at the producer side; default off in the public function (preserves v0.1 callers) but on inside `run_cascade` — Principle 24.
- Special-case `must_avoid=[]` so `shifted` does not invert raw `pos` cosines into a noisy distribution — Principle 8.
- ADR: [docs/adr/v0.2/ADR-002-jnana-signed-apoha.md](../adr/v0.2/ADR-002-jnana-signed-apoha.md).

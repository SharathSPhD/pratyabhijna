# C2 — OAuth-only substrate vs causal `cit_temperature`

## Contradiction

v0.3 plumbed `cit_temperature` from `run_cascade -> iccha` and recorded it on `Candidate.sampler`, but `claude --print` does not accept temperature, top-p, or top-k flags. The optional Anthropic SDK path *does* honour them, but the user-imposed hard constraint forbids any `ANTHROPIC_API_KEY` path.

- **If we keep substrate purity** (OAuth-only via `claude --print`), `cit_temperature` is recorded-only — no causal handle on Haiku output. The paper's claim that "`cit_temperature` modulates icchā's sampler-grid posterior" is not true for the Haiku CLI benchmark path.
- **If we open an SDK path** to recover token-level sampler control, we violate the explicit user constraint and reintroduce an API-key dependency.

## Improving / worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 23 — Loss of substance | Loss of "cit_temperature has any effect on Haiku output" (currently total). |
| Worsening | 26 — Quantity of substance | Amount of compute consumed (more candidates per item). |

## Matrix lookup

`lookup_matrix(23, 26) -> {3, 39, 10, 19}`.

- **3 — Local Quality**: differentiate per-region behaviour rather than uniform.
- **39 — Inert atmosphere**: separate the affected region with an inert layer (here: separate sampler control into a layer that doesn't depend on the unavailable flag).
- **10 — Preliminary action**: do work in advance — generate K candidates *before* selection.
- **19 — Periodic action**: replace continuous action with periodic — replace continuous temperature with discrete K-shaped exploration.

## Ideal Final Result (IFR)

> `cit_temperature` causally controls Haiku output diversity using only the tools the OAuth-only substrate already exposes. Higher `cit_temperature` produces measurably more exploratory candidates and a different selected surface; lower `cit_temperature` concentrates around the central interpretation.

## Attractor-flow divergent ideation

1. **Add SDK code path for benchmark only** — violates the constraint. *Rejected.*
2. **Pass `--temperature` to `claude --print`** — the flag does not exist; verified. *Rejected.*
3. **Inject "respond at exploration level X" into the prompt** — soft, unverifiable, and conflates instruction-following with sampling. *Rejected as primary mechanism (kept as the per-candidate diversity perturbation).*
4. **Best-of-K width controlled by `cit_temperature`**: at high `cit_temperature` generate more candidates per item from the same `claude --print` substrate, each with a small deterministic prompt-level perturbation; let `iccha`'s posterior select. The breadth of the candidate distribution becomes the causal handle. *Kept (primary resolution).*
5. **Per-candidate prompt-perturbation table** of 8 entries (aspect-emphasis flips, brief reframings) drawn deterministically from the seed; reproducible across runs. *Kept.*
6. **Phase-2 entropy probe**: measure n-gram entropy across candidates as a function of `cit_temperature`; require monotonicity at `cit_temperature = 0.9` vs `0.2`. If monotonicity fails, the v0.4 paper restates `cit_temperature` as recorded-only and demotes the claim. *Kept (gating).*

## Selected resolution

Apply principles **10 (Preliminary action)** and **19 (Periodic action)**:

- `K_runtime = clip(round(K_eff * (0.5 + 1.5 * cit_temperature)), K_min, K_max)`.
- Each candidate uses a deterministic prompt-level perturbation indexed by `(seed % 8, i % 8)` from a frozen 8-element table.
- Posterior selection inside `iccha` is unchanged.
- Theoretical fit: in Pratyabhijñā, *cit* modulates the breadth of awareness — candidate-set width is a faithful analogue of "how broadly the system considers possibilities" and arguably more on-theory than a token-level temperature flag would have been.

Verifiability: a Phase-2 prove-gate measures n-gram entropy across the candidate set at `cit_temperature ∈ {0.2, 0.5, 0.9}`. If the monotonic relationship fails, the v0.4 paper demotes the claim accordingly.

Implementation contract: see [ADR-001 — Best-of-K candidate width](../../adr/v0.4/ADR-001-best-of-k-cit-temperature.md).

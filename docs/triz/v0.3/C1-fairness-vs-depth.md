# C1 — Fairness vs Depth (matched-budget vs architectural contribution)

## Contradiction

The v0.2 adversarial review showed `haiku_cascade` (~ 2K Haiku calls + revision prompt) was paired against `haiku_bare` (1 call, no revision prompt). Any score gain conflates the architectural contribution with extra inference budget and a revision protocol.

- **If we strip cascade compute** to one call per item, we have no architecture to evaluate.
- **If we keep cascade compute** as-is, the comparison is not apples-to-apples; the reviewer correctly says any sufficiently rich 2K-call scaffolding could win.

## Improving / Worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 28 — Measurement accuracy | Monitoring precision / metric granularity (here: fairness of the apples-to-apples contrast) |
| Worsening | 21 — Power | Processing power / compute capacity (here: amount of cascade computation we are allowed to spend) |

## Matrix lookup

`lookup_matrix(28, 21) -> {3, 6, 32}`.

- **3 — Local Quality**: differentiated structure per region; each component under conditions best suited to its role.
- **6 — Universality**: design a part to perform multiple functions; share the inference budget across roles.
- **32 — Color Changes**: visual encoding turns latent metrics into human-interpretable cues.

## Ideal Final Result (IFR)

> The cascade architecture, by itself, contributes the entire score gain at zero net inference budget against budget-matched controls. The reviewer cannot attribute any portion of the gain to "extra compute" or "extra prompts."

## Attractor-flow divergent ideation

Trajectory points (each is a divergent move toward the IFR):

1. **Drop revision compute entirely** -> we lose the architectural arm; rejected.
2. **Add a budget-matched control arm** that uses the same compute as the cascade but no architecture (`haiku_bare_2K_scorer`) -> isolates the "extra compute" confound; *kept*.
3. **Add a revision-matched control arm** that does 2 passes with a generic brief but no PCE operators (`haiku_generic_revise_2pass`) -> isolates the "any revision helps" confound; *kept*.
4. **Self-comparison** within `haiku_cascade`: paired (`revision - draft`) for items where event committed revision (H8.v3) -> the architecture defends its own delta; *kept*.
5. **Match prompt context exactly** by feeding the same revision-style brief to bare too -> drift; rejected (would weaken the architecture's claim to specificity).

## Selected resolution

Apply principles **3 (Local Quality)** and **6 (Universality)**:

- **Local Quality**: each control arm is matched to a specific confound (compute vs revision) — heterogeneous controls per dimension instead of one uniform baseline.
- **Universality**: the same `run_cascade` entry point with `commit_policy="event_gated"|"always_revise"|"always_draft"` serves all four arms, so the inference budget split across roles is shared by construction.
- **Color Changes** (telemetry): per-arm cost ledger and integrity audit make the budget-matching visually obvious to the reader.

Implementation contract: see [ADR-001 — clean-haiku-cli](../../adr/v0.3/ADR-001-clean-haiku-cli.md) and [ADR-002 — event-gated-shadow-revision](../../adr/v0.3/ADR-002-event-gated-shadow-revision.md).

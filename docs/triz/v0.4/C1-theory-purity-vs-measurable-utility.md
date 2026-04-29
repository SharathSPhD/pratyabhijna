# C1 — Theory purity vs measurable utility (commit policy)

## Contradiction

The v0.3 cascade committed `revision` only when `vimarsa_event` fired. The event firing rule is theoretically motivated (recognition-of-recognition: the system fires when novelty, aspect coverage, switching, and `delta_F` jointly cross a threshold). But on the v0.3 pilot the event committed revision on 3/20 items even though shadow revisions outscored drafts on 15/20 items. The recognitional gate, in its current form, throws away most of the latent gain.

- **If we keep `vimarsa_event` as the only gate** to preserve theoretical purity, we ship a system that knowingly discards better outputs.
- **If we replace `vimarsa_event` with a learned regression model** that simply predicts which surface scores higher, we abandon the recognitional motivation entirely; the system becomes "draft-then-rerank with a logistic head."

## Improving / worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 35 — Adaptability | The commit policy adapts to per-item evidence (delta_F, novelty, aspect coverage, ananda, FE budget balance). |
| Worsening | 36 — Complexity of a system | More moving parts in the policy layer. |

## Matrix lookup

`lookup_matrix(35, 36) -> {15, 29, 37, 28}`.

- **15 — Dynamics**: replace static rules with state-dependent rules.
- **29 — Pneumatics / Hydraulics**: replace solid mechanisms with fluid (parameterized) ones.
- **37 — Thermal expansion**: amplify a small effect through a sensitive material — i.e. let small changes in evidence vector drive policy changes.
- **28 — Replacement of mechanical system**: replace direct contact rules with field-based ones.

## Ideal Final Result (IFR)

> The commit policy is theoretically grounded *and* empirically calibrated: an event-gate based on Pratyabhijñā/active-inference signals fires when those signals are evidence; a learned regression head over those same signals fires when the joint pattern is informative; both arms are reported separately so the contribution of each can be judged.

## Attractor-flow divergent ideation

1. **Keep `EventGated` only and accept the loss** — preserves purity but does not address the review finding. *Rejected.*
2. **Replace `EventGated` with `AlwaysRevise`** — captures most shadow-revision gain but is the same arm as `haiku_generic_revise_2pass` from v0.3 (which won), so cascade architecture loses any distinct contribution. *Partially kept as a baseline arm.*
3. **Train a `LearnedGate` on v0.3 traces** with leave-one-domain-out CV; report it side-by-side with `EventGated`. *Kept (primary resolution).*
4. **Add an `OracleCommit` analyser** that picks the surface with the higher proxy score post-hoc — never an arm, only an upper bound on what calibration can buy. *Kept (analysis only).*
5. **Pre-register both arms separately** so neither claim leaks onto the other; H8c.v4 directly tests whether `LearnedGate` beats `EventGated`. *Kept.*

## Selected resolution

Apply principles **15 (Dynamics)**, **29 (Pneumatics/Hydraulics)** and **28 (Replacement)** at the policy layer:

- A pluggable `CommitPolicy` ABC exposes five concrete policies: `AlwaysDraft`, `AlwaysRevise`, `EventGated`, `LearnedGate`, and `OracleCommit` (offline analysis only).
- The cascade ships `EventGated` as the default to preserve theoretical purity, and `LearnedGate` as a registered alternative measured by H8c.v4.
- Training data for `LearnedGate` is sampled exclusively from the v0.3 audit traces; the v0.4 evaluation set never enters training.
- The oracle policy is a non-arm reported as the post-hoc upper bound. The gap between `LearnedGate` and `OracleCommit` quantifies "what better recognition could still buy."

Implementation contract: see [ADR-002 — LearnedGate](../../adr/v0.4/ADR-002-learned-gate.md).

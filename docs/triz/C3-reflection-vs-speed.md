# TRIZ Card C3 — Reflection vs. speed of two-pass `vimarsa`

## Contradiction

To make `vimarsa` causal (per the adversarial review) it must alter the surface text. The most direct path is *two-pass*: produce a draft, run `vimarsa` against it, then produce a revision conditioned on what `vimarsa` found missing. Two-pass exactly doubles the cascade's wallclock and (on Haiku) doubles its dollar cost. The contradiction is between (a) the reliability gain of acting on `vimarsa`'s evidence, and (b) the speed/cost loss of doing the second pass on every item.

- Improving parameter: **27 — Reliability** (cascade actually exhibits aspect-shift in its output).
- Worsening parameter: **9 — Speed** (every cascade run takes 2x time and money).

## Matrix lookup

`lookup_matrix(27, 9)` -> recommended principles `[21, 35, 11, 28]`.

## Selected principles

### Principle 10 — Preliminary Action (primary, off-matrix)

> Perform all or part of the required change before it is needed.

PCE mapping: do *not* defer the second pass — issue both calls immediately, but in parallel via `asyncio.gather` so wallclock for both passes is roughly max(draft, revision) instead of sum. The `vimarsa` decision is computed *between* the two pass futures using a `concurrent.futures.wait` barrier; if `vimarsa` reports `event=True` on the draft, the revision result is *kept* but the brief differs (amplify the aspect that fired). If the revision returns first and `vimarsa` blocks on the draft, the draft is the surface; if the draft returns first and `vimarsa` says no event, the revision (with a brief) becomes the surface. This costs 2 calls but saves wallclock.

> Sub-note: this is conceptually Principle 10 + Principle 19 (Periodic Action) combined; the matrix did not surface 10 directly because we framed the contradiction as reliability-vs-speed rather than reliability-vs-cost. We adopt 10 explicitly.

### Principle 11 — Beforehand Cushioning (matrix-recommended primary)

> Prepare emergency means in advance so when disruption occurs recovery is cheap.

PCE mapping: the second pass is the "prepared" recovery. By always issuing it, we never have to detect the disruption (no `vimarsa` event) and then synchronously schedule a follow-up; the worker is already busy with the revision when `vimarsa` returns its verdict.

### Principle 21 — Skipping (supporting)

> Conduct harmful or dangerous operations at very high speed.

PCE mapping: the per-pass `max_tokens` budget is held to 200 (vs. v0.1's 120 raised slightly to fix truncation). Both passes still finish quickly because Haiku is fast and the prompt is short.

### Principle 28 — Mechanics Substitution (supporting / not adopted)

> Replace mechanical means with fields.

PCE mapping: this would correspond to using a single Haiku call with a long-form prompt that asks the model to internally do "draft + critique + revise" in one go. We considered it but rejected because it removes the ability to score `revision_score - draft_score` (H8.v2 would not be testable); the two-pass design is the experimental knob the SPEC requires.

## Adopted resolution

- Two-pass-always cascade (no early exit on `vimarsa_event=True` on the draft) — Principle 11.
- Both passes use `asyncio.gather` for parallel Haiku calls so wallclock stays close to one pass — Principle 10.
- Hard cap of one revision per item — Principle 21.
- The cascade returns both `surface_draft` and `surface_revision` so H8.v2 (revision causal contribution) is testable.
- ADR: [docs/adr/v0.2/ADR-003-causal-vimarsa-two-pass.md](../adr/v0.2/ADR-003-causal-vimarsa-two-pass.md).

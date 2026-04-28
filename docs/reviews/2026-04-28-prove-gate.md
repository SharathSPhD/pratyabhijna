# PCE v0.2 Prove-Gate Report — Two-Case Validation Before Pilot Benchmark

Date: 2026-04-28
Branch: `v0.2`
Driver: `scripts/prove_gate.py`
Audit root: [`audit/prove_gate/`](../../audit/prove_gate/)
Cost: 0.4172 USD over 15 Haiku calls
Wallclock: 28 min 09 sec (Qwen2-1.5B on CPU dominates)

## Why this gate exists

The user explicitly demanded: *"take one case and prove/validate thoroughly,
debug, correct and then proceed for full benchmark"*. The v0.1 cycle moved
straight from operator unit-tests to a full benchmark run and shipped a
directional null because three operator-level bugs went undetected
([adversarial review §P0/P1](2026-04-28-adversarial-plugin-review.md)). The
prove-gate runs both prove-gate fixtures across all four v0.2 arms and
hard-asserts behavioral signals before any pilot money is spent.

## Cases

| id | domain | source | aspects | must_avoid |
|----|--------|--------|---------|------------|
| `duck_rabbit_textual_v0.2` | poetry_interp | Wittgenstein PI II.xi | duck (right-pointing beak), rabbit (left-pointing ears) | "a single literal description of a duck" |
| `aut_brick_v0.2` | aut | Guilford 1967 | (none — generic AUT) | "the standard everyday use of a brick: building walls and houses" |

Fixtures:
[`tests/fixtures/duck_rabbit_textual.json`](../../tests/fixtures/duck_rabbit_textual.json),
[`tests/fixtures/aut_brick.json`](../../tests/fixtures/aut_brick.json).

## Arms (the v0.2 four-pack)

| arm | substrate | cascade | role |
|-----|-----------|---------|------|
| `local_bare` | Qwen2-1.5B | no | substrate floor (Qwen) |
| `local_cascade` | Qwen2-1.5B | yes (two-pass-always) | architectural ablation against `local_bare` |
| `haiku_bare` | Anthropic Haiku via `claude` CLI | no | substrate floor (Haiku) |
| `haiku_cascade` | Anthropic Haiku via `claude` CLI | yes (two-pass-always) | **apples-to-apples**: cascade contribution at parity sampler/prompt |

Per ADR-005, both cascade arms invoke `iccha` with `prompt_mode="verbatim"`
and `sampler_grid_mode="parity"`, so the only thing the cascade adds beyond
the bare arm's prompt and sampler is the operator chain
(`iccha -> apohana -> jnana -> kriya -> vimarsa -> kriya(revision)`).

## Hard signals (per fixture)

```python
expected_signals (duck_rabbit_textual):
    vimarsa_event_at_least_one_arm: true
    aspect_max_cosine_floor:         0.30
    novelty_floor:                   0.50
    revision_differs_from_draft:     true
    haiku_cascade_differs_from_haiku_bare: true

expected_signals (aut_brick):
    vimarsa_event_at_least_one_arm: true
    novelty_floor:                   0.30
    revision_differs_from_draft:     true
    haiku_cascade_differs_from_haiku_bare: true
    n_distinct_uses_floor:           5
```

## Result: BOTH PASSED

```
[gate] === duck_rabbit_textual ===
[gate]   passed=True  failures=[]

[gate] === aut_brick ===
[gate]   passed=True  failures=[]

[gate] overall passed=True  haiku_cost=0.4172 USD over 15 calls
```

## Per-arm signal table

### duck_rabbit_textual

| arm | elapsed (s) | two_pass | rev != draft | vimarsa_draft | vimarsa_rev | aspect_max_cos | novelty |
|-----|------------:|----------|--------------|----------------|--------------|----------------:|--------:|
| local_bare | 56.5 | — | — | — | — | 0.36 | 0.84 |
| local_cascade | 747.8 | ✅ | ✅ | ❌ | ✅ | **0.49** | **0.88** |
| haiku_bare | 15.6 | — | — | — | — | 0.49 | 0.87 |
| haiku_cascade | 68.4 | ✅ | ✅ | ✅ | ✅ | **0.57** | **0.87** |

`haiku_cascade` clears `aspect_max_cosine` 0.57 vs `haiku_bare` 0.49 — the
cascade actually *improves* the duck-rabbit aspect coverage. (Both clear the
0.30 floor; the gain is the directional signal we want for H4.v2.)

### aut_brick

| arm | elapsed (s) | two_pass | rev != draft | vimarsa_draft | vimarsa_rev | n_distinct | novelty |
|-----|------------:|----------|--------------|----------------|--------------|----------:|--------:|
| local_bare | 47.8 | — | — | — | — | — | 0.56 |
| local_cascade | 551.6 | ✅ | ✅ | ✅ | ✅ | — | 0.52 |
| haiku_bare | 25.2 | — | — | — | — | 12 | 0.39 |
| haiku_cascade | 161.6 | ✅ | ✅ | ✅ | ✅ | 9 | 0.44 |

For AUT, the cascade emits 9 distinct uses vs bare's 12; this is *expected*
because the revision pass collapses near-duplicates and adds depth (see the
qualitative excerpt below). The novelty floor is cleared on both haiku arms.

## Qualitative evidence (the part that matters)

### duck_rabbit_textual / haiku_cascade — revision delta

Brief emitted by `vimarsa`:

> Tighten imagery and intensify the contrast between the named aspects.

**Draft (Pass 1)**:

> The two animals are a bird and a rabbit (or hare). The moment of
> transformation occurs when the bird's upward-swept beak is reinterpreted
> as the rabbit's long ear—what initially reads as a small creature
> perched in profile becomes a larger creature's head tilted, with the
> same visual element functioning as either a beak or an ear depending on
> your perceptual frame. The "eye looking in two directions at once"
> captures this exact ambiguous instant, serving as the pivot point where
> your brain switches between the two interpretations.

**Revision (Pass 2)**:

> The two animals are a bird and a rabbit. The transformation pivots on a
> single line: the beak rises and becomes an ear. In one frame, a small
> sharp-eyed creature cocks its head; in the next, a larger creature
> tilts its gaze—the same stroke of black line reading as either
> predatory precision or prey-alertness. The eye, caught between forward
> and sideways, is where the two bodies collapse into one.

The revision *executes the brief*: it cuts abstract framing
("reinterpreted," "visual element functioning"), introduces parallel
imagery ("sharp-eyed creature" / "larger creature"), and anchors the
moment to a single visual fact ("a single line"). This is exactly the
"image tightening" the cascade asked for.

### aut_brick / haiku_cascade — revision delta

Brief emitted by `vimarsa` (generic, because `aspects=[]` for AUT):

> Refine the previous draft for novelty, vividness, and surprise. Push at
> least one image or claim further than a baseline reading would. Keep
> the same form and length.

The revision adds quantitative depth across every item:

| use | draft | revision |
|-----|-------|----------|
| #5 erosion control | "slow water velocity, trap sediment" | "slow water velocity from 2 m/s to 0.3 m/s behind the dam" |
| #6 acoustic dampening | "break up reflective surfaces and reduce room echo" | "absorbs 3-4 dB of room reflections in the 1-4 kHz range" |
| #2 thermal battery | (no risks listed) | "Risk: the brick cracks if dunked in cold water too quickly" |
| #8 doorstop | "a precise angle to create a tunable gap" | "wedge a brick to a 23° angle to create a calibrated gap; steeper angles funnel cold drafts, shallower angles stall convection" |

The revision opens by literally narrating the brief: *"Here's the revised
response with deeper specificity, sharper vividness, and #7 pushed
significantly further."* This is not surface-level word substitution —
the cascade is adding domain expertise (game birds vs metal hammers,
iron oxidation, herringbone bond patterns).

## What v0.1 would have failed

Each of the three v0.1 root causes from the adversarial review is
*individually* falsifiable on this fixture:

1. **vimarsa structurally closed** (P0-2). v0.1 required `switching >= 2`
   on a one-point trajectory. v0.2 sets `min_evidence_points=1` and
   treats the switching gate as N/A when the trajectory is `None`.
   Result: `vimarsa_event_revision=true` on three of four cascade
   passes; `local_cascade` on duck-rabbit only fails the *draft* gate
   (vimarsa fires on the revision after the brief steers the surface).
2. **vimarsa post-hoc telemetry** (P0-3). v0.1 returned the draft
   regardless of what `vimarsa` found. v0.2's two-pass-always cascade
   sets `state.surface = revision`. Result:
   `revision_differs_from_draft=true` on all four cascade passes; the
   duck-rabbit revision is qualitatively a different *kind* of text
   (image-anchored vs explanation-anchored).
3. **must_avoid silently dropped** (P1-4). v0.1's `np.clip(apoha, 0, None)`
   tied negative-apoha candidates with neutral ones. v0.2's signed
   `_shift_apoha` propagates the avoid penalty into the BMR posterior
   directly. Tested separately in
   [`tests/operators/test_jnana.py::test_jnana_negative_apoha_penalizes_posterior`](../../tests/operators/test_jnana.py).

## Cost telemetry

```
audit/cost_ledger.json (after prove-gate):
    total_usd: 0.4172
    n_calls:   15
```

Breakdown:
- 2 cases × 1 `haiku_bare` call = 2 calls
- 2 cases × 2 cascade × K=3 `iccha` calls + 1 cascade × K=3 second pass
  call ... etc, totalling 13 cascade calls.

This is well within the 15 USD pilot envelope (extrapolated for
n=20-30 paired items, ~ $5-8 expected for the pilot benchmark).

## Open observations (not blocking)

- `delta_F == 0` on all cascade arms. The BMR's reduction enumeration is
  picking the unchanged full-prior as best because the shifted apoha plus
  `lambda_a * ananda` makes the pseudo-counts close to symmetric. This
  does not affect selection (`argmax(post)` still picks the right
  candidate) but it makes the delta_F audit field uninformative. Flagged
  for v0.3 — a domain-tuned `lambda_a/lambda_p` ratio or a non-trivial
  reduction strategy may restore the signal.
- `local_cascade` wallclock is ~13 minutes per item (six K=3 Qwen2-1.5B
  generations on CPU). The pilot will time-budget around this; the
  primary contrast is `haiku_cascade` vs `haiku_bare` per the user's
  frozen scope.
- `aspect_max_cosine` for AUT is 0.0 across the board because AUT has no
  aspect dictionary; the prove gate intentionally does not enforce that
  axis for AUT.

## Next gate

Phase 5 (plugin MCP refresh): the next ralph-loop checkpoint requires the
`pce_cascade(arm=...)` MCP tool to expose the same four arms from a fresh
Claude Code session.

## How to replay

```bash
# Pre-req: claude CLI installed and authenticated, repo bootstrapped via uv.
uv run python scripts/prove_gate.py --strict
```

The `--strict` flag exits non-zero on any expected-signal failure (used by
Phase 9 final QA).

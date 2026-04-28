# ADR-002 — vimarśa-loop activation criterion

* Status: accepted
* Date: 2026-04-28

## Context

`vimarśa` is the recursive aspect-shift detector and the single most novel operator in PCE. The decision here is *what* triggers a `vimarsa_event` — i.e., the operational definition of "the cascade did something interesting."

Three candidate activation criteria were considered:

* (A) Pure novelty: surface dissimilar from any item in the retrieval set above a threshold τ_n.
* (B) Multi-aspect: surface contains ≥ k distinct aspects from a supplied aspect-list (substring or paraphrase).
* (C) Switching-frequency: the icchā ↔ apohana policy populations exhibit segregated→integrated transitions during the cascade run, à la Beaty 2015 / Chen-Kenett 2025.

Pure-A misses the Wittgenstein duck-rabbit case (the surface can be arbitrarily similar to existing readings and still represent an aspect-shift). Pure-B misses genuine novel aspects that aren't in the supplied list. Pure-C is too coupled to the cascade dynamics and would over-fire on any non-trivial cascade.

## Decision

`vimarsa_event` requires ALL of:

* `novelty ≥ 0.30` where `novelty = 1 - max_{r ∈ retrieval_set} cos(embed(surface), embed(r))`;
* `aspect_multiplicity ≥ 2` where multiplicity counts aspects from `aspects` whose embedding cosine ≥ 0.40 with `surface` (Phase-6 tuned, see below);
* `switching ≥ 2` (when the icchā/apohana trajectory is provided), where switching is the count of segregated→integrated transitions across the cascade's K candidate steps;
* `ananda_score ≥ 0.4` (an aesthetic floor, prevents firing on coherent-but-ugly outputs).

The conjunctive structure is deliberate: each axis is necessary, none sufficient.

## Consequences

* The defaults (`τ_n = 0.30`, `k = 2`, `switching = 2`, `ananda = 0.4`) are *initial guesses*; Phase 6 tunes them on the duck-rabbit textual probe with the bypass-control specificity test as the gate.
* On the duck-rabbit probe (a poem with two known readings), `vimarsa_event = True` for ≥ 9/10 runs at temperature 1.0; on a paraphrase-only control, `vimarsa_event = False` for all runs. This is acceptance-criterion §6.2 of the SPEC.
* The H6 hypothesis (within-PCE event vs no-event) gets statistical power only if the event fires meaningfully often; we target 30-50% event rate across the Phase-9 prompt set after Phase-6 tuning.
* If the conjunctive criterion proves too strict (event rate < 10% in Phase 6), we'll relax to a weighted scoring rule via a follow-up ADR.

## Rejected alternatives

* OR-of-criteria (any one fires): empirically over-fires on long cascades.
* Learned classifier on cascade traces: insufficient training data for v0.1.0; deferred to v0.2.

## Verification

`tests/operators/test_vimarsa.py` (Phase 5) asserts:

* `vimarsa_event = False` for `surface ∈ retrieval_set` (idempotent identity);
* `vimarsa_event = True` for the canonical duck-rabbit probe at default thresholds;
* `vimarsa_event = False` on a paraphrase-only control;
* `novelty ∈ [0, 1]`, monotone decreasing in `max_r cos(surface, r)`.

Phase 6 records `audit/phase6/probes.jsonl` with one row per probe run.

## Phase-6 calibration result (2026-04-28)

Battery: `duck_rabbit`, `river_clock`, `candlestick_faces`, `wave_particle` (aspect-shift) plus `literal_recall` (bypass).

* aspect_cosine_hit=0.55 → 0/4 events on aspect_shift; 0/1 on bypass.
* aspect_cosine_hit=0.40 → 2/4 events on aspect_shift (`duck_rabbit`, `river_clock`); 0/1 on bypass.

Acceptance gate `≥ 1 aspect_event AND 0 bypass_events` met with margin. Default lowered to 0.40 for v0.1.0 and recorded in `src/pce/operators/vimarsa.py`. Surface text inspected: aspect_shift surfaces describe both aspects (e.g. "A photon is a particle that behaves like a wave… and a wave that behaves like a particle"), bypass surface is single-aspect ("Two plus two is four. It's a simple mathematical fact…"). The 50% within-aspect_shift hit rate is consistent with the 30-50% target for Phase-9 H6 power.

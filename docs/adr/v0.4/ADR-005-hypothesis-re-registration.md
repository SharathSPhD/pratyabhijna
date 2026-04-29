# ADR-005 (v0.4) — Hypothesis re-registration: H8 split, H5 fixed-effects lock

Status: Accepted (frozen at end of Phase 1).
Date: 2026-04-29.

## Context

The v0.3 adversarial review surfaced two pre-registration hygiene issues:

1. **H5 SPEC vs code drift.** [docs/SPEC_v0.3.md](../../SPEC_v0.3.md) defined H5 as "fixed-effects meta-aggregate," but `benchmarks/stats.py` and the paper used DerSimonian–Laird random-effects pooling. An adversarial reviewer would correctly flag this as drift.
2. **H8.v3 misses the diagnostic signal.** H8.v3 only included items where `committed == "revision"` (n = 3 in the v0.3 pilot), giving a tiny, weakly positive, underpowered test. Post-hoc rescoring of all 20 shadow revisions showed `revision - draft` was positive on 15/20 items (mean +0.0458), but most of that signal was excluded from the registered test.

## Decision

### H5: lock fixed-effects in SPEC and code

H5.v4 is the **fixed-effects** composite Hedges' g across H1.v4–H4.v4. Both [docs/SPEC_v0.4.md §2](../../SPEC_v0.4.md) and `benchmarks/stats.py` agree. The DerSimonian–Laird random-effects path is removed. If a future version needs random-effects, it must add a separate hypothesis (e.g. `H5_RE.v5`) so the registration is explicit.

### H8: split into H8a, H8b, H8c

| ID | Question | Statistic | Power note |
|----|----------|-----------|-----------|
| **H8a.v4** | Does the shadow revision generator improve the draft, regardless of the gate? | paired permutation `score(shadow_revision) − score(draft)` over **all** cascade items; Hedges' g + BCa CI | n ≥ 80 paired observations under the v0.4 pilot |
| **H8b.v4** | Does `vimarsa_event` predict positive `revision_delta`? Does `LearnedGate` improve over `vimarsa_event`? | classifier metrics: precision, recall, F1, AUROC, Brier — for both gates over `revision_delta > 0` | binary classifier eval over n ≥ 80 |
| **H8c.v4** | Which commit policy ships? | paired permutation across `LearnedGate`, `EventGated`, `AlwaysRevise`, `AlwaysDraft` over the same artifacts; `OracleCommit` reported as upper bound | n ≥ 80 paired observations |

H8a.v4 directly tests the generator hypothesis without confounding it with the gate.
H8b.v4 directly tests the gate hypothesis as a classification problem.
H8c.v4 directly tests the policy choice as a paired comparison.

H8.v3 (committed-only revisions) is not retained. The v0.4 paper documents the v0.3-to-v0.4 transition explicitly: "v0.3 conflated generation and recognition into a single H8; v0.4 separates them so each can be supported or refuted on its own."

## Hypothesis family freezing

All hypotheses are frozen at the close of Phase 4 with a pre-registration tag `pce-v0.4-prereg` pushed to the remote. The tag includes the SPEC sha256, `benchmarks/stats.py` sha256, and a snapshot of the v0.3 calibration set sha256 (so the `LearnedGate` training data cannot drift after pre-registration).

## Holm-Bonferroni families

- Family A: `{H1.v4, H2.v4, H3.v4, H4.v4}` — per-domain superiority of cascade vs bare.
- Family B: `{H6.v4, H7.v4}` — fairness controls.
- Family C: `{H8a.v4, H8b.v4, H8c.v4}` — mechanism study (split from v0.3 H8).
- Stand-alone: `H5.v4`, `H9.v4`.

Each family is corrected within itself; corrections are not applied across families because the families test conceptually distinct claims.

## Consequences

Positive:

- Resolves the v0.3 SPEC vs code drift explicitly.
- The diagnostic signal the v0.3 review flagged (15/20 shadow revisions positive) becomes the headline test (H8a.v4) instead of being hidden post-hoc.
- The classifier framing of H8b.v4 makes the recognition-policy question testable independently of the policy-choice question.

Negative:

- Three new hypotheses inflate the multiple-comparison surface. Mitigated by separating into the family C correction so the cascade-vs-bare family is unaffected.
- The fixed-effects lock removes the random-effects perspective from H5. Documented as a v0.5 add-on if needed.

## Implementation files

- `docs/SPEC_v0.4.md` — H1.v4..H9.v4 frozen; H5 fixed-effects lock; H8 split.
- `benchmarks/stats.py` — implements all H1.v4..H9.v4; emits `allow_nan=False`; family-aware Holm correction; SPEC and code agreed at the close of Phase 4.

## Acceptance gate (Phase 4)

- `benchmarks/stats.py` runs on synthetic mock data and emits all keys for H1.v4..H9.v4.
- SPEC §2 and `benchmarks/stats.py` agree on fixed-effects for H5.v4 (sha256 cross-check).
- Pre-registration tag `pce-v0.4-prereg` pushed.
- `tests/test_stats_emits_all_v04_hypotheses.py` passes (mock data path).

# ADR-002 (v0.3) — Event-gated commit + always-shadow revision

Status: Accepted (frozen during planning round 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.3/C4-vimarsa-event-vs-guarantee.md](../../triz/v0.3/C4-vimarsa-event-vs-guarantee.md).

## Context

The v0.2 cascade was "two-pass-always": every cascade item ran draft -> vimarsa brief -> revision and committed the revision. The v0.2 review correctly noted that vimarsa was therefore a *scaffold*, not a *control*: the `vimarsa_event` boolean did not gate any commit decision. Worse, H8 (revision-vs-draft causal contribution) was pre-registered but not implemented.

The v0.3 design goal is twofold:

1. Make vimarsa a *real* causal control: the commit depends on the event.
2. Keep H8 measurable on every item: the shadow revision must always be scored even when not committed.

## Decision

Rewrite `run_cascade` in [src/pce/cascade.py](../../../src/pce/cascade.py) so it always runs both passes, but the commit decision is gated:

```python
def run_cascade(
    prompt: str,
    constraint: Constraint,
    *,
    lm: GeneratorProtocol,
    embed: Embedder,
    K: int = 4,
    cit_temperature: float = 1.0,
    max_tokens: int = 200,
    base_seed: int = 0,
    aspects: list[str] | None = None,
    retrieval_set: list[str] | None = None,
    commit_policy: Literal["event_gated", "always_revise", "always_draft"] = "event_gated",
    delta_F_threshold: float = 0.05,
    hopfield_store: HopfieldStore | None = None,
    fe_budget: FreeEnergyBudget | None = None,
    ...
) -> CascadeState: ...
```

Pipeline:

1. Pass 1 (draft) always runs (`iccha -> apohana(hopfield_query) -> jnana -> kriya`).
2. `vimarsa(draft, ..., evidence_points=[delta_F_draft])` runs with `return_brief=True`. The event fires when `|delta_F_draft| > delta_F_threshold` AND aspect coverage / novelty / aesthetic gates pass.
3. Pass 2 (shadow revision) always runs unless the free-energy ledger (ADR-005) is underwater. It uses the brief from step 2 to construct a revision prompt. Surface persisted on `state.surface_revision`.
4. `vimarsa.consolidate(state, mode)` writes the committed surface back into the `HopfieldStore` (ADR-004).
5. Commit:
   - `commit_policy="event_gated"` (default): `state.surface = revision if event else draft; state.committed = "revision" if event else "draft"`.
   - `commit_policy="always_revise"`: `state.surface = revision; state.committed = "revision"`. Used by the matched-revision control arm with a generic brief.
   - `commit_policy="always_draft"`: `state.surface = draft; state.committed = "draft"`. Used as an internal sanity / single-pass arm.

The `bypass_vimarsa` flag is dropped in favor of `commit_policy`. The same `run_cascade` entry point now serves three of the four benchmark arms (`haiku_cascade`, `haiku_generic_revise_2pass`, and an optional `always_draft` ablation). The fourth arm (`haiku_bare_2K_scorer`) does not call `run_cascade`; it goes through `iccha + jnana` over Haiku K times with no revision.

`vimarsa` in [src/pce/operators/vimarsa.py](../../../src/pce/operators/vimarsa.py) gains a kwarg `evidence_points: list[float] | None = None`. When `|delta_F_draft|` exceeds `delta_F_threshold`, it counts as one evidence point that contributes to the firing decision.

## Consequences

Positive:

- Vimarsa becomes a *real* causal control: when it does not fire, the user gets the draft (single-pass behavior), not the revision.
- H8 is measurable on every item where the shadow revision runs (i.e., free-energy budget did not abort it). The paired statistic `revision_score - draft_score` for items where event committed revision is now well-defined.
- The four benchmark arms share one cascade entry point with `commit_policy` switching, simplifying the driver and statistics.
- Internal validity: per-arm specificity reports (e.g., what fraction of items committed revision) become easy to compare.

Negative:

- Cost: the cascade pays for two passes on every item (modulo budget aborts), even when the commit is the draft. Mitigation: the free-energy budget (ADR-005) aborts the shadow pass when underwater; abort rate logged on each row.
- The H8 sample size is now `n_event_committed`, which can be smaller than `n_total`. Reported as such in `stats.json`.

## Implementation files (forecast)

- [src/pce/cascade.py](../../../src/pce/cascade.py) — rewrite `run_cascade` per the pipeline above; drop `bypass_vimarsa`; add `commit_policy`, `delta_F_threshold`, `hopfield_store`, `fe_budget` kwargs.
- [src/pce/operators/vimarsa.py](../../../src/pce/operators/vimarsa.py) — add `evidence_points`, hook `consolidate(state, mode)`.
- [src/pce/types.py](../../../src/pce/types.py) — `CascadeState` gains `committed: Literal["draft", "revision"]`, `commit_policy: str`, `delta_F_draft: float`, `fe_budget_state: dict | None`.

## Acceptance gate (Phase 4)

- `revision_differs_from_draft` is true on >= 80% of paired runs on the prove-gate items (duck-rabbit textual + AUT brick).
- The commit decision honors the event in 100% of cases (no silent revision committed when event=False under `event_gated`).
- `tests/cascade_event_gated_test.py` covers all three commit policies.

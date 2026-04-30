# ADR-003 (v0.4) — Free-energy budget as hard gate inside `run_cascade`

Status: Accepted (frozen at end of Phase 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.4/C3-budget-ledger-vs-authority.md](../../triz/v0.4/C3-budget-ledger-vs-authority.md).

## Context

The v0.3 cascade constructs a `FreeEnergyBudget`, calls `earn_jnana(...)`, `earn_tokens(...)`, `earn_aspect(...)`, and `to_audit()`. But it never calls `should_continue_revision()`. The shadow-revision pass always runs whenever `commit_policy ∈ {event_gated, always_revise}`. The v0.3 paper claimed the budget "gates abort/continue decisions" — that claim was not true in the code.

## Decision

`run_cascade` consults `budget.should_continue_revision()` between the draft pass and the shadow-revision pass. When underwater, the cascade commits the draft, sets `state.audit["revision_skipped_reason"] = "fe_budget_underwater"`, sets `state.surface_revision = None`, and returns. When healthy, the cascade runs the shadow-revision pass exactly as in v0.3.

```python
# inside run_cascade, after draft pass + ledger updates + apohana trajectory + vimarsa brief
if not budget.should_continue_revision():
    state.committed = "draft"
    state.audit["revision_skipped_reason"] = "fe_budget_underwater"
    state.audit["fe_budget"] = budget.to_audit()
    state.surface_revision = None
    return state

# else: run shadow revision pass exactly as v0.3
```

The hierarchy is:

1. **Budget gate** (generation-level) — decides whether to *generate* a shadow revision at all.
2. **Commit policy** (selection-level) — decides whether to *commit* an already-generated revision.

The commit policy never overrides a budget abort.

## Floor and tunability

The default `floor = -2.0` is permissive (aborts only when the per-item ledger drops below -2.0 free-energy units in the v0.3 weighting). Env var `PCE_FE_BUDGET_FLOOR` overrides at instantiation time. The Phase-2 prove-gate fixture (`budget_starved`) injects a synthetic high-cost / low-coverage item that triggers the abort, verifying the gate is wired.

## Pre-registration

The abort rate, cost saved, and per-item budget balance are reported on `audit/cost_ledger_v0_4.json` and `state.audit["fe_budget"]`. H8a.v4, H8b.v4, H8c.v4 all *exclude* items where the budget aborted (i.e. `state.surface_revision is None`); the exclusion is reported on the per-hypothesis JSON as `n_excluded_fe_abort`. With the default floor, the v0.4 pilot is expected to abort 0–2 items / domain; the ledger will surface any unexpected abort surge.

## Consequences

Positive:

- The v0.4 paper can honestly say the FE budget is causal in the cascade.
- The two-tier hierarchy (budget gate → commit policy) cleanly separates "should we generate a revision?" from "should we commit an existing revision?".
- Cost savings on items where revision was never going to help — measurable on the cost ledger.

Negative:

- A new failure mode: if the floor is set too high, useful revisions are aborted. Mitigated by conservative default and prove-gate observability.
- H8.v4 hypotheses lose a small number of items to budget aborts; reported honestly.

## Implementation files

- `src/pce/cascade.py` — wire `budget.should_continue_revision()`; emit `revision_skipped_reason`; set `state.surface_revision = None` on abort.
- `tests/test_fe_budget_gating.py` — synthetic budget-starved fixture aborts shadow revision; healthy fixture runs both passes.
- `scripts/prove_gate.py` — adds `budget_starved` fixture assertion.

## Acceptance gate (Phase 2)

- `tests/test_fe_budget_gating.py` passes (abort branch + healthy branch).
- Prove-gate v0.4-α observes ≥ 1 `revision_skipped_reason="fe_budget_underwater"` event on the budget-starved fixture.
- `state.audit["fe_budget"]` snapshot is present on every cascade row regardless of abort outcome.
- Pilot abort rate (Phase 7) reported on `benchmarks/results_v0.4/stats.json` per domain.

# C3 — Free-energy budget: audit ledger vs decision authority

## Contradiction

`FreeEnergyBudget` exists, exposes `should_continue_revision()`, and is updated through the cascade via `earn_jnana`, `earn_tokens`, `earn_aspect`. But `run_cascade()` *never* calls `should_continue_revision()`. The shadow-revision pass always runs whenever `commit_policy ∈ {event_gated, always_revise}`. The discussion section of the v0.3 paper claims the budget "gates abort/continue decisions" — that is not true in the v0.3 code.

- **If we keep the budget as audit-only**, we cannot honestly call it active inference; the paper claim must retract.
- **If we make the budget abort revisions**, we may abort revisions that would have improved the score, and the abort policy becomes another tunable surface.

## Improving / worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 38 — Extent of automation | The system makes its own abort decisions instead of relying on after-the-fact audit. |
| Worsening | 30 — External harm affecting the object | Risk of aborting useful revisions when the floor is set badly. |

## Matrix lookup

`lookup_matrix(38, 30) -> {2, 21, 35, 11}`.

- **2 — Taking out**: separate the abort decision from the budget-tracking; abort is a derived property.
- **21 — Skipping / hurried action**: skip a step that would otherwise be taken.
- **35 — Parameter changes**: tune the floor parameter conservatively; document via env var.
- **11 — Beforehand cushioning**: build a safety margin into the floor.

## Ideal Final Result (IFR)

> The free-energy budget gates the shadow-revision pass when (and only when) the ledger drops below a floor that is conservative enough to never abort items where revision would have helped, while saving compute on items where the draft already covers all aspects.

## Attractor-flow divergent ideation

1. **Strip the budget from v0.4 entirely** — paper retracts the claim, codepath simplifies. *Rejected — review explicitly asked for either wiring or retraction; we choose wiring.*
2. **Make `should_continue_revision()` advisory** (logged only) — same problem v0.3 already had. *Rejected.*
3. **Make `should_continue_revision()` a hard gate** with the existing `floor=-2.0` and emit `revision_skipped_reason="fe_budget_underwater"` into the audit. *Kept (primary resolution).*
4. **Keep the abort soft** — let the commit policy override an abort with `always_revise`. *Rejected — the budget is supposed to gate the *generation* call, not the policy decision over an already-generated revision.*
5. **Tunable floor via env var `PCE_FE_BUDGET_FLOOR`** — already in v0.3, kept; v0.4 adds prove-gate observability.

## Selected resolution

Apply principles **38 (Automation)** and **2 (Taking out)** at the cascade level:

- `run_cascade` calls `budget.should_continue_revision()` between draft and shadow-revision passes.
- When underwater: cascade commits the draft, sets `state.audit["revision_skipped_reason"] = "fe_budget_underwater"`, and reports the saved spend. `state.surface_revision = None`.
- When healthy: cascade runs the shadow-revision pass exactly as in v0.3.
- Phase-2 prove-gate fixture (`budget_starved`) confirms that at least one fixture item triggers the abort. The abort rate is reported on `audit/cost_ledger_v0_4.json` and `state.audit["fe_budget"]`.
- The commit policy never overrides a budget abort. The hierarchy is: budget gate (generation-level) → commit policy (selection-level).

Implementation contract: see [ADR-003 — FE budget gating](../../adr/v0.4/ADR-003-fe-budget-gating.md).

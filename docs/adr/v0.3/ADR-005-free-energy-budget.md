# ADR-005 (v0.3) — Per-item free-energy budget

Status: Accepted (frozen during planning round 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.3/C3-active-inference-vs-cli.md](../../triz/v0.3/C3-active-inference-vs-cli.md).

## Context

The C3 contradiction asks for active-inference rigor *despite* the CLI substrate exposing no logprobs. ADR-003 makes BMR `delta_F` informative through aspect-conditioned reductions. This ADR closes the loop by introducing a per-item free-energy ledger that integrates `delta_F`, embedding-distance error, and per-call cost into a single budget that gates revision.

The result is a real active-inference loop: each operator pays / earns F based on whether its action reduces prediction error against the aspect prior; when the ledger goes underwater, the cascade aborts the next costly action (the shadow revision pass).

## Decision

Add a new module `src/pce/active_inference/budget.py` (and `src/pce/active_inference/__init__.py`) defining:

```python
@dataclass
class FreeEnergyBudget:
    floor: float = -2.0           # abort threshold
    delta_F_weight: float = 1.0
    embed_err_weight: float = 0.5
    token_cost_weight: float = 0.1
    history: list[FreeEnergyTick] = field(default_factory=list)

    def credit_delta_F(self, delta_F: float) -> None: ...
    def debit_embed_error(self, surface_emb, aspect_prior_emb) -> None: ...
    def debit_token_cost(self, n_tokens: int) -> None: ...
    @property
    def balance(self) -> float: ...
    @property
    def underwater(self) -> bool: return self.balance < self.floor
    def snapshot(self) -> dict[str, Any]: ...
```

Tick dataclass:

```python
@dataclass
class FreeEnergyTick:
    op: str               # "draft.jnana" | "draft.embed" | "draft.kriya_tokens" | ...
    delta: float          # signed contribution to balance
    cumulative: float
    detail: dict[str, Any]
```

Wiring in [src/pce/cascade.py](../../../src/pce/cascade.py):

1. `FreeEnergyBudget` instantiated at the top of `run_cascade` (or supplied by caller).
2. After draft `jnana`: `budget.credit_delta_F(delta_F_draft)`.
3. After draft `kriya`: `budget.debit_embed_error(draft_embedding, aspect_prior_embedding)` and `budget.debit_token_cost(draft_token_count)`.
4. *Before* the shadow revision pass: if `budget.underwater`, raise a structured `BudgetUnderwaterError` (caught by the cascade), `state.surface_revision = None`, `state.audit["fe_budget_aborted"] = True`. The cascade still commits the draft and reports the abort on the row.
5. After revision (if it ran): same `credit_delta_F` / `debit_embed_error` / `debit_token_cost` for the revision pass.
6. `state.audit["fe_budget"] = budget.snapshot()` always.

The default `floor=-2.0` is permissive (very few aborts at K=4, max_tokens=200). Env var `PCE_FE_BUDGET_FLOOR` overrides.

H8.v3 only includes items where `state.surface_revision is not None` (i.e., the budget did not abort). Per-row abort logging makes this explicit in the stats.

## Consequences

Positive:

- The active-inference framing is now load-bearing: each operator's contribution is quantified against a single ledger that gates the next costly operator.
- The free-energy budget is the natural place to plug in additional signals later (e.g., real logprobs from a future SDK substrate, calibrated entropy from an ensemble of K candidates).
- Cost control: when `delta_F` is degenerate or aspect coverage is hopeless, the cascade saves a Haiku call by aborting the revision pass.
- The per-row `fe_budget` snapshot makes the active-inference behavior auditable in the paper.

Negative:

- Tunable weights (`delta_F_weight`, `embed_err_weight`, `token_cost_weight`, `floor`) are an extra hyperparameter surface. Defaults frozen by this ADR; overrides via env vars only.
- An aborted shadow revision means H8.v3 has fewer paired observations than `n_total`. Reported honestly.

## Implementation files (forecast)

- [src/pce/active_inference/__init__.py](../../../src/pce/active_inference/__init__.py) — new package.
- [src/pce/active_inference/budget.py](../../../src/pce/active_inference/budget.py) — `FreeEnergyBudget` class and helpers.
- [src/pce/cascade.py](../../../src/pce/cascade.py) — wire budget per the pipeline above; raise / catch `BudgetUnderwaterError`; snapshot to `state.audit`.
- [tests/test_free_energy_budget.py](../../../tests/test_free_energy_budget.py) — new test file: budget arithmetic; abort fires when underwater; snapshot serializes.

## Acceptance gate (Phase 3)

- `tests/test_free_energy_budget.py` passes.
- On the prove-gate fixtures, the cascade either runs both passes (budget healthy) or commits the draft with `fe_budget_aborted=True` (budget underwater); both cases produce a valid `state.audit["fe_budget"]` snapshot.

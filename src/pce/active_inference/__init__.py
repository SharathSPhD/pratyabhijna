"""Active-inference machinery used by the v0.3 cascade.

Three pieces live here:

* :class:`pce.active_inference.hopfield.HopfieldStore` -- the storehouse
  consciousness (ālaya-vijñāna) component the cascade calls into during
  :func:`pce.operators.apohana.apohana` for warm-start aspect priors and that
  :func:`pce.operators.vimarsa.consolidate` writes back to at the end of every
  prompt (REM-style fast write or SWS-style slow consolidation).
* :class:`pce.active_inference.budget.FreeEnergyBudget` -- a per-item ledger
  that pays/earns free energy from (a) ΔF reported by jñāna BMR, (b) embedding
  distance between the surface and the constraint/aspect prior, and (c) the
  committed token count. The cascade aborts the revision pass if the ledger
  drops below the abort threshold.
* The aspect-conditioned BMR reductions live in :mod:`pce.operators.jnana`
  itself (``reduction_target="aspect_conditioned"``) so the BMR posterior and
  the aspect prior live in one place; this package only owns the storehouse
  and the budget that the cascade consults from the outside.

Per ADR-003 / ADR-004 / ADR-005 the storehouse is per-domain, REM-mode is
deterministic-on-input (so cascade runs are reproducible), and the budget
ledger is per-item (no cross-prompt budget bleed).
"""
from pce.active_inference.budget import (
    BudgetExceededError,
    BudgetLedgerEntry,
    FreeEnergyBudget,
)
from pce.active_inference.hopfield import HopfieldStore

__all__ = [
    "BudgetExceededError",
    "BudgetLedgerEntry",
    "FreeEnergyBudget",
    "HopfieldStore",
]

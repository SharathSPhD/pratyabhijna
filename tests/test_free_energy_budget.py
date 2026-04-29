"""Phase 3 ADR-005 gate: per-item free-energy budget ledger.

Verifies that:

* A fresh ledger starts at ``initial_bits`` and ``should_continue_revision`` is True.
* ``earn_jnana`` adds positive ΔF as a credit; clipped at +/- 10 bits.
* ``earn_aspect`` always pays a non-positive cost.
* ``earn_tokens`` always pays a non-positive cost proportional to ``n_tokens``.
* The audit dict is JSON-safe and lists every entry.
* The cascade-side decision threshold ``should_continue_revision`` flips
  False once the balance crosses ``abort_threshold``.
"""
from __future__ import annotations

import json

import pytest

from pce.active_inference.budget import (
    BudgetExceededError,
    BudgetLedgerEntry,
    FreeEnergyBudget,
)


def test_default_construction_starts_with_initial_bits() -> None:
    b = FreeEnergyBudget()
    assert b.balance() == pytest.approx(b.initial_bits)
    assert b.should_continue_revision() is True


def test_earn_jnana_credits_balance() -> None:
    b = FreeEnergyBudget(initial_bits=1.0)
    after = b.earn_jnana(0.5, note="draft pass ΔF")
    assert after == pytest.approx(1.5)
    assert b.entries[-1] == BudgetLedgerEntry(
        source="jnana", bits=0.5, note="draft pass ΔF"
    )


def test_earn_jnana_clipped() -> None:
    b = FreeEnergyBudget(initial_bits=0.0)
    b.earn_jnana(50.0)
    assert b.balance() == pytest.approx(10.0)
    b2 = FreeEnergyBudget(initial_bits=0.0)
    b2.earn_jnana(-50.0)
    assert b2.balance() == pytest.approx(-10.0)


def test_aspect_cost_is_nonpositive() -> None:
    b = FreeEnergyBudget()
    before = b.balance()
    b.earn_aspect(0.4)
    assert b.balance() <= before
    assert b.entries[-1].source == "aspect"
    assert b.entries[-1].bits <= 0.0


def test_token_cost_scales_linearly() -> None:
    b = FreeEnergyBudget(token_cost_bits=0.02)
    before = b.balance()
    b.earn_tokens(50)
    assert b.balance() == pytest.approx(before - 1.0)


def test_should_continue_revision_flips_at_threshold() -> None:
    b = FreeEnergyBudget(initial_bits=0.0, abort_threshold=-1.0)
    assert b.should_continue_revision() is True
    b.add_manual(-0.5)
    assert b.should_continue_revision() is True
    b.add_manual(-1.0)  # now at -1.5, below -1.0 threshold
    assert b.should_continue_revision() is False


def test_audit_dict_is_json_safe() -> None:
    b = FreeEnergyBudget()
    b.earn_jnana(0.2, note="draft")
    b.earn_aspect(0.5)
    b.earn_tokens(120)
    audit = b.to_audit()
    s = json.dumps(audit, allow_nan=False)
    assert '"jnana"' in s
    assert '"aspect"' in s
    assert '"tokens"' in s
    assert '"balance_bits"' in s


def test_budget_exceeded_error_is_raisable() -> None:
    """Caller-facing sentinel exists and is a RuntimeError subclass."""
    err = BudgetExceededError("budget exhausted")
    assert isinstance(err, RuntimeError)


def test_manual_adjustment_recorded() -> None:
    b = FreeEnergyBudget()
    b.add_manual(0.3, note="boost from prior")
    assert b.entries[-1].source == "manual"
    assert b.entries[-1].bits == pytest.approx(0.3)
    assert b.entries[-1].note == "boost from prior"

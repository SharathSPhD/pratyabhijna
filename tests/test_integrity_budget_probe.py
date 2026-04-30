"""v0.4 Phase 2 (ADR-003) gate: IntegrityProbe.probe_budget_abort.

The probe runs a synthetic budget-starved cascade against an in-memory
fake LM and asserts the FE-budget hard gate fires. It must:

* Pass with ``passed=True`` on the canonical starved budget.
* Record ``revision_skipped_reason="fe_budget_underwater"``.
* Commit the draft (``committed="draft"``).
* Leave ``surface_revision is None``.
* Use exactly K_runtime LM calls (one pass only).

This is the v0.4-α gate's evidence that ``FreeEnergyBudget.should_continue_revision()``
is now causally wired into ``run_cascade``.
"""
from __future__ import annotations

from pce.substrate.integrity import IntegrityProbe


def test_probe_budget_abort_passes_under_starved_budget() -> None:
    probe = IntegrityProbe()
    result = probe.probe_budget_abort()
    assert result.passed, f"probe failed; notes={result.notes!r}"
    assert result.revision_skipped is True
    assert result.revision_skipped_reason == "fe_budget_underwater"
    assert result.fe_budget_underwater is True
    assert result.committed == "draft"
    assert result.surface_revision_was_none is True
    assert result.n_lm_calls > 0  # at least the draft pass ran


def test_probe_budget_abort_result_serializes_to_json() -> None:
    """Audit-friendly: result.to_json() returns a plain dict serializable by json.dumps."""
    import json

    probe = IntegrityProbe()
    result = probe.probe_budget_abort()
    payload = result.to_json()
    assert isinstance(payload, dict)
    text = json.dumps(payload)
    assert "fe_budget_underwater" in text
    assert "revision_skipped_reason" in text

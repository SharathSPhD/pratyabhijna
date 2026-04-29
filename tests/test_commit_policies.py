"""v0.4 Phase 3 (ADR-002) gate: commit-policy decide table.

Each policy must return the expected decision on a frozen
:class:`PolicyFeatures` fixture. ``OracleCommit`` requires ``set_scores``
before ``decide`` — calling without scores must raise so the policy can't
silently leak labels at evaluation time.

``LearnedGate`` is exercised in two paths:

* Model loaded -> ``last_proba`` is set; decision is deterministic.
* Model missing -> falls back to ``EventGated`` and surfaces a fallback
  reason so the audit log preserves the regression.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pce.policies.commit import (
    AlwaysDraft,
    AlwaysRevise,
    DEFAULT_LEARNED_GATE_PATH,
    EventGated,
    LearnedGate,
    OracleCommit,
    PolicyFeatures,
    extract_features_from_audit,
    policy_for_name,
)


@pytest.fixture()
def features() -> PolicyFeatures:
    return PolicyFeatures(
        delta_F=0.5,
        novelty=0.8,
        aspect_count=2.0,
        ananda=0.6,
        budget_balance=2.0,
    )


def test_always_draft_returns_false(features: PolicyFeatures) -> None:
    assert AlwaysDraft().decide(features, vimarsa_event=True) is False
    assert AlwaysDraft().decide(features, vimarsa_event=False) is False


def test_always_revise_returns_true(features: PolicyFeatures) -> None:
    assert AlwaysRevise().decide(features, vimarsa_event=True) is True
    assert AlwaysRevise().decide(features, vimarsa_event=False) is True


def test_event_gated_passes_through_event(features: PolicyFeatures) -> None:
    g = EventGated()
    assert g.decide(features, vimarsa_event=True) is True
    assert g.decide(features, vimarsa_event=False) is False


def test_learned_gate_loads_artifact_when_present(features: PolicyFeatures) -> None:
    if not DEFAULT_LEARNED_GATE_PATH.exists():
        pytest.skip("artifact not built; run scripts/train_learned_gate.py")
    g = LearnedGate()
    assert g.is_loaded() is True
    assert g.last_fallback_reason is None
    decision = g.decide(features, vimarsa_event=False)
    assert isinstance(decision, bool)
    assert g.last_proba is not None
    assert 0.0 <= g.last_proba <= 1.0


def test_learned_gate_falls_back_to_event_gated_when_artifact_missing(
    features: PolicyFeatures, tmp_path: Path
) -> None:
    missing = tmp_path / "does-not-exist.joblib"
    g = LearnedGate(model_path=missing)
    assert g.is_loaded() is False
    assert g.last_fallback_reason is not None
    # Falls back to EventGated semantics.
    assert g.decide(features, vimarsa_event=True) is True
    assert g.decide(features, vimarsa_event=False) is False


def test_learned_gate_strict_mode_returns_false_without_model(
    features: PolicyFeatures, tmp_path: Path
) -> None:
    missing = tmp_path / "no.joblib"
    g = LearnedGate(model_path=missing, fallback_event_gated=False)
    assert g.is_loaded() is False
    assert g.decide(features, vimarsa_event=True) is False


def test_oracle_requires_scores_before_decide(features: PolicyFeatures) -> None:
    o = OracleCommit()
    with pytest.raises(RuntimeError):
        o.decide(features, vimarsa_event=False)


def test_oracle_picks_higher_scoring_surface(features: PolicyFeatures) -> None:
    o = OracleCommit()
    o.set_scores(draft_score=0.3, revision_score=0.5)
    assert o.decide(features, vimarsa_event=False) is True
    o.set_scores(draft_score=0.5, revision_score=0.3)
    assert o.decide(features, vimarsa_event=True) is False


def test_policy_for_name_factory() -> None:
    assert isinstance(policy_for_name("always_draft"), AlwaysDraft)
    assert isinstance(policy_for_name("always_revise"), AlwaysRevise)
    assert isinstance(policy_for_name("event_gated"), EventGated)
    assert isinstance(policy_for_name("learned_gate"), LearnedGate)
    assert isinstance(policy_for_name("oracle"), OracleCommit)
    with pytest.raises(ValueError):
        policy_for_name("does_not_exist")
    with pytest.raises(TypeError):
        policy_for_name("event_gated", model_path="x")


def test_extract_features_from_audit_handles_missing_keys() -> None:
    """Sparse v0.3 audits get default fallbacks rather than NaNs."""
    feats = extract_features_from_audit({})
    assert feats.delta_F == 0.0
    assert feats.novelty == 0.0
    assert feats.aspect_count == 0.0
    assert feats.ananda == 0.0
    assert feats.budget_balance == 0.0


def test_extract_features_uses_v0_3_keys() -> None:
    audit = {
        "delta_F_draft": 0.42,
        "novelty": 0.7,
        "vimarsa_diag_draft": {"aspect_count": 2.0, "ananda": 0.6},
        "budget_ledger": {"balance_bits": 1.5},
    }
    feats = extract_features_from_audit(audit)
    assert feats.delta_F == pytest.approx(0.42)
    assert feats.novelty == pytest.approx(0.7)
    assert feats.aspect_count == pytest.approx(2.0)
    assert feats.ananda == pytest.approx(0.6)
    assert feats.budget_balance == pytest.approx(1.5)


def test_policy_features_as_vector_order() -> None:
    feats = PolicyFeatures(
        delta_F=1.0, novelty=2.0, aspect_count=3.0, ananda=4.0, budget_balance=5.0
    )
    assert feats.as_vector() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert PolicyFeatures.feature_names() == [
        "delta_F",
        "novelty",
        "aspect_count",
        "ananda",
        "budget_balance",
    ]

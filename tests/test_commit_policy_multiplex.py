"""Phase 3 gate test for the v0.4 commit-policy multiplex (driver-level).

The driver synthesises four commit-policy arms (``always_draft``,
``always_revise``, ``event_gated``, ``learned_gate``) and one analysis arm
(``oracle``) from a single ``haiku_cascade`` row, with no extra LM cost.
This module exercises that path with a fully synthetic ``item_rows`` dict
so we never touch a real LM.

Gate invariants tested here:

* All four ``COMMIT_POLICY_ARMS_V4`` are populated.
* ``oracle`` is also populated and its composite score is ≥ every other
  policy's composite score on the same item (post-hoc upper bound).
* Every synthesised row carries ``policy_features``, ``score_draft``, and
  ``score_revision`` so downstream stats can replay the decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from benchmarks.driver import (
    COMMIT_POLICY_ARMS_V4,
    ORACLE_ANALYSIS_ARM,
    _multiplex_commit_policies,
)
from pce.substrate.embed import Embedder


@dataclass(frozen=True)
class _FakeScore:
    composite: float
    axes: dict[str, float]


def _make_scorer(
    revision_score: float,
    draft_score: float,
) -> Any:
    """Return a stub scorer that returns ``revision_score`` for the revision
    string and ``draft_score`` for the draft string. Other strings raise so
    we catch accidental calls."""

    def _scorer(text: str, *, item: dict[str, Any], embed: Embedder) -> _FakeScore:
        if text == "REV":
            return _FakeScore(composite=revision_score, axes={"x": revision_score})
        if text == "DRA":
            return _FakeScore(composite=draft_score, axes={"x": draft_score})
        raise AssertionError(f"unexpected text {text!r}")

    return _scorer


@pytest.fixture()
def cascade_meta() -> dict[str, Any]:
    return {
        "ok": True,
        "vimarsa_event": True,
        "vimarsa_event_draft": True,
        "delta_F_draft": 0.1,
        "novelty": 0.7,
        "aspect_count": 2.0,
        "ananda": 0.4,
        "budget_balance": 1.5,
        "surface_draft": "DRA",
        "surface_revision": "REV",
    }


def test_multiplex_populates_all_policy_arms(cascade_meta: dict[str, Any]) -> None:
    item_rows: dict[str, dict[str, Any]] = {
        "item": {"id": "x"},
        "haiku_cascade": {"text": "REV", "composite": 1.2, "meta": cascade_meta},
    }
    embed = Embedder()
    scorer = _make_scorer(revision_score=1.2, draft_score=0.7)

    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )

    for arm in COMMIT_POLICY_ARMS_V4:
        assert arm in item_rows, arm
        row = item_rows[arm]
        assert "composite" in row
        meta = row["meta"]
        assert "policy_features" in meta
        assert "score_draft" in meta
        assert "score_revision" in meta
        assert meta["score_draft"] == pytest.approx(0.7)
        assert meta["score_revision"] == pytest.approx(1.2)

    assert item_rows["haiku_cascade_always_draft"]["text"] == "DRA"
    assert item_rows["haiku_cascade_always_revise"]["text"] == "REV"
    # event_gated commits revision because vimarsa_event_draft=True.
    assert item_rows["haiku_cascade_event_gated"]["text"] == "REV"


def test_multiplex_oracle_dominates_other_policies(
    cascade_meta: dict[str, Any],
) -> None:
    """ADR-002 sanity check: oracle composite ≥ every committed-policy composite."""
    item_rows: dict[str, dict[str, Any]] = {
        "item": {"id": "x"},
        "haiku_cascade": {"text": "REV", "composite": 1.2, "meta": cascade_meta},
    }
    embed = Embedder()
    scorer = _make_scorer(revision_score=1.2, draft_score=0.7)

    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )

    oracle_score = float(item_rows[ORACLE_ANALYSIS_ARM]["composite"])
    for arm in COMMIT_POLICY_ARMS_V4:
        policy_score = float(item_rows[arm]["composite"])
        assert oracle_score >= policy_score - 1e-9, (
            f"oracle({oracle_score}) < {arm}({policy_score})"
        )


def test_multiplex_oracle_picks_draft_when_draft_better(
    cascade_meta: dict[str, Any],
) -> None:
    """When draft scores higher, oracle commits draft and beats always_revise."""
    item_rows: dict[str, dict[str, Any]] = {
        "item": {"id": "x"},
        "haiku_cascade": {"text": "REV", "composite": 0.4, "meta": cascade_meta},
    }
    embed = Embedder()
    scorer = _make_scorer(revision_score=0.4, draft_score=0.9)

    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )

    oracle = item_rows[ORACLE_ANALYSIS_ARM]
    assert oracle["text"] == "DRA"
    assert oracle["composite"] == pytest.approx(0.9)
    # always_revise lost relative to oracle because it took the worse surface.
    assert (
        float(item_rows["haiku_cascade_always_revise"]["composite"])
        < oracle["composite"] + 1e-9
    )


def test_multiplex_skips_when_no_revision(cascade_meta: dict[str, Any]) -> None:
    """If revision is empty, multiplex still emits all four arms by reusing draft."""
    cascade_meta = dict(cascade_meta)
    cascade_meta["surface_revision"] = ""
    item_rows: dict[str, dict[str, Any]] = {
        "item": {"id": "x"},
        "haiku_cascade": {"text": "DRA", "composite": 0.5, "meta": cascade_meta},
    }
    embed = Embedder()
    scorer = _make_scorer(revision_score=float("nan"), draft_score=0.5)

    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )
    for arm in COMMIT_POLICY_ARMS_V4:
        assert arm in item_rows
        row = item_rows[arm]
        # Every policy must commit draft because revision is unavailable.
        assert row["text"] == "DRA"
    # Oracle is *not* synthesised when revision is missing (analysis-only).
    assert ORACLE_ANALYSIS_ARM not in item_rows


def test_multiplex_idempotent(cascade_meta: dict[str, Any]) -> None:
    """Calling the multiplexer twice does not duplicate or overwrite arms."""
    item_rows: dict[str, dict[str, Any]] = {
        "item": {"id": "x"},
        "haiku_cascade": {"text": "REV", "composite": 1.2, "meta": cascade_meta},
    }
    embed = Embedder()
    scorer = _make_scorer(revision_score=1.2, draft_score=0.7)

    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )
    snapshot = {
        arm: dict(item_rows[arm]) for arm in (*COMMIT_POLICY_ARMS_V4, ORACLE_ANALYSIS_ARM)
    }

    # Second call must be a no-op.
    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )

    for arm, row in snapshot.items():
        assert item_rows[arm]["text"] == row["text"]
        assert (
            float(item_rows[arm]["composite"])
            == pytest.approx(float(row["composite"]))
        )


def test_multiplex_emits_finite_composites(cascade_meta: dict[str, Any]) -> None:
    item_rows: dict[str, dict[str, Any]] = {
        "item": {"id": "x"},
        "haiku_cascade": {"text": "REV", "composite": 1.2, "meta": cascade_meta},
    }
    embed = Embedder()
    scorer = _make_scorer(revision_score=1.2, draft_score=0.7)

    _multiplex_commit_policies(
        domain="poetry_gen",
        item={"id": "x"},
        item_rows=item_rows,
        embed=embed,
        scorer=scorer,
    )
    for arm in (*COMMIT_POLICY_ARMS_V4, ORACLE_ANALYSIS_ARM):
        c = item_rows[arm]["composite"]
        assert c is not None and np.isfinite(c)

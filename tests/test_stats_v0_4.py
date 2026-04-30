"""Phase 4 gate (ADR-005): v0.4 hypothesis registry + fixed-effects H5.

Covers the schema invariants of the v0.4 stats pipeline without running
the full pilot:

* All H1.v4-H9.v4 keys are emitted by the synthetic-data path.
* Synthetic payload survives ``json.dumps(..., allow_nan=False)``.
* Fixed-effects pooler agrees with hand-computed inverse-variance pool.
* :class:`Hypothesis` enum covers exactly the 11 v0.4 hypotheses.
* :func:`hypothesis_label` is exhaustive (no ``assert_never``).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Add benchmarks + src to sys.path for direct imports
SRC = REPO_ROOT / "src"
for _p in (str(SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.stats import (  # noqa: E402
    Hypothesis,
    _classifier_metrics,
    _clean_json,
    _meta_aggregate_fixed_effects,
    _synthetic_v0_4_payload,
    hypothesis_label,
)


def test_hypothesis_enum_covers_v0_4_set() -> None:
    expected = {
        "H1.v4",
        "H2.v4",
        "H3.v4",
        "H4.v4",
        "H5.v4",
        "H6.v4",
        "H7.v4",
        "H8a.v4",
        "H8b.v4",
        "H8c.v4",
        "H9.v4",
    }
    assert {h.value for h in Hypothesis} == expected
    assert len(Hypothesis) == 11


def test_hypothesis_label_exhaustive() -> None:
    """All enum members get a non-empty label without raising."""
    for h in Hypothesis:
        label = hypothesis_label(h)
        assert isinstance(label, str)
        assert len(label) > 0


def test_fixed_effects_meta_pool_matches_hand_calc() -> None:
    """Two equally-sized studies with the same g pool back to that g exactly."""
    out = _meta_aggregate_fixed_effects([(0.5, 20), (0.5, 20)])
    assert out["pooled_g"] == pytest.approx(0.5)
    assert out["method"] == "fixed_effects_inverse_variance"
    assert out["n_studies"] == 2
    weights = out["weights"]
    assert sum(weights) == pytest.approx(1.0)


def test_fixed_effects_pool_inverse_variance_dominance() -> None:
    """A larger study should dominate the pool."""
    out = _meta_aggregate_fixed_effects([(0.0, 5), (0.5, 200)])
    pooled = out["pooled_g"]
    assert pooled is not None and pooled > 0.4  # tilted toward the big study


def test_fixed_effects_pool_handles_empty() -> None:
    out = _meta_aggregate_fixed_effects([])
    assert out["pooled_g"] is None
    assert out["n_studies"] == 0


def test_classifier_metrics_handles_empty() -> None:
    import numpy as np

    out = _classifier_metrics(
        y_true=np.zeros(0, dtype=int), y_pred=np.zeros(0, dtype=int)
    )
    # NaN must serialise (downstream _clean_json maps to None) — verify floats only.
    assert out["n"] == 0
    for k in ("accuracy", "precision", "recall", "f1"):
        v = out[k]
        assert isinstance(v, float)


def test_classifier_metrics_perfect() -> None:
    import numpy as np

    out = _classifier_metrics(
        y_true=np.array([1, 1, 0, 0]),
        y_pred=np.array([1, 1, 0, 0]),
    )
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)


def test_synthetic_payload_emits_all_v0_4_keys() -> None:
    args = argparse.Namespace(
        treatment="haiku_cascade",
        control="haiku_bare",
        control_2K="haiku_bare_2K_scorer",
        control_generic="haiku_generic_revise_2pass",
        seed=4242,
        n_permutations=1000,
        n_bootstrap=500,
    )
    payload = _synthetic_v0_4_payload(args)
    expected_top = {
        "config",
        "primary",
        "H5",
        "H6_v4_extra_compute",
        "H6_v4_extra_compute_meta",
        "H7_v4_generic_revise",
        "H7_v4_generic_revise_meta",
        "H8a_v4_shadow_revision_vs_draft",
        "H8b_v4_gate_calibration",
        "H8c_v4_commit_policy_comparison",
        "H9_v4_judge_proxy_agreement",
        "arm_means_per_domain",
    }
    assert set(payload.keys()) == expected_top
    assert payload["config"]["status"] == "synthetic"
    assert payload["config"]["hypotheses"] == [h.value for h in Hypothesis]


def test_synthetic_payload_survives_strict_json_serialisation() -> None:
    """``_clean_json`` -> ``allow_nan=False`` is the contract the CLI uses; we
    test the same composition (raw payload may contain NaN from empty classifier
    arrays, but ``_clean_json`` maps those to ``None`` before the writer)."""
    args = argparse.Namespace(
        treatment="haiku_cascade",
        control="haiku_bare",
        control_2K="haiku_bare_2K_scorer",
        control_generic="haiku_generic_revise_2pass",
        seed=4242,
        n_permutations=1000,
        n_bootstrap=500,
    )
    payload = _synthetic_v0_4_payload(args)
    cleaned = _clean_json(payload)
    blob = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in blob and "Infinity" not in blob


def test_h5_synthetic_uses_fixed_effects_method_string() -> None:
    args = argparse.Namespace(
        treatment="haiku_cascade",
        control="haiku_bare",
        control_2K="haiku_bare_2K_scorer",
        control_generic="haiku_generic_revise_2pass",
        seed=4242,
        n_permutations=1000,
        n_bootstrap=500,
    )
    payload = _synthetic_v0_4_payload(args)
    h5 = payload["H5"]
    assert h5["method"] == "fixed_effects_inverse_variance"
    assert h5["name"] == "H5.v4"
    assert "per_domain_g" in h5
    assert "per_domain_n" in h5


def test_synthetic_cli_writes_strict_json(tmp_path: Path) -> None:
    """End-to-end: CLI synthesises payload and writes valid JSON file."""
    out = tmp_path / "stats.json"
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "stats.py"),
            "--version",
            "v0.4",
            "--synthetic",
            "--out",
            str(out),
            "--results-dir",
            str(tmp_path),  # empty -> forces synthetic
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["version"] == "v0.4"
    assert payload["config"]["status"] == "synthetic"
    # Sanity: every numeric leaf is finite OR None (no NaN literals leaked).
    blob = out.read_text(encoding="utf-8")
    assert "NaN" not in blob
    assert "Infinity" not in blob
    # Hypothesis registry round-trips.
    assert payload["config"]["hypotheses"] == [h.value for h in Hypothesis]


def test_h8c_leader_board_synthetic_is_empty() -> None:
    args = argparse.Namespace(
        treatment="haiku_cascade",
        control="haiku_bare",
        control_2K="haiku_bare_2K_scorer",
        control_generic="haiku_generic_revise_2pass",
        seed=4242,
        n_permutations=1000,
        n_bootstrap=500,
    )
    payload = _synthetic_v0_4_payload(args)
    h8c = payload["H8c_v4_commit_policy_comparison"]
    assert h8c["leader_board"] == []
    assert h8c["pairwise_p"] == {}


def test_h9_synthetic_status_missing() -> None:
    args = argparse.Namespace(
        treatment="haiku_cascade",
        control="haiku_bare",
        control_2K="haiku_bare_2K_scorer",
        control_generic="haiku_generic_revise_2pass",
        seed=4242,
        n_permutations=1000,
        n_bootstrap=500,
    )
    payload = _synthetic_v0_4_payload(args)
    assert payload["H9_v4_judge_proxy_agreement"]["status"] == "missing"
    assert payload["H9_v4_judge_proxy_agreement"]["supported"] is False


def test_fixed_effects_ci_brackets_pooled_g() -> None:
    """The 95% CI must straddle (or include) the pooled estimate."""
    out = _meta_aggregate_fixed_effects([(0.5, 25), (0.4, 30), (0.6, 40)])
    pooled = out["pooled_g"]
    lo, hi = out["ci_95"]
    assert pooled is not None and lo is not None and hi is not None
    assert lo < pooled < hi
    assert math.isfinite(lo) and math.isfinite(hi)

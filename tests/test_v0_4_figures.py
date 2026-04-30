"""Smoke tests for the v0.4 figure pack.

Each test runs the actual matplotlib rendering against the committed
``benchmarks/results_v0.4`` artefacts and asserts that the expected PNG
exists and is non-trivial. We do not pixel-compare; we only enforce the
gate "figure renders, file exists, file > 4 KB".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks import figures as F

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results_v0.4"
AUDIT = REPO_ROOT / "audit" / "v0.4"
STATS = RESULTS / "stats.json"


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("v04_figs")
    return d


def _has_artefacts() -> bool:
    return STATS.exists() and (RESULTS / "judge.jsonl").exists()


pytestmark = pytest.mark.skipif(
    not _has_artefacts(),
    reason="v0.4 Phase 7 result tree absent; skipping figure render smoke tests",
)


def _assert_real_png(p: Path) -> None:
    assert p.exists(), f"figure not emitted: {p}"
    size = p.stat().st_size
    assert size > 4096, f"figure suspiciously small ({size} B): {p}"
    head = p.read_bytes()[:8]
    assert head[:4] == b"\x89PNG", f"not a real PNG: {p} (header={head!r})"


def test_h5_fixed_forest(out_dir: Path) -> None:
    F._figure_v04_h5_fixed_forest(STATS, out_dir)
    _assert_real_png(out_dir / "fig_v04_h5_fixed_forest.png")


def test_h8a_revision_vs_draft(out_dir: Path) -> None:
    F._figure_v04_h8a_revision_vs_draft(RESULTS, STATS, out_dir)
    _assert_real_png(out_dir / "fig_v04_h8a_revision_vs_draft.png")


def test_h8b_gate_calibration(out_dir: Path) -> None:
    F._figure_v04_h8b_gate_calibration(STATS, out_dir)
    _assert_real_png(out_dir / "fig_v04_h8b_gate_calibration.png")


def test_h8c_policy_leaderboard(out_dir: Path) -> None:
    F._figure_v04_h8c_policy_leaderboard(STATS, out_dir)
    _assert_real_png(out_dir / "fig_v04_h8c_policy_leaderboard.png")


def test_h9_judge_scatter(out_dir: Path) -> None:
    F._figure_v04_h9_judge_scatter(
        RESULTS / "judge.jsonl",
        RESULTS / "judge_agreement.json",
        out_dir,
    )
    _assert_real_png(out_dir / "fig_v04_h9_judge_scatter.png")


def test_cost_per_domain(out_dir: Path) -> None:
    F._figure_v04_cost_per_domain(AUDIT, out_dir)
    _assert_real_png(out_dir / "fig_v04_cost_per_domain.png")

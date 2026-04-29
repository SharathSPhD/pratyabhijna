"""End-to-end pipeline tests over synthetic results.

These tests cover stats.py + figures.py + autoreport.py without invoking the
benchmark driver itself (which requires loading the LLM and is exercised by
the live Phase 9 audit).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def synth_results(tmp_path: Path) -> Path:
    """Generate a synthetic results directory in a tmpdir."""
    out = tmp_path / "results"
    rc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "synthesize_results.py"), "--out-dir", str(out)],
        capture_output=True, check=True,
    )
    assert rc.returncode == 0
    assert (out / "poetry_gen.json").exists()
    return out


def test_stats_runs_and_produces_six_hypotheses(synth_results: Path, tmp_path: Path) -> None:
    out = tmp_path / "stats.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "stats.py"),
         "--results-dir", str(synth_results),
         "--out", str(out),
         "--n-permutations", "5000",
         "--n-bootstrap", "2000"],
        capture_output=True, check=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    primary = data["primary"]
    for h in ("H1", "H2", "H3", "H4"):
        assert h in primary
        for key in ("n", "estimate", "hedges_g", "bca_ci_95",
                    "permutation_p_one_sided", "wilcoxon_p_one_sided",
                    "holm_p", "power_apriori", "power_retrospective", "supported"):
            assert key in primary[h], f"{h} missing {key}"
    # H5.v3 = pooled effect-size meta-aggregation under primary
    assert "H5" in primary or "H5" in data
    # v0.3 contrasts replace v0.2 names.
    assert "H6_v3_extra_compute" in data, "H6_v3_extra_compute missing (v0.3 fairness vs +K compute)"
    assert "H7_v3_generic_revise" in data, "H7_v3_generic_revise missing (v0.3 generic 2-pass control)"
    assert "H8_v3_revision_vs_draft" in data, "H8_v3_revision_vs_draft missing (within-cascade)"


def test_figures_produces_six_pngs(synth_results: Path, tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    fig_dir = tmp_path / "figs"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "stats.py"),
         "--results-dir", str(synth_results),
         "--out", str(stats_path),
         "--n-permutations", "5000",
         "--n-bootstrap", "1000"],
        check=True, capture_output=True,
    )
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "figures.py"),
         "--results-dir", str(synth_results),
         "--stats", str(stats_path),
         "--out-dirs", str(fig_dir)],
        capture_output=True, check=True,
    )
    assert proc.returncode == 0
    for fname in (
        "fig_per_domain_box.png", "fig_paired_deltas.png",
        "fig_effects_forest.png", "fig_h6_event_vs_no_event.png",
        "fig_power.png", "fig_axes_breakdown.png",
    ):
        assert (fig_dir / fname).exists(), f"missing {fname}"
        assert (fig_dir / fname).stat().st_size > 1000


def test_autoreport_substitutes_placeholders(synth_results: Path, tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    paper_dir = tmp_path / "paper"
    shutil.copytree(REPO_ROOT / "paper", paper_dir)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "stats.py"),
         "--results-dir", str(synth_results),
         "--out", str(stats_path),
         "--n-permutations", "2000",
         "--n-bootstrap", "1000"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "autoreport.py"),
         "--stats", str(stats_path),
         "--paper-dir", str(paper_dir),
         "--items", str(REPO_ROOT / "benchmarks" / "items.py")],
        check=True, capture_output=True,
    )
    autoreport = (paper_dir / "autoreport.tex").read_text()
    assert "\\begin{tabular}" in autoreport
    assert "H1" in autoreport
    assert "H2" in autoreport
    assert "H3" in autoreport
    assert "H4" in autoreport
    main_tex = (paper_dir / "main.tex").read_text()
    # Placeholders must have been replaced
    for ph in ("{N_PAIRED}", "{HEADLINE_RESULT}",
               "{POETRY_GEN_N}", "{POETRY_INTERP_N}", "{AUT_N}", "{SCI_N}"):
        assert ph not in main_tex, f"unreplaced placeholder: {ph}"
    # No stray garbled \text command
    assert "\\text{POETRY" not in main_tex

"""v0.4.3 paper polish: the 5 TikZ flowcharts ship as PNGs on both sides.

``scripts/build_paper_flowchart_pngs.py`` exports each ``F<N>_*.png`` into
both:

* ``paper/figures/v0.4/flowcharts/`` — paper-side mirror so a future PDF
  build that wants the raster (e.g. for IEEE camera-ready) has the file.
* ``docs/site/public/figures/v0.4/flowcharts/`` — site-side asset that
  the placeholder figures on architecture / background / results /
  hypotheses pages auto-swap into.

This test verifies every expected slug exists at both paths and is large
enough not to be a byte-zero artefact.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO / "paper" / "figures" / "v0.4" / "flowcharts"
SITE_DIR = REPO / "docs" / "site" / "public" / "figures" / "v0.4" / "flowcharts"

FLOWCHART_PNGS = [
    "F1_panchashakti_cascade.png",
    "F2_active_inference_loop.png",
    "F3_commit_policy_multiplexer.png",
    "F4_phase7_pipeline.png",
    "F5_hypothesis_tree.png",
]

MIN_SIZE_BYTES = 5_000


def test_paper_flowchart_pngs_exist_in_paper_tree() -> None:
    for name in FLOWCHART_PNGS:
        p = PAPER_DIR / name
        assert p.exists(), f"missing paper-tree flowchart PNG: {p.relative_to(REPO)}"
        assert (
            p.stat().st_size >= MIN_SIZE_BYTES
        ), f"flowchart PNG suspiciously small: {p.relative_to(REPO)} ({p.stat().st_size}b)"


def test_paper_flowchart_pngs_exist_in_site_tree() -> None:
    for name in FLOWCHART_PNGS:
        p = SITE_DIR / name
        assert p.exists(), f"missing site-tree flowchart PNG: {p.relative_to(REPO)}"
        assert (
            p.stat().st_size >= MIN_SIZE_BYTES
        ), f"site flowchart PNG suspiciously small: {p.relative_to(REPO)} ({p.stat().st_size}b)"

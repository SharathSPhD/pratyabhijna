"""v0.4.2 content-expansion: paper visuals are wired into the sources.

Asserts that:

* The five v0.4.2 TikZ flowcharts (F1–F5) exist as ``\\begin{figure*}``
  blocks with TikZ markers in the expected paper sections.
* The four LaTeX table snippets (T1–T4) exist on disk under
  ``paper/sections/_tables/`` and are ``\\input{}``-loaded from the right
  paper sections.
* The two new matplotlib chart PNGs (C1, C2) exist on disk under
  ``paper/figures/v0.4/``.

This is an artefact-existence test; it does not re-run TikZ compilation
or rebuild the matplotlib charts.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
SECTIONS = PAPER / "sections"
TABLES = SECTIONS / "_tables"
FIGS = PAPER / "figures" / "v0.4"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _has_tikz_block(tex_path: Path) -> bool:
    text = _read(tex_path)
    if r"\begin{tikzpicture}" not in text and r"\begin{tikzcd}" not in text:
        return False
    return r"\begin{figure*}" in text or r"\begin{figure}" in text


# ---------------------------------------------------------------------------
# F1 – F5: TikZ flowcharts in their canonical sections
# ---------------------------------------------------------------------------


def test_F1_panchashakti_cascade_in_pratyabhijna_section() -> None:
    tex = SECTIONS / "03_pratyabhijna_background.tex"
    assert _has_tikz_block(tex), f"F1 (5-śakti cascade) TikZ block missing in {tex}"


def test_F2_active_inference_loop_in_active_inference_section() -> None:
    tex = SECTIONS / "04_active_inference_background.tex"
    assert _has_tikz_block(tex), f"F2 (active-inference / BMR loop) TikZ block missing in {tex}"


def test_F3_F4_F5_in_methods_section() -> None:
    tex = SECTIONS / "07_methods.tex"
    text = _read(tex)
    assert r"\begin{tikzpicture}" in text, "methods.tex has no TikZ block at all"
    n_tikz = text.count(r"\begin{tikzpicture}")
    assert n_tikz >= 3, (
        f"methods.tex must contain ≥3 TikZ blocks (F3 commit-policy, F4 Phase 7 pipeline, "
        f"F5 hypothesis tree); found {n_tikz}"
    )


# ---------------------------------------------------------------------------
# T1 – T4: LaTeX table snippets
# ---------------------------------------------------------------------------


def test_table_snippets_exist_on_disk() -> None:
    for fname in (
        "tab_per_axis_effects.tex",
        "tab_per_domain_raw.tex",
        "tab_cost_split.tex",
        "tab_showcase_registry.tex",
    ):
        p = TABLES / fname
        assert p.exists(), f"table snippet missing: {p.relative_to(REPO)}"
        text = p.read_text(encoding="utf-8")
        assert r"\begin{tabular}" in text, f"{fname} does not contain a tabular env"
        assert r"\caption{" in text, f"{fname} does not contain a caption"


def test_results_section_inputs_T1_T2_T3() -> None:
    tex = SECTIONS / "09_results.tex"
    text = _read(tex)
    for snippet in (
        "tab_per_axis_effects",
        "tab_per_domain_raw",
        "tab_cost_split",
    ):
        assert snippet in text, f"results section does not \\input{{{snippet}}}"


def test_showcase_section_inputs_T4() -> None:
    tex = SECTIONS / "10c_showcase_examples.tex"
    text = _read(tex)
    assert "tab_showcase_registry" in text, "showcase section does not \\input{tab_showcase_registry}"


# ---------------------------------------------------------------------------
# C1 – C2: matplotlib charts on disk
# ---------------------------------------------------------------------------


def test_C1_C2_chart_pngs_present() -> None:
    for fname in (
        "fig_v04_axes_breakdown.png",
        "fig_v04_power_vs_realised.png",
    ):
        p = FIGS / fname
        assert p.exists(), f"chart PNG missing: {p.relative_to(REPO)}"
        assert p.stat().st_size > 5_000, f"chart PNG suspiciously small: {p}"


def test_results_section_includegraphics_C1_C2() -> None:
    tex = SECTIONS / "09_results.tex"
    text = _read(tex)
    for needle in ("fig_v04_axes_breakdown", "fig_v04_power_vs_realised"):
        assert needle in text, f"results section does not \\includegraphics {needle}"

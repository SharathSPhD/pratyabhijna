"""v0.4.2 content-expansion: site-level active-inference content is present.

Asserts:

* ``docs/site/src/pages/background.astro`` contains the expanded
  active-inference section with at least four ``<p>`` tags discussing
  Friston's free-energy framework, the variational identity, BMR-as-apohana,
  and the recognition framing.
* ``docs/site/src/pages/architecture.astro`` contains the new
  "What 'computation' looks like" section heading and the F1 cascade
  flowchart placeholder.
* Both pages cross-link each other.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE_PAGES = REPO / "docs" / "site" / "src" / "pages"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_background_active_inference_section_expanded() -> None:
    text = _read(SITE_PAGES / "background.astro")
    assert text, "background.astro missing"
    assert "Active inference and Bayesian Model Reduction" in text, (
        "background.astro missing the Active-inference / BMR section heading"
    )
    # Locate the section block: from the heading to the next <h2> or end.
    m = re.search(r'id="active-inference"', text)
    assert m, "background.astro active-inference heading should have id='active-inference'"

    # Slice from the active-inference heading onward to find the next h2 boundary.
    start = m.start()
    next_h2 = text.find("<h2", start + 1)
    if next_h2 == -1:
        next_h2 = len(text)
    block = text[start:next_h2]
    p_tags = block.count("<p>")
    assert p_tags >= 4, (
        f"active-inference section should have ≥4 <p> tags; found {p_tags}"
    )
    # Topical anchors that prove the content is substantive, not just stub-padding.
    for needle in (
        "free energy",
        "BMR",
        "apohana",
        "ADR-003",
    ):
        assert needle in block, f"active-inference section missing topical anchor: {needle}"


def test_background_embeds_active_inference_flowchart_placeholder() -> None:
    text = _read(SITE_PAGES / "background.astro")
    assert "PlaceholderFigure" in text, "background.astro should import PlaceholderFigure"
    assert "F2_active_inference_loop" in text or "F2-active-inference" in text, (
        "background.astro should reference the F2 active-inference flowchart"
    )


def test_architecture_what_computation_section_present() -> None:
    text = _read(SITE_PAGES / "architecture.astro")
    assert text, "architecture.astro missing"
    assert "What \"computation\" looks like" in text or "what-computation-looks-like" in text, (
        "architecture.astro missing the new 'What computation looks like' section"
    )
    assert "PlaceholderFigure" in text, "architecture.astro should import PlaceholderFigure"
    assert "F1_panchashakti_cascade" in text or "F1-cascade" in text, (
        "architecture.astro should reference the F1 5-śakti cascade flowchart"
    )


def test_pages_cross_link_each_other() -> None:
    bg = _read(SITE_PAGES / "background.astro")
    arch = _read(SITE_PAGES / "architecture.astro")
    assert "/architecture" in bg, "background.astro should link to /architecture"
    assert "/background" in arch, "architecture.astro should link to /background"

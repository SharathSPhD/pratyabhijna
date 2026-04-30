"""v0.4.2 content-expansion: docs/figures/PROMPTS.md is shippable.

Asserts:

* ``docs/figures/PROMPTS.md`` exists with the required section headings.
* It documents at least seven figure entries (1 hero + F1–F5 + C1–C2 minimum).
* It names the recommended Pratyabhijñā root verse for the hero image.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "docs" / "figures" / "PROMPTS.md"


def test_prompts_md_exists_and_substantial() -> None:
    assert PROMPTS.exists(), f"docs/figures/PROMPTS.md missing"
    text = PROMPTS.read_text(encoding="utf-8")
    assert len(text) > 2000, f"PROMPTS.md suspiciously small ({len(text)}b)"


def test_prompts_md_has_required_sections() -> None:
    text = PROMPTS.read_text(encoding="utf-8")
    for needle in (
        "## How to use this file",
        "## Hero image",
        "## Figure prompts",
    ):
        assert needle in text, f"PROMPTS.md missing required section: {needle!r}"


def test_prompts_md_documents_hero_verse() -> None:
    text = PROMPTS.read_text(encoding="utf-8")
    # The hero verse should be Utpaladeva's Īśvarapratyabhijñākārikā I.1.1
    # (transliteration of "kathaṃcid āsādya maheśvarasya...").
    assert "Īśvarapratyabhijñākārikā" in text or "Isvarapratyabhijnakarika" in text, (
        "PROMPTS.md should name Īśvarapratyabhijñākārikā as the hero verse source"
    )
    assert "Utpaladeva" in text, "PROMPTS.md should name Utpaladeva as the hero verse author"


def test_prompts_md_lists_seven_figure_slots() -> None:
    text = PROMPTS.read_text(encoding="utf-8")
    # Look for the F1–F5 + C1 + C2 anchors plus the hero entry.
    needed = ("F1", "F2", "F3", "F4", "F5", "C1", "C2", "Hero")
    missing = [n for n in needed if n not in text]
    assert not missing, f"PROMPTS.md missing figure anchors: {missing}"


def test_prompts_md_documents_canonicality() -> None:
    text = PROMPTS.read_text(encoding="utf-8")
    # Should explicitly say which figures are TikZ-canonical, matplotlib-
    # canonical, and AI-generated.
    assert "TikZ" in text, "PROMPTS.md should document TikZ-canonical sources"
    assert "matplotlib" in text, "PROMPTS.md should document matplotlib-canonical sources"

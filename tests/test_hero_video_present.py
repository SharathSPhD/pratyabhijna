"""v0.4.3 paper polish: hero video is committed and wired into the site.

Asserts that:

* ``docs/site/public/figures/v0.4/hero.mp4`` exists and is at least 1 MB
  (the committed file is ~4.4 MB; the threshold guards against
  accidentally committing a placeholder stub).
* ``docs/site/src/pages/index.astro`` references both ``videoSrc`` (the
  ``PlaceholderFigure`` prop that turns the hero into a video) and the
  ``hero.mp4`` filename, so a future refactor that drops either side
  fails loudly.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERO_VIDEO = REPO / "docs" / "site" / "public" / "figures" / "v0.4" / "hero.mp4"
HERO_POSTER = REPO / "docs" / "site" / "public" / "figures" / "v0.4" / "hero.png"
INDEX_ASTRO = REPO / "docs" / "site" / "src" / "pages" / "index.astro"

MIN_VIDEO_SIZE = 1_000_000  # 1 MB


def test_hero_video_committed_to_site_public() -> None:
    assert HERO_VIDEO.exists(), f"hero video missing at {HERO_VIDEO.relative_to(REPO)}"
    size = HERO_VIDEO.stat().st_size
    assert size >= MIN_VIDEO_SIZE, (
        f"hero video suspiciously small ({size}b < {MIN_VIDEO_SIZE}b); "
        "expected the full ~4.4 MB committed asset"
    )


def test_hero_poster_present() -> None:
    assert HERO_POSTER.exists(), (
        f"hero poster (autoplay-blocked fallback) missing at "
        f"{HERO_POSTER.relative_to(REPO)}"
    )


def test_index_astro_references_hero_video() -> None:
    assert INDEX_ASTRO.exists(), f"index.astro missing at {INDEX_ASTRO.relative_to(REPO)}"
    text = INDEX_ASTRO.read_text(encoding="utf-8")
    assert "videoSrc" in text, "index.astro must pass `videoSrc` to PlaceholderFigure"
    assert "hero.mp4" in text, "index.astro must reference hero.mp4"

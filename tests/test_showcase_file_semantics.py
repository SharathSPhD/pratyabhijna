"""File-semantics regression for the v0.4 showcase artefacts.

The v0.4.1 amend made the trace-vs-file invariant explicit: each surface
written to disk under ``benchmarks/showcase_v0.4/<slug>/`` must equal
the corresponding key in ``trace.json``. Specifically:

* ``draft.txt`` equals ``trace["draft"]``.
* ``shadow_revision.txt`` equals ``trace["shadow_revision"]``.
* ``revised.txt`` equals ``trace["shadow_revision"]`` (the curated/legacy
  filename for the shadow-revision surface; the file exists for backwards
  compatibility with the v0.4 site components).
* ``committed.txt`` equals ``trace["committed"]``, and that committed
  surface equals either the draft or the shadow-revision according to
  ``trace["committed_choice"]``.

This is the test the v0.4 adversarial review flagged as missing under
``tests/test_showcase_file_semantics.py``. It runs without launching the
cascade: it only inspects committed artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_ROOT = REPO_ROOT / "benchmarks" / "showcase_v0.4"


def _slugs() -> list[str]:
    if not SHOWCASE_ROOT.exists():
        return []
    return sorted(p.name for p in SHOWCASE_ROOT.iterdir() if p.is_dir())


pytestmark = pytest.mark.skipif(
    not SHOWCASE_ROOT.exists() or not _slugs(),
    reason="benchmarks/showcase_v0.4 not present",
)


@pytest.fixture(scope="module")
def slugs() -> list[str]:
    return _slugs()


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_draft_txt_equals_trace_draft(slugs: list[str]) -> None:
    for slug in slugs:
        d = SHOWCASE_ROOT / slug
        trace = json.loads(_read(d / "trace.json"))
        on_disk = _read(d / "draft.txt")
        assert on_disk == trace.get("draft", ""), f"{slug}: draft.txt != trace.draft"


def test_shadow_revision_txt_equals_trace_shadow(slugs: list[str]) -> None:
    for slug in slugs:
        d = SHOWCASE_ROOT / slug
        trace = json.loads(_read(d / "trace.json"))
        on_disk = _read(d / "shadow_revision.txt")
        assert on_disk == trace.get("shadow_revision", ""), (
            f"{slug}: shadow_revision.txt != trace.shadow_revision"
        )


def test_revised_txt_equals_trace_shadow(slugs: list[str]) -> None:
    """``revised.txt`` is the legacy filename the v0.4 site reads as the
    revised surface; it must equal the same shadow-revision string as
    ``shadow_revision.txt``."""
    for slug in slugs:
        d = SHOWCASE_ROOT / slug
        trace = json.loads(_read(d / "trace.json"))
        on_disk = _read(d / "revised.txt")
        assert on_disk == trace.get("shadow_revision", ""), (
            f"{slug}: revised.txt != trace.shadow_revision"
        )


def test_committed_txt_equals_trace_committed(slugs: list[str]) -> None:
    for slug in slugs:
        d = SHOWCASE_ROOT / slug
        trace = json.loads(_read(d / "trace.json"))
        on_disk = _read(d / "committed.txt")
        assert on_disk == trace.get("committed", ""), (
            f"{slug}: committed.txt != trace.committed"
        )


def test_committed_matches_committed_choice(slugs: list[str]) -> None:
    """``trace.committed`` must equal either ``trace.draft`` or
    ``trace.shadow_revision`` according to ``trace.committed_choice``."""
    for slug in slugs:
        d = SHOWCASE_ROOT / slug
        trace = json.loads(_read(d / "trace.json"))
        choice = trace.get("committed_choice")
        committed = trace.get("committed", "")
        assert choice in ("draft", "revision"), f"{slug}: unknown committed_choice {choice!r}"
        if choice == "draft":
            assert committed == trace.get("draft", ""), (
                f"{slug}: committed != draft despite committed_choice=draft"
            )
        else:
            assert committed == trace.get("shadow_revision", ""), (
                f"{slug}: committed != shadow_revision despite committed_choice=revision"
            )
